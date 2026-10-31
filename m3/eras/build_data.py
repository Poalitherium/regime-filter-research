"""Build the eras study's proxy dataset (PREREG section 3) into data/eras/.

    uv run --with xlrd python -m m3.eras.build_data       # xlrd reads EIA's .xls

Sources (all public, no key):
    Yahoo Finance            equity indices and the S&P GSCI (requested to 2014-12-31)
    Federal Reserve H.10     daily FX (bulk file Output.aspx?rel=H10&filetype=zip)
    Federal Reserve H.15     daily Treasury constant-maturity yields and 3-month bill
    EIA                      WTI spot price, daily (RWTCd.xls)

The Fed and EIA files cover dates after 2014; those rows are dropped in memory as the
files are parsed and are never saved or used (PREREG: the hold-out stays closed).

Every series becomes a data/final-style file: date, close, close_adj, ret, volume, roll,
kind. `close` = `close_adj` = an index of cumulative returns starting at 100, `roll` is
always False (no contracts), `kind` = proxy. Raw values (to 2014-12-31) go to
data/raw/eras/, and data/eras/BUILD_LOG.md records rows, gaps and every removed bad tick.
Moves the bad-tick rule flags but that are verified real (VERIFIED_REAL) are kept:
deviation #1 in m3/eras/DEVIATIONS.md. BUILD_LOG.md keeps the command and paths the
committed data was built with (python -m eras.build_data, eras/DEVIATIONS.md).
"""

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

END = pd.Timestamp("2014-12-31")
CORE_START = pd.Timestamp("1973-04-02")  # generalised floating of JPY, CHF, GBP
UA = {"User-Agent": "Mozilla/5.0"}
FED = "https://www.federalreserve.gov/datadownload/Output.aspx?rel={rel}&filetype=zip"
EIA_WTI = "https://www.eia.gov/dnav/pet/hist_xls/RWTCd.xls"
TBILL = "RIFSGFSM03_N.B"
MISSING = -9999.0  # the Fed's code for a day with no data (ND) in its bulk XML files
# Deviation #1 (m3/eras/DEVIATIONS.md): real market moves that the bad-tick rule flags because
# they reversed sharply the next day. Kept, not removed. Decided before any strategy was run.
VERIFIED_REAL = {("NKY", "1987-10-20"),   # Nikkei 225, October 1987 crash in Tokyo
                 ("DAX", "1989-10-16")}   # DAX, October 1989 mini-crash

# code: (name, asset class, pool, source, series, transform, start)
UNIVERSE = {
    "SPX": ("S&P 500", "equity", "core", "yahoo", "^GSPC", "price", CORE_START),
    "NKY": ("Nikkei 225", "equity", "core", "yahoo", "^N225", "price", CORE_START),
    "US5": ("US 5-year Treasury", "rates", "core", "fed_h15", "RIFLGFCY05_N.B", "bond5", CORE_START),
    "US10": ("US 10-year Treasury", "rates", "core", "fed_h15", "RIFLGFCY10_N.B", "bond10", CORE_START),
    "JPY": ("Japanese yen", "fx", "core", "fed_h10", "RXI_N.B.JA", "fx_invert", CORE_START),
    "CHF": ("Swiss franc", "fx", "core", "fed_h10", "RXI_N.B.SZ", "fx_invert", CORE_START),
    "GBP": ("British pound", "fx", "core", "fed_h10", "RXI$US_N.B.UK", "fx", CORE_START),
    "CAD": ("Canadian dollar", "fx", "core", "fed_h10", "RXI_N.B.CA", "fx_invert", CORE_START),
    "FTSE": ("FTSE 100", "equity", "extended", "yahoo", "^FTSE", "price", None),
    "DAX": ("DAX", "equity", "extended", "yahoo", "^GDAXI", "price", None),
    "HSI": ("Hang Seng", "equity", "extended", "yahoo", "^HSI", "price", None),
    "AUD": ("Australian dollar", "fx", "extended", "fed_h10", "RXI$US_N.B.AL", "fx", pd.Timestamp("1984-01-02")),
    "WTI": ("WTI crude oil spot", "energy", "extended", "eia", "RWTC", "price", None),
    "GSCI": ("S&P GSCI", "energy", "extended", "yahoo", "^SPGSCI", "price", None),
}


# ---------------------------------------------------------------- transforms

def par_bond_price(coupon: np.ndarray, yld: np.ndarray, years: float) -> np.ndarray:
    """Price (per 100 face) of an N-year semiannual bond with annual coupon rate `coupon`
    at annual yield `yld` (both decimals). Equal coupon and yield give 100."""
    n = 2 * years
    v = (1 + yld / 2) ** (-n)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(yld) > 1e-12, (1 - v) / (yld / 2), n)
    return 100 * (coupon / 2 * annuity + v)


def bond_excess_returns(yld: pd.Series, tbill: pd.Series, years: float) -> pd.Series:
    """Daily excess return of a constant-maturity par bond (PREREG 3.3), yields in percent.

    Each day the bond is the par bond of the previous day (coupon y_{t-1}), repriced at y_t,
    plus accrued coupon over the calendar days since, minus the 3-month bill rate (as of
    the previous observation, carried forward) over the same days."""
    y = yld.dropna().sort_index() / 100
    tb = tbill.dropna().sort_index().reindex(y.index.union(tbill.index)).ffill().reindex(y.index) / 100
    days = y.index.to_series().diff().dt.days
    price = par_bond_price(y.shift(1).to_numpy(), y.to_numpy(), years)
    total = (price - 100) / 100 + y.shift(1) * days / 365
    return (total - tb.shift(1) * days / 365).rename(yld.name)


def price_returns(level: pd.Series) -> pd.Series:
    return level / level.shift(1) - 1


def bad_ticks(level: pd.Series, to_returns) -> pd.DatetimeIndex:
    """Single pass (PREREG 3.3): the observation at t is a bad tick when |r_t| exceeds 10x the
    standard deviation of the previous 60 returns AND r_{t+1} has the opposite sign with more
    than half its size."""
    r = to_returns(level)
    sd = r.shift(1).rolling(60, min_periods=60).std()
    nxt = r.shift(-1)
    bad = (r.abs() > 10 * sd) & (np.sign(nxt) == -np.sign(r)) & (nxt.abs() > 0.5 * r.abs())
    return r.index[bad.fillna(False).to_numpy()]


# ---------------------------------------------------------------- sources

def fetch_yahoo(ticker: str) -> pd.Series:
    import yfinance as yf
    h = yf.Ticker(ticker).history(start="1960-01-01", end="2015-01-01", auto_adjust=False)
    s = h["Close"]
    s.index = pd.DatetimeIndex(s.index.tz_localize(None).normalize())
    return s[s > 0].rename(ticker)


def fetch_fed(rel: str, series: list[str]) -> dict[str, pd.Series]:
    r = requests.get(FED.format(rel=rel), timeout=300, headers=UA)
    r.raise_for_status()
    z = zipfile.ZipFile(io.BytesIO(r.content))
    txt = z.read(next(n for n in z.namelist() if n.endswith("_data.xml"))).decode("utf-8", "ignore")
    out = {}
    for sid in series:
        i = txt.find(f'SERIES_NAME="{sid}"')
        if i < 0:
            raise ValueError(f"{rel}: series {sid} not found")
        block = txt[i:txt.find("</kf:Series>", i)]
        obs = re.findall(r'OBS_VALUE="([^"]+)"\s+TIME_PERIOD="([0-9-]+)"', block)
        s = pd.to_numeric(pd.Series({pd.Timestamp(d): v for v, d in obs}), errors="coerce")
        out[sid] = s[s != MISSING].dropna().sort_index().rename(sid)  # the Fed codes ND (no data) as -9999
    return out


def fetch_eia_wti() -> pd.Series:
    r = requests.get(EIA_WTI, timeout=300, headers=UA)
    r.raise_for_status()
    df = pd.read_excel(io.BytesIO(r.content), sheet_name="Data 1", skiprows=2)
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").to_numpy(), index=pd.to_datetime(df.iloc[:, 0]))
    return s.dropna().sort_index().rename("RWTC")


# ---------------------------------------------------------------- build

def build_one(code: str, raw: pd.Series, tbill: pd.Series | None):
    name, asset, pool, source, series, transform, start = UNIVERSE[code]
    level = raw.loc[:END]
    level = level[~level.index.duplicated(keep="last")]
    if start is not None:
        level = level.loc[start:]
    if transform == "fx_invert":
        level = 1 / level
    if transform.startswith("bond"):
        years = int(transform[4:])
        to_ret = lambda lv: bond_excess_returns(lv, tbill, years)  # noqa: E731
    else:
        to_ret = price_returns
    flagged = bad_ticks(level, to_ret)
    kept = [d for d in flagged if (code, str(d.date())) in VERIFIED_REAL]
    bad = flagged.difference(pd.DatetimeIndex(kept))
    removed = [(d.date(), float(raw.loc[d])) for d in bad]
    level = level.drop(bad)
    ret = to_ret(level)
    ret.iloc[0] = np.nan
    close = 100 * (1 + ret.fillna(0)).cumprod()
    out = pd.DataFrame({"close": close, "close_adj": close, "ret": ret, "volume": np.nan,
                        "roll": False, "kind": "proxy"}, index=level.index.rename("date"))
    gaps = level.index.to_series().diff().dt.days
    info = {"code": code, "name": name, "asset_class": asset, "pool": pool, "source": source, "series": series,
            "transform": transform, "first": out.index[0].date(), "last": out.index[-1].date(), "rows": len(out),
            "bad_ticks_removed": len(removed), "flagged_but_kept": ";".join(str(d.date()) for d in kept),
            "max_gap_days": int(gaps.max()),
            "gaps_over_7_days": int((gaps > 7).sum()), "max_abs_ret": float(ret.abs().max())}
    return out, info, removed


def main(out_dir: Path = Path("data/eras"), raw_dir: Path = Path("data/raw/eras")):
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    raws: dict[str, pd.Series] = {}
    by_src: dict[str, list[str]] = {}
    for code, spec in UNIVERSE.items():
        by_src.setdefault(spec[3], []).append(code)
    for code in by_src.get("yahoo", []):
        raws[code] = fetch_yahoo(UNIVERSE[code][4])
    h10 = fetch_fed("H10", [UNIVERSE[c][4] for c in by_src["fed_h10"]])
    for c in by_src["fed_h10"]:
        raws[c] = h10[UNIVERSE[c][4]]
    h15 = fetch_fed("H15", [UNIVERSE[c][4] for c in by_src["fed_h15"]] + [TBILL])
    for c in by_src["fed_h15"]:
        raws[c] = h15[UNIVERSE[c][4]]
    tbill = h15[TBILL].loc[:END]
    raws["WTI"] = fetch_eia_wti()

    tbill.rename("value").rename_axis("date").to_csv(raw_dir / "TBILL3M_raw.csv")
    infos, removed_all = [], []
    for code in UNIVERSE:
        raw = raws[code].loc[:END]
        raw.rename("value").rename_axis("date").to_csv(raw_dir / f"{code}_raw.csv")
        out, info, removed = build_one(code, raw, tbill)
        out.to_csv(out_dir / f"{code}.csv")
        infos.append(info)
        removed_all += [{"code": code, "date": d, "raw_value": v} for d, v in removed]
        print(f"{code:5s} {info['first']} .. {info['last']}  rows {info['rows']:6d}  bad ticks {info['bad_ticks_removed']}"
              f"  max gap {info['max_gap_days']}d  max |ret| {info['max_abs_ret']:.3f}", flush=True)
    uni = pd.DataFrame(infos)
    uni.to_csv(out_dir / "universe.csv", index=False)
    rem = pd.DataFrame(removed_all, columns=["code", "date", "raw_value"])
    rem.to_csv(out_dir / "bad_ticks_removed.csv", index=False)
    write_log(out_dir, uni, rem)
    return uni, rem


def write_log(out_dir: Path, uni: pd.DataFrame, rem: pd.DataFrame):
    lines = ["# Eras proxy data: build log", "",
             "Built by `uv run --with xlrd python -m eras.build_data` (see the script's docstring and PREREG §3).",
             "Data end at 2014-12-31: rows after that in the Fed and EIA files were dropped while parsing.", "",
             "| Code | Pool | Asset class | Source | First | Last | Rows | Bad ticks removed | Max gap (days) | Gaps > 7 days | Max abs return |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in uni.itertuples():
        lines.append(f"| {r.code} | {r.pool} | {r.asset_class} | {r.source} `{r.series}` | {r.first} | {r.last} | {r.rows} | "
                     f"{r.bad_ticks_removed} | {r.max_gap_days} | {r.gaps_over_7_days} | {r.max_abs_ret:.4f} |")
    lines += ["", f"Bad ticks removed: {len(rem)} (listed in `bad_ticks_removed.csv`)."]
    kept = uni[uni.flagged_but_kept.fillna("") != ""]
    if len(kept):
        lines += ["", "Flagged by the bad-tick rule but kept as verified real moves (deviation #1, `eras/DEVIATIONS.md`):"]
        for r in kept.itertuples():
            d = pd.read_csv(out_dir / f"{r.code}.csv", index_col="date", parse_dates=True).ret
            for day in r.flagged_but_kept.split(";"):
                t = d.index.get_loc(pd.Timestamp(day))
                lines.append(f"- {r.code} {day}: return {d.iloc[t]:+.4f}, next day {d.iloc[t + 1]:+.4f}")
    (out_dir / "BUILD_LOG.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
