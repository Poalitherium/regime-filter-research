"""Stand-in series for contracts with no free futures history (FESX, FGBL).

FESX: Euro Stoxx 50 price index (^STOXX50E, 2007+). Before that, FEZ (US-listed ETF)
      converted to EUR with EURUSD=X and spliced on at the first common date.
FGBL: price of a notional 6% annual-coupon German government bond with 9 years to
      maturity (Bund futures CTD sits near the short end of the 8.5-10.5y basket),
      re-priced every day off the Bundesbank Svensson zero curve (1997+).

Neither is a futures price: both miss futures carry (index: dividends minus financing;
bond: coupon minus financing, roll-down, CTD switches). The same constructions are run
on markets where we DO have futures (ES from ^GSPC, ZN from US yields) to measure how
far a stand-in drifts from the real thing; see the printed validation table.

Usage:
    uv run python data_pipeline/build_standins.py
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pull_yfinance import pull

ROOT = Path(__file__).resolve().parents[1]
RAW_BBK = ROOT / "data" / "raw" / "bundesbank"
RAW_YF = ROOT / "data" / "raw" / "yfinance" / "standins"
OUT_DIR = ROOT / "data" / "processed" / "standins"
ADJ_DIR = ROOT / "data" / "processed" / "yfinance"

BBK_URL = ("https://api.statistiken.bundesbank.de/rest/data/BBSIS/"
           "D.I.ZST.ZI.EUR.S1311.B.A604.R{m:02d}XX.R.A.A._Z._Z.A?format=csv&lang=en")
BBK_MATURITIES = range(1, 11)  # years
YF_TICKERS = ["^STOXX50E", "FEZ", "EURUSD=X", "^GSPC", "^FVX", "^TNX"]

BUND = {"coupon": 0.06, "maturity": 9.0, "freq": 1}
TNOTE = {"coupon": 0.06, "maturity": 7.0, "freq": 2}  # ZN CTD ~ short end of 6.5-10y


# ---------------------------------------------------------------- downloads

def fetch_bundesbank() -> pd.DataFrame:
    """Daily zero yields (%) by maturity in years, columns 1..10."""
    RAW_BBK.mkdir(parents=True, exist_ok=True)
    cols = {}
    for m in BBK_MATURITIES:
        text = requests.get(BBK_URL.format(m=m), timeout=120).text
        (RAW_BBK / f"zero_{m:02d}y.csv").write_text(text)
        df = pd.read_csv(io.StringIO(text), header=None, skiprows=1, names=["date", "y", "flag"])
        df = df[df["date"].str.match(r"\d{4}-\d{2}-\d{2}$", na=False)]
        s = pd.to_numeric(df["y"], errors="coerce")
        s.index = pd.to_datetime(df["date"])
        cols[m] = s
    return pd.DataFrame(cols).dropna(how="all")


def fetch_yahoo() -> dict[str, pd.Series]:
    RAW_YF.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in YF_TICKERS:
        df = pull(t)  # unadjusted: price indices / ETF price, like a futures price
        df.to_csv(RAW_YF / (t.replace("^", "_").replace("=", "_") + ".csv"))
        out[t] = df["close"]
    return out


# ---------------------------------------------------------------- constructions

def bond_price(curve: pd.DataFrame, coupon: float, maturity: float, freq: int) -> pd.Series:
    """Price per 100 of a fixed-coupon bond off a zero curve (% p.a., annual compounding),
    linearly interpolated between curve nodes and flat outside them."""
    nodes = np.array(curve.columns, dtype=float)
    times = np.arange(maturity, 0, -1 / freq)[::-1]
    flows = np.full(len(times), 100 * coupon / freq)
    flows[-1] += 100
    y = curve.to_numpy(dtype=float) / 100
    zeros = np.vstack([np.interp(times, nodes, row) for row in y])
    price = (flows * (1 + zeros) ** -times).sum(axis=1)
    return pd.Series(price, index=curve.index)


def us_par_curve(yf_data: dict[str, pd.Series]) -> pd.DataFrame:
    """5y and 10y CMT yields (%) as a two-node curve; treated as zero yields."""
    curve = pd.concat({5: yf_data["^FVX"], 10: yf_data["^TNX"]}, axis=1).dropna()
    return curve[(curve > 0).all(axis=1)]


def fesx_standin(yf_data: dict[str, pd.Series]) -> tuple[pd.Series, pd.Series]:
    sx5e = yf_data["^STOXX50E"].dropna()
    fez_eur = (yf_data["FEZ"] / yf_data["EURUSD=X"].reindex(yf_data["FEZ"].index).ffill(limit=3)).dropna()
    splice = sx5e.index[0]
    early = fez_eur[fez_eur.index < splice]
    early = early * sx5e.iloc[0] / fez_eur[fez_eur.index <= splice].iloc[-1]
    price = pd.concat([early, sx5e])
    source = pd.Series(np.where(price.index < splice, "FEZ/EURUSD (rescaled)", "^STOXX50E"),
                       index=price.index)
    return price, source


# ---------------------------------------------------------------- validation

def compare(standin: pd.Series, real_ret: pd.Series, start: str = "2007") -> dict:
    """How close is a stand-in to the real futures series (back-adjusted returns)?"""
    s = standin.pct_change(fill_method=None)
    d = pd.concat([s, real_ret], axis=1, keys=["s", "f"], sort=True).loc[start:].dropna()
    lvl = (1 + d).cumprod()
    m = lvl.resample("ME").last().pct_change().dropna()
    yrs = (d.index[-1] - d.index[0]).days / 365.25
    sig = np.sign(lvl.pct_change(60)).dropna()  # 60-day trend direction
    return {
        "from": str(d.index[0].date()),
        "daily_corr": round(d.s.corr(d.f), 3),
        "monthly_corr": round(m.s.corr(m.f), 3),
        "vol_ratio": round(d.s.std() / d.f.std(), 3),
        "drift_diff_pa": round(lvl.s.iloc[-1] ** (1 / yrs) - lvl.f.iloc[-1] ** (1 / yrs), 4),
        "trend_sign_agree": round((sig.s == sig.f).mean(), 3),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    yf_data = fetch_yahoo()
    bbk = fetch_bundesbank()

    fgbl = bond_price(bbk.dropna(), **BUND)
    fesx, fesx_src = fesx_standin(yf_data)
    for code, price, src in [("FGBL", fgbl, "Bundesbank zero curve, 6% 9y bond"),
                             ("FESX", fesx, fesx_src)]:
        pd.DataFrame({"close": price, "ret": price.pct_change(fill_method=None),
                      "source": src}).rename_axis("date").to_csv(OUT_DIR / f"{code}.csv")
        print(f"{code}: {price.index[0].date()} -> {price.index[-1].date()}  rows={len(price)}")

    # same constructions on markets with real futures, to size the stand-in error
    zn_real = pd.read_csv(ADJ_DIR / "ZN.csv", index_col=0, parse_dates=True)["ret"]
    es_real = pd.read_csv(ADJ_DIR / "ES.csv", index_col=0, parse_dates=True)["ret"]
    validation = {
        "FGBL method on ZN (US 5y/10y yields -> 6% 7y bond)":
            compare(bond_price(us_par_curve(yf_data), **TNOTE), zn_real),
        "FESX method on ES (^GSPC price index)": compare(yf_data["^GSPC"], es_real),
    }
    print("\nStand-in vs real futures (2007+):")
    print(pd.DataFrame(validation).T.to_string())
    (OUT_DIR / "validation.json").write_text(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
