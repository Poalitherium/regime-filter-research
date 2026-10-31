"""Assemble the 13-contract research dataset from roll-adjusted Yahoo series and stand-ins.

Output (data/final/):
    {CODE}.csv     date, close (raw / stand-in level), close_adj, ret, volume, roll, kind
    ret.csv        wide panel of daily returns, union calendar (NaN = market closed)
    close_adj.csv  wide panel of back-adjusted prices
    universe.csv   one row per contract: source, coverage, quality checks, reliability

`ret` is the daily return of the position actually held (roll gaps removed). Strategies
should use `ret` / `close_adj` only; `close` is kept for reference and cost models.

Usage:
    uv run python data_pipeline/build_dataset.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ADJ_DIR = ROOT / "data" / "processed" / "yfinance"
STANDIN_DIR = ROOT / "data" / "processed" / "standins"
OUT_DIR = ROOT / "data" / "final"

# code: (name, asset class, kind, source, reliability, note)
UNIVERSE = {
    "ES": ("S&P 500 E-mini", "equity", "futures", "ES=F", "good", ""),
    "FESX": ("Euro Stoxx 50", "equity", "standin", "^STOXX50E; FEZ/EURUSD before 2007-03-30", "standin",
             "index level, no futures carry (~-0.8%/yr drift on ES test); pre-2007 part is a US-listed ETF"),
    "NK": ("Nikkei 225", "equity", "futures", "NKD=F (CME, USD)", "good",
           "CME USD contract stands in for OSE yen contract; small rolls left in"),
    "CL": ("WTI crude oil", "energy", "futures", "CL=F", "good",
           "2020-04-20/21 patched with June contract settlements"),
    "NG": ("Natural gas", "energy", "futures", "NG=F", "medium",
           "roll estimates uncertain; drift differs from UNG by several %/yr"),
    "GC": ("Gold", "metal", "futures", "GC=F", "good", "rolls under ~3 sigma left in"),
    "SI": ("Silver", "metal", "futures", "SI=F", "medium", "drift ~2%/yr above SLV-implied"),
    "ZN": ("US 10y T-note", "rates", "futures", "ZN=F", "good", ""),
    "FGBL": ("Euro-Bund", "rates", "standin", "Bundesbank zero curve -> 6% 9y bond", "standin",
             "no futures carry (~-1.6%/yr drift on ZN test); German business-day calendar"),
    "6E": ("Euro FX", "fx", "futures", "6E=F", "good", "small (serial-month) rolls left in"),
    "6J": ("Japanese yen", "fx", "futures", "6J=F", "good", "small (serial-month) rolls left in"),
    "ZC": ("Corn", "agri", "futures", "ZC=F", "medium", "drift differs from CORN ETF by ~4%/yr"),
    "ZS": ("Soybeans", "agri", "futures", "ZS=F", "UNRELIABLE",
           "adjusted drift +8%/yr vs ~0% implied by SOYB; do not trust long-run returns"),
}


def load(code: str, kind: str) -> pd.DataFrame:
    if kind == "futures":
        df = pd.read_csv(ADJ_DIR / f"{code}.csv", index_col=0, parse_dates=True)
        return df[["close", "close_adj", "ret", "volume", "roll"]]
    df = pd.read_csv(STANDIN_DIR / f"{code}.csv", index_col=0, parse_dates=True)
    return pd.DataFrame({"close": df["close"], "close_adj": df["close"], "ret": df["ret"],
                         "volume": np.nan, "roll": False}, index=df.index)


def trim_sparse_start(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's first months are sometimes weekly; start after the last >6-day hole in year one."""
    gaps = df.index.to_series().diff() > pd.Timedelta(days=6)
    early = gaps & (df.index < df.index[0] + pd.Timedelta(days=365))
    return df.loc[early[early].index[-1]:].iloc[1:] if early.any() else df


def checks(df: pd.DataFrame) -> dict:
    c = df["close_adj"]
    gaps = df.index.to_series().diff().dt.days
    stale = (c.diff() == 0).astype(int)
    longest_stale = int(stale.groupby((stale == 0).cumsum()).sum().max())
    return {
        "start": df.index[0].date(), "end": df.index[-1].date(), "rows": len(df),
        "missing_ret": int(df["ret"].iloc[1:].isna().sum()),
        "duplicate_dates": int(df.index.duplicated().sum()),
        "max_calendar_gap_days": int(gaps.max()),
        "longest_unchanged_run": longest_stale,
        "max_abs_ret": round(float(df["ret"].abs().max()), 4),
        "ann_vol": round(float(df["ret"].std() * np.sqrt(252)), 4),
        "rolls": int(df["roll"].sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows, rets, closes = [], {}, {}
    for code, (name, asset, kind, source, reliability, note) in UNIVERSE.items():
        df = load(code, kind)
        df = trim_sparse_start(df[~df.index.duplicated(keep="last")].sort_index())
        df["ret"] = df["ret"].mask(df.index == df.index[0])  # no return on the first kept day
        df["kind"] = kind
        df.rename_axis("date").to_csv(OUT_DIR / f"{code}.csv")
        rets[code], closes[code] = df["ret"], df["close_adj"]
        rows.append({"code": code, "name": name, "asset_class": asset, "kind": kind,
                     "source": source, "reliability": reliability, **checks(df), "note": note})

    pd.DataFrame(rets).sort_index().rename_axis("date").to_csv(OUT_DIR / "ret.csv")
    pd.DataFrame(closes).sort_index().rename_axis("date").to_csv(OUT_DIR / "close_adj.csv")
    uni = pd.DataFrame(rows)
    uni.to_csv(OUT_DIR / "universe.csv", index=False)

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(uni.drop(columns=["name", "source", "note"]).to_string(index=False))
    common = max(r["start"] for r in rows)
    print(f"\nAll 13 contracts available from {common} to {min(r['end'] for r in rows)}")


if __name__ == "__main__":
    main()
