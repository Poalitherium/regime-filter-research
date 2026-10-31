"""CFTC Commitments of Traders (legacy, futures only) for M4-3 (PREREG_M4-3.md).

    python -m m4.cot --src <folder with FUT86_06 and FUT07_14 as .xls or .csv>

Writes the 11-contract subset to m4/data/cot_legacy_2014.csv.
Only FUT86_06 and FUT07_14 are read; later years are downloaded by backtest.resplit_data.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .common import M4_DIR, asof

COT_FILE = M4_DIR / "data" / "cot_legacy_2014.csv"
CFTC_CODES = {"ES": "13874A", "NK": "240741", "CL": "067651", "NG": "023651", "GC": "088691", "SI": "084691",
              "ZN": "043602", "6E": "099741", "6J": "097741", "ZC": "002602", "ZS": "005602"}
SOURCES = ("FUT86_06", "FUT07_14")  # never FUT15_16
COLS = {"CFTC_Contract_Market_Code": "cftc_code", "Market_and_Exchange_Names": "market",
        "Report_Date_as_MM_DD_YYYY": "report_date", "Open_Interest_All": "oi",
        "NonComm_Positions_Long_All": "nc_long", "NonComm_Positions_Short_All": "nc_short",
        "Comm_Positions_Long_All": "comm_long", "Comm_Positions_Short_All": "comm_short"}

AVAILABLE_AFTER_DAYS = 7          # report (Tuesday positions) usable at the close 7 calendar days later
FFILL_BDAYS = 10                  # a stale report expires after this many business days
SHUTDOWN = (pd.Timestamp("2013-10-01"), pd.Timestamp("2013-11-12"))  # delayed reports, dropped
WINDOW, MIN_WEEKS = 156, 104
HI, LO = 0.90, 0.10


def build_subset(src: Path, out: Path = COT_FILE, end: str = "2014-12-31") -> pd.DataFrame:
    parts = []
    for name in SOURCES:
        f = src / f"{name}.csv"
        raw = (pd.read_csv(f, usecols=list(COLS), dtype={"CFTC_Contract_Market_Code": str}) if f.is_file()
               else pd.read_excel(src / f"{name}.xls", usecols=list(COLS), dtype={"CFTC_Contract_Market_Code": str}))
        parts.append(raw.rename(columns=COLS))
    d = pd.concat(parts)
    d["cftc_code"] = d.cftc_code.str.strip()
    inv = {v: k for k, v in CFTC_CODES.items()}
    d = d[d.cftc_code.isin(inv)].copy()
    d["code"] = d.cftc_code.map(inv)
    d["report_date"] = pd.to_datetime(d.report_date)
    d = d[d.report_date <= pd.Timestamp(end)]
    d = d[~((d.code == "6E") & ~d.market.str.contains("EURO FX"))]  # 099741 was the ECU before 1999
    d = d.drop_duplicates(["code", "report_date"]).sort_values(["code", "report_date"])
    cols = ["code", "cftc_code", "market", "report_date", "oi", "nc_long", "nc_short", "comm_long", "comm_short"]
    out.parent.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(out, index=False)
    return d[cols]


def load(path: Path = COT_FILE) -> pd.DataFrame:
    return pd.read_csv(path, parse_dates=["report_date"], dtype={"cftc_code": str})


def rolling_pct(x: pd.Series, window: int = WINDOW, min_weeks: int = MIN_WEEKS) -> pd.Series:
    """Percentile of the latest value within the trailing window (inclusive), mean rank for ties."""
    def pct(a):
        last = a[-1]
        return ((a < last).sum() + 0.5 * ((a == last).sum() - 1)) / (len(a) - 1)
    return x.rolling(window, min_periods=min_weeks).apply(pct, raw=True)


def weekly_features(cot: pd.DataFrame, who: str = "nc") -> pd.DataFrame:
    """Per contract and report: net share of open interest and its 156-week percentile,
    indexed by the date the report becomes usable. who = "nc" (speculators) or "comm" (hedgers)."""
    out = []
    for code, g in cot.groupby("code"):
        g = g[~g.report_date.between(*SHUTDOWN)].sort_values("report_date")
        net = (g[f"{who}_long"] - g[f"{who}_short"]) / g.oi
        f = pd.DataFrame({"code": code, "report_date": g.report_date.to_numpy(),
                          "available": (g.report_date + pd.Timedelta(days=AVAILABLE_AFTER_DAYS)).to_numpy(),
                          "net": net.to_numpy()})
        f["pct"] = rolling_pct(f.net).to_numpy()
        out.append(f)
    return pd.concat(out, ignore_index=True)


def daily_pct(feat: pd.DataFrame, code: str, index: pd.DatetimeIndex) -> pd.Series:
    """The latest usable percentile on each of a contract's dates (carried forward, expiring)."""
    f = feat[feat.code == code].set_index("available")[["pct"]]
    if f.empty:
        return pd.Series(np.nan, index=index)
    return asof(f, index, limit=FFILL_BDAYS)["pct"]


def crowded(target: pd.Series, pct: pd.Series, hi: float = HI, lo: float = LO) -> pd.Series:
    c = ((target > 0) & (pct >= hi)) | ((target < 0) & (pct <= lo))
    return c.astype(float).where(pct.notna() & target.notna())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    a = p.parse_args()
    s = build_subset(a.src)
    print(s.groupby("code").report_date.agg(["min", "max", "size"]))
