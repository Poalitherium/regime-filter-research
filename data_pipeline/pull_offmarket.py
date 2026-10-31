"""Provisional off-market (macro / news) features for M3, until M4's data lands.

Output: data/processed/offmarket/offmarket_daily.csv, one row per business day. Each
value is what was *known* on that day: every source is shifted by its publication lag
before being carried forward, so a contract on date t can use the row dated t as is.

    vix   log VIX close (^VIX)                                  lag 1 business day
    term  10y - 13-week Treasury yield, ^TNX - ^IRX (pp)        lag 1
    epu   log 20-day mean of the US daily EPU index             lag 2 (published next day)
    gpr   log 20-day mean of the daily Geopolitical Risk index  lag 2

Usage (GPR is an .xls file, so xlrd is needed only here):
    uv run --with xlrd python data_pipeline/pull_offmarket.py
"""

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pandas.tseries.offsets import BDay

from pull_yfinance import pull

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "offmarket"
OUT = ROOT / "data" / "processed" / "offmarket"
EPU_URL = "https://www.policyuncertainty.com/media/All_Daily_Policy_Data.csv"
GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
LAGS = {"vix": 1, "term": 1, "epu": 2, "gpr": 2}


def fetch(url: str, name: str) -> bytes:
    RAW.mkdir(parents=True, exist_ok=True)
    content = requests.get(url, timeout=120).content
    (RAW / name).write_bytes(content)
    return content


def available_from(s: pd.Series, lag: int, days: pd.DatetimeIndex) -> pd.Series:
    """Re-date each observation to the first business day it can be used, then carry forward."""
    s = s.dropna()
    s.index = s.index + BDay(lag)
    s = s[~s.index.duplicated(keep="last")]
    return s.reindex(s.index.union(days)).ffill().reindex(days)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    vix = pull("^VIX")["close"]
    tnx, irx = pull("^TNX")["close"], pull("^IRX")["close"]
    for t, s in [("VIX", vix), ("TNX", tnx), ("IRX", irx)]:
        s.rename("close").to_csv(RAW / f"_{t}.csv")

    epu = pd.read_csv(io.BytesIO(fetch(EPU_URL, "epu_daily.csv")))
    epu.index = pd.to_datetime(dict(year=epu.year, month=epu.month, day=epu.day))
    epu = epu["daily_policy_index"].astype(float).sort_index()

    gpr = pd.read_excel(io.BytesIO(fetch(GPR_URL, "gpr_daily.xls")))
    gpr.index = pd.to_datetime(gpr["DAY"].astype(int).astype(str), format="%Y%m%d")
    gpr = gpr["GPRD"].astype(float).sort_index()

    raw = {
        "vix": np.log(vix.where(vix > 0)),
        "term": (tnx - irx.reindex(tnx.index)).dropna(),
        # 20-day mean on each source's own (calendar-daily) dates, then log
        "epu": np.log(epu.rolling(20, min_periods=20).mean()),
        "gpr": np.log(gpr.rolling(20, min_periods=20).mean().where(lambda x: x > 0)),
    }
    start = min(s.dropna().index[0] for s in raw.values())
    days = pd.bdate_range(start, pd.Timestamp.today().normalize())
    out = pd.DataFrame({k: available_from(v, LAGS[k], days) for k, v in raw.items()})
    out = out.dropna(how="all").rename_axis("date")
    out.round(6).to_csv(OUT / "offmarket_daily.csv")
    meta = {k: {"lag_bdays": LAGS[k], "first_valid": str(out[k].first_valid_index().date()),
                "last_valid": str(out[k].last_valid_index().date())} for k in out}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(pd.DataFrame(meta).T.to_string())
    print(out.loc["2004":].describe().round(3).to_string())


if __name__ == "__main__":
    main()
