"""Pull daily front-month futures data from Yahoo Finance for the CME contracts.

Yahoo's `=F` series are the raw front month stitched together: NOT back-adjusted,
so every roll shows up as a price jump. `roll_adjust.py` detects and removes those
jumps; it needs the roll-free proxy series (indices / ETFs / spot FX) pulled here too.
FESX and FGBL (Eurex) are not available on Yahoo.

Usage:
    uv run python data_pipeline/pull_yfinance.py
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yfinance as yf

# plan code -> (Yahoo ticker, asset class, note)
CONTRACTS = {
    "ES": ("ES=F", "index", "S&P 500 E-mini"),
    "NK": ("NKD=F", "index", "CME Nikkei 225 (USD) - proxy for OSE NK"),
    "CL": ("CL=F", "energy", "WTI crude oil"),
    "NG": ("NG=F", "energy", "Henry Hub natural gas"),
    "GC": ("GC=F", "metal", "Gold"),
    "SI": ("SI=F", "metal", "Silver"),
    "ZN": ("ZN=F", "rates", "10-year T-note"),
    "6E": ("6E=F", "fx", "Euro FX"),
    "6J": ("6J=F", "fx", "Japanese yen"),
    "ZC": ("ZC=F", "agri", "Corn"),
    "ZS": ("ZS=F", "agri", "Soybeans"),
}

# Roll-free series that co-move with each contract; used only to estimate roll gaps.
PROXIES = {
    "ES": ["^GSPC"],
    "NK": ["^N225", "EWJ"],
    "CL": ["USO", "XLE"],
    "NG": ["UNG", "XLE"],
    "GC": ["GLD", "^XAU"],
    "SI": ["SLV", "GLD"],
    "ZN": ["IEF", "^TNX"],
    "6E": ["EURUSD=X"],
    "6J": ["JPY=X"],
    "ZC": ["CORN", "DBA"],
    "ZS": ["SOYB", "DBA"],
}

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "yfinance"
PROXY_DIR = OUT_DIR / "proxies"
JUMP_SIGMA = 6.0  # flag daily moves larger than this many rolling std devs


def proxy_file(ticker: str) -> str:
    return ticker.replace("^", "_").replace("=", "_") + ".csv"


def pull(ticker: str, auto_adjust: bool = False) -> pd.DataFrame:
    df = yf.download(ticker, period="max", interval="1d", progress=False, auto_adjust=auto_adjust)
    if df.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
    df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    df = df[~df.index.duplicated(keep="last")].sort_index()
    # today's bar is still live (and was seen to splice in the next contract)
    return df[df.index < pd.Timestamp.today().normalize()]


def quality(df: pd.DataFrame) -> dict:
    close = df["close"]
    # simple returns: log returns break on CL's negative print in Apr 2020
    ret = close.pct_change()
    sd = ret.rolling(60, min_periods=20).std().shift(1)
    jumps = ret[(ret.abs() > JUMP_SIGMA * sd)]
    return {
        "start": str(df.index[0].date()),
        "end": str(df.index[-1].date()),
        "rows": len(df),
        "missing_close": int(close.isna().sum()),
        "nonpositive_close": int((close <= 0).sum()),
        "zero_volume_days": int((df["volume"] == 0).sum()),
        "high_lt_low": int((df["high"] < df["low"]).sum()),
        "close_outside_range": int(((close > df["high"]) | (close < df["low"])).sum()),
        "jump_days": [str(d.date()) for d in jumps.index],
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PROXY_DIR.mkdir(parents=True, exist_ok=True)
    pulled_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {}
    for code, (ticker, asset_class, note) in CONTRACTS.items():
        df = pull(ticker)
        df.to_csv(OUT_DIR / f"{code}.csv")
        q = quality(df)
        meta[code] = {
            "ticker": ticker,
            "asset_class": asset_class,
            "note": note,
            "source": "Yahoo Finance via yfinance",
            "back_adjusted": False,
            "proxies": PROXIES[code],
            "pulled_at": pulled_at,
            **q,
        }
        print(
            f"{code:3s} {ticker:6s} {q['start']} -> {q['end']}  rows={q['rows']:5d}  "
            f"zero-vol={q['zero_volume_days']:4d}  nonpos={q['nonpositive_close']}  "
            f"bad-close={q['close_outside_range']}  jumps={len(q['jump_days'])}"
        )
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    # dividend/split-adjusted closes are what we want for the ETF proxies
    for ticker in sorted({t for ts in PROXIES.values() for t in ts}):
        pull(ticker, auto_adjust=True).to_csv(PROXY_DIR / proxy_file(ticker))
    print(f"\nSaved {len(meta)} contracts and proxies to {OUT_DIR}")


if __name__ == "__main__":
    main()
