"""Detect contract rolls in Yahoo's stitched front-month series and back-adjust them.

Yahoo `=F` switches to the next contract at (roughly) expiry without telling us when,
and expired contracts are not downloadable, so the roll gap (new - old contract price)
is never observed directly. We estimate it:

1. Clean bad ticks (close outside the day's high-low range).
2. Explain each day's futures return with roll-free proxies (index / ETF / spot FX,
   same day and next day to absorb close-time mismatches) via a per-year robust OLS.
   The residual e_t is what the proxies cannot explain.
3. Only inside calendar windows around each contract's first-notice / last-trade date
   can a day be a roll. Within each window pick the largest residual that is
   (a) > K robust sigmas and (b) persistent: not reversed the next day. (b) rejects
   close-time artifacts such as FOMC days, which jump on t and reverse on t+1.
   A volume jump on t-1 (Yahoo switches volume a day before price) counts as extra
   evidence, which helps catch small rolls in ES / ZN / grains.
4. Roll gap = e_t * |P_{t-1}| (the move the proxies say did not happen in the market).
   Prices before the roll are shifted by the gap ("Panama" / additive back-adjustment),
   so the adjusted series ends at the last raw price and has no roll jumps.

Limitations: the gap estimate carries one day of residual noise; small rolls (gap below
K sigma, typical for FX and serial months) are left in; pre-proxy history (e.g. grains
before 2007) only catches large gaps. Yahoo holds the expiring contract until its last
trade date, so near-expiry days can be illiquid (CL on 2020-04-20 is patched by hand).

Usage:
    uv run python data_pipeline/roll_adjust.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from pull_yfinance import CONTRACTS, OUT_DIR as RAW_DIR, PROXIES, PROXY_DIR, proxy_file

OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "yfinance"
K_SIGMA = 3.5  # roll must stand out this many robust sigmas from proxy-explained noise
VOL_JUMP = np.log(3)  # Yahoo's volume switches to the new contract the day BEFORE the price
VOL_BONUS = 2.0  # sigmas credited to a day whose previous day had such a volume jump
WINDOW = (-5, 3)  # business days around each calendar anchor
BAD_TICK = 0.2  # close this far outside the day's high-low range is a typo, not a settlement
SPIKE = 0.05  # close >5% outside its bar AND a >5% move that reverses next day = bad print
ANOMALY_SIGMA = 8.0  # report (but do not touch) residuals this large outside windows

# Yahoo holds the expiring contract to its last trade date. In April 2020 that put the
# May WTI squeeze (-37.63 settle) into CL=F; replace with the June contract (CLM20)
# settlements, i.e. what a holder who rolled before expiry would have seen.
MANUAL_CLOSE = {"CL": {"2020-04-20": 20.43, "2020-04-21": 11.57}}


# ---------------------------------------------------------------- calendar anchors

def _nth_weekday(y: int, m: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(y, m, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_bday(y: int, m: int) -> pd.Timestamp:
    return pd.Timestamp(y, m, 1) + pd.offsets.BMonthEnd(0)


def _bday_on_or_before(d: pd.Timestamp) -> pd.Timestamp:
    return d if d.weekday() < 5 else d - BDay(1)


QUARTERLY = (3, 6, 9, 12)
ALL = tuple(range(1, 13))


def anchors(code: str, years: range) -> list[pd.Timestamp]:
    """First-notice / last-trade dates (weekday calendar, holidays ignored)."""
    out = []
    for y in years:
        for m in ALL:
            prev = pd.Timestamp(y, m, 1) - pd.offsets.MonthBegin(1)  # month before m
            if code == "ES" and m in QUARTERLY:
                out.append(_nth_weekday(y, m, 4, 3))  # 3rd Friday
            elif code == "NK" and m in QUARTERLY:
                out.append(_nth_weekday(y, m, 4, 2) - BDay(1))  # day before 2nd Friday
            elif code == "CL":  # contract m+1: 3 bdays before the 25th of m
                out.append(_bday_on_or_before(pd.Timestamp(y, m, 25)) - BDay(3))
            elif code == "NG":  # contract m+1: 3 bdays before the 1st of m+1
                out.append(pd.Timestamp(y, m, 1) + pd.offsets.MonthBegin(1) - BDay(3))
            elif code in ("GC", "SI"):  # FND (last bday of m) and LTD (3rd last) both
                out.append(_last_bday(y, m))  # window covers the 3rd-last bday too
            elif code == "ZN" and m in QUARTERLY:
                out.append(_last_bday(prev.year, prev.month))  # first notice
                out.append(_last_bday(y, m) - BDay(7))  # last trade
            elif code in ("6E", "6J"):  # serial months exist, so every month
                out.append(_nth_weekday(y, m, 2, 3) - BDay(2))
            elif (code == "ZC" and m in (3, 5, 7, 9, 12)) or (
                code == "ZS" and m in (1, 3, 5, 7, 8, 9, 11)
            ):
                out.append(_last_bday(prev.year, prev.month))  # first notice
                out.append(_bday_on_or_before(pd.Timestamp(y, m, 15)) - BDay(1))  # last trade
    return out


def window_ids(index: pd.DatetimeIndex, code: str) -> pd.Series:
    """Label each trading day with the id of the roll window it falls in (-1 = none)."""
    ids = pd.Series(-1, index=index)
    years = range(index[0].year - 1, index[-1].year + 2)
    for i, a in enumerate(sorted(set(anchors(code, years)))):
        lo, hi = a + BDay(WINDOW[0]), a + BDay(WINDOW[1])
        sel = (index >= lo) & (index <= hi) & (ids.values == -1)
        ids[sel] = i
    return ids


# ---------------------------------------------------------------- residuals

def clean(code: str, df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # settlements often sit slightly outside the session range; only gross misses are typos
    df = df.copy()
    c = df["close"]
    tol = BAD_TICK * c.abs().rolling(20, min_periods=1).median()
    bad = (c > df["high"] + tol) | (c < df["low"] - tol)
    # one-day spike: close well outside its own bar, big move, undone the next day
    outside = ((c - df["high"]).clip(lower=0) + (df["low"] - c).clip(lower=0)) / c.abs()
    r = c.pct_change(fill_method=None)
    r_next = r.shift(-1)
    bad |= (outside > SPIKE) & (r.abs() > SPIKE) & ((r + r_next).abs() < 0.5 * r.abs())
    df.loc[bad, "close"] = np.nan
    df["close"] = df["close"].interpolate(method="time")
    fixed = [str(d.date()) for d in df.index[bad]]
    for day, px in MANUAL_CLOSE.get(code, {}).items():
        df.loc[pd.Timestamp(day), ["open", "high", "low", "close"]] = px
        fixed.append(f"{day} (manual)")
    return df, fixed


def proxy_features(code: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    feats = {}
    for t in PROXIES[code]:
        s = pd.read_csv(PROXY_DIR / proxy_file(t), index_col=0, parse_dates=True)["close"]
        s = s.reindex(s.index.union(index)).ffill(limit=3).reindex(index)
        x = s.diff() if t == "^TNX" else s.pct_change(fill_method=None)  # yield in pct points
        feats[t] = x
        feats[t + "(t+1)"] = x.shift(-1)
    return pd.DataFrame(feats)


def residuals(r: pd.Series, X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Per-year robust OLS of r on available proxies (fit on year +-1). Returns (e, rhat)."""
    rhat = pd.Series(0.0, index=r.index)
    for y in sorted(set(r.index.year)):
        this = r.index.year == y
        cols = [c for c in X.columns if X.loc[this, c].notna().mean() > 0.8]
        if not cols:
            continue
        fit = (r.index.year >= y - 1) & (r.index.year <= y + 1)
        A, yy = X.loc[fit, cols], r[fit]
        ok = yy.notna() & A.notna().all(axis=1)
        A, yy = A[ok].values, yy[ok].values
        mad = np.median(np.abs(yy - np.median(yy))) / 0.6745
        keep = np.abs(yy - np.median(yy)) < 5 * mad  # drop roll gaps / crashes from the fit
        A1 = np.column_stack([np.ones(len(A)), A])
        beta = np.linalg.lstsq(A1[keep], yy[keep], rcond=None)[0]
        Xp = X.loc[this, cols].fillna(0.0).values
        rhat[this] = beta[0] + Xp @ beta[1:]
    return r - rhat, rhat


# ---------------------------------------------------------------- main

def adjust(code: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    raw = pd.read_csv(RAW_DIR / f"{code}.csv", index_col=0, parse_dates=True)
    df, bad_ticks = clean(code, raw)
    p = df["close"]
    r = p.diff() / p.shift().abs()
    e, rhat = residuals(r, proxy_features(code, df.index))
    sig = (e.abs().rolling(250, min_periods=60, center=True).median() / 0.6745).bfill().ffill()
    z = e / sig

    win = window_ids(df.index, code)
    e_next = e.shift(-1).fillna(0.0)
    persistent = (np.sign(e + e_next) == np.sign(e)) & ((e + e_next).abs() > 0.5 * e.abs())
    v = df["volume"].replace(0, np.nan)
    vol_jump = (np.log(v / v.shift()) > VOL_JUMP).shift(1, fill_value=False)
    score = z.abs() + VOL_BONUS * vol_jump
    cand = (win >= 0) & (score > K_SIGMA) & persistent
    roll_days = score[cand].groupby(win[cand]).idxmax().values

    gap = pd.Series(0.0, index=df.index)
    gap[roll_days] = e[roll_days] * p.shift().abs()[roll_days]
    # shift everything before each roll by that roll's gap
    shift = gap[::-1].cumsum()[::-1].shift(-1, fill_value=0.0)
    out = df.copy()
    for col in ("open", "high", "low", "close"):
        out[f"{col}_adj"] = df[col] + shift
    out["roll"] = gap != 0
    out["roll_gap"] = gap
    out["ret"] = out["close_adj"].diff() / p.shift().abs()  # return of the contract held

    rolls = pd.DataFrame({
        "code": code, "date": roll_days, "gap": gap[roll_days].values,
        "gap_pct": (gap[roll_days] / p.shift().abs()[roll_days]).values,
        "z": z[roll_days].values,
    })
    off = (win < 0) & (z.abs() > ANOMALY_SIGMA)
    anomalies = pd.DataFrame({"code": code, "date": df.index[off], "ret": r[off].values,
                              "resid": e[off].values, "z": z[off].values})
    return out, rolls, anomalies, bad_ticks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rolls, all_anom, report = [], [], {}
    for code in CONTRACTS:
        out, rolls, anomalies, bad_ticks = adjust(code)
        out.to_csv(OUT_DIR / f"{code}.csv")
        all_rolls.append(rolls)
        all_anom.append(anomalies)
        years = (out.index[-1] - out.index[0]).days / 365.25
        report[code] = {"rolls": len(rolls), "rolls_per_year": round(len(rolls) / years, 2),
                        "mean_abs_gap_pct": round(float(rolls.gap_pct.abs().mean()), 5),
                        "bad_ticks_fixed": bad_ticks, "anomalies_outside_windows": len(anomalies)}
        print(f"{code:3s} rolls={len(rolls):4d} ({len(rolls) / years:4.1f}/yr)  "
              f"mean |gap|={rolls.gap_pct.abs().mean():6.2%}  bad ticks={len(bad_ticks)}  "
              f"anomalies={len(anomalies)}")
    pd.concat(all_rolls).to_csv(OUT_DIR / "rolls.csv", index=False)
    pd.concat(all_anom).to_csv(OUT_DIR / "anomalies.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved adjusted series, rolls.csv and anomalies.csv to {OUT_DIR}")


if __name__ == "__main__":
    main()
