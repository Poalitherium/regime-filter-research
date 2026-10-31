"""Shared M4 inputs and statistics (PREREG sections 2-4).

Per contract, on its own calendar: the A2 combo target and its gross P&L (M5's loader,
m5.common.load_all), so a target set at close t earns from t + 2 rows (delay = 1).
A regressor for the P&L of day tau is taken at the decision close, LEAD rows earlier.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from m3.predictability import OOS_START, asof, holm, norm_sf, ols  # noqa: F401  (re-exported)
from m5.calfilter import proportional, state_means
from m5.common import DEV_END, LEAD, load_all  # noqa: F401  (re-exported)

M4_DIR = Path(__file__).resolve().parent
FOMC_FILE = M4_DIR / "data" / "fomc_scheduled.csv"
OFFMARKET_FILE = Path("data/processed/offmarket/offmarket_daily.csv")
MIN_YEARS = 3

# ---------------------------------------------------------------- M4-1 calendar

CAL_START = pd.Timestamp("2001-01-01")  # first date the FOMC file covers (REPORT deviation D1)
EUROPE_CLOSE = {"FGBL": None, "FESX": pd.Timestamp("2007-03-30")}  # FESX: FEZ (US close) before this date


def fomc_dates(path: Path = FOMC_FILE) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(pd.read_csv(path, parse_dates=["statement_date"]).statement_date)


# Post hoc (REPORT deviation D2): settlement times the pre-registration missed. COMEX gold and
# silver settle at 13:30 ET, before every statement; CBOT corn/soy closed at 14:15 ET (13:15 CT),
# at or before the 14:15 release, until the release moved to 14:00 in 2013.
SETTLE_BEFORE = {"GC": None, "SI": None, "ZC": pd.Timestamp("2013-01-01"), "ZS": pd.Timestamp("2013-01-01")}


def event_days(code: str, dates: pd.DatetimeIndex | None = None, timing: str = "prereg") -> pd.DatetimeIndex:
    """Days whose close-to-close return contains the statement.

    timing "prereg" (PREREG 4.1, the primary): US-afternoon closes on the statement day;
    European closes (FGBL; FESX from 2007-03-30) the next weekday.
    timing "settle" (post hoc): also the next weekday for contracts that settle before the release."""
    dates = fomc_dates() if dates is None else dates
    rules = dict(EUROPE_CLOSE)
    if timing == "settle":
        rules.update({k: ("until", v) for k, v in SETTLE_BEFORE.items()})
    elif timing != "prereg":
        raise ValueError(timing)
    if code not in rules:
        return dates
    rule = rules[code]
    if isinstance(rule, tuple):  # next day before the given date (None = always)
        nxt = np.ones(len(dates), bool) if rule[1] is None else (dates < rule[1])
    else:  # next day from the given date on (None = always)
        nxt = np.ones(len(dates), bool) if rule is None else (dates >= rule)
    shifted = dates + pd.offsets.BDay(1)
    return pd.DatetimeIndex(np.where(nxt, shifted, dates)).sort_values()


def event_state(index: pd.DatetimeIndex, code: str, lead: int = 0, timing: str = "prereg") -> pd.Series:
    """1 if the day `lead` weekdays after each date is an event day for `code`."""
    target = index + pd.offsets.BDay(lead) if lead else index
    return pd.Series(np.asarray(target.isin(event_days(code, timing=timing))), index=index)


# ---------------------------------------------------------------- M4-2 states

RISK_DIR = {"ES": 1, "FESX": 1, "NK": 1, "CL": 1, "ZN": -1, "FGBL": -1, "GC": -1, "6J": -1}
STRESS_Q = 0.80


def vix_stress(exog: pd.DataFrame, q: float = STRESS_Q) -> pd.Series:
    """1[vix_t >= expanding q-quantile of vix up to and including t] on the off-market dates."""
    v = exog["vix"].dropna()
    return (v >= v.expanding().quantile(q)).astype(float)


def on_calendar(s: pd.Series, index: pd.DatetimeIndex, limit: int = 5) -> pd.Series:
    """Carry an off-market series forward to a contract's dates (never backward)."""
    return asof(s.to_frame("x"), index, limit=limit)["x"]


# ---------------------------------------------------------------- panel tests (as M5-4)

def fe_test(df: pd.DataFrame, x: list, controls: tuple = ("logvol",), y: str = "y") -> dict:
    """Within-contract OLS of y on controls + x, SEs clustered by calendar month."""
    cols = [y] + list(controls) + list(x)
    d = df.dropna(subset=cols).copy()
    d[cols] = d[cols] - d.groupby("contract")[cols].transform("mean")
    b, V, _ = ols(d[y].to_numpy(), d[cols[1:]].to_numpy(), groups=d["period"].to_numpy())
    out = {"n": len(d), "months": int(d.period.nunique())}
    for j, name in enumerate(cols[1:]):
        t = b[j] / np.sqrt(V[j, j])
        out[name] = {"coef": float(b[j]), "t": float(t), "p": 2 * norm_sf(abs(t))}
    return out


def panel_oos_r2(df: pd.DataFrame, base: list, add: list, start: str = OOS_START) -> dict:
    """Monthly expanding re-estimation from `start`: FE + base + add vs FE + base."""
    df = df.dropna(subset=["y"] + base + add)
    codes = sorted(df.contract.unique())
    dummies = lambda d: np.column_stack([(d.contract == k).to_numpy(float) for k in codes])  # noqa: E731
    ys, ff, bb = [], [], []
    for m in sorted(df.period[df.period >= pd.Timestamp(start)].unique()):
        tr, te = df[df.period < m], df[df.period == m]
        Xb = np.column_stack([dummies(tr)] + [tr[c] for c in base])
        Tb = np.column_stack([dummies(te)] + [te[c] for c in base])
        Xf, Tf = np.column_stack([Xb] + [tr[c] for c in add]), np.column_stack([Tb] + [te[c] for c in add])
        ff += list(Tf @ np.linalg.lstsq(Xf, tr.y, rcond=None)[0])
        bb += list(Tb @ np.linalg.lstsq(Xb, tr.y, rcond=None)[0])
        ys += list(te.y)
    y, f, b = map(np.array, (ys, ff, bb))
    return {"oos_r2": float(1 - ((y - f) ** 2).sum() / ((y - b) ** 2).sum()), "days": len(y)}


# ---------------------------------------------------------------- proportional filter (PREREG 4.3)

def proportional_filter(contracts: dict, members: list, pnl_state: dict, filter_state: dict,
                        refit: str = "Q", min_years: int = MIN_YEARS, dev_end: str | None = None):
    """M5's quarterly proportional rule with a state that may depend on the position.

    pnl_state[code]   : bool Series on the contract's P&L days (state relevant to that day's P&L)
    filter_state[code]: bool Series on the contract's decision days (state the target at t is for)
    Contracts outside `members` get filter 1. Returns (filter table, estimate log)."""
    rows, log = [], []
    for code, c in contracts.items():
        if code not in members:
            rows.append(pd.DataFrame({"date": c["data"].index, "contract": code,
                                      "filter_close_t": 1.0, "vintage": pd.NaT}))
    obs = []
    for code in members:
        c = contracts[code]
        idx = c["live"].index[c["live"]]
        if dev_end is not None:
            idx = idx[idx <= pd.Timestamp(dev_end)]
        st = pnl_state[code].reindex(idx)
        ok = st.notna()
        obs.append(pd.DataFrame({"pnl": c["gross"].reindex(idx)[ok], "state": st[ok].astype(bool)}))
    obs = pd.concat(obs).sort_index().dropna()
    first = obs.index.min()
    dates = pd.DatetimeIndex(sorted(set().union(*(contracts[k]["data"].index for k in members))))
    est = {}
    for period in dates.to_period(refit).unique():
        s = period.start_time
        if first > s - pd.DateOffset(years=min_years):
            est[period] = (1.0, 1.0, s)
            continue
        past = obs[obs.index < s]
        m_in, m_out, n_in, n_out = state_means(past.pnl, past.state)
        f_in, f_out = proportional(m_in, m_out)
        est[period] = (f_in, f_out, s)
        log.append({"vintage": s.date(), "m_in_ann": m_in * 252, "m_out_ann": m_out * 252,
                    "n_in": n_in, "n_out": n_out, "f_in": f_in, "f_out": f_out})
    for code in members:
        idx = contracts[code]["data"].index
        per = idx.to_period(refit)
        f_in = np.array([est[p][0] for p in per])
        f_out = np.array([est[p][1] for p in per])
        st = filter_state[code].reindex(idx).fillna(False).astype(bool).to_numpy()
        rows.append(pd.DataFrame({"date": idx, "contract": code, "filter_close_t": np.where(st, f_in, f_out),
                                  "vintage": [est[p][2] for p in per]}))
    filt = pd.concat(rows, ignore_index=True).sort_values(["contract", "date"]).reset_index(drop=True)
    return filt, pd.DataFrame(log)
