"""Pre-registered proportional filter for calendar states (PREREG 4.6; M5-3 and M5-4).

Built only for a group whose gate test passed. At the start of every quarter s, for
each group G, using daily gross A2 combo P&L dated before s, pooled over G's contracts:

    m_in  = mean P&L on days inside the state (season / roll window)
    m_out = mean P&L on days outside it

    if max(m_in, m_out) <= 0:  filter = 1 in both states (no usable relative information)
    else: the better state keeps 1, the worse one gets clip(m_worse / m_better, 0, 1)

The estimates switch on once some member of G has 3 full years of P&L before s;
until then the filter is 1. A target set at close t first earns on t + 2 weekdays,
so the filter at t takes the value of the state of t + 2 weekdays (calendar known in
advance). Contracts outside every group get filter 1.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

MIN_YEARS = 3


def state_means(pnl: pd.Series, state: pd.Series) -> tuple[float, float, int, int]:
    m_in, m_out = pnl[state].mean(), pnl[~state].mean()
    return float(m_in), float(m_out), int(state.sum()), int((~state).sum())


def proportional(m_in: float, m_out: float) -> tuple[float, float]:
    if not (np.isfinite(m_in) and np.isfinite(m_out)) or max(m_in, m_out) <= 0:
        return 1.0, 1.0
    if m_in >= m_out:
        return 1.0, float(np.clip(m_out / m_in, 0, 1))
    return float(np.clip(m_in / m_out, 0, 1)), 1.0


def build(contracts: dict, groups: dict[str, list[str]], state_fn: Callable, lead: int = 2,
          refit: str = "Q", min_years: int = MIN_YEARS):
    """contracts: m5.common.load_all output. state_fn(index, code, lead) -> bool Series.

    Returns (filter DataFrame date, contract, filter_close_t, vintage; estimate log)."""
    rows, log = [], []
    grouped = {c for codes in groups.values() for c in codes}
    for code, c in contracts.items():
        if code not in grouped:
            rows.append(pd.DataFrame({"date": c["data"].index, "contract": code,
                                      "filter_close_t": 1.0, "vintage": pd.NaT}))
    for gname, codes in groups.items():
        obs = []
        for code in codes:
            c = contracts[code]
            live = c["live"]
            pnl = c["gross"][live]
            obs.append(pd.DataFrame({"pnl": pnl, "state": state_fn(pnl.index, code, 0).to_numpy()}))
        obs = pd.concat(obs).sort_index()
        first_pnl = obs.index.min()
        dates = pd.DatetimeIndex(sorted(set().union(*(contracts[k]["data"].index for k in codes))))
        est = {}
        for period in dates.to_period(refit).unique():
            s = period.start_time
            if first_pnl > s - pd.DateOffset(years=min_years):
                est[period] = (1.0, 1.0, s)
                continue
            past = obs[obs.index < s]
            m_in, m_out, n_in, n_out = state_means(past.pnl, past.state.astype(bool))
            f_in, f_out = proportional(m_in, m_out)
            est[period] = (f_in, f_out, s)
            log.append({"group": gname, "vintage": s.date(), "m_in_ann": m_in * 252, "m_out_ann": m_out * 252,
                        "n_in": n_in, "n_out": n_out, "f_in": f_in, "f_out": f_out})
        for code in codes:
            idx = contracts[code]["data"].index
            per = idx.to_period(refit)
            f_in = np.array([est[p][0] for p in per])
            f_out = np.array([est[p][1] for p in per])
            vint = [est[p][2] for p in per]
            st = state_fn(idx, code, lead).to_numpy()
            rows.append(pd.DataFrame({"date": idx, "contract": code,
                                      "filter_close_t": np.where(st, f_in, f_out), "vintage": vint}))
    filt = pd.concat(rows, ignore_index=True).sort_values(["contract", "date"]).reset_index(drop=True)
    return filt, pd.DataFrame(log)
