"""Performance statistics and the paired Approach 3 vs Approach 2 comparison.

Provisional: M6 owns the final significance and robustness tests.
"""

import numpy as np
import pandas as pd
from math import erf, sqrt

ANN = 252


def _norm_sf(z: float) -> float:
    return 0.5 * (1 - erf(z / sqrt(2)))


def max_drawdown(r: pd.Series) -> float:
    equity = (1 + r.fillna(0)).cumprod()
    return float((equity / equity.cummax() - 1).min())


def perf(r: pd.Series) -> dict:
    r = r.dropna()
    ann_ret, vol = r.mean() * ANN, r.std(ddof=1) * np.sqrt(ANN)
    mdd = max_drawdown(r)
    return {"ann_ret": ann_ret, "ann_vol": vol, "sharpe": ann_ret / vol if vol > 0 else np.nan,
            "max_dd": mdd, "calmar": ann_ret / abs(mdd) if mdd < 0 else np.nan,
            "start": r.index[0].date(), "end": r.index[-1].date(), "days": len(r)}


def newey_west_t(x: pd.Series, lags: int = 10) -> float:
    """t-statistic of the mean with a Newey-West (Bartlett) long-run variance."""
    x = x.dropna().to_numpy()
    e = x - x.mean()
    lrv = e @ e / len(x)
    for L in range(1, lags + 1):
        lrv += 2 * (1 - L / (lags + 1)) * (e[L:] @ e[:-L]) / len(x)
    if lrv <= 1e-30:  # constant series: no variance, no test
        return float("nan")
    return float(x.mean() / np.sqrt(lrv / len(x)))


def sharpe_diff_test(ra: pd.Series, rb: pd.Series) -> dict:
    """Jobson-Korkie test with Memmel's correction (iid normal returns assumed)."""
    d = pd.concat([ra, rb], axis=1).dropna()
    a, b = d.iloc[:, 0], d.iloc[:, 1]
    T = len(d)
    sa, sb = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    rho = a.corr(b)
    theta = (2 - 2 * rho + 0.5 * (sa**2 + sb**2 - 2 * sa * sb * rho**2)) / T
    if theta <= 0:  # identical series
        return {"d_sharpe": 0.0, "z": 0.0, "p_two_sided": 1.0, "corr": rho}
    z = (sa - sb) / np.sqrt(theta)
    return {"d_sharpe": (sa - sb) * np.sqrt(ANN), "z": z, "p_two_sided": 2 * _norm_sf(abs(z)),
            "corr": rho}


Z_975 = 1.959964


def block_bootstrap_sharpe_diff(ra: pd.Series, rb: pd.Series, block: int = 20, reps: int = 5000,
                                seed: int = 0, chunk: int = 500, return_boot: bool = False) -> dict:
    """Circular block bootstrap of the annualised Sharpe difference SR(a) - SR(b).

    Days are resampled in blocks of `block` consecutive days, the same days for both
    series, so fat tails, volatility clustering and the a-b correlation are kept.
    Returns the observed difference, a 95% percentile interval, the bootstrap standard
    error, the smallest difference that would be significant at 5% (two-sided) given
    that error, and a two-sided p-value from the null-centred bootstrap distribution.
    """
    d = pd.concat([ra, rb], axis=1).dropna()
    A, B = d.iloc[:, 0].to_numpy(), d.iloc[:, 1].to_numpy()
    T = len(A)
    n_blocks = -(-T // block)
    rng = np.random.default_rng(seed)
    sr = lambda x: x.mean(axis=-1) / x.std(axis=-1, ddof=1)
    obs = float((sr(A) - sr(B)) * np.sqrt(ANN))
    boot = np.empty(reps)
    for s in range(0, reps, chunk):
        n = min(chunk, reps - s)
        starts = rng.integers(0, T, (n, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(n, -1)[:, :T]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[s:s + n] = (sr(A[idx]) - sr(B[idx])) * np.sqrt(ANN)
    boot = np.nan_to_num(boot)  # identical series give 0/0
    se = float(boot.std(ddof=1))
    lo, hi = np.quantile(boot, [0.025, 0.975])
    p = float(np.mean(np.abs(boot - boot.mean()) >= abs(obs) - 1e-12))
    out = {"d_sharpe": obs, "ci_low": float(lo), "ci_high": float(hi), "se_boot": se,
           "min_gap_5pct": Z_975 * se, "p_boot": p, "block": block, "reps": reps}
    if return_boot:
        out["boot"] = boot
    return out


def paired(r_treat: pd.Series, r_ctrl: pd.Series, lags: int = 10, block: int = 20,
           reps: int = 5000, seed: int = 0) -> dict:
    """Treatment (A3) vs control (A2) on the same days.

    Sharpe difference: block bootstrap (main result), Jobson-Korkie p kept for reference.
    Mean difference: the treatment is first rescaled to the control's realised vol
    (ex post), because A3 runs at a fraction of A2's exposure and a raw mean difference
    would mostly measure that.
    """
    d = pd.concat([r_treat, r_ctrl], axis=1).dropna()
    t, c = d.iloc[:, 0], d.iloc[:, 1]
    out = block_bootstrap_sharpe_diff(t, c, block, reps, seed)
    diff_vm = vol_matched(t, c) - c
    out.update({"p_jk": sharpe_diff_test(t, c)["p_two_sided"], "corr": float(t.corr(c)),
                "ann_mean_diff_vm": diff_vm.mean() * ANN,
                "nw_t_vm": newey_west_t(diff_vm, lags),
                "tracking_vol_vm": diff_vm.std(ddof=1) * np.sqrt(ANN)})
    return out


def vol_matched(r: pd.Series, target: pd.Series) -> pd.Series:
    """Ex-post rescale r to target's realised vol over the same window (display only)."""
    d = pd.concat([r, target], axis=1).dropna()
    return r * (d.iloc[:, 1].std() / d.iloc[:, 0].std())
