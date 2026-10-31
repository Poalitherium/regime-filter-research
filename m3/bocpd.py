"""Bayesian online changepoint detection (Adams & MacKay 2007), try 3 of the regime adaptor.

Each segment has an unknown mean and variance with a Normal-Inverse-Gamma prior, so the
one-step predictive is Student-t. The hazard (probability that a new segment starts on
any day) is constant. Only the forward recursion is used, so every output at t depends
on observations up to t only.

Applied to vol-standardised returns z_t = ret_t / sigma_{t-1} (60-day realised vol up to
the previous day), so that the detector reacts to breaks in drift rather than in variance.

Outputs per day:
    cp_recent : P(run length <= k), a changepoint within the last k days
    seg_mu    : posterior mean drift of the current segment (mixture over run lengths)
    exp_rl    : expected run length
"""

from dataclasses import dataclass
from math import lgamma, log, pi

import numba
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BOCPDConfig:
    hazard: float = 1 / 250
    mu0: float = 0.0
    kappa0: float = 1.0
    alpha0: float = 3.0
    beta0: float = 2.0  # prior variance beta0 / (alpha0 - 1) = 1 on standardised returns
    r_max: int = 1500  # run lengths beyond this are folded into the last bucket
    k_recent: int = 20
    vol_window: int = 60


@numba.njit(cache=True)
def _bocpd(x, hazard, mu0, kappa0, alpha0, beta0, r_max, k_recent):
    T = len(x)
    p = np.zeros(r_max)
    p[0] = 1.0
    mu = np.full(r_max, mu0)
    ka = np.full(r_max, kappa0)
    al = np.full(r_max, alpha0)
    be = np.full(r_max, beta0)
    cp_recent = np.full(T, np.nan)
    seg_mu = np.full(T, np.nan)
    exp_rl = np.full(T, np.nan)
    pred = np.empty(r_max)
    newp = np.empty(r_max)
    n_live = 1  # run lengths 0..n_live-1 can carry mass
    started = False
    for t in range(T):
        xt = x[t]
        if np.isnan(xt):
            if started:  # no observation: carry the outputs, keep the posterior
                cp_recent[t] = cp_recent[t - 1]
                seg_mu[t] = seg_mu[t - 1]
                exp_rl[t] = exp_rl[t - 1]
            continue
        started = True
        # one-step predictive density of x_t under each run length (Student-t)
        for r in range(n_live):
            nu = 2.0 * al[r]
            s2 = be[r] * (ka[r] + 1.0) / (al[r] * ka[r])
            d = (xt - mu[r]) ** 2 / (nu * s2)
            pred[r] = np.exp(lgamma((nu + 1.0) / 2.0) - lgamma(nu / 2.0)
                             - 0.5 * log(nu * pi * s2) - (nu + 1.0) / 2.0 * np.log1p(d))
        # run-length recursion: grow (no break) or reset to 0 (break)
        cp_mass = 0.0
        for r in range(r_max):
            newp[r] = 0.0
        for r in range(n_live):
            w = p[r] * pred[r]
            cp_mass += w * hazard
            if r + 1 < r_max:
                newp[r + 1] += w * (1.0 - hazard)
            else:
                newp[r_max - 1] += w * (1.0 - hazard)
        newp[0] = cp_mass
        total = 0.0
        for r in range(r_max):
            total += newp[r]
        n_new = min(n_live + 1, r_max)
        # posterior parameters: run length r >= 1 has absorbed x_t; r = 0 restarts at the prior.
        # Walk downwards so each source (r - 1) is read before it is overwritten. Once the
        # folded last bucket is live it keeps its own statistics (mass there is ~exp(-H*r_max)).
        last = r_max - 1
        for r in range(n_new - 1, 0, -1):
            src = last if (r == last and n_live == r_max) else r - 1
            m, k, a, b = mu[src], ka[src], al[src], be[src]
            mu[r] = (k * m + xt) / (k + 1.0)
            ka[r] = k + 1.0
            al[r] = a + 0.5
            be[r] = b + k * (xt - m) ** 2 / (2.0 * (k + 1.0))
        mu[0], ka[0], al[0], be[0] = mu0, kappa0, alpha0, beta0
        n_live = n_new
        cr, sm, er = 0.0, 0.0, 0.0
        for r in range(n_live):
            p[r] = newp[r] / total
            if r <= k_recent:
                cr += p[r]
            sm += p[r] * mu[r]
            er += p[r] * r
        cp_recent[t] = cr
        seg_mu[t] = sm
        exp_rl[t] = er
    return cp_recent, seg_mu, exp_rl


def run_bocpd(x: np.ndarray, cfg: BOCPDConfig = BOCPDConfig()):
    """Forward BOCPD over a 1-d array (NaN = no observation that day)."""
    return _bocpd(np.asarray(x, dtype=float), cfg.hazard, cfg.mu0, cfg.kappa0, cfg.alpha0,
                  cfg.beta0, cfg.r_max, cfg.k_recent)


def standardise(ret: pd.Series, window: int = 60) -> pd.Series:
    """z_t = ret_t / realised daily vol over the `window` days up to t-1 (no use of ret_t)."""
    sd = ret.rolling(window, min_periods=window).std(ddof=1).shift(1)
    return (ret / sd.where(sd > 0)).rename("z")


def contract_features(ret: pd.Series, cfg: BOCPDConfig = BOCPDConfig()) -> pd.DataFrame:
    z = standardise(ret, cfg.vol_window)
    cp, mu, rl = run_bocpd(z.to_numpy(), cfg)
    return pd.DataFrame({"z": z, f"cp{cfg.k_recent}": cp, "seg_mu": mu, "exp_rl": rl}, index=ret.index)
