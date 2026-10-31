"""Causal HMM regime filter for one contract (try 1: fixed number of states).

Schedule: at the start of every refit period (year / quarter / month) the HMM is refitted
(Baum-Welch) on all feature rows dated before that period, standardised with that
training window's mean/std, warm-started from the previous fit. The forward filter with
those parameters gives P(state_t | x_1..x_t) for every day of the period. Nothing dated
in the period or later touches its parameters.

States are relabelled after every refit by their mean of `order_by` (ascending), so state
0 is always the least trend-friendly and state K-1 the most. The filter in [0, 1] is

    mapping="top":     filter_t = P(state_t = K-1)                   (trending-state prob.)
    mapping="linear":  filter_t = sum_k P(state_t = k) * k / (K - 1)

Both are identical for K = 2. Approach 3 = approach 2 target x filter.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import DEFAULT_FEATURES, build_features
from .hmm import GaussianHMM

REFIT_FREQ = {"year": "Y", "quarter": "Q", "month": "M"}


def period_seed(base: int, period: pd.Period) -> int:
    """Seed of a refit period: base + the period's ordinal (counted from 1970). Periods before
    1970 have negative ordinals, which numpy rejects, so a negative seed is taken mod 2**32;
    every seed from 1970 on is unchanged (eras study deviation #4)."""
    s = base + period.ordinal
    return s if s >= 0 else s % 2**32


@dataclass(frozen=True)
class AdaptorConfig:
    n_states: int = 2
    features: tuple = DEFAULT_FEATURES
    order_by: str = "er60"
    mapping: str = "top"
    refit: str = "year"
    min_train: int = 750  # ~3 years of valid feature rows before the first refit
    n_init: int = 4
    seed: int = 0


def _rescale(params, mu_old, sd_old, mu_new, sd_new):
    """Re-express fitted parameters in a new standardisation (for warm starts)."""
    pi, A, means, covs = params
    means = (means * sd_old + mu_old - mu_new) / sd_new
    r = sd_old / sd_new
    covs = covs * np.outer(r, r)[None]
    return pi, A, means, covs


def _scores(cfg: AdaptorConfig) -> np.ndarray:
    K = cfg.n_states
    if K == 1:
        return np.ones(1)
    if cfg.mapping == "top":
        return np.eye(K)[K - 1]
    if cfg.mapping == "linear":
        return np.arange(K) / (K - 1)
    raise ValueError(f"unknown mapping {cfg.mapping!r}")


def run_contract(ret: pd.Series, cfg: AdaptorConfig = AdaptorConfig(),
                 exog: pd.DataFrame | None = None):
    """Returns (daily output DataFrame, list of per-refit fit records).

    `exog` is the off-market table used by any "x:<column>" feature in cfg.features.
    """
    if cfg.order_by not in cfg.features:
        raise ValueError("order_by must be one of the features")
    F = build_features(ret, cfg.features, exog)
    valid = F.notna().all(axis=1)
    if not valid.any():
        raise ValueError("no valid feature rows")
    F = F.loc[valid.idxmax():]  # drop the leading warm-up block
    valid = valid.loc[F.index]
    K, k_order, scores = cfg.n_states, cfg.features.index(cfg.order_by), _scores(cfg)

    out = pd.DataFrame(np.nan, index=ret.index,
                       columns=["filter"] + [f"p_state_{k}" for k in range(K)])
    out["vintage"] = pd.NaT
    fits, prev = [], None
    for period in F.index.to_period(REFIT_FREQ[cfg.refit]).unique():
        start, end = period.start_time, (period + 1).start_time
        n_train = int((F.index < start).sum())
        train_ok = valid.iloc[:n_train]
        if train_ok.sum() < cfg.min_train:
            continue
        tv = F.iloc[:n_train][train_ok]
        mu, sd = tv.mean().to_numpy(), tv.std(ddof=1).to_numpy()
        Z = ((F[F.index < end] - mu) / sd).to_numpy()

        model = GaussianHMM(n_states=K, n_init=cfg.n_init, seed=period_seed(cfg.seed, period))
        warm = _rescale(prev[0], prev[1], prev[2], mu, sd) if prev else None
        model.fit(Z[:n_train], warm_start=warm)
        model.reorder(np.argsort(model.means[:, k_order]))
        prev = (model.params(), mu, sd)

        probs = model.filter(Z)[n_train:]  # rows of this period
        days = F.index[n_train:len(Z)]
        probs[~valid.loc[days].to_numpy()] = np.nan  # no output on days with missing features
        out.loc[days, "filter"] = probs @ scores
        out.loc[days, [f"p_state_{k}" for k in range(K)]] = probs
        out.loc[days, "vintage"] = start

        means = model.means * sd + mu  # back to feature units
        stat = np.linalg.matrix_power(model.A, 1000)[0]
        rec = {"vintage": start.date(), "train_start": F.index[0].date(),
               "train_end": F.index[n_train - 1].date(), "n_train": int(train_ok.sum()),
               "loglik": model.loglik, "bic": model.bic(int(train_ok.sum()), len(cfg.features)),
               "n_iter": model.n_iter, "converged": model.converged}
        for k in range(K):
            rec[f"dur_{k}"] = 1 / max(1 - model.A[k, k], 1e-12)
            rec[f"share_{k}"] = stat[k]
            for j, f in enumerate(cfg.features):
                rec[f"{f}_{k}"] = means[k, j]
        fits.append(rec)
    return out, fits
