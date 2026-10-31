"""Trial 4: portfolio-level, correlation-aware volatility targeting.

A2 sizes each contract to its own vol and ignores co-movement. Here the whole book is
rescaled every day so the *portfolio's* ex-ante vol, from an EWMA covariance of contract
returns, stays at its typical level:

    sigma_p,t = sqrt(252 * w_t' Sigma_t w_t),  w = A2 combo targets / N in the book
    k_t       = clip(median(sigma_p,<=t) / sigma_p,t, 0, 3)

Sigma_t uses returns up to and including t (known at t's close), and the target is the
expanding median of sigma_p, so nothing after t is used. A4 = A2 x k_t.

    python -m m3.portfolio_vol      # writes m3_outputs/portvol_filter.csv (filter = k)
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import in_book
from m2.pipeline import build_m2

from .run_real_data import CODES

FACTORS = ["tsmom", "ma_crossover", "donchian"]
HALFLIFE = 60
CAP = 3.0
MIN_OBS = 250


def a2_targets(d: pd.DataFrame, code: str) -> pd.DataFrame:
    m2 = build_m2(pd.DataFrame({"date": d.index, "contract": code,
                                "close": d.close_adj.to_numpy(), "return": d.ret.to_numpy()}))
    a2 = m2.pivot(index="date", columns="factor", values="approach2_target_close_t")[FACTORS]
    a2["combo"] = a2.mean(axis=1, skipna=False)
    return a2


def ewma_portfolio_vol(rets: pd.DataFrame, weights: pd.DataFrame, halflife: int = HALFLIFE) -> pd.Series:
    """Annualised ex-ante portfolio vol; Sigma_t = lam*Sigma_{t-1} + (1-lam) r_t r_t' (zero mean)."""
    lam = 0.5 ** (1 / halflife)
    R, W = rets.to_numpy(float), weights.to_numpy(float)
    S = np.zeros((R.shape[1], R.shape[1]))
    out = np.full(len(R), np.nan)
    for t in range(len(R)):
        r = R[t]
        S = lam * S + (1 - lam) * np.outer(r, r)
        if t >= halflife:
            w = W[t]
            out[t] = np.sqrt(max(w @ S @ w, 0.0) * 252)
    return pd.Series(out, index=rets.index)


def scale_factor(sigma_p: pd.Series, cap: float = CAP, min_obs: int = MIN_OBS) -> pd.Series:
    s = sigma_p.where(sigma_p > 0)
    target = s.expanding(min_periods=min_obs).median()
    return (target / s).clip(upper=cap)


def compute(data_dir: Path = Path("data/final")):
    data = {c: pd.read_csv(data_dir / f"{c}.csv", index_col="date", parse_dates=True) for c in CODES}
    targets = {c: a2_targets(d, c)["combo"] for c, d in data.items()}
    union = sorted(set().union(*(d.index for d in data.values())))
    book = in_book(pd.DataFrame({c: t.notna() for c, t in targets.items()}).reindex(union))
    rets = pd.DataFrame({c: d.ret for c, d in data.items()}).reindex(union).fillna(0.0)
    pos = pd.DataFrame(targets).reindex(union).ffill(limit=5).where(book, 0.0).fillna(0.0)
    n = book.sum(axis=1).replace(0, np.nan)
    sigma_p = ewma_portfolio_vol(rets, pos.div(n, axis=0).fillna(0.0))
    k = scale_factor(sigma_p)
    return data, sigma_p, k


def main(out: Path = Path("m3_outputs/portvol_filter.csv")):
    data, sigma_p, k = compute()
    rows = []
    for code, d in data.items():
        kk = k.reindex(d.index)
        rows.append(pd.DataFrame({"date": d.index, "contract": code, "filter_close_t": kk.values,
                                  "sigma_p_close_t": sigma_p.reindex(d.index).values}))
    res = pd.concat(rows, ignore_index=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    kd = k.loc[:"2014-12-31"].dropna()
    print(f"k from {kd.index[0].date()}: mean {kd.mean():.2f}, median {kd.median():.2f}, "
          f"p5 {kd.quantile(.05):.2f}, p95 {kd.quantile(.95):.2f}, at cap {(kd >= CAP).mean():.1%}")
    print("OUTPUT", out)
    return res


if __name__ == "__main__":
    main()
