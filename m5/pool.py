"""Pooled HMM regime filter: one HMM shared by a group of contracts (trials M5-1, M5-2).

    python -m m5.pool --pool class     # M5-1: one HMM per asset class
    python -m m5.pool --pool global    # M5-2: one HMM for all 13 contracts

Same schedule and settings as M3 trial #1 (m3/adaptor.py), except that the parameters
are shared within the pool:

- at the start of every refit period, every contract in the pool with at least
  `min_train` valid feature rows dated before the period joins the fit;
- each member is standardised with its own training-window mean/std, then all members'
  sequences are fitted together by multi-sequence Baum-Welch (m5/hmm_pool.py);
- states are relabelled by their mean of `order_by` (ascending);
- each member runs the forward filter with the pooled parameters on its own standardised
  feature sequence; the rows of the period give its filter.

Nothing dated in the period or later touches the parameters of that period, for any
member. With a single contract the output is identical to m3.adaptor.run_contract.

Writes m5_outputs/pool_{class|global}_k{K}_{refit}_{mapping}_filter.csv (date, contract,
filter_close_t, p_state_k_close_t, vintage) and ..._fits.csv (one row per contract x
refit, same columns as M3's fits plus the pool's name and size).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from m3.adaptor import REFIT_FREQ, AdaptorConfig, _rescale, _scores
from m3.features import build_features

from .hmm_pool import MultiSeqGaussianHMM

M3_TRIAL1 = AdaptorConfig(n_states=3, refit="quarter", mapping="top")  # frozen M3 settings


def _prepare(ret: pd.Series, cfg: AdaptorConfig):
    F = build_features(ret, cfg.features)
    valid = F.notna().all(axis=1)
    if not valid.any():
        raise ValueError("no valid feature rows")
    F = F.loc[valid.idxmax():]  # drop the leading warm-up block
    return F, valid.loc[F.index]


def _pooled_warm(prev, members):
    """Previous pooled parameters re-expressed in each continuing member's new
    standardisation, averaged over those members (= M3's _rescale for one member)."""
    if prev is None:
        return None
    params, old = prev
    re = [_rescale(params, *old[code], mu, sd) for code, _, _, mu, sd in members if code in old]
    if not re:
        return params
    if len(re) == 1:
        return re[0]
    pi, A = params[0], params[1]
    return pi, A, np.mean([r[2] for r in re], axis=0), np.mean([r[3] for r in re], axis=0)


def run_pool(rets: dict[str, pd.Series], cfg: AdaptorConfig = M3_TRIAL1, name: str = "pool"):
    """Returns ({contract: daily output DataFrame}, list of per-contract x refit fit records)."""
    if cfg.order_by not in cfg.features:
        raise ValueError("order_by must be one of the features")
    K, k_order, scores = cfg.n_states, cfg.features.index(cfg.order_by), _scores(cfg)
    prepared = {code: _prepare(r, cfg) for code, r in rets.items()}
    outs = {}
    for code, r in rets.items():
        o = pd.DataFrame(np.nan, index=r.index, columns=["filter"] + [f"p_state_{k}" for k in range(K)])
        o["vintage"] = pd.NaT
        outs[code] = o
    union = pd.DatetimeIndex(sorted(set().union(*(F.index for F, _ in prepared.values()))))

    fits, prev = [], None
    for period in union.to_period(REFIT_FREQ[cfg.refit]).unique():
        start, end = period.start_time, (period + 1).start_time
        members = []
        for code, (F, valid) in prepared.items():
            n_train = int((F.index < start).sum())
            train_ok = valid.iloc[:n_train]
            if train_ok.sum() < cfg.min_train:
                continue
            tv = F.iloc[:n_train][train_ok]
            mu, sd = tv.mean().to_numpy(), tv.std(ddof=1).to_numpy()
            Z = ((F[F.index < end] - mu) / sd).to_numpy()
            members.append((code, Z, n_train, mu, sd))
        if not members:
            continue

        model = MultiSeqGaussianHMM(n_states=K, n_init=cfg.n_init, seed=cfg.seed + period.ordinal)
        model.fit_multi([Z[:n] for _, Z, n, _, _ in members], warm_start=_pooled_warm(prev, members))
        model.reorder(np.argsort(model.means[:, k_order]))
        prev = (model.params(), {code: (mu, sd) for code, _, _, mu, sd in members})

        n_pool = sum(int(prepared[c][1].iloc[:n].sum()) for c, _, n, _, _ in members)
        stat = np.linalg.matrix_power(model.A, 1000)[0]
        for code, Z, n_train, mu, sd in members:
            F, valid = prepared[code]
            probs = model.filter(Z)[n_train:]  # rows of this period
            days = F.index[n_train:len(Z)]
            probs[~valid.loc[days].to_numpy()] = np.nan  # no output on days with missing features
            o = outs[code]
            o.loc[days, "filter"] = probs @ scores
            o.loc[days, [f"p_state_{k}" for k in range(K)]] = probs
            o.loc[days, "vintage"] = start

            means = model.means * sd + mu  # back to this contract's feature units
            rec = {"contract": code, "pool": name, "pool_size": len(members), "vintage": start.date(),
                   "train_start": F.index[0].date(), "train_end": F.index[n_train - 1].date(),
                   "n_train": int(valid.iloc[:n_train].sum()), "n_train_pool": n_pool,
                   "loglik": model.loglik, "bic": model.bic(n_pool, len(cfg.features)),
                   "n_iter": model.n_iter, "converged": model.converged}
            for k in range(K):
                rec[f"dur_{k}"] = 1 / max(1 - model.A[k, k], 1e-12)
                rec[f"share_{k}"] = stat[k]
                for j, f in enumerate(cfg.features):
                    rec[f"{f}_{k}"] = means[k, j]
            fits.append(rec)
    return outs, fits


def groups(universe: pd.DataFrame, pool: str) -> dict[str, list[str]]:
    if pool == "global":
        return {"global": list(universe.code)}
    if pool == "class":
        return {c: list(g.code) for c, g in universe.groupby("asset_class", sort=False)}
    raise ValueError(f"unknown pool {pool!r}")


def run(data_dir: Path, out_dir: Path | None, pool: str, cfg: AdaptorConfig = M3_TRIAL1,
        universe: pd.DataFrame | None = None, tag: str = ""):
    """universe: optional table (code, asset_class[, data_dir]) instead of data_dir/universe.csv;
    out_dir None = return the outputs without writing them."""
    uni = pd.read_csv(data_dir / "universe.csv") if universe is None else universe
    data_dirs = uni.set_index("code")["data_dir"] if "data_dir" in uni else None
    filters, fits = [], []
    for gname, codes in groups(uni, pool).items():
        t0 = time.perf_counter()
        rets = {}
        for c in codes:
            dd = data_dirs[c] if data_dirs is not None else None
            d_dir = Path(dd) if isinstance(dd, str) and dd else data_dir
            rets[c] = pd.read_csv(d_dir / f"{c}.csv", index_col="date", parse_dates=True)["ret"]
        outs, recs = run_pool(rets, cfg, gname)
        for code, out in outs.items():
            out = out.rename(columns=lambda c: c if c == "vintage" else f"{c}_close_t")
            filters.append(out.assign(contract=code).rename_axis("date").reset_index())
        fits += recs
        print(f"{gname:7s} {','.join(codes):18s} refits={len({r['vintage'] for r in recs}):3d}  "
              f"{time.perf_counter() - t0:6.1f}s")
    filt = pd.concat(filters, ignore_index=True)
    filt = filt[["date", "contract"] + [c for c in filt.columns if c not in ("date", "contract")]]
    if out_dir is None:
        return filt, pd.DataFrame(fits)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pool_{pool}_k{cfg.n_states}_{cfg.refit}_{cfg.mapping}" + (f"_{tag}" if tag else "")
    filt.to_csv(out_dir / f"{stem}_filter.csv", index=False)
    pd.DataFrame(fits).to_csv(out_dir / f"{stem}_fits.csv", index=False)
    print("OUTPUT", out_dir / f"{stem}_filter.csv")
    return filt, pd.DataFrame(fits)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pool", choices=["class", "global"], required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    a = p.parse_args()
    run(a.data_dir, a.out_dir, a.pool)
