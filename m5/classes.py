"""Phase 1 (descriptive): approaches 1/2/3 broken down by asset class.

    python -m m5.classes --filter-file m3_outputs/hmm_k3_quarter_top_filter.csv --label hmm
    python -m m5.classes --filter-file m3_outputs/bocpd_h250_filter.csv --label bocpd
    python -m m5.classes --diagnostics m3_outputs/hmm_k3_quarter_top --label hmm   # fit diagnostics

Same book, engine and window as `python -m backtest.run` (dev window only). Within each
asset class, the in-book contracts are equally weighted (a class sub-portfolio). Dates
are those of the 13-contract portfolio (at least `min_contracts` in the book), so the
class results add up to the headline result:

    full_t = sum_c (n_c,t / n_t) * class_c,t

A3 - A2 Sharpe differences of all classes are bootstrapped jointly (circular blocks,
the same resampled days for every class), which gives their covariance and so a test
of whether the differences are equal across classes:

    Wald (equality):  W = (Cd)' (C S C')^-1 (Cd) ~ chi2(C-1),  C = [I | -1]
    Cochran's Q:      sum_c w_c (d_c - d_w)^2, w_c = 1 / se_c^2 (ignores the covariance)
    Wald (all zero):  d' S^-1 d ~ chi2(C)

With 2-3 contracts per class these tests have little power; not rejecting does not
show that the classes behave alike.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from backtest.metrics import ANN, Z_975, perf
from backtest.run import load_targets
from m3.features import build_features
from m3.predictability import chi2_sf

CLASSES = ("equity", "energy", "metal", "rates", "fx", "agri")
Z_80 = 0.841621  # one-sided 80% power quantile
DEV_END = "2014-12-31"


def class_returns(data_dir: Path, filter_file: Path, factor: str = "combo", delay: int = 1,
                  costs: bool = True, dev_end: str = DEV_END, min_contracts: int = 10):
    """Daily net returns of the full book and of every class sub-portfolio, A1/A2/A3.

    Returns (DataFrame with columns '<class>_<approach>' and 'all_<approach>',
    DataFrame '<class>_n' of in-book counts, universe)."""
    contracts, uni = load_targets(data_dir, filter_file)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    valid = pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union)
    book = in_book(valid)
    classes = [c for c in CLASSES if c in set(uni.asset_class)] + sorted(set(uni.asset_class) - set(CLASSES))
    members = {cl: list(uni.code[uni.asset_class == cl]) for cl in classes}
    out, counts = {}, {}
    for ap in ("A1", "A2", "A3"):
        pnl = {k: contract_pnl(c[ap][factor], c["data"].ret, c["data"].roll,
                               COST_BPS[c["asset"]] if costs else 0.0, delay)
               for k, c in contracts.items()}
        full = portfolio(pnl, book, min_contracts).loc[:dev_end]
        out[f"all_{ap}"] = full.net
        counts["all_n"] = full.n_contracts
        for cl, codes in members.items():
            sub = portfolio({k: pnl[k] for k in codes}, book[codes], 1).reindex(full.index)
            out[f"{cl}_{ap}"] = sub.net
            counts[f"{cl}_n"] = sub.n_contracts.fillna(0)
    return pd.DataFrame(out), pd.DataFrame(counts), classes


def joint_bootstrap(treat: pd.DataFrame, ctrl: pd.DataFrame, block: int = 20, reps: int = 5000,
                    seed: int = 0, chunk: int = 500):
    """Annualised Sharpe differences treat - ctrl for every column, bootstrapped jointly.

    Same resampling scheme (and random stream) as backtest.metrics.block_bootstrap_sharpe_diff,
    so each column's marginal results equal a separate call on that column."""
    A, B = treat.to_numpy(), ctrl.to_numpy()
    T, C = A.shape
    n_blocks = -(-T // block)
    rng = np.random.default_rng(seed)
    sr = lambda x: x.mean(axis=1) / x.std(axis=1, ddof=1)  # noqa: E731  (axis 1 = days)
    obs = (sr(A[None]) - sr(B[None]))[0] * np.sqrt(ANN)
    boot = np.empty((reps, C))
    for s in range(0, reps, chunk):
        n = min(chunk, reps - s)
        starts = rng.integers(0, T, (n, n_blocks))
        idx = ((starts[:, :, None] + np.arange(block)) % T).reshape(n, -1)[:, :T]
        with np.errstate(invalid="ignore", divide="ignore"):
            boot[s:s + n] = (sr(A[idx]) - sr(B[idx])) * np.sqrt(ANN)
    return obs, np.nan_to_num(boot)


def heterogeneity(d: np.ndarray, boot: np.ndarray) -> dict:
    S = np.cov(boot, rowvar=False)
    C = len(d)
    Cm = np.hstack([np.eye(C - 1), -np.ones((C - 1, 1))])
    cd = Cm @ d
    w_eq = float(cd @ np.linalg.solve(Cm @ S @ Cm.T, cd))
    w_zero = float(d @ np.linalg.solve(S, d))
    w = 1 / np.diag(S)
    dw = float((w * d).sum() / w.sum())
    q = float((w * (d - dw) ** 2).sum())
    return {"wald_equal": w_eq, "df_equal": C - 1, "p_equal": chi2_sf(w_eq, C - 1),
            "cochran_q": q, "p_cochran": chi2_sf(q, C - 1), "pooled_d_ivw": dw,
            "wald_zero": w_zero, "df_zero": C, "p_zero": chi2_sf(w_zero, C),
            "corr_matrix": np.corrcoef(boot, rowvar=False).round(4).tolist()}


def breakdown(rets: pd.DataFrame, counts: pd.DataFrame, classes: list, reps: int = 5000):
    groups = ["all"] + classes
    rows = []
    for g in groups:
        r = {"group": g, "n_contracts": int(counts[f"{g}_n"].max()),
             "start": rets.index[0].date(), "end": rets.index[-1].date(), "days": len(rets)}
        for ap in ("A1", "A2", "A3"):
            p = perf(rets[f"{g}_{ap}"])
            r.update({f"sharpe_{ap}": p["sharpe"], f"vol_{ap}": p["ann_vol"], f"maxdd_{ap}": p["max_dd"]})
        rows.append(r)
    res = pd.DataFrame(rows).set_index("group")
    d, boot = joint_bootstrap(rets[[f"{g}_A3" for g in groups]], rets[[f"{g}_A2" for g in groups]], reps=reps)
    se = boot.std(axis=0, ddof=1)
    res["d_sharpe"] = d
    res["ci_low"], res["ci_high"] = np.quantile(boot, 0.025, axis=0), np.quantile(boot, 0.975, axis=0)
    res["se_boot"] = se
    res["min_gap_5pct"] = Z_975 * se
    res["mde_80"] = (Z_975 + Z_80) * se
    res["p_boot"] = (np.abs(boot - boot.mean(axis=0)) >= np.abs(d) - 1e-12).mean(axis=0)
    res["corr_A3_A2"] = [rets[f"{g}_A3"].corr(rets[f"{g}_A2"]) for g in groups]
    het = heterogeneity(d[1:], boot[:, 1:])
    het["classes"] = classes
    return res.reset_index(), het


def run(data_dir: Path, filter_file: Path, label: str, out_dir: Path, reps: int = 5000):
    rets, counts, classes = class_returns(data_dir, filter_file)
    # the class sub-portfolios must add up to the 13-contract portfolio
    for ap in ("A1", "A2", "A3"):
        recon = sum(rets[f"{c}_{ap}"].fillna(0) * counts[f"{c}_n"] for c in classes) / counts["all_n"]
        assert np.allclose(recon, rets[f"all_{ap}"], atol=1e-12), ap
    res, het = breakdown(rets, counts, classes, reps)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / f"classes_{label}.csv", index=False)
    (out_dir / f"classes_{label}_heterogeneity.json").write_text(json.dumps(het, indent=2))
    ex = pd.DataFrame({g: (rets[f"{g}_A3"] - rets[f"{g}_A2"]).cumsum() for g in ["all"] + classes})
    ex.resample("W-FRI").last().rename_axis("date").to_csv(out_dir / f"classes_{label}_excess_weekly.csv")
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(f"=== {label}: combo, dev window {res.start.iloc[0]} .. {res.end.iloc[0]}")
        print(res[["group", "n_contracts", "sharpe_A1", "sharpe_A2", "sharpe_A3", "d_sharpe", "ci_low", "ci_high",
                   "min_gap_5pct", "mde_80", "p_boot", "corr_A3_A2"]].round(3).to_string(index=False))
    print(f"heterogeneity: Wald(equal) = {het['wald_equal']:.2f}, df {het['df_equal']}, p = {het['p_equal']:.3f}; "
          f"Cochran Q = {het['cochran_q']:.2f}, p = {het['p_cochran']:.3f}; "
          f"Wald(all zero) = {het['wald_zero']:.2f}, p = {het['p_zero']:.3f}")
    return res, het


# ---------------------------------------------------------------- HMM fit diagnostics

def _inv_logit(v):
    return 1 / (1 + np.exp(-v))


def fit_diagnostics(data_dir: Path, stem: Path, label: str, out_dir: Path, dev_end: str = DEV_END):
    """Per contract and per class, dev-window vintages only; no strategy returns used."""
    uni = pd.read_csv(data_dir / "universe.csv").set_index("code")
    fits = pd.read_csv(f"{stem}_fits.csv", parse_dates=["vintage"])
    fits = fits[fits.vintage <= dev_end]
    filt = pd.read_csv(f"{stem}_filter.csv", parse_dates=["date"])
    K = 1 + max(int(c.split("_")[1]) for c in fits.columns if c.startswith("dur_"))
    top = K - 1
    rows = []
    for code, g in fits.groupby("contract", sort=False):
        ret = pd.read_csv(data_dir / f"{code}.csv", index_col="date", parse_dates=True).ret
        lv = build_features(ret, ("logvol20",)).logvol20
        f = filt[filt.contract == code].set_index("date").loc[:dev_end, "filter_close_t"].dropna()
        rows.append({
            "contract": code, "asset_class": uni.loc[code, "asset_class"], "reliability": uni.loc[code, "reliability"],
            "vintages": len(g),
            "trend_duration": g[f"dur_{top}"].median(),
            "other_duration": g[[f"dur_{k}" for k in range(top)]].median().mean(),
            "trend_share": g[f"share_{top}"].median(),
            "er60_trend": _inv_logit(g[f"er60_{top}"]).median(),
            "er60_next": _inv_logit(g[f"er60_{top - 1}"]).median(),
            "er_gap": (_inv_logit(g[f"er60_{top}"]) - _inv_logit(g[f"er60_{top - 1}"])).median(),
            "er20_trend": _inv_logit(g[f"er20_{top}"]).median(),
            "vol_trend_vs_low": np.exp(g[f"logvol20_{top}"] - g["logvol20_0"]).median(),
            "mean_filter": f.mean(),
            "corr_filter_vol": f.corr(lv.reindex(f.index)),
        })
    per = pd.DataFrame(rows)
    num = per.columns.drop(["contract", "asset_class", "reliability"])
    by_class = per.groupby("asset_class", sort=False)[list(num)].mean().reset_index()
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / f"diag_{label}_contracts.csv", index=False)
    by_class.to_csv(out_dir / f"diag_{label}_classes.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(per.round(3).to_string(index=False))
        print(by_class.round(3).to_string(index=False))
    return per, by_class


def filter_profile(data_dir: Path, filter_file: Path, label: str, out_dir: Path, dev_end: str = DEV_END):
    """Mean filter level and share of days cut (< 1), per contract, dev window (no returns)."""
    uni = pd.read_csv(data_dir / "universe.csv").set_index("code")
    filt = pd.read_csv(filter_file, parse_dates=["date"])
    filt = filt[filt.date <= dev_end].dropna(subset=["filter_close_t"])
    per = filt.groupby("contract", sort=False).filter_close_t.agg(
        mean_filter="mean", cut_share=lambda s: (s < 1 - 1e-12).mean()).reset_index()
    per.insert(1, "asset_class", per.contract.map(uni.asset_class))
    out_dir.mkdir(parents=True, exist_ok=True)
    per.to_csv(out_dir / f"profile_{label}.csv", index=False)
    return per


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--filter-file", type=Path)
    p.add_argument("--diagnostics", type=Path, help="stem of a *_fits.csv / *_filter.csv pair")
    p.add_argument("--label", required=True)
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    a = p.parse_args()
    if a.diagnostics is not None:
        fit_diagnostics(a.data_dir, a.diagnostics, a.label, a.out_dir)
    if a.filter_file is not None:
        filter_profile(a.data_dir, a.filter_file, a.label, a.out_dir)
        run(a.data_dir, a.filter_file, a.label, a.out_dir)
