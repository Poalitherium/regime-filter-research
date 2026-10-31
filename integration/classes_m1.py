"""The asset-class breakdown of M5 phase 1, on M1's engine (M5 and M6 report it on the
provisional engine only).

    python -m integration.classes_m1        # needs the M3/M5 filter files, ~2 min

For each filter (M3 #1 HMM, M3 #3 BOCPD, M5-1, M5-2) and cost model: the trial's A2 and A3
on its own window (first day with 10 contracts in the book, as in the trial), then the same
targets split into one sub-portfolio per asset class. M1's portfolio weights its members
equally and fixes membership, so the 13-contract portfolio is exactly
sum_c (n_c / 13) x class_c, which is checked, and its A3 - A2 must equal the trial table.
Sharpe differences of all classes are bootstrapped jointly with M5's own functions
(m5.classes.joint_bootstrap, heterogeneity: same blocks, resamples and seed), giving the
equality test across classes. Writes integration/results/classes_m1.csv and
classes_m1_heterogeneity.csv.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.m1_bridge import COST_MODELS, DEV_END, filter_trial, m1_book, window_start
from backtest.metrics import Z_975, perf
from m5.classes import CLASSES, heterogeneity, joint_bootstrap

from .trial_family import OUT, ROOT, Z_80

DATA = ROOT / "data/final"
FILTERS = {"M3-1": ("M3 #1 HMM", ROOT / "m3_outputs/hmm_k3_quarter_top_filter.csv"),
           "M3-3": ("M3 #3 BOCPD", ROOT / "m3_outputs/bocpd_h250_filter.csv"),
           "M5-1": ("M5-1 class-pooled HMM", ROOT / "m5_outputs/pool_class_k3_quarter_top_filter.csv"),
           "M5-2": ("M5-2 global-pooled HMM", ROOT / "m5_outputs/pool_global_k3_quarter_top_filter.csv")}


def class_nets(trial: str, path: Path, costs: str):
    data, asset, targets, book = filter_trial(DATA, path)
    start = window_start(book)
    uni = pd.read_csv(DATA / "universe.csv")
    members = {cl: [c for c in uni.code[uni.asset_class == cl] if c in data] for cl in CLASSES}
    nets = {}
    for leg, short in (("control", "A2"), ("treatment", "A3")):
        full = m1_book(targets[leg], data, asset, book, start, DEV_END, costs).net
        nets[f"all_{short}"] = full
        recon = 0.0
        for cl, codes in members.items():
            sub = m1_book({c: targets[leg][c] for c in codes}, {c: data[c] for c in codes}, asset,
                          book[codes], start, DEV_END, costs).net.reindex(full.index).fillna(0.0)
            nets[f"{cl}_{short}"] = sub
            recon = recon + sub * len(codes) / len(data)
        if not np.allclose(recon, full, atol=1e-12):
            raise AssertionError(f"{trial} {costs} {short}: classes do not add up to the portfolio")
    return pd.DataFrame(nets), members


def main() -> pd.DataFrame:
    family = pd.read_csv(OUT / "trial_family.csv").set_index(["trial", "cost_model"])
    rows, hets = [], []
    for trial, (name, path) in FILTERS.items():
        for costs in COST_MODELS:
            nets, members = class_nets(trial, path, costs)
            groups = ["all", *CLASSES]
            d, boot = joint_bootstrap(nets[[f"{g}_A3" for g in groups]], nets[[f"{g}_A2" for g in groups]])
            expected = family.loc[(trial, costs), "d_sharpe"]
            if abs(d[0] - expected) > 1e-9:
                raise AssertionError(f"{trial} {costs}: portfolio A3 - A2 {d[0]:.6f}, trial table {expected:.6f}")
            se = boot.std(axis=0, ddof=1)
            lo, hi = np.quantile(boot, [0.025, 0.975], axis=0)
            p = (np.abs(boot - boot.mean(axis=0)) >= np.abs(d) - 1e-12).mean(axis=0)
            for j, g in enumerate(groups):
                rows.append({"trial": trial, "method": name, "cost_model": costs, "group": g,
                             "n_contracts": sum(map(len, members.values())) if g == "all" else len(members[g]),
                             "start": nets.index[0].date(), "end": nets.index[-1].date(),
                             "sharpe_A2": perf(nets[f"{g}_A2"])["sharpe"], "sharpe_A3": perf(nets[f"{g}_A3"])["sharpe"],
                             "d_sharpe": d[j], "ci_low": lo[j], "ci_high": hi[j], "se_boot": se[j],
                             "mde_80": (Z_975 + Z_80) * se[j], "p_boot": p[j]})
            h = heterogeneity(d[1:], boot[:, 1:])
            hets.append({"trial": trial, "method": name, "cost_model": costs,
                         **{k: v for k, v in h.items() if k != "corr_matrix"}})
    t, het = pd.DataFrame(rows), pd.DataFrame(hets)
    t.to_csv(OUT / "classes_m1.csv", index=False)
    het.to_csv(OUT / "classes_m1_heterogeneity.csv", index=False)
    cls = t[t.group != "all"]
    for costs in COST_MODELS:
        c = cls[cls.cost_model == costs]
        print(f"{costs}: {len(c)} class comparisons, {int((c.d_sharpe > 0).sum())} positive, "
              f"{int((c.p_boot < 0.05).sum())} with p < 0.05; equality test p = "
              + ", ".join(f"{r.trial} {r.p_equal:.2f}" for r in het[het.cost_model == costs].itertuples()))
    print("wrote", OUT / "classes_m1.csv")
    return t


if __name__ == "__main__":
    main()
