"""Primary test of every M5 trial and the Holm correction across them (PREREG section 3).

    python -m m5.trials

For each trial whose filter exists, runs the provisional backtest on the development
window (never --include-test) and reads its combo A3 vs A2 row. A calendar trial whose
gate failed has no filter: its method is the identity filter, so A3 = A2, the
difference is 0 and the primary p-value is 1 (PREREG section 3).

Writes m5_outputs/trials.csv and backtest_outputs/m5_<trial>/ (same layout as M3's).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import Z_975
from backtest.run import run as backtest_run
from m3.predictability import holm

from .classes import Z_80

DEV_END = "2014-12-31"
TRIALS = [  # id, name, filter file, gate json (None = no gate), backtest folder
    ("M5-1", "Class-pooled HMM", "pool_class_k3_quarter_top_filter.csv", None, "m5_pool_class"),
    ("M5-2", "Global-pooled HMM", "pool_global_k3_quarter_top_filter.csv", None, "m5_pool_global"),
    ("M5-3", "Seasonality (ZC/ZS weather, NG withdrawal)", "season_filter.csv", "season_gate.json", "m5_season"),
    ("M5-4", "Contract expiry window", "expiry_filter.csv", "expiry_gate.json", "m5_expiry"),
]


def primary(data_dir: Path, filter_file: Path, bt_dir: Path, rerun: bool) -> dict:
    if rerun or not (bt_dir / "a3_vs_a2.csv").is_file():
        backtest_run(data_dir, filter_file, delay=1, costs=True, dev_end=DEV_END, include_test=False,
                     min_contracts=10, out_dir=bt_dir)
    pr = pd.read_csv(bt_dir / "a3_vs_a2.csv").query("window == 'dev' and factor == 'combo'").iloc[0]
    sm = pd.read_csv(bt_dir / "summary.csv").query("window == 'dev' and factor == 'combo'").set_index("approach")
    return {"start": sm.loc["A2", "start"], "end": sm.loc["A2", "end"],
            "sharpe_A2": sm.loc["A2", "sharpe"], "sharpe_A3": sm.loc["A3", "sharpe"],
            "gross_exp_A3": sm.loc["A3", "gross_exp"], "d_sharpe": pr.d_sharpe, "ci_low": pr.ci_low,
            "ci_high": pr.ci_high, "min_gap_5pct": pr.min_gap_5pct, "mde_80": (Z_975 + Z_80) * pr.se_boot,
            "p_boot": pr.p_boot, "corr": pr["corr"], "A3_volmatched_max_dd": pr.A3_volmatched_max_dd,
            "A2_max_dd": sm.loc["A2", "max_dd"]}


def main(data_dir: Path, m5_dir: Path, bt_root: Path, rerun: bool = False):
    rows = []
    for tid, name, ffile, gate, folder in TRIALS:
        row = {"trial": tid, "name": name, "filter_built": True, "gate_p": np.nan}
        if gate is not None:
            g = json.loads((m5_dir / gate).read_text())
            row["filter_built"] = bool(g["filter_built"])
            gt = pd.read_csv(m5_dir / gate.replace(".json", ".csv"))
            row["gate_p"] = float(gt["p_holm"].min() if "p_holm" in gt else gt["p"].min())
        if row["filter_built"]:
            row.update(primary(data_dir, m5_dir / ffile, bt_root / folder, rerun))
        else:  # identity filter: A3 = A2 by construction
            row.update({"d_sharpe": 0.0, "p_boot": 1.0})
        rows.append(row)
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res.p_boot.to_numpy())
    res["significant_holm_5pct"] = res.p_holm < 0.05
    m5_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(m5_dir / "trials.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(res.round(4).to_string(index=False))
    return res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--m5-dir", type=Path, default=Path("m5_outputs"))
    p.add_argument("--bt-root", type=Path, default=Path("backtest_outputs"))
    p.add_argument("--rerun", action="store_true", help="re-run backtests even if outputs exist")
    a = p.parse_args()
    main(a.data_dir, a.m5_dir, a.bt_root, a.rerun)
