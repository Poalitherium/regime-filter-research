"""Primary test of the five M4 trials on M1's engine, and the exploratory runs (PREREG section 3).

    python -m m4.trials            # after python -m m4.gates, m4.crowding and m4.impvol

A trial whose gate failed is the identity filter: A3 = A2, difference 0, primary p 1.
Every filter is still run once through M1's engine (both cost models) as an exploratory
result; it never enters the Holm family when the gate failed.

Writes m4_outputs/trials_m1_engine.csv (M3/M5 column layout plus `role`) and
m4_outputs/<trial>_daily_net.csv.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.m1_bridge import COST_MODELS, filter_trial, run_trial
from backtest.metrics import Z_975
from m3.predictability import holm

Z_80 = 0.841621
TRIALS = {"M4-1": "FOMC scheduled statement days", "M4-2": "VIX stress x risk-on position",
          "M4-3": "COT speculative crowding",            # PREREG_M4-3.md (python -m m4.crowding)
          "M4-4": "Implied / realised vol sizing",       # PREREG_M4-4_5.md (python -m m4.impvol), no gate
          "M4-5": "VIX term-structure inversion x risk-on position"}


def main(data_dir: Path, out_dir: Path, min_contracts: int = 10):
    gates = json.loads((out_dir / "gates.json").read_text())
    rows = []
    for tid, name in TRIALS.items():
        g = gates[tid]
        gate_txt = g.get("gate", "passed" if g["passed"] else "failed")
        print(f"{tid}: {name} (gate {gate_txt}; run is {g['role']})", flush=True)
        res, daily = run_trial(*filter_trial(data_dir, Path(g["filter_file"])), min_contracts=min_contracts)
        daily.rename_axis("date").to_csv(out_dir / f"{tid}_daily_net.csv")
        for r in res.to_dict("records"):
            rows.append({"trial": tid, "name": name, "role": g["role"], "engine": "m1", **r,
                         "mde_80": (Z_975 + Z_80) * r["se_boot"]})
        if not g["passed"]:  # primary = identity filter
            for costs in COST_MODELS:
                rows.append({"trial": tid, "name": f"{name} (gate failed, A3 = A2)", "role": "primary",
                             "engine": "m1", "cost_model": costs, "d_sharpe": 0.0, "p_boot": 1.0})
    table = pd.DataFrame(rows)
    prim = table.role == "primary"
    table.loc[prim, "p_holm"] = table[prim].groupby("cost_model").p_boot.transform(lambda p: holm(p.to_numpy()))
    table.to_csv(out_dir / "trials_m1_engine.csv", index=False)
    with pd.option_context("display.width", 240, "display.max_columns", 40):
        print(table[["trial", "role", "cost_model", "start", "sharpe_control", "sharpe_treatment", "d_sharpe",
                     "ci_low", "ci_high", "min_gap_5pct", "mde_80", "p_boot", "p_holm", "corr",
                     "turnover_pa_control", "turnover_pa_treatment", "gross_exp_treatment"]].round(4).to_string(index=False))
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m4_outputs"))
    p.add_argument("--min-contracts", type=int, default=10)
    a = p.parse_args()
    main(a.data_dir, a.out_dir, a.min_contracts)
