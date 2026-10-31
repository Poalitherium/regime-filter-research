"""Sharpe, max drawdown and Calmar for A1, A2 and every trial's control and treatment
(plan.md, "检验指标"), on M1's engine and the development window.

    python -m integration.performance

- A1 and A2 (combo factor): backtest/results/m2_baseline.csv, as committed.
- Trials: from the daily net returns that the bridge and M4 write (not tracked):

    backtest_outputs/m1_rerun/trial{1..8}_daily_net.csv  python -m backtest.m1_bridge --all --out-dir backtest_outputs/m1_rerun
    backtest_outputs/m1_m5/M5-{1,2}_daily_net.csv        python -m backtest.m1_bridge --m5 --out-dir backtest_outputs/m1_m5
    m4_outputs/M4-4_daily_net.csv                        python -m m4.trials (after m4.gates, m4.crowding, m4.impvol)

  A trial whose gate failed has A3 = A2 and gets no row of its own. The M4 daily files of
  those trials hold M4's exploratory filters, which are not part of the family.

Each trial is measured on its own window (the first day with 10 contracts in its book), so
its control is not always the baseline's A2. The Sharpe ratios must equal the module tables
(integration/results/trial_family.csv); the run stops if they do not.

Writes integration/results/performance.csv.
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest.metrics import perf

from .trial_family import COST_MODELS, OUT, ROOT, family, load

KEYS = ["ann_ret", "ann_vol", "sharpe", "max_dd", "calmar", "start", "end", "days"]


def daily_files(rerun: Path, m5: Path, m4: Path) -> dict:
    files = {f"M3-{i}": rerun / f"trial{i}_daily_net.csv" for i in range(1, 9)}
    files.update({"M5-1": m5 / "M5-1_daily_net.csv", "M5-2": m5 / "M5-2_daily_net.csv",
                  "M4-4": m4 / "M4-4_daily_net.csv"})
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("missing daily net returns (see the module docstring):\n  " + "\n  ".join(missing))
    return files


def baseline_rows() -> list[dict]:
    b = pd.read_csv(ROOT / "backtest/results/m2_baseline.csv").query("factor == 'combo'")
    return [{"trial": "M2 baseline", "control": "", "cost_model": r.cost_model, "leg": r.approach,
             **{k: getattr(r, k) for k in KEYS if k != "days"}} for r in b.itertuples()]


def trial_rows(files: dict, family_table: pd.DataFrame) -> list[dict]:
    rows = []
    for trial, path in files.items():
        d = pd.read_csv(path, index_col="date", parse_dates=True)
        fam = family_table[family_table["trial"] == trial].set_index("cost_model")
        for cost in COST_MODELS:
            for leg in ("control", "treatment"):
                p = perf(d[f"{cost}_{leg}"])
                expected = fam.loc[cost, f"sharpe_{leg}"]
                if abs(p["sharpe"] - expected) > 1e-6:  # M4's table is written to 6 decimals
                    raise ValueError(f"{trial} {cost} {leg}: Sharpe {p['sharpe']:.6f} here, {expected:.6f} in the table")
                rows.append({"trial": trial, "control": fam.loc[cost, "control"], "cost_model": cost,
                             "leg": leg, **{k: p[k] for k in KEYS}})
    return rows


def main(rerun: Path, m5: Path, m4: Path) -> pd.DataFrame:
    fam = family(load())
    t = pd.DataFrame(baseline_rows() + trial_rows(daily_files(rerun, m5, m4), fam))
    identity = sorted(set(fam.loc[fam["a3_equals_a2"], "trial"].astype(str)))
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "performance.csv", index=False)
    wide = t.pivot_table(index=["trial", "cost_model"], columns="leg", values=["sharpe", "max_dd", "calmar"],
                         sort=False)
    print("\n" + wide.round(3).to_string())
    print("\nA3 = A2 (gate failed), no row of their own:", ", ".join(identity))
    print("wrote", OUT / "performance.csv")
    return t


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--rerun-dir", type=Path, default=ROOT / "backtest_outputs/m1_rerun")
    a.add_argument("--m5-dir", type=Path, default=ROOT / "backtest_outputs/m1_m5")
    a.add_argument("--m4-dir", type=Path, default=ROOT / "m4_outputs")
    args = a.parse_args()
    main(args.rerun_dir, args.m5_dir, args.m4_dir)
