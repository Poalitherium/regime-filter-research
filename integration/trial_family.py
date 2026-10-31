"""The project's trial family: the 17 pre-registered regime-adaptor trials of M3, M4 and M5
on M1's engine, with Holm over all 17.

    python -m integration.trial_family

Reads each module's own table and changes none of them:

- m3/report/tables/trials_m1_engine.csv: trials 1-8 (rows with engine == "m1");
- m4/report/tables/trials_m1_engine.csv: M4-1 to M4-5 (rows with role == "primary"; a
  failed gate is the identity filter, A3 = A2, p = 1);
- m5/report/tables/trials_m1_engine.csv: M5-1 to M5-4 (rows with engine == "m1"; M5-3 and
  M5-4 failed their gates).

M6 (m6/results/all_trials.csv) corrects over the 12 M3/M5 trials only. Its estimates and
adjusted p-values must equal the 12-trial Holm computed here; the run stops if they do not.

Writes integration/results/trial_family.csv: one row per trial x cost model with the
control, the estimate, its 95% interval and bootstrap SE, the smallest effect detectable
with 80% power (MDE), the p-value, Holm over the 12 M3/M5 trials and Holm over all 17.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import Z_975
from m3.predictability import holm

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "integration" / "results"
TABLES = {"M3": ROOT / "m3/report/tables/trials_m1_engine.csv",
          "M4": ROOT / "m4/report/tables/trials_m1_engine.csv",
          "M5": ROOT / "m5/report/tables/trials_m1_engine.csv"}
M6_TABLE = ROOT / "m6/results/all_trials.csv"
COST_MODELS = ("official", "research")
Z_80 = 0.841621
STATIC_CONTROL = {"M3-5", "M3-8"}  # dynamic signal / pure HMM strategy vs static 12-month TSMOM
TRIAL_IDS = [f"M3-{i}" for i in range(1, 9)] + [f"M4-{i}" for i in range(1, 6)] + [f"M5-{i}" for i in range(1, 5)]
COLUMNS = ["trial", "module", "name", "control", "a3_equals_a2", "cost_model", "start", "end", "days",
           "sharpe_control", "sharpe_treatment", "d_sharpe", "ci_low", "ci_high", "se_boot", "mde_80", "corr",
           "p_boot", "p_holm_12", "p_holm_17"]


def load() -> pd.DataFrame:
    m3 = pd.read_csv(TABLES["M3"]).query("engine == 'm1'").assign(module="M3")
    m3["trial"] = "M3-" + m3["trial"].astype(str)
    m4 = pd.read_csv(TABLES["M4"]).query("role == 'primary'").assign(module="M4")
    m5 = pd.read_csv(TABLES["M5"]).query("engine == 'm1'").assign(module="M5")
    t = pd.concat([m3, m4, m5], ignore_index=True)
    t["control"] = np.where(t["trial"].isin(STATIC_CONTROL), "static 12-month TSMOM", "A2")
    t["a3_equals_a2"] = t["sharpe_treatment"].isna() & (t["d_sharpe"] == 0) & (t["p_boot"] == 1)
    t["mde_80"] = (Z_975 + Z_80) * t["se_boot"]
    return t


def family(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for cost in COST_MODELS:
        s = t[t["cost_model"] == cost].copy()
        if sorted(s["trial"]) != sorted(TRIAL_IDS):
            raise ValueError(f"{cost}: expected trials {TRIAL_IDS}, found {sorted(s['trial'])}")
        if s["p_boot"].isna().any():
            raise ValueError(f"{cost}: missing p-values")
        s["p_holm_17"] = holm(s["p_boot"].to_numpy())
        m35 = s["module"].isin(["M3", "M5"])
        s.loc[m35, "p_holm_12"] = holm(s.loc[m35, "p_boot"].to_numpy())
        out.append(s)
    f = pd.concat(out, ignore_index=True)
    f["trial"] = pd.Categorical(f["trial"], TRIAL_IDS, ordered=True)
    return f.sort_values(["cost_model", "trial"]).reset_index(drop=True)[COLUMNS]


def check_m6(f: pd.DataFrame, tol: float = 1e-9) -> int:
    """M6's 12-trial table must match this family's M3/M5 rows."""
    m6 = pd.read_csv(M6_TABLE).rename(columns={"model_id": "trial"})
    m = f.merge(m6[["trial", "cost_model", "d_sharpe", "p_boot", "p_holm"]], on=["trial", "cost_model"],
                suffixes=("", "_m6"))
    if len(m) != 24:
        raise ValueError(f"M6 table: expected 24 rows (12 trials x 2 cost models), matched {len(m)}")
    for a, b in (("d_sharpe", "d_sharpe_m6"), ("p_boot", "p_boot_m6"), ("p_holm_12", "p_holm")):
        bad = m[(m[a] - m[b]).abs() > tol]
        if len(bad):
            raise ValueError(f"M6 disagrees on {a}:\n{bad[['trial', 'cost_model', a, b]]}")
    return len(m)


def main() -> pd.DataFrame:
    f = family(load())
    n = check_m6(f)
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_csv(OUT / "trial_family.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(f[["trial", "control", "cost_model", "d_sharpe", "ci_low", "ci_high", "mde_80", "p_boot",
                 "p_holm_12", "p_holm_17"]].round(4).to_string(index=False))
    print(f"\nM6 check: {n} rows agree (estimate, p, Holm over 12).")
    for cost in COST_MODELS:
        s = f[f["cost_model"] == cost]
        best = s.loc[s["p_boot"].idxmin()]
        print(f"{cost}: smallest p is {best.trial}'s, {best.p_boot:.4f}; Holm over 12 {best.p_holm_12:.3f}, "
              f"over 17 {best.p_holm_17:.3f}; rejected at 5% after Holm: {int((s['p_holm_17'] < 0.05).sum())}")
    print("wrote", OUT / "trial_family.csv")
    return f


if __name__ == "__main__":
    main()
