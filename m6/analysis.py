
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "m6" / "results"

COST_MODELS = ("research", "official")
ALPHA = 0.05

# M3-5 and M3-8 use static 12-month TSMOM,
# rather than A2, as their control.
M3_CONTROLS = {
    5: "Static 12-month TSMOM",
    8: "Static 12-month TSMOM",
}


def load_trials():
    m3 = pd.read_csv(
        ROOT / "m3/report/tables/trials_m1_engine.csv"
    )
    m5 = pd.read_csv(
        ROOT / "m5/report/tables/trials_m1_engine.csv"
    )

    m3 = m3[
        (m3["engine"] == "m1")
        & (m3["cost_model"].isin(COST_MODELS))
    ].copy()

    m5 = m5[
        (m5["engine"] == "m1")
        & (m5["cost_model"].isin(COST_MODELS))
    ].copy()

    m3["module"] = "M3"
    m3["model_id"] = "M3-" + m3["trial"].astype(str)
    m3["control"] = m3["trial"].map(
        M3_CONTROLS
    ).fillna("A2 (volatility targeting)")

    m5["module"] = "M5"
    m5["model_id"] = m5["trial"].astype(str)
    m5["control"] = "A2 (volatility targeting)"

    results = pd.concat(
        [m3, m5],
        ignore_index=True
    )

    # Failed-gate M5 trials are identity filters.
    # Their Sharpe difference is zero and p = 1.
    for trial_id in ("M5-3", "M5-4"):
        for cost_model in COST_MODELS:
            exists = (
                (results["model_id"] == trial_id)
                & (results["cost_model"] == cost_model)
            )

            if not exists.any():
                row = {
                    "module": "M5",
                    "model_id": trial_id,
                    "name": (
                        "Seasonality"
                        if trial_id == "M5-3"
                        else "Contract expiry window"
                    ),
                    "engine": "m1",
                    "cost_model": cost_model,
                    "control": "A2 (volatility targeting)",
                    "d_sharpe": 0.0,
                    "p_boot": 1.0,
                }

                results = pd.concat(
                    [results, pd.DataFrame([row])],
                    ignore_index=True
                )

    columns = [
        "module",
        "model_id",
        "name",
        "engine",
        "cost_model",
        "control",
        "start",
        "end",
        "sharpe_control",
        "sharpe_treatment",
        "d_sharpe",
        "ci_low",
        "ci_high",
        "se_boot",
        "p_boot",
    ]

    results = results.reindex(columns=columns)

    # Ensure exactly one result per trial and cost model.
    duplicates = results.duplicated(
        ["model_id", "cost_model"]
    )

    if duplicates.any():
        raise ValueError(
            "Duplicate trial and cost-model combinations."
        )

    return results


def evaluate_trials():
    results = load_trials()

    expected_ids = {
        *(f"M3-{i}" for i in range(1, 9)),
        *(f"M5-{i}" for i in range(1, 5)),
    }

    output = []

    for cost_model in COST_MODELS:
        subset = results[
            results["cost_model"] == cost_model
        ].copy()

        if set(subset["model_id"]) != expected_ids:
            raise ValueError(
                f"Unexpected trial family: {cost_model}"
            )

        if len(subset) != 12:
            raise ValueError(
                f"Expected 12 trials for {cost_model}"
            )

        if subset["p_boot"].isna().any():
            raise ValueError(
                f"Missing p-values: {cost_model}"
            )

        reject, adjusted, _, _ = multipletests(
            subset["p_boot"].to_numpy(),
            alpha=ALPHA,
            method="holm",
        )

        subset["p_holm"] = adjusted
        subset["significant_raw"] = (
            subset["p_boot"] < ALPHA
        )
        subset["significant_holm"] = reject

        output.append(subset)

    evaluated = pd.concat(
        output, ignore_index=True
    )

    evaluated = evaluated.sort_values(
        ["cost_model", "module", "model_id"]
    ).reset_index(drop=True)

    evaluated.to_csv(
        OUTPUT / "all_trials.csv",
        index=False
    )

    evaluated.to_csv(
        OUTPUT / "valid_trials.csv",
        index=False
    )

    print("\n=== M1 Strategy Comparison ===")

    for cost_model in COST_MODELS:
        sub = evaluated[
            evaluated["cost_model"] == cost_model
        ]

        print(f"\nCost model: {cost_model}")
        print("Total trials:", len(sub))
        print(
            "Significant (raw):",
            int(sub["significant_raw"].sum())
        )
        print(
            "Significant (Holm):",
            int(sub["significant_holm"].sum())
        )

        print(
            sub[
                [
                    "model_id",
                    "name",
                    "control",
                    "d_sharpe",
                    "p_boot",
                    "p_holm",
                ]
            ].round(4).to_string(index=False)
        )

    return evaluated


def evaluate_predictability():
    individual = pd.read_csv(
        ROOT / "m3/report/tables/predictability.csv"
    )

    joint = pd.read_csv(
        ROOT / "m3/report/tables/predictability_joint.csv"
    )

    individual["significant_raw"] = (
        individual["p"] < ALPHA
    )

    individual["significant_holm"] = (
        individual["p_holm"] < ALPHA
    )

    joint["significant_joint"] = (
        joint["joint_p"] < ALPHA
    )

    joint["positive_oos_r2"] = (
        joint["oos_r2_portfolio"] > 0
    )

    individual.to_csv(
        OUTPUT / "predictability_individual.csv",
        index=False
    )

    joint.to_csv(
        OUTPUT / "predictability_joint.csv",
        index=False
    )

    print("\n=== Predictive Power ===")
    print("Individual tests:", len(individual))
    print(
        "Significant after Holm:",
        int(individual["significant_holm"].sum())
    )

    print("\nJoint tests:")
    print(
        joint[
            [
                "round",
                "horizon",
                "level",
                "joint_p",
                "oos_r2_portfolio",
            ]
        ].to_string(index=False)
    )


def evaluate_robustness():
    classes = pd.read_csv(
        ROOT / "m5/report/tables/classes.csv"
    )

    heterogeneity = pd.read_csv(
        ROOT / "m5/report/tables/heterogeneity.csv"
    )

    # These are M5's provisional-engine class-level
    # results, retained as supplementary evidence.
    classes["engine"] = "provisional"
    heterogeneity["engine"] = "provisional"

    classes["significant_raw"] = (
        classes["p_boot"] < ALPHA
    )

    classes["positive_improvement"] = (
        classes["d_sharpe"] > 0
    )

    heterogeneity["significant_heterogeneity"] = (
        heterogeneity["p_equal"] < ALPHA
    )

    classes.to_csv(
        OUTPUT / "asset_class_robustness.csv",
        index=False
    )

    heterogeneity.to_csv(
        OUTPUT / "heterogeneity_tests.csv",
        index=False
    )

    groups = classes[
        classes["group"] != "all"
    ]

    print("\n=== Cross-Asset Robustness ===")
    print(
        "Provisional-engine class comparisons:",
        len(groups)
    )
    print(
        "Positive improvements:",
        int(groups["positive_improvement"].sum())
    )
    print(
        "Significant raw comparisons:",
        int(groups["significant_raw"].sum())
    )


def main():
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True
    )

    evaluate_trials()
    evaluate_predictability()
    evaluate_robustness()

    print("\nM6 results saved to:", OUTPUT)


if __name__ == "__main__":
    main()
