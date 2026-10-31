
from pathlib import Path
from math import erf, sqrt

import numpy as np
import pandas as pd

from backtest.metrics import (
    ANN,
    Z_975,
    block_bootstrap_sharpe_diff,
    perf,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "m6" / "results"

MODELS = ("M5-1", "M5-2")
COST_MODELS = ("research", "official")

BLOCK = 20
REPS = 5000
SEED = 0

ALPHA = 0.05
Z_80 = 0.8416212335729143


def normal_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def load_returns(model_id, cost_model):
    filename = (
        f"{model_id}_{cost_model}_daily_returns.csv"
    )

    path = RESULTS / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing daily returns: {path}\n"
            "Run m6.bootstrap_robustness first."
        )

    df = pd.read_csv(
        path,
        index_col="date",
        parse_dates=["date"],
    )

    df = df[["control", "treatment"]].copy()

    df = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna().sort_index()

    if df.empty or df.index.has_duplicates:
        raise ValueError(
            f"Invalid return data: {path}"
        )

    if df.index.max() > pd.Timestamp("2014-12-31"):
        raise ValueError(
            "Holdout data must not be used."
        )

    return df


def evaluate_period(model_id, cost_model, period, df):
    if len(df) < 100:
        raise ValueError(
            f"Insufficient observations: "
            f"{model_id}, {cost_model}, {period}"
        )

    control = df["control"]
    treatment = df["treatment"]

    test = block_bootstrap_sharpe_diff(
        ra=treatment,
        rb=control,
        block=BLOCK,
        reps=REPS,
        seed=SEED,
    )

    sharpe_control = perf(control)["sharpe"]
    sharpe_treatment = perf(treatment)["sharpe"]

    se = test["se_boot"]
    effect = abs(test["d_sharpe"])

    # Normal-approximation MDE for 80% target power.
    mde_80 = (Z_975 + Z_80) * se

    # Descriptive power approximation at the
    # observed effect; not a formal post-hoc test.
    if se > 0:
        ncp = effect / se

        approx_power = (
            1.0 - normal_cdf(Z_975 - ncp)
            + normal_cdf(-Z_975 - ncp)
        )
    else:
        approx_power = np.nan

    return {
        "model_id": model_id,
        "engine": "m1",
        "cost_model": cost_model,
        "period": period,
        "start": df.index.min().date(),
        "end": df.index.max().date(),
        "n_days": len(df),
        "sharpe_control": sharpe_control,
        "sharpe_treatment": sharpe_treatment,
        "d_sharpe": test["d_sharpe"],
        "ci_low": test["ci_low"],
        "ci_high": test["ci_high"],
        "p_boot": test["p_boot"],
        "se_boot": se,
        "mde_80": mde_80,
        "approx_power_observed": approx_power,
        "significant_raw": test["p_boot"] < ALPHA,
    }


def main():
    RESULTS.mkdir(
        parents=True,
        exist_ok=True
    )

    rows = []

    for model_id in MODELS:
        for cost_model in COST_MODELS:
            df = load_returns(
                model_id,
                cost_model,
            )

            midpoint = len(df) // 2

            periods = {
                "Full development": df,
                "Early development": df.iloc[:midpoint],
                "Late development": df.iloc[midpoint:],
            }

            for period, sample in periods.items():
                rows.append(
                    evaluate_period(
                        model_id,
                        cost_model,
                        period,
                        sample,
                    )
                )

    output = pd.DataFrame(rows)

    output.to_csv(
        RESULTS / "subperiod_power.csv",
        index=False,
    )

    print("\n=== M1 Subperiod Robustness ===")

    print(
        output[
            [
                "model_id",
                "cost_model",
                "period",
                "n_days",
                "d_sharpe",
                "ci_low",
                "ci_high",
                "p_boot",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n=== M1 Statistical Power and MDE ===")

    print(
        output[
            [
                "model_id",
                "cost_model",
                "period",
                "se_boot",
                "mde_80",
                "approx_power_observed",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nSaved:",
        RESULTS / "subperiod_power.csv",
    )


if __name__ == "__main__":
    main()
