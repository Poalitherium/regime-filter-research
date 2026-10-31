
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.metrics import block_bootstrap_sharpe_diff


ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "backtest_outputs" / "m1_m5"
OUTPUT_DIR = ROOT / "m6" / "results"

MODELS = {
    "M5-1": "M5-1_daily_net.csv",
    "M5-2": "M5-2_daily_net.csv",
}

COST_MODELS = ("research", "official")
BLOCK_LENGTHS = (10, 20, 40)

REPS = 5000
SEED = 0


def load_returns(filename, cost_model):
    path = INPUT_DIR / filename

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing M1 daily returns: {path}\n"
            "Run: uv run --frozen python -m "
            "backtest.m1_bridge --m5 "
            "--out-dir backtest_outputs/m1_m5"
        )

    df = pd.read_csv(path)

    control_col = f"{cost_model}_control"
    treatment_col = f"{cost_model}_treatment"

    required = [control_col, treatment_col]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns in {path}: {missing}"
        )

    date_col = "date"

    if date_col not in df.columns:
        raise ValueError(
            f"Missing date column in {path}"
        )

    df[date_col] = pd.to_datetime(df[date_col])

    df = df.set_index(date_col).sort_index()

    df = df[required].rename(columns={
        control_col: "control",
        treatment_col: "treatment",
    })

    df = df.replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if df.empty:
        raise ValueError(
            f"No valid observations in {path}"
        )

    if df.index.has_duplicates:
        raise ValueError(
            f"Duplicate dates in {path}"
        )

    # Keep the original development sample only.
    df = df.loc[: "2014-12-31"]

    if len(df) < 100:
        raise ValueError(
            f"Insufficient development observations: {path}"
        )

    return df


def evaluate_model(model_id, filename, cost_model):
    returns = load_returns(filename, cost_model)

    control = returns["control"]
    treatment = returns["treatment"]

    rows = []

    for block in BLOCK_LENGTHS:
        result = block_bootstrap_sharpe_diff(
            ra=treatment,
            rb=control,
            block=block,
            reps=REPS,
            seed=SEED,
        )

        rows.append({
            "model_id": model_id,
            "engine": "m1",
            "cost_model": cost_model,
            "block": block,
            "reps": REPS,
            "n_days": len(returns),
            "start": returns.index.min().date(),
            "end": returns.index.max().date(),
            "d_sharpe": result["d_sharpe"],
            "ci_low": result["ci_low"],
            "ci_high": result["ci_high"],
            "se_boot": result["se_boot"],
            "p_boot": result["p_boot"],
            "significant_raw": (
                result["p_boot"] < 0.05
            ),
        })

    # Save aligned M1-engine daily returns.
    output_name = (
        f"{model_id}_{cost_model}_daily_returns.csv"
    )

    returns.to_csv(
        OUTPUT_DIR / output_name,
        index_label="date",
    )

    return rows


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    all_rows = []

    for model_id, filename in MODELS.items():
        for cost_model in COST_MODELS:
            print(
                f"\nEvaluating {model_id} "
                f"({cost_model})..."
            )

            rows = evaluate_model(
                model_id,
                filename,
                cost_model,
            )

            all_rows.extend(rows)

    summary = pd.DataFrame(all_rows)

    summary.to_csv(
        OUTPUT_DIR / "bootstrap_sensitivity.csv",
        index=False,
    )

    print("\n=== M1 Bootstrap Sensitivity ===")

    print(
        summary[
            [
                "model_id",
                "cost_model",
                "block",
                "n_days",
                "d_sharpe",
                "ci_low",
                "ci_high",
                "p_boot",
            ]
        ].round(4).to_string(index=False)
    )

    print(
        "\nResults saved to:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
