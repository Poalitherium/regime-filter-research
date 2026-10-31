
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "m6" / "results"
FIGURES = ROOT / "m6" / "figures"

COST_MODELS = ("research", "official")
MODEL_IDS = ("M5-1", "M5-2")

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 120,
    "savefig.dpi": 300,
})


def save_figure(fig, filename):
    FIGURES.mkdir(parents=True, exist_ok=True)
    path = FIGURES / filename
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path)


def plot_sharpe_improvement():
    df = pd.read_csv(RESULTS / "valid_trials.csv")

    # Failed-gate identity filters have no estimated CI.
    df = df.dropna(
        subset=["d_sharpe", "ci_low", "ci_high"]
    ).copy()

    fig, axes = plt.subplots(
        1, 2, figsize=(15, 7), sharey=True
    )

    order = (
        df[df["cost_model"] == "research"]
        .sort_values("d_sharpe")["model_id"]
        .tolist()
    )

    for ax, cost_model in zip(axes, COST_MODELS):
        sub = (
            df[df["cost_model"] == cost_model]
            .set_index("model_id")
            .loc[order]
        )

        y = np.arange(len(sub))
        values = sub["d_sharpe"].to_numpy()

        lower = values - sub["ci_low"].to_numpy()
        upper = sub["ci_high"].to_numpy() - values

        ax.errorbar(
            values,
            y,
            xerr=[lower, upper],
            fmt="o",
            capsize=3,
            linewidth=1.4,
        )

        ax.axvline(
            0, color="black", linestyle="--", linewidth=1
        )

        ax.set_yticks(y)
        ax.set_yticklabels(
            [
                f"{idx} {row['name']}"
                for idx, row in sub.iterrows()
            ],
            fontsize=9,
        )

        ax.set_title(cost_model.capitalize())
        ax.set_xlabel("Annualized Sharpe Difference")
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "M1 Engine: Sharpe Differences and 95% Bootstrap CI"
    )

    fig.text(
        0.08, 0.01,
        "M3-5 and M3-8 use static 12-month TSMOM "
        "as control; other trials use A2. "
        "Intervals are unadjusted.",
        fontsize=9,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    save_figure(fig, "sharpe_improvement.png")


def plot_bootstrap_sensitivity():
    df = pd.read_csv(
        RESULTS / "bootstrap_sensitivity.csv"
    )

    fig, axes = plt.subplots(
        2, 2, figsize=(12, 9),
        sharex=True, sharey=True
    )

    for i, model_id in enumerate(MODEL_IDS):
        for j, cost_model in enumerate(COST_MODELS):
            ax = axes[i, j]

            sub = df[
                (df["model_id"] == model_id)
                & (df["cost_model"] == cost_model)
            ].sort_values("block")

            x = np.arange(len(sub))
            values = sub["d_sharpe"].to_numpy()

            lower = values - sub["ci_low"].to_numpy()
            upper = sub["ci_high"].to_numpy() - values

            ax.errorbar(
                x,
                values,
                yerr=[lower, upper],
                fmt="o",
                capsize=5,
                linewidth=1.5,
            )

            ax.axhline(
                0, color="black", linestyle="--"
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                sub["block"].astype(str)
            )

            ax.set_title(
                f"{model_id} — {cost_model.capitalize()}"
            )

            ax.set_xlabel("Block Length (days)")
            ax.set_ylabel("Annualized Sharpe Difference")
            ax.grid(axis="y", alpha=0.25)

            ax.set_xlim(-0.5, len(sub) - 0.5)

            for k, (_, row) in enumerate(sub.iterrows()):
                ax.annotate(
                    f"p={row['p_boot']:.3f}",
                    (k, row["ci_high"]),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                )

    ymin = df["ci_low"].min()
    ymax = df["ci_high"].max()
    margin = (ymax - ymin) * 0.18

    for ax in axes.flat:
        ax.set_ylim(
            ymin - margin * 0.3,
            ymax + margin
        )

    fig.suptitle(
        "M1 Engine: Bootstrap Block-Length Sensitivity"
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, "bootstrap_sensitivity.png")


def plot_cross_asset_robustness():
    df = pd.read_csv(
        RESULTS / "asset_class_robustness.csv"
    )

    # M5 class-level decomposition remains provisional.
    df = df[df["group"] != "all"].copy()

    model_order = [
        "hmm", "bocpd", "pool_class", "pool_global"
    ]

    class_order = [
        "equity", "energy", "metal",
        "rates", "fx", "agri"
    ]

    matrix = df.pivot(
        index="label",
        columns="group",
        values="d_sharpe",
    ).reindex(
        index=model_order,
        columns=class_order,
    )

    fig, ax = plt.subplots(figsize=(10, 5))

    values = matrix.to_numpy()
    limit = np.nanmax(np.abs(values))

    im = ax.imshow(
        values,
        cmap="RdBu",
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )

    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels([
        "Equity", "Energy", "Metals",
        "Rates", "FX", "Agriculture"
    ])

    ax.set_yticks(np.arange(len(model_order)))
    ax.set_yticklabels([
        "M3 HMM",
        "M3 BOCPD",
        "M5 Class-pooled HMM",
        "M5 Global-pooled HMM",
    ])

    for i in range(len(model_order)):
        for j in range(len(class_order)):
            value = matrix.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j, i, f"{value:+.3f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color=(
                        "white"
                        if abs(value) > 0.6 * limit
                        else "black"
                    ),
                )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Annualized Sharpe Difference")

    ax.set_title(
        "Cross-Asset Robustness (Provisional Engine)"
    )

    fig.text(
        0.12, 0.01,
        "Descriptive class-level results from M5. "
        "None of the 24 comparisons has p < 0.05.",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    save_figure(fig, "cross_asset_robustness.png")


def plot_subperiod_robustness():
    df = pd.read_csv(
        RESULTS / "subperiod_power.csv"
    )

    periods = [
        "Early development",
        "Late development",
    ]

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )

    for ax, cost_model in zip(axes, COST_MODELS):
        for model_id, offset in zip(
            MODEL_IDS, (-0.12, 0.12)
        ):
            sub = (
                df[
                    (df["model_id"] == model_id)
                    & (df["cost_model"] == cost_model)
                ]
                .set_index("period")
                .loc[periods]
            )

            x = np.arange(len(periods)) + offset
            values = sub["d_sharpe"].to_numpy()

            lower = values - sub["ci_low"].to_numpy()
            upper = sub["ci_high"].to_numpy() - values

            ax.errorbar(
                x,
                values,
                yerr=[lower, upper],
                fmt="o",
                capsize=5,
                linewidth=1.5,
                label=model_id,
            )

        ax.axhline(
            0, color="black", linestyle="--"
        )

        ax.set_xticks(np.arange(len(periods)))
        ax.set_xticklabels(["Early", "Late"])
        ax.set_title(cost_model.capitalize())
        ax.set_xlabel("Development Subperiod")
        ax.grid(axis="y", alpha=0.25)
        ax.legend()

    axes[0].set_ylabel(
        "Annualized Sharpe Difference (A3 - A2)"
    )

    fig.suptitle("M1 Engine: Subperiod Robustness")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    save_figure(fig, "subperiod_robustness.png")


def plot_mde():
    df = pd.read_csv(
        RESULTS / "subperiod_power.csv"
    )

    df = df[
        df["period"] == "Full development"
    ].copy()

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), sharey=True
    )

    for ax, cost_model in zip(axes, COST_MODELS):
        sub = (
            df[df["cost_model"] == cost_model]
            .set_index("model_id")
            .loc[list(MODEL_IDS)]
        )

        x = np.arange(len(sub))
        width = 0.32

        bars1 = ax.bar(
            x - width / 2,
            sub["d_sharpe"].abs(),
            width,
            label="Observed |ΔSharpe|",
        )

        bars2 = ax.bar(
            x + width / 2,
            sub["mde_80"],
            width,
            label="80% Power MDE",
        )

        ax.bar_label(
            bars1, fmt="%.3f", padding=3
        )

        ax.bar_label(
            bars2, fmt="%.3f", padding=3
        )

        ax.set_xticks(x)
        ax.set_xticklabels(list(MODEL_IDS))
        ax.set_title(cost_model.capitalize())
        ax.set_xlabel("Model")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 0.85)

    axes[0].set_ylabel(
        "Annualized Sharpe Difference"
    )

    fig.legend(
        ["Observed |ΔSharpe|", "80% Power MDE"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
    )

    fig.suptitle(
        "M1 Engine: Observed Effect vs Approximate MDE",
        y=1.02,
    )

    fig.text(
        0.10, 0.01,
        "MDE: two-sided 5% test, 80% target power, "
        "normal approximation.",
        fontsize=8,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.88])
    save_figure(fig, "statistical_power_mde.png")


def main():
    plot_sharpe_improvement()
    plot_bootstrap_sensitivity()
    plot_cross_asset_robustness()
    plot_subperiod_robustness()
    plot_mde()

    print("\nAll five M6 figures generated successfully.")


if __name__ == "__main__":
    main()
