"""Figure of the 17-trial family, from integration/results/trial_family.csv.

    python -m integration.make_figures      # after python -m integration.trial_family
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .trial_family import OUT, ROOT, TRIAL_IDS  # noqa: E402

FIG = ROOT / "integration" / "figures"
COLOR = {"M3": "#2563eb", "M4": "#ea580c", "M5": "#16a34a"}
INK, MUTED, REF = "#1f2937", "#6b7280", "#9ca3af"
COST_TITLE = {"official": "Exact fees", "research": "Research bps"}


def short_name(name: str) -> str:
    return name.split(" (gate failed")[0]


def main() -> Path:
    f = pd.read_csv(OUT / "trial_family.csv")
    names = f.drop_duplicates("trial").set_index("trial")["name"].map(short_name)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": REF, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK})
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.6), sharey=True)
    y = {t: len(TRIAL_IDS) - 1 - i for i, t in enumerate(TRIAL_IDS)}
    for ax, cost in zip(axes, ("official", "research")):
        s = f[f["cost_model"] == cost].set_index("trial")
        for t in TRIAL_IDS:
            r, c = s.loc[t], COLOR[t[:2]]
            if r.a3_equals_a2:
                ax.plot(0, y[t], "o", ms=6, mfc="white", mec=c, mew=1.4)
                ax.text(0.06, y[t], "gate failed: A3 = A2", va="center", fontsize=7.5, color=MUTED)
                continue
            ax.plot([r.ci_low, r.ci_high], [y[t], y[t]], color=c, lw=2, solid_capstyle="round")
            ax.plot(r.d_sharpe, y[t], "o", ms=6, color=c, mec="white", mew=1)
            ax.text(1.02, y[t], f"p {r.p_boot:.3f} · Holm {r.p_holm_17:.2f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=7.5, color=MUTED)
        ax.axvline(0, color=REF, lw=1, ls="--")
        ax.set_title(COST_TITLE[cost], loc="left", fontsize=10, color=INK, fontweight="bold")
        ax.set_xlabel("Treatment − control Sharpe, 95% bootstrap interval")
        ax.grid(axis="x", color="#e5e7eb", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    labels = [f"{t}  {names[t]}" + ("  [vs static TSMOM]" if t in ("M3-5", "M3-8") else "") for t in TRIAL_IDS]
    axes[0].set_yticks([y[t] for t in TRIAL_IDS], labels)
    fig.suptitle("The 17 pre-registered regime-adaptor trials on M1's engine (development window, to 2014-12-31); "
                 "Holm over all 17", x=0.01, ha="left", fontsize=11, fontweight="bold", color=INK)
    handles = [plt.Line2D([], [], color=c, marker="o", lw=2, label=m) for m, c in COLOR.items()]
    fig.legend(handles=handles, loc="lower left", ncol=3, frameon=False, bbox_to_anchor=(0.01, 0.0))
    fig.tight_layout(rect=(0, 0.04, 0.93, 0.96))
    FIG.mkdir(parents=True, exist_ok=True)
    path = FIG / "trial_family.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("wrote", path)
    return path


if __name__ == "__main__":
    main()
