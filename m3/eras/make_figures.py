"""Draw the eras report figures from m3/eras/results/ into m3/eras/figures/.

    python -m m3.eras.make_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R, F = Path("m3/eras/results"), Path("m3/eras/figures")
BLUE, ORANGE, GREEN, VIOLET = "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
MUTED, REF, INK = "#94a3b8", "#64748b", "#1e293b"
ERA_COLOR = {"I": BLUE, "II": VIOLET, "III": ORANGE}
ERA_LABEL = {"I": "I  1977–1989 (pit trading)", "II": "II 1990–1999 (transition)", "III": "III 2000–2014 (electronic)"}
METHOD_LABEL = {"bocpd": "M3 #3 BOCPD (primary)", "hmm": "M3 #1 HMM", "pool": "M5-2 global-pooled HMM"}
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7, "xtick.color": REF, "ytick.color": REF,
    "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.axisbelow": True,
})


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / name)
    plt.close(fig)
    print("wrote", F / name)


def forest(ax, rows):
    """rows: dicts(label, d, lo, hi, color); top to bottom."""
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["lo"], r["hi"]], [y, y], color=r["color"], lw=2, solid_capstyle="round")
        ax.plot(r["d"], y, "o", color=r["color"], ms=6, mec="white", mew=1.2, zorder=3)
    ax.axvline(0, color=REF, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1])
    ax.grid(axis="y", visible=False)


def fig_primary():
    e = pd.read_csv(R / "eras.csv").query("method == 'bocpd' and pool == 'core' and cost == 'x1'").set_index("era")
    p = pd.read_csv(R / "primary.csv").set_index("test")
    rows = [{"label": ERA_LABEL[k], "d": e.loc[k, "d_sharpe"], "lo": e.loc[k, "ci_low"], "hi": e.loc[k, "ci_high"],
             "color": ERA_COLOR[k]} for k in ("I", "II", "III")]
    h1 = p.loc["H1"]
    rows.append({"label": "Pre-electronic pooled\n1977–1999 (H1)", "d": h1.d_sharpe, "lo": h1.ci_low, "hi": h1.ci_high,
                 "color": INK})
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    forest(ax, rows)
    h2 = p.loc["H2"]
    ax.set_xlabel("Sharpe difference, A3 (A2 × BOCPD filter) − A2, core pool, research bps")
    ax.set_title("Primary test: BOCPD's increment over vol sizing by era (95% block-bootstrap CI)", loc="left")
    ax.text(0.0, -0.21, f"H1 (pooled 1977–1999 > 0): {h1.d_sharpe:+.3f}, one-sided p = {h1.p_one_sided:.3f}, "
                        f"Holm {h1.p_holm:.3f}     H2 (pre − post > 0): {h2.d_sharpe:+.3f}, one-sided p = "
                        f"{h2.p_one_sided:.3f}, Holm {h2.p_holm:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=REF)
    save(fig, "fig1_primary.png")


def fig_methods():
    e = pd.read_csv(R / "eras.csv").query("pool == 'core' and cost == 'x1'")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), sharey=True)
    for ax, m in zip(axes, ("bocpd", "hmm", "pool")):
        g = e[e.method == m].set_index("era")
        rows = [{"label": ERA_LABEL[k], "d": g.loc[k, "d_sharpe"], "lo": g.loc[k, "ci_low"], "hi": g.loc[k, "ci_high"],
                 "color": ERA_COLOR[k]} for k in ("I", "II", "III")]
        forest(ax, rows)
        ax.set_title(METHOD_LABEL[m], loc="left")
        ax.set_xlabel("Sharpe difference vs A2")
    fig.suptitle("Increment over vol sizing by era, core pool, research bps (95% CI; note the different x-scales)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig2_methods_by_era.png")


def fig_cumulative():
    d = pd.read_csv(R / "daily_core_x1.csv", index_col="date", parse_dates=True)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (lo, hi), c in zip([("1977", "1990"), ("1990", "2000"), ("2000", "2015")], (BLUE, VIOLET, ORANGE)):
        ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=c, alpha=0.06, lw=0)
    top = 0.0
    for m, col in zip(("bocpd", "hmm", "pool"), (INK, BLUE, GREEN)):
        t, c = d[f"{m}_core_x1_treatment"], d[f"{m}_core_x1_control"]
        x = (t * c.std() / t.std() - c).cumsum() * 100
        top = max(top, float(x.max()))
        ax.plot(x.index, x, color=col, lw=1.4, label=METHOD_LABEL[m])
    ax.set_ylim(top=top + 18)
    ax.axhline(0, color=REF, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Cumulative excess over A2 (%)")
    ax.set_title("Cumulative return of A3 over A2, A3 scaled to A2's volatility (core pool, research bps);\n"
                 "shading: era I (pit), II (transition), III (electronic)", loc="left")
    ax.legend(loc="upper left", ncol=3)
    save(fig, "fig3_cumulative_excess.png")


def fig_baseline():
    b = pd.read_csv(R / "baseline.csv").set_index("era").loc[["I", "II", "III"]]
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(7, 3.4))
    for off, col, lab, key in ((-0.18, MUTED, "A1 raw trend signals", "sharpe_A1"),
                               (0.18, BLUE, "A2 with vol sizing", "sharpe_A2")):
        bars = ax.bar(x + off, b[key], width=0.34, color=col, label=lab)
        for rect, v in zip(bars, b[key]):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (0.02 if v >= 0 else -0.06), f"{v:.2f}",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xticks(x, [ERA_LABEL[k] for k in b.index])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Context: the trend strategy itself by era (M2 combo, core pool, research bps)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig4_baseline_by_era.png")


def fig_costs():
    t = pd.read_csv(R / "eras.csv").query("pool == 'core'")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    for ax, m in zip(axes, ("bocpd", "hmm")):
        g = t[t.method == m]
        for i, era in enumerate(("I", "II", "III")):
            for j, (cost, mk) in enumerate((("x0", "s"), ("x1", "o"), ("x3", "^"))):
                r = g[(g.era == era) & (g.cost == cost)].iloc[0]
                y = i * 4 + j
                ax.plot([r.ci_low, r.ci_high], [y, y], color=ERA_COLOR[era], lw=1.5, alpha=0.7)
                ax.plot(r.d_sharpe, y, mk, color=ERA_COLOR[era], ms=5, mec="white", mew=0.8)
        ax.set_yticks([i * 4 + j for i in range(3) for j in range(3)],
                      [f"{era}, cost {c}" for era in ("I", "II", "III") for c in ("×0", "×1", "×3")])
        ax.invert_yaxis()
        ax.axvline(0, color=REF, lw=1, ls="--")
        ax.set_title(METHOD_LABEL[m], loc="left")
        ax.set_xlabel("Sharpe difference vs A2")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Cost sensitivity: research bps ×0, ×1, ×3 (pit-era costs were higher)", x=0.01, ha="left",
                 fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig5_cost_sensitivity.png")


def fig_power():
    t = pd.read_csv(R / "tests.csv").query("pool == 'core' and cost == 'x1'")
    rows = []
    for m in ("bocpd", "hmm", "pool"):
        for test in ("H1", "H2"):
            r = t[(t.method == m) & (t.test == test)].iloc[0]
            rows.append((f"{METHOD_LABEL[m].split(' (')[0]}, {test}", abs(r.d_sharpe), r.mde_80))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[2] for r in rows], color=MUTED, alpha=0.5, height=0.55, label="Smallest effect detectable with 80% power")
    ax.plot([r[1] for r in rows], y, "o", color=BLUE, ms=6, mec="white", mew=1, label="|estimated effect|")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlabel("Sharpe")
    ax.set_title("Power: what each test could detect vs what it found (core pool, research bps)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig6_power.png")


def fig_extended():
    e = pd.read_csv(R / "eras.csv").query("cost == 'x1' and era != 'I'")
    rows = []
    for m in ("bocpd", "hmm", "pool"):
        for pool in ("core", "extended"):
            for era in ("II", "III"):
                r = e[(e.method == m) & (e.pool == pool) & (e.era == era)].iloc[0]
                rows.append({"label": f"{METHOD_LABEL[m].split(' (')[0]} · {pool} · era {era}", "d": r.d_sharpe,
                             "lo": r.ci_low, "hi": r.ci_high, "color": ERA_COLOR[era], "pool": pool})
    fig, ax = plt.subplots(figsize=(8, 5.4))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["lo"], r["hi"]], [y, y], color=r["color"], lw=2, solid_capstyle="round",
                ls="-" if r["pool"] == "core" else (0, (3, 1.5)))
        ax.plot(r["d"], y, "o" if r["pool"] == "core" else "s", color=r["color"], ms=6, mec="white", mew=1.2, zorder=3)
    ax.axvline(0, color=REF, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Sharpe difference vs A2 (research bps)")
    ax.set_title("Core pool (8, circles, solid) vs extended pool (14, squares, dashed: adds FTSE, DAX, HSI, AUD,\n"
                 "WTI, GSCI); colour = era (violet II 1990s, orange III 2000–2014)", loc="left")
    save(fig, "fig7_extended_pool.png")


if __name__ == "__main__":
    fig_primary()
    fig_methods()
    fig_cumulative()
    fig_baseline()
    fig_costs()
    fig_power()
    fig_extended()
