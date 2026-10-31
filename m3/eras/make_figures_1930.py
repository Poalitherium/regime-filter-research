"""Draw the 1930 study figures from m3/eras/results_1930/ into m3/eras/figures_1930/.

    python -m m3.eras.make_figures_1930
"""

from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from m3.eras.make_figures import BLUE, GREEN, INK, MUTED, ORANGE, REF, VIOLET, forest  # also sets the style

R, F = Path("m3/eras/results_1930"), Path("m3/eras/figures_1930")
CYAN = "#0891b2"
ERAS = ("E0", "E1", "I", "II", "III")
ERA_COLOR = {"E0": GREEN, "E1": CYAN, "I": BLUE, "II": VIOLET, "III": ORANGE}
ERA_LABEL = {"E0": "E0 1930–1959 (before CTAs)", "E1": "E1 1960–1976", "I": "I  1977–1989 (pit trading)",
             "II": "II 1990–1999 (transition)", "III": "III 2000–2014 (electronic)"}
SPANS = {"E0": ("1930", "1960"), "E1": ("1960", "1977"), "I": ("1977", "1990"), "II": ("1990", "2000"), "III": ("2000", "2015")}
METHOD_LABEL = {"bocpd": "M3 #3 BOCPD (primary)", "hmm": "M3 #1 HMM", "pool": "M5-2 global-pooled HMM"}


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / name)
    plt.close(fig)
    print("wrote", F / name)


def era_rows(g):
    return [{"label": ERA_LABEL[k], "d": g.loc[k, "d_sharpe"], "lo": g.loc[k, "ci_low"], "hi": g.loc[k, "ci_high"],
             "color": ERA_COLOR[k]} for k in ERAS]


def fig_primary():
    e = pd.read_csv(R / "eras.csv").query("method == 'bocpd' and cost == 'x1'").set_index("era")
    p = pd.read_csv(R / "primary.csv").set_index("test")
    rows = era_rows(e)
    rows[0]["label"] = "E0 1930–1959 = H3 window"
    h3, h4 = p.loc["H3"], p.loc["H4"]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    forest(ax, rows)
    ax.set_xlabel("Sharpe difference, A3 (A2 × BOCPD filter) − A2, 10 US industries, research bps")
    ax.set_title("Primary test, 1930 addendum: BOCPD's increment over vol sizing by era (95% CI)", loc="left")
    ax.text(0.0, -0.19, f"H3 (1930–1959 > 0): {h3.d_sharpe:+.3f}, one-sided p = {h3.p_one_sided:.3f}, Holm (H1–H4) "
                        f"{h3.p_holm:.3f}     H4 (1930–59 − 2000–14 > 0): {h4.d_sharpe:+.3f}, one-sided p = "
                        f"{h4.p_one_sided:.3f}, Holm {h4.p_holm:.3f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8, color=REF)
    save(fig, "fig1_primary.png")


def fig_family():
    p = pd.read_csv(R / "primary.csv")
    label = {"H1": "H1  1977–1999 increment > 0\n(cross-asset proxies, first run)",
             "H2": "H2  1977–99 minus 2000–14 > 0\n(first run)",
             "H3": "H3  1930–1959 increment > 0\n(US industries, this run)",
             "H4": "H4  1930–59 minus 2000–14 > 0\n(this run)"}
    rows = [{"label": label[r.test], "d": r.d_sharpe, "lo": r.d_sharpe - 1.96 * r.se_boot,
             "hi": r.d_sharpe + 1.96 * r.se_boot, "color": BLUE if r.test in ("H1", "H2") else GREEN} for r in p.itertuples()]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    forest(ax, rows)
    for i, r in enumerate(p.itertuples()):
        y = len(rows) - 1 - i
        ax.text(1.01, y, f"p = {r.p_one_sided:.3f}\nHolm {r.p_holm:.3f}", transform=ax.get_yaxis_transform(),
                va="center", fontsize=8, color=REF)
    ax.set_xlabel("Estimate (Sharpe) with ±1.96 bootstrap SE; p one-sided")
    ax.set_title("The pre-registered family: H1–H4, Holm-corrected together", loc="left")
    save(fig, "fig2_primary_family.png")


def fig_methods():
    e = pd.read_csv(R / "eras.csv").query("cost == 'x1'")
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), sharey=True)
    for ax, m in zip(axes, ("bocpd", "hmm", "pool")):
        forest(ax, era_rows(e[e.method == m].set_index("era")))
        ax.set_title(METHOD_LABEL[m], loc="left")
        ax.set_xlabel("Sharpe difference vs A2")
    fig.suptitle("Increment over vol sizing by era, 10 US industries, research bps (95% CI; different x-scales)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig3_methods_by_era.png")


def fig_cumulative():
    d = pd.read_csv(R / "daily_industry_x1.csv", index_col="date", parse_dates=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, (k, (lo, hi)) in enumerate(SPANS.items()):  # neutral bands, so the line colours stay unambiguous
        if i % 2 == 0:
            ax.axvspan(pd.Timestamp(lo), pd.Timestamp(hi), color=MUTED, alpha=0.08, lw=0)
        mid = pd.Timestamp(lo) + (pd.Timestamp(hi) - pd.Timestamp(lo)) / 2
        ax.text(mid, 0.97, k, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=8, color=REF)
    top = 0.0
    for m, col in zip(("bocpd", "hmm", "pool"), (INK, BLUE, GREEN)):
        t, c = d[f"{m}_industry_x1_treatment"], d[f"{m}_industry_x1_control"]
        x = (t * c.std() / t.std() - c).cumsum() * 100
        top = max(top, float(x.max()))
        ax.plot(x.index, x, color=col, lw=1.3, label=METHOD_LABEL[m])
    ax.set_ylim(top=top + 25)
    ax.axhline(0, color=REF, lw=0.8)
    ax.xaxis.set_major_locator(mdates.YearLocator(10))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Cumulative excess over A2 (%)")
    ax.set_title("Cumulative return of A3 over A2, A3 scaled to A2's volatility (10 US industries, research bps);\n"
                 "bands: E0 1930–59, E1 1960–76, I 1977–89, II 1990–99, III 2000–14", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save(fig, "fig4_cumulative_excess.png")


def fig_baseline():
    b = pd.read_csv(R / "baseline.csv").set_index("era").loc[list(ERAS)]
    x = np.arange(len(ERAS))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for off, col, lab, key in ((-0.18, MUTED, "A1 raw trend signals", "sharpe_A1"),
                               (0.18, BLUE, "A2 with vol sizing", "sharpe_A2")):
        bars = ax.bar(x + off, b[key], width=0.34, color=col, label=lab)
        for rect, v in zip(bars, b[key]):
            ax.text(rect.get_x() + rect.get_width() / 2, v + (0.02 if v >= 0 else -0.07), f"{v:.2f}",
                    ha="center", fontsize=8, color=INK)
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xticks(x, [ERA_LABEL[k].replace(" (", "\n(") for k in ERAS], fontsize=8)
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Context: the trend strategy itself by era (M2 combo on 10 US industries, research bps)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig5_baseline_by_era.png")


def fig_costs():
    t = pd.read_csv(R / "eras.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, m in zip(axes, ("bocpd", "hmm")):
        g = t[t.method == m]
        labels = []
        for i, era in enumerate(ERAS):
            for j, (cost, mk, lab) in enumerate((("x0", "s", "×0"), ("x1", "o", "×1"), ("x3", "^", "×3"))):
                r = g[(g.era == era) & (g.cost == cost)].iloc[0]
                y = i * 4 + j
                ax.plot([r.ci_low, r.ci_high], [y, y], color=ERA_COLOR[era], lw=1.5, alpha=0.7)
                ax.plot(r.d_sharpe, y, mk, color=ERA_COLOR[era], ms=5, mec="white", mew=0.8)
                labels.append((y, f"{era}, cost {lab}"))
        ax.set_yticks([y for y, _ in labels], [s for _, s in labels], fontsize=7)
        ax.invert_yaxis()
        ax.axvline(0, color=REF, lw=1, ls="--")
        ax.set_title(METHOD_LABEL[m], loc="left")
        ax.set_xlabel("Sharpe difference vs A2")
        ax.grid(axis="y", visible=False)
    fig.suptitle("Cost sensitivity: research bps ×0, ×1, ×3 (fixed commissions until 1975 were higher still)",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout()
    save(fig, "fig6_cost_sensitivity.png")


def fig_power():
    t = pd.read_csv(R / "tests.csv").query("cost == 'x1'")
    rows = []
    for m in ("bocpd", "hmm", "pool"):
        for test in ("H3", "H4"):
            r = t[(t.method == m) & (t.test == test)].iloc[0]
            rows.append((f"{METHOD_LABEL[m].split(' (')[0]}, {test}", r.d_sharpe, r.mde_80))
    fig, ax = plt.subplots(figsize=(8, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[2] for r in rows], color=MUTED, alpha=0.5, height=0.55, label="Smallest effect detectable with 80% power")
    ax.plot([abs(r[1]) for r in rows], y, "o", color=BLUE, ms=6, mec="white", mew=1, label="|estimated effect|")
    for yy, r in zip(y, rows):
        ax.text(abs(r[1]), yy + 0.28, f"{r[1]:+.3f}", fontsize=7, color=INK, ha="center")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xlim(left=-0.03)
    ax.set_xlabel("Sharpe")
    ax.set_title("Power, 1930 addendum: detectable vs estimated effect (signed estimate above each dot)", loc="left")
    ax.legend(loc="upper right")
    ax.grid(axis="y", visible=False)
    save(fig, "fig7_power.png")


if __name__ == "__main__":
    fig_primary()
    fig_family()
    fig_methods()
    fig_cumulative()
    fig_baseline()
    fig_costs()
    fig_power()
