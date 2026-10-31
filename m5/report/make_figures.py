"""Draw the M5 report figures from m5/report/tables/ into m5/report/figures/.

    python -m m5.report.make_figures
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T, F = Path("m5/report/tables"), Path("m5/report/figures")
BLUE, ORANGE, GREEN, VIOLET = "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
MUTED, REF, INK, RED = "#94a3b8", "#64748b", "#1e293b", "#dc2626"
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.labelsize": 9,
    "axes.edgecolor": "#cbd5e1", "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e2e8f0", "grid.linewidth": 0.7, "xtick.color": REF, "ytick.color": REF,
    "axes.labelcolor": INK, "text.color": INK, "legend.frameon": False, "figure.dpi": 160, "savefig.bbox": "tight",
    "axes.axisbelow": True,
})
CLASSES = ["equity", "energy", "metal", "rates", "fx", "agri"]
METHODS = {"hmm": ("M3 #1 HMM", MUTED), "bocpd": ("M3 #3 BOCPD", VIOLET),
           "pool_class": ("M5-1 class pool", BLUE), "pool_global": ("M5-2 global pool", GREEN)}


def save(fig, name):
    F.mkdir(parents=True, exist_ok=True)
    fig.savefig(F / name)
    plt.close(fig)
    print("wrote", F / name)


def forest_rows(ax, rows, xlabel):
    """rows: list of dicts(label, d, lo, hi, gap|None, color[, hollow])."""
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r.get("gap"):
            ax.add_patch(plt.Rectangle((-r["gap"], y - 0.32), 2 * r["gap"], 0.64, color=MUTED, alpha=0.22, lw=0))
        if r.get("lo") is not None:
            ax.plot([r["lo"], r["hi"]], [y, y], color=r["color"], lw=2, solid_capstyle="round")
        ax.plot(r["d"], y, "o", color="white" if r.get("hollow") else r["color"], ms=6, mec=r["color"],
                mew=1.4, zorder=3)
    ax.axvline(0, color=REF, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig_trials():
    t = pd.read_csv(T / "trials.csv").set_index("trial")
    m3 = pd.read_csv(T / "classes.csv").query("label == 'hmm' and group == 'all'").iloc[0]
    rows = [{"label": f"M3 #1 HMM per contract (reference)\n{m3.start[:4]}–{m3.end[:4]}, p = {m3.p_boot:.2f}",
             "d": m3.d_sharpe, "lo": m3.ci_low, "hi": m3.ci_high, "gap": m3.min_gap_5pct, "color": MUTED}]
    for tid in ("M5-1", "M5-2"):
        r = t.loc[tid]
        rows.append({"label": f"{tid} {r['name']}\n{r.start[:4]}–{r.end[:4]}, p = {r.p_boot:.2f} (Holm {r.p_holm:.2f})",
                     "d": r.d_sharpe, "lo": r.ci_low, "hi": r.ci_high, "gap": r.min_gap_5pct, "color": BLUE})
    sg = pd.read_csv(T / "season_gate.csv")
    eg = pd.read_csv(T / "expiry_gate.csv").iloc[0]
    for tid, note in [("M5-3", "gate failed: Holm p = " + " / ".join(f"{p:.2f}" for p in sg.p_holm)),
                      ("M5-4", f"gate failed: p = {eg.p:.2f}")]:
        r = t.loc[tid]
        rows.append({"label": f"{tid} {r['name']}\n{note} → no filter, A3 = A2",
                     "d": 0.0, "lo": None, "hi": None, "gap": None, "color": ORANGE, "hollow": True})
    fig, ax = plt.subplots(figsize=(8, 4.4))
    forest_rows(ax, rows, "Sharpe difference A3 − A2 (combo, 13-contract book, development window)")
    ax.set_title("M5 trials: point estimate, 95% block-bootstrap CI,\n"
                 "grey band = differences not distinguishable from 0 at 5%", loc="left")
    save(fig, "fig01_trials_forest.png")


def fig_class_forest():
    c = pd.read_csv(T / "classes.csv")
    h = pd.read_csv(T / "heterogeneity.csv").set_index("label")
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharex=True, sharey=True)
    for ax, (label, (name, color)) in zip(axes, METHODS.items()):
        g = c[c.label == label].set_index("group")
        rows = [{"label": f"{grp} ({int(g.loc[grp, 'n_contracts'])})", "d": g.loc[grp, "d_sharpe"],
                 "lo": g.loc[grp, "ci_low"], "hi": g.loc[grp, "ci_high"], "gap": g.loc[grp, "min_gap_5pct"],
                 "color": INK if grp == "all" else color} for grp in ["all"] + CLASSES]
        forest_rows(ax, rows, "A3 − A2 Sharpe")
        ax.set_title(f"{name}\nequality Wald p = {h.loc[label, 'p_equal']:.2f}", fontsize=9)
    fig.suptitle("By asset class (equal weight within class, development window): no class stands out",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig02_class_forest.png")


def fig_class_sharpe():
    c = pd.read_csv(T / "classes.csv").query("label == 'hmm'").set_index("group")
    x = np.arange(len(CLASSES) + 1)
    groups = ["all"] + CLASSES
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for k, (ap, col, name) in enumerate([("A1", MUTED, "A1 raw signal"), ("A2", BLUE, "A2 vol sizing"),
                                         ("A3", ORANGE, "A3 = A2 × M3 HMM filter")]):
        v = c.loc[groups, f"sharpe_{ap}"].to_numpy()
        ax.bar(x + (k - 1) * 0.26, v, width=0.26, color=col, label=name)
    ax.axhline(0, color=REF, lw=0.8)
    ax.set_xticks(x, [f"{g}\n({int(c.loc[g, 'n_contracts'])})" for g in groups])
    ax.set_ylabel("Sharpe (net, combo)")
    ax.legend(ncol=3, loc="upper left", fontsize=8)
    ax.set_title(f"Sharpe by asset class, {c.start.iloc[0][:7]} – {c.end.iloc[0][:7]}", loc="left")
    save(fig, "fig03_class_sharpe.png")


def fig_diagnostics():
    d = pd.read_csv(T / "diag_classes.csv")
    labels = [("hmm", "M3 per contract", MUTED), ("pool_class", "M5-1 class pool", BLUE),
              ("pool_global", "M5-2 global pool", GREEN)]
    panels = [("trend_duration", "Trend-state duration (days)"), ("er_gap", "Efficiency-ratio gap: trend vs next state"),
              ("corr_filter_vol", "corr(filter, 20d log vol)"), ("mean_filter", "Mean filter (exposure kept)")]
    x = np.arange(len(CLASSES))
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (col, title) in zip(axes, panels):
        for k, (lab, name, color) in enumerate(labels):
            g = d[d.label == lab].set_index("asset_class").reindex(CLASSES)
            ax.bar(x + (k - 1) * 0.27, g[col], width=0.27, color=color, label=name)
        ax.axhline(0, color=REF, lw=0.8)
        ax.set_xticks(x, CLASSES, rotation=30)
        ax.set_title(title, fontsize=9)
    axes[0].legend(fontsize=7.5, loc="upper right")
    fig.suptitle("HMM fit diagnostics by asset class (vintages to 2014, no strategy returns)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    save(fig, "fig04_hmm_diagnostics.png")


def fig_calendar():
    sg = pd.read_csv(T / "season_gate.csv")
    ed = pd.read_csv(T / "expiry_describe.csv")
    eg = pd.read_csv(T / "expiry_gate.csv").iloc[0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    rows = []
    for r in sg.itertuples():
        se = abs(r.raw_diff_ann / r.t)
        rows.append({"label": f"{r.group} ({r.contracts})\nHolm p = {r.p_holm:.2f}, OOS R² = {r.oos_r2:+.3f}",
                     "d": r.raw_diff_ann, "lo": r.raw_diff_ann - 1.96 * se, "hi": r.raw_diff_ann + 1.96 * se,
                     "gap": None, "color": ORANGE})
    forest_rows(a1, rows, "in-season − off-season A2 return (per year)")
    a1.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    a1.set_title("M5-3 gate: season (next month)", loc="left")

    se = abs(eg.coef_ann / eg.t)
    rows = [{"label": f"GATE: 11 futures, ±3 days\np = {eg.p:.2f}, OOS R² = {eg.oos_r2:+.4f}", "d": eg.coef_ann,
             "lo": eg.coef_ann - 1.96 * se, "hi": eg.coef_ann + 1.96 * se, "gap": None, "color": ORANGE}]
    pick = ed[ed.outcome == "y"]
    for r in pick.itertuples():
        name = r.analysis.replace("all futures, pre/post halves", "halves").replace("vendor monthly cycle for GC/SI/6E/6J", "monthly cycle GC/SI/6E/6J")
        term = {"W_pre": " pre (−3..−1)", "W_post": " post (0..+3)"}.get(r.term, "")
        s = abs(r.coef_ann / r.t)
        rows.append({"label": f"{name}{term}  (p = {r.p:.2f})", "d": r.coef_ann, "lo": r.coef_ann - 1.96 * s,
                     "hi": r.coef_ann + 1.96 * s, "gap": None, "color": MUTED})
    forest_rows(a2, rows, "in-window − out-of-window A2 gross return (per year)")
    a2.xaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    a2.set_title("M5-4 gate (orange) and descriptive splits (grey, uncorrected)", loc="left")
    fig.tight_layout()
    save(fig, "fig05_calendar_gates.png")


def fig_excess():
    ex = pd.read_csv(T / "excess_weekly.csv", parse_dates=["date"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for tid, color in [("M3 #1", MUTED), ("M5-1", BLUE), ("M5-2", GREEN)]:
        g = ex[ex.trial == tid]
        ax.plot(g.date, g.excess, color=color, lw=1.5, label=tid)
    ax.axhline(0, color=REF, lw=0.8, ls="--")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend()
    ax.set_title("Cumulative return difference A3 − A2 (not vol-matched; A3 holds ~30% of A2's exposure)", loc="left")
    save(fig, "fig06_cumulative_excess.png")


def fig_class_excess():
    ex = pd.read_csv(T / "classes_excess_monthly.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 3, figsize=(11, 4.6), sharey=True)
    lo, hi = ex[ex.group != "all"].excess.min(), ex[ex.group != "all"].excess.max()
    for ax, cl in zip(axes.ravel(), CLASSES):
        for label, color in [("hmm", MUTED), ("pool_class", BLUE)]:
            g = ex[(ex.group == cl) & (ex.label == label)]
            ax.plot(g.date, g.excess, color=color, lw=1.3, label=METHODS[label][0])
        ax.axhline(0, color=REF, lw=0.8, ls="--")
        ax.set_ylim(lo * 1.1, hi * 1.1)
        ax.set_title(cl, fontsize=9)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[0, 0].legend(fontsize=7.5)
    fig.suptitle("Cumulative A3 − A2 by asset class (development window, month ends)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig07_class_excess.png")


def fig_event_profile():
    e = pd.read_csv(T / "posthoc_roll_event_profile.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.3))
    for ax, outcome, title, fmt in [
            (a1, "y", "A2 gross P&L vs days outside ±5 (per year)", lambda v, _: f"{v:+.0%}"),
            (a2, "abs_ret", "|ret| relative to days outside ±5", lambda v, _: f"{v:.2f}×")]:
        g = e[e.outcome == outcome]
        if outcome == "y":
            ax.bar(g.offset, g["diff"], color=[ORANGE if k == 0 else MUTED for k in g.offset], width=0.7)
            ax.errorbar(g.offset, g["diff"], yerr=1.96 * g.se, fmt="none", ecolor=INK, lw=0.8, capsize=2)
            ax.axhline(0, color=REF, lw=0.8)
        else:
            base = g["diff"] / (g.ratio_to_outside - 1)
            ax.bar(g.offset, g.ratio_to_outside - 1, bottom=1, color=[ORANGE if k == 0 else MUTED for k in g.offset], width=0.7)
            ax.errorbar(g.offset, g.ratio_to_outside, yerr=1.96 * g.se / base, fmt="none", ecolor=INK, lw=0.8, capsize=2)
            ax.axhline(1, color=REF, lw=0.8)
        ax.yaxis.set_major_formatter(fmt)
        ax.set_xticks(list(g.offset))
        ax.set_xlabel("weekdays from the scheduled switch day (0 = first day of the new contract)")
        ax.set_title(title, fontsize=9)
    fig.suptitle("POST HOC: roll event profile, 11 futures, development window (95% CI, clustered by month)",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig08_roll_event_posthoc.png")


if __name__ == "__main__":
    fig_trials()
    fig_class_forest()
    fig_class_sharpe()
    fig_diagnostics()
    fig_calendar()
    fig_excess()
    fig_class_excess()
    fig_event_profile()
