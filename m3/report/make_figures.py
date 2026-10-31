"""Draw the M3 report figures from m3/report/tables/ into m3/report/figures/.

    python -m m3.report.make_figures
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

T, F = Path("m3/report/tables"), Path("m3/report/figures")
BLUE, ORANGE, GREEN, VIOLET = "#2563eb", "#ea580c", "#16a34a", "#7c3aed"
MUTED, REF, INK, RED = "#94a3b8", "#64748b", "#1e293b", "#dc2626"
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


def forest_rows(ax, rows, xlabel):
    """rows: list of dicts(label, d, lo, hi, gap|None, color)."""
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        if r.get("gap"):
            ax.add_patch(plt.Rectangle((-r["gap"], y - 0.32), 2 * r["gap"], 0.64, color=MUTED, alpha=0.22, lw=0))
        ax.plot([r["lo"], r["hi"]], [y, y], color=r["color"], lw=2, solid_capstyle="round")
        ax.plot(r["d"], y, "o", color=r["color"], ms=6, mec="white", mew=1.2, zorder=3)
    ax.axvline(0, color=REF, lw=1, ls="--")
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1])
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)


def fig_trials():
    t = pd.read_csv(T / "trials.csv").sort_values("trial")
    c = pd.read_csv(T / "confirm.csv").iloc[0]
    rows = [{"label": f"#{r.trial} {r.name}\n(vs {r.control}, {r.start[:4]}–{r.end[:4]})", "d": r.d_sharpe, "lo": r.ci_low,
             "hi": r.ci_high, "gap": r.min_gap_5pct, "color": BLUE} for r in t.itertuples()]
    rows.append({"label": f"#3 BOCPD on the hold-out\n({c.start[:4]}–{c.end[:4]}, one-sided p = {c.p_one_sided:.2f})",
                 "d": c.d_sharpe, "lo": c.ci_low, "hi": c.ci_high, "gap": None, "color": ORANGE})
    fig, ax = plt.subplots(figsize=(8, 6.2))
    forest_rows(ax, rows, "Sharpe difference vs control (combo of the three M2 factors)")
    ax.set_title("All regime methods vs their controls: point estimate, 95% block-bootstrap CI,\n"
                 "grey band = differences not distinguishable from 0 at 5%", loc="left")
    save(fig, "fig01_trials_forest.png")


def fig_excess():
    ex = pd.read_csv(T / "excess_weekly.csv", parse_dates=["date"])
    t = pd.read_csv(T / "trials.csv").set_index("trial")
    lo, hi = min(0, ex.excess.min()), max(0, ex.excess.max())
    fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), sharey=True)
    for ax, (tid, g) in zip(axes.ravel(), sorted(ex.groupby("trial"))):
        ax.plot(g.date, g.excess, color=BLUE, lw=1.4)
        ax.axhline(0, color=REF, lw=0.8, ls="--")
        ax.set_title(f"#{tid} {t.loc[tid, 'name']}", fontsize=8.5)
        ax.set_ylim(lo * 1.1, hi * 1.1)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:+.0%}")
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Cumulative return difference vs control (development window), weekly", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig02_cumulative_excess.png")


def fig_horizon():
    h = pd.read_csv(T / "horizons.csv").set_index("strategy")
    order = [("hmm", "Pure HMM\nregime strategy", VIOLET), ("m2_A2", "M2 combo\n(A2)", MUTED), ("tsmom_21", "TSMOM\n1m", BLUE),
             ("tsmom_63", "TSMOM\n3m", BLUE), ("tsmom_126", "TSMOM\n6m", BLUE), ("tsmom_252", "TSMOM\n12m", BLUE)]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    vals = [h.loc[k, "sharpe"] for k, _, _ in order]
    ax.bar(range(len(order)), vals, color=[c for _, _, c in order], width=0.6)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(len(order)), [l for _, l, _ in order])
    ax.set_ylabel("Sharpe (net, vol-sized like A2)")
    ax.set_title(f"Signal horizon matters more than regimes ({h.loc['hmm', 'start'][:7]} – {h.loc['hmm', 'end'][:7]})", loc="left")
    save(fig, "fig03_horizon.png")


def fig_hmm_diag():
    d = pd.read_csv(T / "hmm_diagnostics.csv")
    lab = {"hmm_k2_year_top": "K=2\nyearly", "hmm_k3_year_top": "K=3\nyearly", "hmm_k3_quarter_top": "K=3\nquarterly",
           "hmm_k3_month_top": "K=3\nmonthly"}
    x = np.arange(len(d))
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
    for ax, col, title, fmt in [(axes[0], "max_abs_corr_vol", "Max |corr(filter, 20d vol)| across contracts", "{:.2f}"),
                                (axes[1], "mean_filter", "Median average filter level (exposure kept)", "{:.2f}"),
                                (axes[2], "trend_duration", "Median trend-state duration (days)", "{:.0f}")]:
        ax.bar(x, d[col], color=[MUTED] + [BLUE] * (len(d) - 1), width=0.6)
        for i, v in enumerate(d[col]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, [lab[v] for v in d.variant])
        ax.set_title(title, fontsize=9)
    fig.suptitle("HMM diagnostics on data up to 2012 (no strategy returns used)", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig04_hmm_diagnostics.png")


def fig_predictability():
    p = pd.read_csv(T / "predictability.csv")
    p = p[p.in_family | (p.predictor == "er60")].copy()
    p["row"] = "R" + p["round"].astype(str) + " " + p.horizon + " · " + p.predictor
    rows = list(dict.fromkeys(p.row))
    fig, ax = plt.subplots(figsize=(7.2, 0.28 * len(rows) + 1.2))
    for level, color, mk in [("portfolio", ORANGE, "o"), ("panel", BLUE, "s")]:
        q = p[p.level == level].set_index("row").reindex(rows)
        ax.scatter(q.t, range(len(rows))[::-1], color=color, marker=mk, s=26, label=level, zorder=3)
    for v in (-1.96, 1.96):
        ax.axvline(v, color=RED, lw=0.9, ls="--")
    ax.axvline(0, color=REF, lw=0.8)
    ax.set_yticks(range(len(rows)), rows[::-1], fontsize=7.5)
    ax.set_xlabel("t-statistic (vol-controlled; dashed = ±1.96, before multiple-testing correction)")
    ax.set_title("Pre-registered predictability tests: can anything predict next-period trend returns?", loc="left")
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig05_predictability.png")


def fig_meta_auc():
    a = pd.read_csv(T / "meta_auc.csv")
    fig, ax = plt.subplots(figsize=(7, 3.3))
    for (model, g), color in zip(a.groupby("model", sort=False), [BLUE, MUTED, VIOLET]):
        ax.plot(g.year, g.auc, "-o", color=color, ms=4, lw=1.6, label=model)
    ax.axhline(0.5, color=REF, ls="--", lw=1)
    ax.set_ylabel("Out-of-sample AUC")
    ax.set_title("Meta-labeling: walk-forward AUC by year (0.5 = coin flip)", loc="left")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.15))
    save(fig, "fig06_meta_auc.png")


def fig_noise():
    s = pd.read_csv(T / "noise_share.csv")
    acf = pd.read_csv(T / "strategy_acf.csv")
    test = json.loads((T / "strategy_acf_test.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    x = np.arange(len(s))
    axes[0].bar(x - 0.18, s.raw, 0.36, color=MUTED, label="raw returns")
    axes[0].bar(x + 0.18, s.garch_std, 0.36, color=BLUE, label="GARCH-standardised")
    axes[0].set_xticks(x, s.test)
    axes[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_title("Share of 13 contracts rejecting 'no structure' at 5%", fontsize=9)
    axes[0].legend()
    band = acf.band.iloc[0]
    axes[1].bar(acf.lag, acf.acf, color=[BLUE if abs(v) > band else MUTED for v in acf.acf], width=0.55)
    for b in (band, -band):
        axes[1].axhline(b, color=REF, ls="--", lw=0.9)
    axes[1].axhline(0, color=REF, lw=0.8)
    axes[1].set_xlabel("lag (months)")
    axes[1].set_title(f"Exploratory: autocorrelation of monthly A2 returns (robust joint p = {test['monthly_A2_robust_p']:.3f})", fontsize=9)
    fig.tight_layout()
    save(fig, "fig07_noise.png")


def fig_bound():
    ref = pd.read_csv(T / "bound_reference.csv")
    ins = json.loads((T / "bound_insample.json").read_text())
    b = pd.read_csv(T / "bound.csv")
    oos = b[(b.level == "contract") & (b.model == "ridge")].iloc[0]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(ref.r2, ref.max_gain_contract, "-o", color=BLUE, ms=4, label="contract level")
    ax.plot(ref.r2, ref.max_gain_portfolio, "-o", color=ORANGE, ms=4, label="portfolio level")
    ax.plot(ins["adj_r2"], ins["max_gain_at_adj_r2"], "D", color=VIOLET, ms=8, label="in-sample adjusted R² (infeasible)")
    ax.annotate(f"adj. R² {ins['adj_r2']:.1%} → ≤ +{ins['max_gain_at_adj_r2']:.2f} Sharpe\n(only if coefficients known in advance)",
                (ins["adj_r2"], ins["max_gain_at_adj_r2"]), xytext=(12, -28), textcoords="offset points", fontsize=8)
    ax.text(0.0005, ax.get_ylim()[1] * 0.92, f"real-time (walk-forward) R² = {oos.oos_r2:+.1%}, 95% upper bound {oos.r2_upper95:+.1%}\n"
            "→ achievable gain 0", fontsize=8, color=RED, va="top")
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("monthly predictive R² of next-month A2 returns")
    ax.set_ylabel("max Sharpe gain (Campbell-Thompson)")
    ax.set_title("What predictability would be worth, and what we actually have", loc="left")
    ax.legend(loc="lower right")
    save(fig, "fig08_bound.png")


def fig_power():
    p = pd.read_csv(T / "power.csv")
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for dur, color, label in [(60, BLUE, "regimes last 60 days on average"), (250, GREEN, "250 days (positive control)")]:
        g = p[p.duration == dur].sort_values("d_oracle")
        ax.plot(g.d_oracle, g.power, "-o", color=color, ms=6, lw=1.6, label=label)
        for r in g.itertuples():
            ax.annotate(f"μ={r.mu}", (r.d_oracle, r.power), xytext=(4, 6 if dur == 60 else -13),
                        textcoords="offset points", fontsize=7.5, color=color)
    ax.set_ylim(-0.05, 1.05)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Sharpe gain of an oracle that knows the true state")
    ax.set_ylabel("share of worlds where our pipeline detects it")
    ax.set_title("Planted regimes: valuable but fast-switching regimes are not detectable", loc="left")
    ax.legend(loc="upper left")
    save(fig, "fig09_power.png")


def fig_race():
    r = pd.read_csv(T / "detection_race.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.3))
    sh = sorted(r.sharpe.unique())
    x = np.arange(len(sh))
    for k, (det, color) in enumerate([("BOCPD (H=1/250)", BLUE), ("TSMOM 60d", ORANGE)]):
        g = r[r.detector == det].set_index("sharpe").loc[sh]
        off = (k - 0.5) * 0.36
        axes[0].bar(x + off, g.median_days, 0.36, color=color, label=det)
        axes[0].errorbar(x + off, g.median_days, yerr=[g.median_days - g.p25, g.p75 - g.median_days], fmt="none", ecolor=REF, lw=1)
        axes[1].bar(x + off, g.wrong_before, 0.36, color=color, label=det)
    for ax in axes:
        ax.set_xticks(x, [f"trend Sharpe {s:g}" for s in sh])
    axes[0].set_title("Days until the detector turns after a reversal (median, IQR)", fontsize=9)
    axes[1].set_title("Share of days pointing the wrong way before the reversal", fontsize=9)
    axes[1].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].legend()
    fig.tight_layout()
    save(fig, "fig10_detection_race.png")


def fig_states():
    s = pd.read_csv(T / "regime_strategy_states.csv").sort_values("sharpe_low")
    fig, ax = plt.subplots(figsize=(7, 4.4))
    y = np.arange(len(s))
    ax.hlines(y, s.sharpe_low, s.sharpe_high, color=MUTED, lw=2)
    ax.plot(s.sharpe_low, y, "o", color=RED, ms=6, label="low-mean state")
    ax.plot(s.sharpe_high, y, "o", color=BLUE, ms=6, label="high-mean state")
    for yi, r in zip(y, s.itertuples()):
        ax.text(max(r.sharpe_high, 0) + 0.15, yi, f"vol ×{r.sd_low / r.sd_high:.1f}, long {r.share_long:.0%}", va="center", fontsize=7.5, color=REF)
    ax.axvline(0, color=REF, ls="--", lw=0.9)
    ax.set_yticks(y, [f"{r.contract} ({r.asset_class})" for r in s.itertuples()])
    ax.set_xlabel("annualised Sharpe of each state (median over refits)")
    ax.set_title("Pure HMM strategy: what the two states really are", loc="left")
    ax.set_xlim(s.sharpe_low.min() - 0.4, s.sharpe_high.max() + 2.6)  # room for the right-hand notes
    ax.legend(loc="upper left")
    ax.grid(axis="y", visible=False)
    save(fig, "fig11_pure_hmm_states.png")


if __name__ == "__main__":
    for f in (fig_trials, fig_excess, fig_horizon, fig_hmm_diag, fig_predictability, fig_meta_auc, fig_noise,
              fig_bound, fig_power, fig_race, fig_states):
        f()
