"""Copy M4 results into report/tables/ and draw the report figures.

    python -m m4.report.make_figures      # after python -m m4.gates and python -m m4.trials
"""

import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from m3.features import load_exog  # noqa: E402
from m4.common import OFFMARKET_FILE, vix_stress  # noqa: E402

OUT = Path("m4_outputs")
REP = Path(__file__).resolve().parent
TAB, FIG = REP / "tables", REP / "figures"
BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
TABLES = ["gates.csv", "describe.csv", "offmarket_audit.csv", "trials_m1_engine.csv",
          "fomc_filter_log.csv", "riskoff_filter_log.csv",
          "crowding_gate.csv", "crowding_describe.csv", "crowding_filter_log.csv",
          "inversion_gate.csv", "impvol_describe.csv", "inversion_filter_log.csv"]

plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRAY, "axes.labelcolor": MUTED, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "axes.axisbelow": True})


def ci(df):
    se = (df.coef_ann / df.t).abs()
    return df.coef_ann, 1.96 * se


def fig_gates(gates, desc):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    # M4-1: pooled + classes
    rows = [("pooled (gate)", gates.iloc[0].coef_ann, gates.iloc[0].coef_ann / gates.iloc[0].t)]
    for r in desc[(desc.trial == "M4-1") & desc.analysis.str.startswith("class")].itertuples():
        rows.append((r.analysis.replace("class ", ""), r.coef_ann, r.coef_ann / r.t))
    ph = desc[desc.trial == "M4-1 post hoc"].iloc[0]
    rows.append(("pooled, settle timing\n(post hoc)", ph.coef_ann, ph.coef_ann / ph.t))
    ax = axes[0]
    for i, (lab, c, se) in enumerate(rows):
        col = BLUE if i == 0 else (ORANGE if "post hoc" in lab else GRAY)
        ax.errorbar(c, i, xerr=1.96 * abs(se), fmt="o", color=col, ms=6, capsize=3, lw=1.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("event-day minus other days, gross A2 P&L (annualised)")
    ax.set_title("M4-1  FOMC statement days", loc="left", color=INK)
    # M4-2: W at three quantiles + contracts
    d2 = desc[(desc.trial == "M4-2") & (desc.term == "W")]
    rows = [("q = 0.80 (gate)", gates.iloc[1].coef_ann, gates.iloc[1].coef_ann / gates.iloc[1].t)]
    for r in d2.itertuples():
        rows.append((r.analysis.replace("stress quantile", "q =").replace("contract ", ""), r.coef_ann,
                     r.coef_ann / r.t))
    order = [rows[0]] + sorted([r for r in rows[1:] if r[0].startswith("q")]) + [r for r in rows[1:]
                                                                                 if not r[0].startswith("q")]
    ax = axes[1]
    for i, (lab, c, se) in enumerate(order):
        col = BLUE if i == 0 else (MUTED if lab.startswith("q") else GRAY)
        ax.errorbar(c, i, xerr=1.96 * abs(se), fmt="o", color=col, ms=6, capsize=3, lw=1.5)
    ax.set_yticks(range(len(order)), [r[0] for r in order])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("interaction W = stress x risk-on (annualised)")
    ax.set_title("M4-2  VIX stress x risk-on position", loc="left", color=INK)
    fig.text(0.01, 0.01, "Points: estimate; bars: 95% CI (contract FE, SEs clustered by month). Development window "
             "to 2014-12-31.", color=MUTED, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig01_gates.png", dpi=160)
    plt.close(fig)


def fig_timing(desc):
    d = desc[desc.analysis.str.startswith("|ret| event/other")].copy()
    d["timing"] = np.where(d.analysis.str.contains("settle"), "settle", "prereg")
    w = d.pivot(index="term", columns="timing", values="fe_ratio_in_out").sort_values("prereg")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(w))
    ax.barh(y + 0.2, w.prereg, height=0.38, color=GRAY, label="pre-registered mapping (statement day)")
    ax.barh(y - 0.2, w.settle, height=0.38, color=BLUE, label="settlement-time mapping (post hoc)")
    ax.axvline(1, color=INK, lw=0.8)
    ax.set_yticks(y, w.index)
    ax.set_xlabel("mean |ret| on event days / other days")
    ax.set_title("FOMC days: GC/SI react the next day (COMEX settles 13:30 ET)", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    fig.savefig(FIG / "fig02_event_timing.png", dpi=160)
    plt.close(fig)


def fig_state(exog, l1, l2):
    v = exog["vix"].dropna()
    s = vix_stress(exog).loc["2001":"2014"]
    lvl = np.exp(v.loc["2001":"2014"])
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
    ax = axes[0]
    ax.fill_between(s.index, 0, lvl.max() * 1.05, where=s.astype(bool), color=ORANGE, alpha=0.18, lw=0,
                    label="stress state S (VIX >= expanding 80th pct)")
    ax.plot(lvl.index, lvl, color=INK, lw=1)
    ax.set_ylabel("VIX")
    ax.set_title("M4-2 state and the exploratory filters' quarterly estimates", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax = axes[1]
    for lg, col, lab, ls, lw in ((l2, ORANGE, "M4-2 filter when W = 1", "-", 3.5),
                                 (l1, BLUE, "M4-1 filter on event days", "--", 1.6)):
        g = lg.assign(v=pd.to_datetime(lg.vintage)).set_index("v").loc[:"2014"]
        ax.step(g.index, g.f_in, where="post", color=col, lw=lw, ls=ls, label=lab)
    ax.text(0.99, 0.55, "both are 1 outside their state; before the first vintage the filter is 1",
            transform=ax.transAxes, ha="right", color=MUTED, fontsize=7.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("filter")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_state_filters.png", dpi=160)
    plt.close(fig)


def fig_forest(tr):
    m3 = pd.read_csv("m3/report/tables/trials_m1_engine.csv")
    m3 = m3[(m3.engine == "m1") & (m3.trial == 2)].assign(trial="M3 #2 HMM+macro", role="M3")
    ex = tr[tr.role == "exploratory"]
    d = pd.concat([m3, ex])
    labels = ["M3 #2 HMM+macro", "M4-1", "M4-2", "M4-3", "M4-4", "M4-5"]
    ex = tr[(tr.role == "exploratory") | ((tr.trial == "M4-4") & tr.sharpe_control.notna())]
    d = pd.concat([m3, ex])
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for j, costs in enumerate(("research", "official")):
        sub = d[d.cost_model == costs].set_index("trial")
        for i, lab in enumerate(labels):
            r = sub.loc[lab]
            ax.errorbar(r.d_sharpe, i + (0.15 if j else -0.15), xerr=[[r.d_sharpe - r.ci_low], [r.ci_high - r.d_sharpe]],
                        fmt="o", color=BLUE if j == 0 else ORANGE, ms=6, capsize=3, lw=1.5,
                        label=f"{costs} costs" if i == 0 else None)
    ax.set_yticks(range(6), ["M3 #2  HMM + macro\n(primary)", "M4-1  FOMC days\n(exploratory)",
                             "M4-2  stress x risk-on\n(exploratory)", "M4-3  COT crowding\n(exploratory)",
                             "M4-4  implied / realised vol\n(primary, no gate)",
                             "M4-5  VIX inversion x risk-on\n(exploratory)"])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("A3 - A2 Sharpe, M1 engine (95% block-bootstrap CI)")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.set_title("Off-market filters vs volatility sizing", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig04_trials_forest.png", dpi=160)
    plt.close(fig)


def fig_crowding(gate, desc, log):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.2, 1]})
    g = gate.iloc[0]
    rows = [("pooled, 0.90 / 0.10 (gate)", g.coef_ann, g.coef_ann / g.t)]
    rows += [(r.analysis.replace("class ", "").replace("hedgers (commercial) net instead of speculators",
                                                       "hedgers' net instead"), r.coef_ann, r.coef_ann / r.t)
             for r in desc.itertuples()]
    ax = axes[0]
    for i, (lab, c, se) in enumerate(rows):
        ax.errorbar(c, i, xerr=1.96 * abs(se), fmt="o", color=BLUE if i == 0 else GRAY, ms=6, capsize=3, lw=1.5)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_xlabel("crowded minus other days, gross A2 P&L (annualised)")
    ax.set_title("M4-3  trend aligned with extreme speculator position", loc="left", color=INK)
    ax = axes[1]
    lg = log.assign(v=pd.to_datetime(log.vintage)).set_index("v").loc[:"2014"]
    ax.step(lg.index, lg.f_in, where="post", color=BLUE, lw=2, label="filter when crowded")
    ax.step(lg.index, lg.f_out, where="post", color=ORANGE, lw=2, label="filter otherwise")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("filter (exploratory rule)")
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.set_title("Quarterly estimates", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig05_crowding.png", dpi=160)
    plt.close(fig)


def fig_impvol(desc, tr):
    fc = desc[desc.analysis.str.startswith("forecast")]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    labs = [a.split(", ")[1] for a in fc.analysis]
    x = np.arange(len(fc))
    ax.bar(x - 0.2, fc.r2_rv60_only, width=0.38, color=GRAY, label="trailing 60-day realised vol only")
    ax.bar(x + 0.2, fc.r2_rv60_and_iv, width=0.38, color=BLUE, label="+ implied vol")
    for i, r in enumerate(fc.itertuples()):
        ax.text(i + 0.2, r.r2_rv60_and_iv + 0.01, f"t={r.t:.1f}", ha="center", fontsize=7.5, color=MUTED)
    ax.set_xticks(x, labs)
    ax.set_ylim(0, 0.8)
    ax.set_ylabel("R2, next-20-day log realised vol")
    ax.set_title("Implied vol forecasts risk better ...", loc="left", color=INK)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax = axes[1]
    sc = desc[desc.analysis.str.startswith("single-contract")]
    y = np.arange(len(sc))
    ax.barh(y, sc.d_sharpe, color=[BLUE if v >= 0 else ORANGE for v in sc.d_sharpe], height=0.5)
    ax.set_yticks(y, [a.split(", ")[1] for a in sc.analysis])
    ax.invert_yaxis()
    ax.axvline(0, color=INK, lw=0.8)
    m = tr[(tr.trial == "M4-4") & (tr.cost_model == "research") & tr.sharpe_control.notna()].iloc[0]
    ax.set_xlabel(f"A3 - A2 Sharpe, single contract (portfolio: {m.d_sharpe:+.3f}, p = {m.p_boot:.2f})")
    ax.set_title("... but not into higher Sharpe", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig06_implied_vol.png", dpi=160)
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        df = pd.read_csv(OUT / t)
        df.round(6).to_csv(TAB / t, index=False)
    shutil.copy(OUT / "gates.json", TAB / "gates.json")
    gates, desc = pd.read_csv(TAB / "gates.csv"), pd.read_csv(TAB / "describe.csv")
    fig_gates(gates, desc)
    fig_timing(desc)
    fig_state(load_exog(OFFMARKET_FILE), pd.read_csv(TAB / "fomc_filter_log.csv"),
              pd.read_csv(TAB / "riskoff_filter_log.csv"))
    fig_forest(pd.read_csv(TAB / "trials_m1_engine.csv"))
    fig_impvol(pd.read_csv(TAB / "impvol_describe.csv"), pd.read_csv(TAB / "trials_m1_engine.csv"))
    fig_crowding(pd.read_csv(TAB / "crowding_gate.csv"), pd.read_csv(TAB / "crowding_describe.csv"),
                 pd.read_csv(TAB / "crowding_filter_log.csv"))
    print("tables ->", TAB, "\nfigures ->", FIG)


if __name__ == "__main__":
    main()
