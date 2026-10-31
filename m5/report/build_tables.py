"""Collect every result the M5 report cites into m5/report/tables/ (small, tracked CSVs).

Run from the repository root after the M5 scripts have produced m5_outputs/ and
backtest_outputs/ (see m5/README.md):

    python -m m5.report.build_tables
"""

import json
from pathlib import Path

import pandas as pd

M3, M5, BT = Path("m3_outputs"), Path("m5_outputs"), Path("backtest_outputs")
OUT = Path("m5/report/tables")
DEV_END = "2014-12-31"
LABELS = {"hmm": "M3 #1 HMM (per contract)", "bocpd": "M3 #3 BOCPD",
          "pool_class": "M5-1 class-pooled HMM", "pool_global": "M5-2 global-pooled HMM"}
POOLED = {"pool_class": "pool_class_k3_quarter_top", "pool_global": "pool_global_k3_quarter_top"}


def reproduction():
    rows = []
    for folder, name in [("hmm_base", "M3 #1 HMM"), ("bocpd_h250", "M3 #3 BOCPD")]:
        pr = pd.read_csv(BT / folder / "a3_vs_a2.csv").query("window == 'dev' and factor == 'combo'").iloc[0]
        sm = pd.read_csv(BT / folder / "summary.csv").query("window == 'dev' and factor == 'combo'").set_index("approach")
        rows.append({"method": name, "start": sm.loc["A2", "start"], "end": sm.loc["A2", "end"],
                     "sharpe_A2": sm.loc["A2", "sharpe"], "sharpe_A3": sm.loc["A3", "sharpe"],
                     "d_sharpe": pr.d_sharpe, "ci_low": pr.ci_low, "ci_high": pr.ci_high, "p_boot": pr.p_boot})
    m3 = pd.read_csv("m3/report/tables/trials.csv").set_index("trial")
    r = pd.DataFrame(rows)
    r["m3_report_d_sharpe"] = [m3.loc[1, "d_sharpe"], m3.loc[3, "d_sharpe"]]
    r.round(6).to_csv(OUT / "reproduction.csv", index=False)


def classes():
    tabs, het = [], []
    for label, name in LABELS.items():
        t = pd.read_csv(M5 / f"classes_{label}.csv").assign(label=label, method=name)
        tabs.append(t)
        h = json.loads((M5 / f"classes_{label}_heterogeneity.json").read_text())
        het.append({"label": label, "method": name, **{k: v for k, v in h.items() if k not in ("corr_matrix", "classes")}})
    cols = ["label", "method", "group", "n_contracts", "start", "end", "sharpe_A1", "sharpe_A2", "sharpe_A3",
            "maxdd_A2", "maxdd_A3", "d_sharpe", "ci_low", "ci_high", "se_boot", "min_gap_5pct", "mde_80", "p_boot",
            "corr_A3_A2"]
    pd.concat(tabs)[cols].round(6).to_csv(OUT / "classes.csv", index=False)
    pd.DataFrame(het).round(6).to_csv(OUT / "heterogeneity.csv", index=False)
    ex = []
    for label in ("hmm", "pool_class"):
        w = pd.read_csv(M5 / f"classes_{label}_excess_weekly.csv", index_col="date", parse_dates=True)
        w = w.resample("ME").last().rename_axis("date").reset_index()
        ex.append(w.melt(id_vars="date", var_name="group", value_name="excess").assign(label=label))
    ex = pd.concat(ex).dropna()
    ex.assign(excess=ex.excess.round(5)).to_csv(OUT / "classes_excess_monthly.csv", index=False)


def diagnostics():
    per, cls = [], []
    for label in ("hmm", "pool_class", "pool_global"):
        per.append(pd.read_csv(M5 / f"diag_{label}_contracts.csv").assign(label=label))
        cls.append(pd.read_csv(M5 / f"diag_{label}_classes.csv").assign(label=label))
    pd.concat(per).round(4).to_csv(OUT / "diag_contracts.csv", index=False)
    pd.concat(cls).round(4).to_csv(OUT / "diag_classes.csv", index=False)
    pd.read_csv(M5 / "profile_bocpd.csv").round(4).to_csv(OUT / "profile_bocpd.csv", index=False)

    m3 = pd.read_csv(M3 / "hmm_k3_quarter_top_filter.csv", parse_dates=["date"]).set_index(["contract", "date"]).filter_close_t
    rows = []
    for label, stem in POOLED.items():
        f = pd.read_csv(M5 / f"{stem}_filter.csv", parse_dates=["date"]).set_index(["contract", "date"]).filter_close_t
        d = pd.concat([m3.rename("m3"), f.rename("pool")], axis=1).dropna()
        d = d[d.index.get_level_values("date") <= DEV_END]
        for code, g in d.groupby(level="contract", sort=False):
            rows.append({"label": label, "contract": code, "corr_with_m3": g.m3.corr(g.pool),
                         "mean_m3": g.m3.mean(), "mean_pool": g.pool.mean()})
        rows.append({"label": label, "contract": "all", "corr_with_m3": d.m3.corr(d.pool),
                     "mean_m3": d.m3.mean(), "mean_pool": d.pool.mean()})
    pd.DataFrame(rows).round(4).to_csv(OUT / "filter_corr.csv", index=False)


def trials():
    t = pd.read_csv(M5 / "trials.csv")
    t.round(6).to_csv(OUT / "trials.csv", index=False)
    factors, excess = [], []
    for folder, tid in [("hmm_base", "M3 #1"), ("m5_pool_class", "M5-1"), ("m5_pool_global", "M5-2")]:
        pr = pd.read_csv(BT / folder / "a3_vs_a2.csv").query("window == 'dev'")
        for r in pr.itertuples():
            factors.append({"trial": tid, "factor": r.factor, "d_sharpe": r.d_sharpe, "ci_low": r.ci_low,
                            "ci_high": r.ci_high, "p_boot": r.p_boot})
        d = pd.read_csv(BT / folder / "daily_net_returns.csv", index_col="date", parse_dates=True).loc[:DEV_END]
        x = (d.combo_A3 - d.combo_A2).dropna().cumsum()
        excess.append(pd.DataFrame({"date": x.index, "trial": tid, "excess": x.values}))
    pd.DataFrame(factors).round(6).to_csv(OUT / "trial_factors.csv", index=False)
    ex = pd.concat(excess).set_index("date").groupby("trial").excess.resample("W-FRI").last().reset_index().dropna()
    ex.assign(excess=ex.excess.round(6)).to_csv(OUT / "excess_weekly.csv", index=False)


def calendar_trials():
    for f in ("season_gate", "season_describe", "expiry_gate", "expiry_describe", "expiry_calendar_qa",
              "posthoc_roll_event_profile"):
        pd.read_csv(M5 / f"{f}.csv").round(6).to_csv(OUT / f"{f}.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    reproduction()
    classes()
    diagnostics()
    trials()
    calendar_trials()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:28s} {f.stat().st_size / 1024:6.1f} KB")
