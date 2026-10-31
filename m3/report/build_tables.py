"""Collect every result the M3 report cites into m3/report/tables/ (small, tracked CSVs).

Run from the repository root after the M3 scripts have produced m3_outputs/ and
backtest_outputs/ (see the report's reproduction section):

    python -m m3.report.build_tables
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from m3 import predictability
from m3.bocpd import BOCPDConfig, run_bocpd
from m3.features import build_features

M3, BT = Path("m3_outputs"), Path("backtest_outputs")
OUT = Path("m3/report/tables")
DEV_END = "2014-12-31"
DATA = Path("data/final")
EXOG = Path("data/processed/offmarket/offmarket_daily.csv")

FILTER_TRIALS = [  # trial, backtest folder, name
    (1, "hmm_base", "HMM (in-market features)"),
    (2, "hmm_offmkt", "HMM + macro/news"),
    (3, "bocpd_h250", "BOCPD changepoints"),
    (4, "portvol", "Portfolio covariance vol target"),
    (6, "meta_hand_gbm", "Meta-labeling (GBM)"),
    (7, "meta_sig_gbm", "Signature meta-labeling"),
]


def trials():
    rows, factors, excess = [], [], []
    for tid, folder, name in FILTER_TRIALS:
        pr = pd.read_csv(BT / folder / "a3_vs_a2.csv").query("window == 'dev'")
        sm = pd.read_csv(BT / folder / "summary.csv").query("window == 'dev' and factor == 'combo'").set_index("approach")
        c = pr[pr.factor == "combo"].iloc[0]
        rows.append({"trial": tid, "name": name, "control": "A2 (vol sizing)", "start": sm.loc["A2", "start"],
                     "end": sm.loc["A2", "end"], "sharpe_control": sm.loc["A2", "sharpe"], "sharpe_treat": sm.loc["A3", "sharpe"],
                     "gross_exp_treat": sm.loc["A3", "gross_exp"], "d_sharpe": c.d_sharpe, "ci_low": c.ci_low,
                     "ci_high": c.ci_high, "min_gap_5pct": c.min_gap_5pct, "p_boot": c.p_boot, "corr": c["corr"]})
        for r in pr.itertuples():
            factors.append({"trial": tid, "name": name, "factor": r.factor, "d_sharpe": r.d_sharpe,
                            "ci_low": r.ci_low, "ci_high": r.ci_high, "p_boot": r.p_boot})
        d = pd.read_csv(BT / folder / "daily_net_returns.csv", index_col="date", parse_dates=True).loc[:DEV_END]
        x = (d.combo_A3 - d.combo_A2).dropna().cumsum()
        excess.append(pd.DataFrame({"date": x.index, "trial": tid, "excess": x.values}))
    for tid, prefix, treat, ctrl, name, ctrl_name in [
            (5, "turning_points", "dynamic", "static_slow", "Momentum turning points", "static 12m TSMOM"),
            (8, "regime_strategy", "hmm", "tsmom_252", "Pure HMM regime strategy", "static 12m TSMOM")]:
        sm = pd.read_csv(M3 / f"{prefix}_summary.csv").set_index("strategy")
        c = pd.read_csv(M3 / f"{prefix}_pairs.csv").set_index("control").loc[ctrl]
        rows.append({"trial": tid, "name": name, "control": ctrl_name, "start": sm.loc[treat, "start"], "end": sm.loc[treat, "end"],
                     "sharpe_control": sm.loc[ctrl, "sharpe"], "sharpe_treat": sm.loc[treat, "sharpe"],
                     "gross_exp_treat": sm.loc[treat, "gross_exp"], "d_sharpe": c.d_sharpe, "ci_low": c.ci_low,
                     "ci_high": c.ci_high, "min_gap_5pct": c.min_gap_5pct, "p_boot": c.p_boot, "corr": c["corr"]})
        d = pd.read_csv(M3 / f"{prefix}_daily.csv", index_col="date", parse_dates=True).loc[:DEV_END]
        x = (d[treat] - d[ctrl]).dropna().cumsum()
        excess.append(pd.DataFrame({"date": x.index, "trial": tid, "excess": x.values}))
    t = pd.DataFrame(rows).sort_values("trial")
    t.round(6).to_csv(OUT / "trials.csv", index=False)
    pd.DataFrame(factors).round(6).to_csv(OUT / "trial_factors.csv", index=False)
    ex = pd.concat(excess).set_index("date").groupby("trial").excess.resample("W-FRI").last().reset_index().dropna()
    ex.round(6).to_csv(OUT / "excess_weekly.csv", index=False)
    cf = json.loads((M3 / "confirmatory_test.json").read_text())
    pd.DataFrame([{"test": "BOCPD hold-out confirmation", "start": cf["window"][0], "end": cf["window"][1],
                   "days": cf["days"], "sharpe_A2": cf["sharpe_A2"], "sharpe_A3": cf["sharpe_A3"],
                   "d_sharpe": cf["d_sharpe"], "p_one_sided": cf["p_one_sided"], "lower_95": cf["lower_95_one_sided"],
                   "ci_low": cf["ci95_two_sided"][0], "ci_high": cf["ci95_two_sided"][1]}]).round(6) \
        .to_csv(OUT / "confirm.csv", index=False)


def horizons():
    s = pd.read_csv(M3 / "regime_strategy_summary.csv")
    s.round(6).to_csv(OUT / "horizons.csv", index=False)
    pd.read_csv(M3 / "regime_strategy_signcorr.csv").round(6).to_csv(OUT / "regime_strategy_signcorr.csv", index=False)
    pd.read_csv(M3 / "regime_strategy_states.csv").round(6).to_csv(OUT / "regime_strategy_states.csv", index=False)
    pd.read_csv(M3 / "turning_points_summary.csv").round(6).to_csv(OUT / "turning_points.csv", index=False)
    pd.read_csv(M3 / "turning_points_pairs.csv").round(6).to_csv(OUT / "turning_points_pairs.csv", index=False)


def hmm_diagnostics():
    """K=2 vs K=3 (quarterly/yearly refits), data up to 2012 only, no strategy returns."""
    inv = lambda v: 1 / (1 + np.exp(-v))
    feats = {c: build_features(pd.read_csv(DATA / f"{c}.csv", index_col="date", parse_dates=True).ret)
             for c in pd.read_csv(DATA / "universe.csv").code}
    rows = []
    for stem in ["hmm_k2_year_top", "hmm_k3_year_top", "hmm_k3_quarter_top", "hmm_k3_month_top"]:
        K = int(stem[5])
        fits = pd.read_csv(M3 / f"{stem}_fits.csv", parse_dates=["vintage"]).query("vintage <= '2012-12-31'")
        filt = pd.read_csv(M3 / f"{stem}_filter.csv", parse_dates=["date"])
        cv, mf = [], []
        for c, g in filt.groupby("contract"):
            x = g.set_index("date").loc[:"2012-12-31", "filter_close_t"].dropna()
            cv.append(abs(x.corr(feats[c].logvol20.reindex(x.index))))
            mf.append(x.mean())
        gap = inv(fits[f"er60_{K-1}"]) - inv(fits[f"er60_{K-2}"])
        rows.append({"variant": stem, "K": K, "mean_filter": float(np.median(mf)), "max_abs_corr_vol": float(np.max(cv)),
                     "trend_state_er60": float(inv(fits[f"er60_{K-1}"]).median()), "er_gap_next": float(gap.median()),
                     "weak_label_share": float((gap < 0.03).mean()), "trend_duration": float(fits[f"dur_{K-1}"].median())})
    pd.DataFrame(rows).round(4).to_csv(OUT / "hmm_diagnostics.csv", index=False)


def predictability_tables():
    rows, joint = [], []
    for rnd in (1, 2):
        res, oos = predictability.main(DATA, EXOG, M3, rnd)
        cfg = predictability.ROUNDS[rnd]
        for level, (uni, j) in res.items():
            for x, r in uni.iterrows():
                rows.append({"round": rnd, "horizon": cfg["label"], "level": level, "predictor": x,
                             "in_family": x in cfg["family"], "t": r.t, "p": r.p, "p_holm": r.get("p_holm", np.nan)})
            joint.append({"round": rnd, "horizon": cfg["label"], "level": level, "joint_p": j["p"], "n": j["n"],
                          "oos_r2_portfolio": oos["oos_r2"]})
    r3 = pd.read_csv(M3 / "predictability_r3.csv")
    o3 = pd.read_csv(M3 / "predictability_r3_oos.csv").set_index("horizon")
    for r in r3.itertuples():
        rows.append({"round": 3, "horizon": r.horizon, "level": r.level, "predictor": r.predictor, "in_family": True,
                     "t": r.t, "p": r.p, "p_holm": r.p_holm})
    for (h, lvl), g in r3.groupby(["horizon", "level"]):
        joint.append({"round": 3, "horizon": h, "level": lvl, "joint_p": g.joint_p.iloc[0], "n": g.n.iloc[0],
                      "oos_r2_portfolio": o3.loc[h, "oos_r2"]})
    pd.DataFrame(rows).round(4).to_csv(OUT / "predictability.csv", index=False)
    pd.DataFrame(joint).round(4).to_csv(OUT / "predictability_joint.csv", index=False)


def meta_and_evidence():
    aucs = []
    for tag, model in [("meta_hand_gbm", "hand features, GBM"), ("meta_hand_logit", "hand features, logistic"),
                       ("meta_sig_gbm", "signature, GBM")]:
        a = pd.read_csv(M3 / f"{tag}_auc.csv").assign(model=model)
        aucs.append(a)
    pd.concat(aucs).round(4).to_csv(OUT / "meta_auc.csv", index=False)
    nz = pd.read_csv(M3 / "evidence_noise.csv")
    nz.round(4).to_csv(OUT / "noise_by_contract.csv", index=False)
    share = []
    for label, raw, std in [("LB(10) returns", "lb_r", "lb_z"), ("LB(10) squared", "lb_r2", "lb_z2"),
                            ("BDS m=2", "bds_r_m2", "bds_z_m2"), ("BDS m=3", "bds_r_m3", "bds_z_m3")]:
        share.append({"test": label, "raw": float((nz[raw] < 0.05).mean()), "garch_std": float((nz[std] < 0.05).mean())})
    pd.DataFrame(share).round(4).to_csv(OUT / "noise_share.csv", index=False)
    ns = json.loads((M3 / "evidence_noise_summary.json").read_text())
    pd.DataFrame([{"lag": int(k), "acf": v, "band": 2 / np.sqrt(ns["months"])} for k, v in ns["monthly_A2_acf_1to6"].items()]) \
        .round(4).to_csv(OUT / "strategy_acf.csv", index=False)
    json.dump({k: ns[k] for k in ("monthly_A2_robust_p", "monthly_A2_lb6_p", "months")}, open(OUT / "strategy_acf_test.json", "w"), indent=2)
    pd.read_csv(M3 / "evidence_bound.csv").round(4).to_csv(OUT / "bound.csv", index=False)
    pd.read_csv(M3 / "evidence_bound_reference.csv").round(4).to_csv(OUT / "bound_reference.csv", index=False)
    (OUT / "bound_insample.json").write_text((M3 / "evidence_bound_insample.json").read_text())
    p = pd.concat([pd.read_csv(M3 / "evidence_power_summary.csv").assign(duration=60),
                   pd.read_csv(M3 / "evidence_power_control_summary.csv")], ignore_index=True)
    p[["duration", "mu", "power", "auc", "d_meta", "d_oracle"]].round(4).to_csv(OUT / "power.csv", index=False)


def detection_race():
    """Simulated drift reversal +S -> -S (annual Sharpe S): days until BOCPD / 60-day TSMOM turn."""
    rng = np.random.default_rng(42)
    rows = []
    for S in (0.5, 1.0, 2.0, 4.0):
        dmu = S / np.sqrt(252)
        res = {"BOCPD (H=1/250)": ([], []), "TSMOM 60d": ([], [])}
        for _ in range(300):
            z = np.r_[rng.normal(dmu, 1, 750), rng.normal(-dmu, 1, 500)]
            _, mu, _ = run_bocpd(z, BOCPDConfig())
            mom = pd.Series(z).rolling(60).sum().to_numpy()
            for name, sig in [("BOCPD (H=1/250)", mu), ("TSMOM 60d", mom)]:
                after = sig[750:]
                res[name][0].append(np.argmax(after < 0) if (after < 0).any() else 500)
                res[name][1].append(np.mean(sig[500:750] < 0))
        for name, (delay, wrong) in res.items():
            rows.append({"sharpe": S, "detector": name, "median_days": float(np.median(delay)),
                         "p25": float(np.percentile(delay, 25)), "p75": float(np.percentile(delay, 75)),
                         "wrong_before": float(np.mean(wrong))})
    pd.DataFrame(rows).round(4).to_csv(OUT / "detection_race.csv", index=False)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    trials()
    horizons()
    hmm_diagnostics()
    predictability_tables()
    meta_and_evidence()
    detection_race()
    for f in sorted(OUT.iterdir()):
        print(f"{f.name:32s} {f.stat().st_size / 1024:6.1f} KB")
