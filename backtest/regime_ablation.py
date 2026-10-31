"""Post-hoc frozen-filter ablation: A1, A2, R=A1*f, A3=A2*f.

Run: python -m backtest.regime_ablation
Uses existing filter CSVs, M1 accounting and both existing cost specifications.
Does not fit models, tune parameters, rescale risk ex post or inspect the hold-out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from m1.backtest import performance
from m3.predictability import holm
from .engine import in_book
from .m1_bridge import COST_MODELS, DEV_END, FILTER_TRIALS, M5_TRIALS, m1_book, window_start
from .metrics import block_bootstrap_sharpe_diff
from .run import load_targets

ROOT = Path(__file__).resolve().parents[1]
APPROACHES = ("A1", "A2", "R", "A3")
NEW_PAIRS = (("R", "A2"), ("R", "A3"))


def jobs():
    result = []
    for tid, (name, stem) in FILTER_TRIALS.items():
        result.append(dict(trial=f"M3-{tid}", name=name,
                           kind="volatility_benchmark" if tid == 4 else "regime_filter",
                           filter_file=f"m3_outputs/{stem}_filter.csv",
                           reference=f"backtest_outputs/m1_rerun/trial{tid}_daily_net.csv"))
    result.append(dict(trial="M4-4", name="Implied / realised volatility filter", kind="regime_filter",
                       filter_file="m4_outputs/impvol_filter.csv", reference="m4_outputs/M4-4_daily_net.csv"))
    for tid, (name, stem) in M5_TRIALS.items():
        result.append(dict(trial=tid, name=name, kind="regime_filter",
                           filter_file=f"m5_outputs/{stem}_filter.csv",
                           reference=f"backtest_outputs/m1_m5/{tid}_daily_net.csv"))
    return result


def inputs(job, data_dir):
    path = ROOT / job["filter_file"]
    contracts, _ = load_targets(data_dir, path)
    f = pd.read_csv(path, parse_dates=["date"]).set_index(["contract", "date"]).filter_close_t
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"].combo.notna() for k, c in contracts.items()}).reindex(union))
    targets = {ap: {} for ap in APPROACHES}
    filters = {}
    for code, c in contracts.items():
        ff = f.loc[code].reindex(c["data"].index)
        finite = ff.dropna()
        if not np.isfinite(finite).all() or (finite < 0).any():
            raise ValueError(f"Invalid filter: {job['trial']} {code}")
        if job["kind"] == "regime_filter" and (finite > 1 + 1e-12).any():
            raise ValueError("Expected a bounded attenuation filter")
        for ap in ("A1", "A2", "A3"):
            targets[ap][code] = c[ap].combo
        targets["R"][code] = c["A1"].combo * ff
        np.testing.assert_allclose(targets["A3"][code], targets["A2"][code] * ff, equal_nan=True)
        filters[code] = ff
    return ({k: c["data"] for k, c in contracts.items()},
            {k: c["asset"] for k, c in contracts.items()}, targets, book, filters)


def overlap_diagnostics(targets, book, filters, start):
    rows = []
    for code, signal in targets["A1"].items():
        d = pd.DataFrame({"signal": signal, "a2": targets["A2"][code],
                          "a3": targets["A3"][code], "r": targets["R"][code], "f": filters[code]})
        d = d.loc[start:DEV_END]
        d = d.loc[book[code].reindex(d.index).fillna(False) & d.signal.ne(0)].dropna()
        d["v"] = d.a2 / d.signal
        d["contract"] = code
        rows.append(d)
    x = pd.concat(rows)
    both = (x.v < 1 - 1e-10) & (x.f < 1 - 1e-10)
    w = x.signal.abs()
    return dict(active_contract_days=len(x), both_cut_share=float(both.mean()),
                both_cut_signal_weighted_share=float(w[both].sum() / w.sum()),
                vol_cut_share=float((x.v < 1 - 1e-10).mean()),
                filter_cut_share=float((x.f < 1 - 1e-10).mean()),
                filter_scale_corr=float(x.f.corr(x.v)),
                a3_target_abs_ratio_a2=float(x.a3.abs().sum() / x.a2.abs().sum()),
                r_target_abs_ratio_a1=float(x.r.abs().sum() / x.signal.abs().sum()),
                a3_target_abs_ratio_r=float(x.a3.abs().sum() / x.r.abs().sum()),
                mean_v_both_cut=float(x.loc[both, "v"].mean()),
                mean_f_both_cut=float(x.loc[both, "f"].mean()))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(out_dir):
    data_dir = ROOT / "data/final"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "summary.csv").exists():
        raise FileExistsError("Use a fresh output directory to preserve the previous analysis")
    jj = jobs()
    files = [ROOT / j["filter_file"] for j in jj] + sorted(data_dir.glob("*.csv"))
    files += [Path(__file__), ROOT / "m1/config/fee_schedule.json", ROOT / "m2/volatility.py",
              ROOT / "m2/factors.py", ROOT / "m2/pipeline.py", ROOT / "m1/backtest.py", ROOT / "backtest/m1_bridge.py"]
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="post-hoc exploratory ablation",
                end=DEV_END, factor="combo", min_contracts=10, fixed_weights="1/13 with absent contracts in cash",
                execution="close-t target executes at close t+1 and earns first at t+2; start flat",
                approaches={"A1": "signal", "A2": "signal * volatility target scale",
                            "R": "signal * frozen filter", "A3": "signal * volatility target scale * frozen filter"},
                costs=list(COST_MODELS), refit=False, tune=False, holdout_used=False, ex_post_risk_scaling=False,
                bootstrap=dict(block=20, reps=5000, seed=0, sided="two", ci="unadjusted 95% percentile"),
                multiplicity="Holm over 18 new R-A2 and R-A3 comparisons per cost model, including the volatility benchmark",
                identity_filters=["M4-1", "M4-2", "M4-3", "M4-5", "M5-3", "M5-4"],
                identity_implication="Gate decisions unchanged: f=1 implies R=A1 and A3=A2; no active overlay to ablate",
                excluded_full_strategies=["M3-5", "M3-8"], jobs=jj,
                sha256={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p) for p in files})
    (out_dir / "analysis_spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    summary, pairs, overlaps, validation = [], [], [], []
    for job in jj:
        print(job["trial"], job["name"], flush=True)
        data, asset, targets, book, filters = inputs(job, data_dir)
        start = window_start(book)
        meta = {k: job[k] for k in ("trial", "name", "kind")}
        overlaps.append({**meta, "start": str(start.date()), "end": DEV_END,
                         **overlap_diagnostics(targets, book, filters, start)})
        original = pd.read_csv(ROOT / job["reference"], index_col="date", parse_dates=True)
        daily = {}
        for cost in COST_MODELS:
            ports = {ap: m1_book(t, data, asset, book, start, DEV_END, cost) for ap, t in targets.items()}
            for ap, old in (("A2", "control"), ("A3", "treatment")):
                a, b = ports[ap].net, original[f"{cost}_{old}"]
                pd.testing.assert_index_equal(a.index, b.index, check_names=False)
                err = float((a - b).abs().max())
                np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-12)
                validation.append(dict(trial=job["trial"], cost_model=cost, approach=ap,
                                       reference=job["reference"], max_abs_daily_error=err))
            for ap, port in ports.items():
                metrics = performance(port.net)
                summary.append({**meta, "cost_model": cost, "approach": ap, **metrics,
                                "gross_sharpe": performance(port.gross)["sharpe"],
                                "turnover_pa": port.turnover.mean() * 252,
                                "cost_pa": port.cost.mean() * 252, "gross_exposure": port.gross_exposure.mean()})
                for col in ("net", "gross", "cost", "turnover", "gross_exposure"):
                    daily[f"{cost}_{ap}_{col}"] = port[col]
            for treat, control in NEW_PAIRS + (("A3", "A2"),):
                stats = block_bootstrap_sharpe_diff(ports[treat].net, ports[control].net)
                pairs.append({**meta, "cost_model": cost, "comparison": f"{treat}-{control}",
                              "new_comparison": (treat, control) in NEW_PAIRS,
                              "corr": ports[treat].net.corr(ports[control].net), **stats})
        pd.DataFrame(daily).rename_axis("date").to_csv(out_dir / f"{job['trial']}_daily.csv")
    sm, pr, ov, va = map(pd.DataFrame, (summary, pairs, overlaps, validation))
    pr["p_holm_new"] = np.nan
    for cost in COST_MODELS:
        mask = pr.cost_model.eq(cost) & pr.new_comparison
        assert mask.sum() == 18
        pr.loc[mask, "p_holm_new"] = holm(pr.loc[mask, "p_boot"].to_numpy())
    sm.to_csv(out_dir / "summary.csv", index=False)
    pr.to_csv(out_dir / "comparisons.csv", index=False)
    ov.to_csv(out_dir / "overlap.csv", index=False)
    va.to_csv(out_dir / "reproduction_checks.csv", index=False)
    compact = sm[sm.cost_model.eq("official")].pivot(index="trial", columns="approach", values="sharpe")
    print(compact.round(4).to_string(), flush=True)
    print(pr[pr.cost_model.eq("official") & pr.new_comparison][["trial", "comparison", "d_sharpe", "ci_low", "ci_high", "p_boot", "p_holm_new"]].round(4).to_string(index=False), flush=True)
    print("All 36 A2/A3 daily-return reproductions passed. Max error:", va.max_abs_daily_error.max(), flush=True)
    return sm, pr, ov


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "backtest_outputs/regime_ablation")
    args = parser.parse_args()
    run(args.out_dir)
