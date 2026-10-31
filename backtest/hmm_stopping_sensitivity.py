"""Full causal replay and A2/R/A3 inference for HMM stopping sensitivity.

Commands (project Python):
  python -m backtest.hmm_stopping_sensitivity prepare
  python -m backtest.hmm_stopping_sensitivity train --workers 3
  python -m backtest.hmm_stopping_sensitivity evaluate
  python -m backtest.hmm_stopping_sensitivity summarise

New outputs only. Report, slides, production models, and original results are
hash-protected. Historical proxy extensions are outside this main 80/20 study.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.resplit_common import ROOT, OUT as ORIGINAL, START, DEV_END, TEST_START, END, NAMES
from backtest.resplit_evaluate import load_base, metrics, WINDOWS, PAIRS
from backtest.m1_bridge import m1_book, COST_MODELS
from backtest.metrics import block_bootstrap_sharpe_diff
from m3.adaptor import AdaptorConfig
from m3.features import DEFAULT_FEATURES, OFFMARKET_FEATURES, load_exog
from m3.hmm_stopping import POLICIES, StoppingHMM
from m3.predictability import holm

OUT = ROOT / "backtest_outputs/hmm_stopping_20261031"
TRIALS = ("M3-1", "M3-2", "M5-1", "M5-2")
SOURCE = ["m3/hmm.py", "m3/adaptor.py", "m3/features.py", "m5/hmm_pool.py", "m5/pool.py",
          "m1/backtest.py", "m2/pipeline.py", "backtest/metrics.py", "backtest/m1_bridge.py",
          "backtest/resplit_evaluate.py", "m3/hmm_stopping.py", "backtest/hmm_stopping_sensitivity.py"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_manifest():
    return read_json(ORIGINAL / "evaluated_filter_manifest.json")


def jobs():
    from m5.pool import groups
    universe = pd.read_csv(ROOT / "data/final/universe.csv")
    for policy in POLICIES:
        for tid in TRIALS:
            if tid.startswith("M3"):
                units = {c: [c] for c in universe.code}
            else:
                units = groups(universe, "class" if tid == "M5-1" else "global")
            for group, codes in units.items():
                yield dict(policy=policy, trial=tid, group=group, codes=codes)


def prepare():
    spec_path = OUT / "analysis_spec.json"
    if spec_path.exists():
        verify()
        return
    protected = set(SOURCE + ["pyproject.toml", "uv.lock"])
    protected.update(row["file"] for row in source_manifest().values())
    for folder in ["data/final", "m3_outputs", "m5_outputs", "backtest_outputs/resplit_80_20"]:
        protected.update(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / folder).glob("*.csv"))
    protected.add("data/processed/offmarket/offmarket_daily.csv")
    for p in (ROOT / "report").rglob("*"):
        if p.is_file() and p.suffix in (".pdf", ".pptx", ".typ", ".docx", ".txt"):
            protected.add(str(p.relative_to(ROOT)).replace("\\", "/"))
    protected = [p for p in sorted(protected) if (ROOT / p).is_file()]
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        objective="Assess whether HMM stopping changes the main common-sample incremental-value conclusion; no report/slides edits.",
        trials=list(TRIALS), policies={
            "original": "Signed per-row likelihood gain < 1e-7, including a negative gain; exact original replay.",
            "abs3": "Three consecutive absolute per-row likelihood changes < 1e-7; otherwise continue.",
            "extra50": "After the first original stopping trigger, perform 50 additional EM updates; return terminal parameters."},
        fixed=dict(states=3, n_init=4, warm_start=True, refit="quarter", min_train=750,
                   cov_reg=.001, tol=1e-7, max_iter=500, seed=0, mapping="top ER60 state",
                   sample_start=START, development_end=DEV_END, test_start=TEST_START, end=END,
                   development_days=3984, test_days=997, costs=list(COST_MODELS), approaches=["A2", "R", "A3"]),
        replay="From each contract/pool's first eligible fit through the final quarter; each policy has its own recursively propagated warm starts.",
        selection="Every initialisation follows the same policy; select highest terminal training likelihood, retaining the existing restart rule.",
        inference=dict(block=20, reps=5000, seed=0, sided="two", interval="unadjusted percentile 95%",
                       holm_per_cost=45, holm_joint_costs=90, focused_regime_increment_per_cost=14,
                       across_stopping_policies=138,
                       across_policies_definition="90 original comparisons plus 24 changed-HMM comparisons per alternative, separately by development/test; unchanged hypotheses are not duplicated."),
        no_parameter_search=True, production_python=platform.python_version(),
        versions={p: importlib.metadata.version(p) for p in ["numpy", "pandas", "numba", "scipy"]},
        protected_sha256={p: sha(ROOT / p) for p in protected}, jobs=list(jobs()))
    save_json(spec_path, spec)
    print(f"Prepared {len(spec['jobs'])} full-history training jobs", flush=True)


def verify():
    spec = read_json(OUT / "analysis_spec.json")
    changed = [p for p, h in spec["protected_sha256"].items() if sha(ROOT / p) != h]
    if changed:
        raise RuntimeError(f"Protected files changed: {changed}")
    return spec


def unit_prefix(job):
    return OUT / job["policy"] / "units" / f"{job['trial']}_{job['group']}"


def train_unit(job):
    from m3.adaptor import run_contract
    from m5.pool import run_pool
    prefix = unit_prefix(job)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    done_path = prefix.with_name(prefix.name + "_done.json")
    specification = read_json(OUT / "analysis_spec.json")
    spec_hash = specification.get("training_spec_sha256", sha(OUT / "analysis_spec.json"))
    if done_path.exists():
        done = read_json(done_path)
        if done["spec_sha256"] != spec_hash:
            raise RuntimeError("Cached unit belongs to another specification")
        for p, h in done["output_sha256"].items():
            if sha(OUT / p) != h:
                raise RuntimeError(f"Cached training output changed: {p}")
        return done
    start_time = time.perf_counter()
    candidates, fit_events = [], []
    progress_path = prefix.with_name(prefix.name + "_progress.json")
    def callback(model):
        vintage = str(pd.Period(ordinal=model.seed, freq="Q").start_time.date())
        candidates.extend(dict(**r, vintage=vintage, group=job["group"]) for r in model.candidates)
        fit_events.append(vintage)
        save_json(progress_path, dict(**job, latest_vintage=vintage, refits=len(fit_events),
                                      elapsed_seconds=time.perf_counter() - start_time))
    factory = lambda *args, **kwargs: StoppingHMM(*args, policy=job["policy"], callback=callback, **kwargs)
    features = DEFAULT_FEATURES + (OFFMARKET_FEATURES if job["trial"] == "M3-2" else ())
    cfg = AdaptorConfig(n_states=3, features=features, refit="quarter", mapping="top", n_init=4, seed=0)
    rets = {c: pd.read_csv(ROOT / f"data/final/{c}.csv", index_col="date", parse_dates=True)["ret"] for c in job["codes"]}
    with threadpool_limits(limits=4):
        if job["trial"].startswith("M3"):
            code = job["codes"][0]
            exog = load_exog(ROOT / "data/processed/offmarket/offmarket_daily.csv") if job["trial"] == "M3-2" else None
            with patch("m3.adaptor.GaussianHMM", factory):
                output, fits = run_contract(rets[code], cfg, exog)
            outs = {code: output}
            fits = [dict(contract=code, **r) for r in fits]
        else:
            with patch("m5.pool.MultiSeqGaussianHMM", factory):
                outs, fits = run_pool(rets, cfg, name=job["group"])
    rows = []
    for code, df in outs.items():
        df = df.rename(columns=lambda c: c if c == "vintage" else f"{c}_close_t")
        rows.append(df.assign(contract=code).rename_axis("date").reset_index())
    filters = pd.concat(rows, ignore_index=True)
    filters = filters[["date", "contract"] + [c for c in filters if c not in ("date", "contract")]]
    fit_frame = pd.DataFrame(fits)
    assert (pd.to_datetime(fit_frame.train_end) < pd.to_datetime(fit_frame.vintage)).all()
    old = pd.read_csv(ROOT / source_manifest()[job["trial"]]["file"], parse_dates=["date"]).set_index(["contract", "date"]).filter_close_t
    new = filters.set_index(["contract", "date"]).filter_close_t
    ref = old.reindex(new.index)
    if not np.array_equal(ref.isna(), new.isna()):
        raise AssertionError(f"Coverage changed: {job}")
    diff = (new - ref).abs().dropna()
    if job["policy"] == "original":
        np.testing.assert_allclose(new, ref, rtol=0, atol=1e-8, equal_nan=True,
                                   err_msg=f"Original full-history replay differs for {job}")
    check_range = new[new.index.get_level_values("date").to_series(index=new.index).between(START, END)]
    assert check_range.notna().all() and check_range.between(0, 1 + 1e-12).all()
    outputs = {}
    for suffix, frame in [("filter", filters), ("fits", fit_frame), ("candidates", pd.DataFrame(candidates))]:
        path = prefix.with_name(prefix.name + f"_{suffix}.csv")
        frame.to_csv(path, index=False)
        outputs[str(path.relative_to(OUT)).replace("\\", "/")] = sha(path)
    done = dict(**job, spec_sha256=spec_hash, refits=len(fit_events), candidates=len(candidates),
                max_filter_change=float(diff.max()), mean_filter_change=float(diff.mean()),
                elapsed_seconds=time.perf_counter() - start_time, output_sha256=outputs)
    save_json(done_path, done)
    return done


def train(workers):
    spec = verify()
    tasks = spec["jobs"]
    records = []
    # Each worker writes only its own unit; only the parent assembles outputs.
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(train_unit, job): job for job in tasks}
        for future in as_completed(pending):
            done = future.result()
            records.append(done)
            save_json(OUT / "training_progress.json", dict(completed=len(records), total=len(tasks), latest=done))
            print(f"{len(records)}/{len(tasks)} {done['policy']} {done['trial']} {done['group']}: "
                  f"{done['refits']} refits, max filter change {done['max_filter_change']:.6g}, {done['elapsed_seconds']:.1f}s", flush=True)
    for policy in POLICIES:
        dest = OUT / policy / "filters"
        dest.mkdir(parents=True, exist_ok=True)
        for tid in TRIALS:
            members = [j for j in tasks if j["policy"] == policy and j["trial"] == tid]
            for suffix in ["filter", "fits", "candidates"]:
                parts = [pd.read_csv(unit_prefix(j).with_name(unit_prefix(j).name + f"_{suffix}.csv")) for j in members]
                pd.concat(parts, ignore_index=True).to_csv(dest / f"{tid}_{suffix}.csv", index=False)
    save_json(OUT / "training_complete.json", dict(jobs=records, all_protected_files_unchanged=bool(verify())))


def adjusted(pr):
    pr = pr.copy()
    for window in WINDOWS:
        w = pr.window.eq(window)
        assert w.sum() == 90
        pr.loc[w, "p_holm_joint"] = holm(pr.loc[w, "p_boot"].to_numpy())
        for cost in COST_MODELS:
            m = w & pr.cost_model.eq(cost)
            assert m.sum() == 45
            pr.loc[m, "p_holm_cost"] = holm(pr.loc[m, "p_boot"].to_numpy())
            inc = m & pr.comparison.eq("A3-A2") & pr.trial.ne("M3-4")
            assert inc.sum() == 14
            pr.loc[inc, "p_holm_increment_14"] = holm(pr.loc[inc, "p_boot"].to_numpy())
    return pr


def evaluate():
    verify()
    if not (OUT / "training_complete.json").exists():
        raise RuntimeError("Finish full-history training first")
    data, asset, base, book = load_base()
    baseline = {(w, cost): m1_book(base["A2"], data, asset, book, start, end, cost)
                for w, (start, end) in WINDOWS.items() for cost in COST_MODELS}
    old_sm, old_pr = pd.read_csv(ORIGINAL / "summary.csv"), pd.read_csv(ORIGINAL / "comparisons.csv")
    audits = []
    for policy in POLICIES:
        dest = OUT / policy
        rows, pairs = [], []
        for tid in TRIALS:
            print(f"EVALUATE {policy} {tid}", flush=True)
            filt = pd.read_csv(dest / "filters" / f"{tid}_filter.csv", parse_dates=["date"]).set_index(["contract", "date"]).filter_close_t
            targets = {ap: {c: base[src][c] * filt.loc[c].reindex(d.index) for c, d in data.items()}
                       for ap, src in [("R", "A1"), ("A3", "A2")]}
            daily = {}
            old_daily = pd.read_csv(ORIGINAL / f"{tid}_daily.csv", index_col="date", parse_dates=True)
            for window, (start, end) in WINDOWS.items():
                for cost in COST_MODELS:
                    ports = {"A2": baseline[window, cost]}
                    ports.update({ap: m1_book(tg, data, asset, book, start, end, cost) for ap, tg in targets.items()})
                    for ap, port in ports.items():
                        pd.testing.assert_index_equal(port.index, ports["A2"].index)
                        assert len(port) == (3984 if window == "development" else 997)
                        assert np.isfinite(port.net).all()
                        rows.append(dict(trial=tid, name=NAMES[tid], kind="regime_filter", identity=False,
                                         window=window, cost_model=cost, approach=ap, **metrics(port)))
                        for col in ["net", "gross", "cost", "turnover", "gross_exposure"]:
                            key = f"{window}_{cost}_{ap}_{col}"
                            daily[key] = port[col]
                            if policy == "original" or ap == "A2":
                                previous = old_daily.loc[port.index, key]
                                error = float((port[col] - previous).abs().max())
                                np.testing.assert_allclose(port[col], previous, rtol=0, atol=1e-10)
                                audits.append(dict(policy=policy, trial=tid, check=key, max_abs_error=error))
                    for treat, ctrl in PAIRS:
                        stat = block_bootstrap_sharpe_diff(ports[treat].net, ports[ctrl].net,
                                                         block=20, reps=5000, seed=0)
                        assert np.isclose(stat["d_sharpe"], metrics(ports[treat])["sharpe"] - metrics(ports[ctrl])["sharpe"])
                        pairs.append(dict(trial=tid, name=NAMES[tid], window=window, cost_model=cost,
                                          comparison=f"{treat}-{ctrl}", identity=False, **stat))
            pd.DataFrame(daily).rename_axis("date").to_csv(dest / f"{tid}_daily.csv")
        # Reuse unchanged raw tests for the other 11 overlays, but recompute
        # every Holm family with the four new HMM results included.
        sm = pd.concat([old_sm[~old_sm.trial.isin(TRIALS)], pd.DataFrame(rows)], ignore_index=True)
        pr = adjusted(pd.concat([old_pr[~old_pr.trial.isin(TRIALS)], pd.DataFrame(pairs)], ignore_index=True))
        if policy == "original":
            keys = ["trial", "window", "cost_model", "comparison"]
            columns = ["d_sharpe", "ci_low", "ci_high", "p_boot", "p_holm_joint", "p_holm_cost", "p_holm_increment_14"]
            a, b = pr.set_index(keys).sort_index()[columns], old_pr.set_index(keys).sort_index()[columns]
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-8, equal_nan=True)
        sm.to_csv(dest / "summary.csv", index=False)
        pr.to_csv(dest / "comparisons.csv", index=False)
    pd.DataFrame(audits).to_csv(OUT / "reproduction_checks.csv", index=False)
    save_json(OUT / "evaluation_complete.json", dict(unchanged=bool(verify()), policies=list(POLICIES), bootstrap_reps=5000))


def summarise():
    spec = verify()
    if not (OUT / "evaluation_complete.json").exists():
        raise RuntimeError("Evaluation incomplete")
    pr = pd.concat([pd.read_csv(OUT / p / "comparisons.csv").assign(policy=p) for p in POLICIES], ignore_index=True)
    # Account for inspecting more than one stopping policy without duplicating
    # the 11 unchanged overlays in the expanded hypothesis family.
    sensitivity = pr.policy.eq("original") | pr.trial.isin(TRIALS)
    for window in WINDOWS:
        mask = sensitivity & pr.window.eq(window)
        assert mask.sum() == 138
        pr.loc[mask, "p_holm_sensitivity_138"] = holm(pr.loc[mask, "p_boot"].to_numpy())
    pr.to_csv(OUT / "all_comparisons.csv", index=False)
    headline = pr[pr.window.eq("test") & pr.comparison.eq("A3-A2") & pr.trial.isin(TRIALS)].copy()
    headline["positive_raw_significant"] = (headline.d_sharpe > 0) & (headline.p_boot < .05)
    headline["positive_holm14_significant"] = (headline.d_sharpe > 0) & (headline.p_holm_increment_14 < .05)
    headline["positive_holm90_significant"] = (headline.d_sharpe > 0) & (headline.p_holm_joint < .05)
    headline.to_csv(OUT / "headline_test.csv", index=False)
    diagnostics = []
    for policy in POLICIES:
        for tid in TRIALS:
            d = pd.read_csv(OUT / policy / "filters" / f"{tid}_candidates.csv")
            for scope, subset in [("all_initialisations", d), ("selected_fits", d[d.selected])]:
                diagnostics.append(dict(policy=policy, trial=tid, scope=scope, fits=len(subset),
                    mean_iterations=float(subset.n_iter.mean()), maximum_iterations=int(subset.n_iter.max()),
                    stopped_after_negative_gain=int((subset.final_gain_per_row < 0).sum()),
                    hit_cap=int(subset.reason.eq("iteration_cap").sum()),
                    stable_by_abs3=int((subset.stable_steps >= 3).sum())))
    pd.DataFrame(diagnostics).to_csv(OUT / "stopping_diagnostics.csv", index=False)
    result = dict(scope="Full-history causal re-estimation of four HMM filters in the original 13-contract 80/20 main experiment.",
                  policies={}, original_reproduced=True, sources_and_deliverables_unchanged=bool(verify()),
                  protected_files=len(spec["protected_sha256"]),
                  limitations=["Does not rerun the separate long-history proxy experiments.",
                               "A stopping sensitivity check is not an independent untouched test sample.",
                               "extra50 is a deliberate computational stress check; its terminal fit need not satisfy abs3 stability."])
    for policy in POLICIES:
        m = pr.policy.eq(policy) & pr.window.eq("test")
        inc = m & pr.comparison.eq("A3-A2") & pr.trial.ne("M3-4")
        other = m & pr.comparison.ne("A3-A2")
        result["policies"][policy] = dict(
            positive_raw_increment=pr.loc[inc & (pr.d_sharpe > 0) & (pr.p_boot < .05), ["trial", "cost_model", "d_sharpe", "p_boot"]].to_dict("records"),
            positive_holm14_increment=pr.loc[inc & (pr.d_sharpe > 0) & (pr.p_holm_increment_14 < .05), ["trial", "cost_model", "p_holm_increment_14"]].to_dict("records"),
            positive_holm90_increment=pr.loc[inc & (pr.d_sharpe > 0) & (pr.p_holm_joint < .05), ["trial", "cost_model", "p_holm_joint"]].to_dict("records"),
            other_holm90_significant=pr.loc[other & (pr.p_holm_joint < .05), ["trial", "cost_model", "comparison", "d_sharpe", "p_holm_joint"]].to_dict("records"))
    save_json(OUT / "conclusion.json", result)
    columns = ["policy", "trial", "cost_model", "d_sharpe", "ci_low", "ci_high", "p_boot", "p_holm_increment_14", "p_holm_joint", "p_holm_sensitivity_138"]
    lines = ["HMM stopping-rule sensitivity: full main-experiment replay", "",
             f"Development {START} to {DEV_END}; test {TEST_START} to {END}.",
             "Four restarts plus each policy's prior-quarter warm start; all original features, state mapping, 0.001 covariance regularisation and 500-iteration cap retained.",
             "abs3: three consecutive absolute per-row likelihood changes below 1e-7.",
             "extra50: fifty additional updates after the first original stopping trigger, subject to the same cap.",
             "Paired circular bootstrap: 20-day blocks, 5,000 replications, seed 0; individual percentile 95% intervals.",
             "Holm families: 14 incremental regime tests per cost; 45 overlay comparisons per cost; 90 across both costs; additionally 138 across all stopping policies without duplicating unchanged methods.",
             "", "Test-period A3 minus A2, annualised net Sharpe:", headline[columns].round(6).to_string(index=False),
             "", "Conclusion counts:", json.dumps(result["policies"], indent=2),
             "", "Selected-fit stopping diagnostics:", pd.DataFrame(diagnostics).query("scope == 'selected_fits'").to_string(index=False),
             "", "Original filter, daily return and inference reproduction passed. Original inputs, results, report and slides unchanged.",
             "", "Scope limitations:"] + result["limitations"]
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "train", "evaluate", "summarise", "unit"])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--policy", choices=POLICIES, default="original")
    parser.add_argument("--trial", choices=TRIALS, default="M3-1")
    parser.add_argument("--group", default="ES")
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        if args.stage == "train":
            train(args.workers)
        elif args.stage == "unit":
            verify()
            job = next(j for j in jobs() if j["policy"] == args.policy and j["trial"] == args.trial and j["group"] == args.group)
            print(json.dumps(train_unit(job), indent=2), flush=True)
        else:
            {"prepare": prepare, "evaluate": evaluate, "summarise": summarise}[args.stage]()
