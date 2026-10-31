"""Independent numerical checks against hmmlearn 0.3.3.

Run export with the project's Python; run check in the isolated reference env:
    .venv/Scripts/python -m m3.validate_hmm export
    .venv-hmm/Scripts/python -m m3.validate_hmm check

Export freezes inputs, initialisations, and actual production-code outputs. Check
uses hmmlearn's independent Gaussian density, C++ forward/backward, sufficient
statistics, and M-step. Its small adapter ONLY aligns missing-row semantics,
additive covariance regularisation, dead-state resets, and stopping/restarts.
Unmodified-library, zero-regularisation, one-step tests are also run separately.
These are numerical checks, not a new strategy performance comparison.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from m3.hmm import GaussianHMM, _forward, _forward_backward
from m3.features import DEFAULT_FEATURES, OFFMARKET_FEATURES, build_features, load_exog
from m5.hmm_pool import MultiSeqGaussianHMM

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "m3_outputs" / "hmm_validation"
VINTAGE = pd.Timestamp("2023-01-01")
TEST_END = pd.Timestamp("2023-04-01")
SOURCE_FILES = ["m3/hmm.py", "m3/adaptor.py", "m3/features.py", "m5/hmm_pool.py", "m5/pool.py"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def versions():
    result = {"python": platform.python_version()}
    for name in ["numpy", "scipy", "scikit-learn", "numba", "pandas", "hmmlearn"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def simulate(T, K, D, seed):
    rng = np.random.default_rng(seed)
    A = np.full((K, K), 0.04 / max(K - 1, 1))
    np.fill_diagonal(A, 0.96 if K > 1 else 1.)
    means = np.linspace(-2, 2, K)[:, None] * np.linspace(1, .4, D)[None]
    R = rng.normal(size=(K, D, D))
    covs = R @ R.transpose(0, 2, 1) / (2 * D) + .3 * np.eye(D)
    state = 0
    X = []
    for _ in range(T):
        state = rng.choice(K, p=A[state])
        X.append(rng.multivariate_normal(means[state], covs[state]))
    return np.asarray(X)


def cases():
    for K, D in [(1, 1), (2, 1), (2, 3), (3, 7)]:
        X = simulate(1500, K, D, 100 + D)
        yield f"synthetic_k{K}_d{D}", [X[:1200]], [X], K, {"kind": "synthetic"}
    seqs = [simulate(800, 3, 3, s) for s in (31, 32, 33)]
    yield "synthetic_pool", [x[:650] for x in seqs], seqs, 3, {"kind": "synthetic_pool"}
    gaps = [x.copy() for x in seqs]
    for x in gaps:
        x[0] = np.nan
        x[100:105] = np.nan
        x[350, 1] = np.nan
        x[649] = np.nan
    yield "synthetic_missing_pool", [x[:650] for x in gaps], gaps, 3, {"kind": "synthetic_missing"}

    universe = pd.read_csv(ROOT / "data/final/universe.csv")
    exog = load_exog(ROOT / "data/processed/offmarket/offmarket_daily.csv")
    base = {}
    for names, tag in [(DEFAULT_FEATURES, "market"), (DEFAULT_FEATURES + OFFMARKET_FEATURES, "macro")]:
        for code in universe.code:
            ret = pd.read_csv(ROOT / f"data/final/{code}.csv", index_col="date", parse_dates=True)["ret"]
            F = build_features(ret, names, exog)
            F = F.loc[F.notna().all(axis=1).idxmax():]
            tv = F[F.index < VINTAGE].dropna()
            mu, sd = tv.mean().to_numpy(), tv.std(ddof=1).to_numpy()
            Z = ((F[F.index < TEST_END] - mu) / sd).to_numpy()
            n_train = int((F.index < VINTAGE).sum())
            meta = {"kind": "real_single", "contract": code, "features": list(names),
                    "train_start": str(F.index[0].date()), "train_end": str(F.index[n_train - 1].date()),
                    "evaluation_end": str(F.index[len(Z) - 1].date()), "vintage": str(VINTAGE.date())}
            yield f"{tag}_{code}", [Z[:n_train]], [Z], 3, meta
            if tag == "market":
                base[code] = (Z[:n_train], Z)
    groups = {name: list(g.code) for name, g in universe.groupby("asset_class")}
    groups["global"] = list(universe.code)
    for name, codes in groups.items():
        yield f"pool_{name}", [base[c][0] for c in codes], [base[c][1] for c in codes], 3, {
            "kind": "real_pool", "contracts": codes, "vintage": str(VINTAGE.date()),
            "features": list(DEFAULT_FEATURES)}


def set_custom(params, K):
    model = GaussianHMM(K)
    model.pi, model.A, model.means, model.covs = [x.copy() for x in params]
    return model


def export():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "cases").mkdir(exist_ok=True)
    protected = list(SOURCE_FILES)
    for folder in ["m3_outputs", "m5_outputs", "backtest_outputs/resplit_80_20"]:
        protected += [str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / folder).glob("*.csv")]
    inputs = ["data/final/universe.csv", "data/processed/offmarket/offmarket_daily.csv"]
    inputs += [f"data/final/{c}.csv" for c in pd.read_csv(ROOT / "data/final/universe.csv").code]
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "production_versions": versions(),
                "design": "2023Q1 snapshots; full available pre-quarter histories; four shared random starts; no prior-quarter warm start in real snapshots",
                "protected_hashes": {p: digest(ROOT / p) for p in protected},
                "input_hashes": {p: digest(ROOT / p) for p in inputs}, "cases": []}
    for i, (name, Xs, evals, K, meta) in enumerate(cases()):
        start = time.perf_counter()
        seed = 202310 + i
        factory = GaussianHMM if len(Xs) == 1 else MultiSeqGaussianHMM
        model = factory(K, n_init=4, seed=seed)
        Xo = np.concatenate([x[~np.isnan(x).any(axis=1)] for x in Xs])
        rng = np.random.default_rng(seed)
        inits = [(model._random_init(Xs[0], rng) if len(Xs) == 1 else model._random_init_obs(Xo, rng)) for _ in range(4)]
        if len(Xs) == 1:
            model.fit(Xs[0])
        else:
            model.fit_multi(Xs)
        arrays = {f"train_{j}": x for j, x in enumerate(Xs)}
        arrays.update({f"eval_{j}": x for j, x in enumerate(evals)})
        for j, x in enumerate(evals):
            arrays[f"filter_{j}"] = model.filter(x)
        for j, ps in enumerate(inits):
            arrays.update({f"init_{j}_{k}": p for k, p in enumerate(ps)})
        arrays.update({f"fit_{k}": p for k, p in enumerate(model.params())})
        arrays["history"] = np.asarray(model.history)
        # Independent stock-library M-step oracle will use exactly these starts.
        for reg in (0., .001):
            one = factory(K, n_init=0, max_iter=1, cov_reg=reg, seed=seed)
            if len(Xs) == 1:
                one.fit(Xs[0], warm_start=inits[0])
            else:
                one.fit_multi(Xs, warm_start=inits[0])
            arrays.update({f"step_{reg}_{k}": p for k, p in enumerate(one.params())})
        filename = f"cases/{name}.npz"
        np.savez_compressed(OUT / filename, **arrays)
        record = dict(name=name, file=filename, sha256=digest(OUT / filename), K=K, D=Xs[0].shape[1],
                      n_sequences=len(Xs), train_rows=sum(map(len, Xs)),
                      missing_rows=sum(int(np.isnan(x).any(axis=1).sum()) for x in Xs),
                      evaluation_rows=sum(len(x) - len(t) for x, t in zip(evals, Xs)), seed=seed,
                      n_init=4, max_iter=model.max_iter, tol=model.tol, cov_reg=model.cov_reg,
                      loglik=float(model.loglik), n_iter=model.n_iter, converged=bool(model.converged),
                      runtime_seconds=time.perf_counter() - start, **meta)
        manifest["cases"].append(record)
        dump(OUT / "manifest.json", manifest)
        print(f"EXPORT {name}: T={record['train_rows']} missing={record['missing_rows']} iterations={model.n_iter} {record['runtime_seconds']:.2f}s", flush=True)
    print(f"Exported {len(manifest['cases'])} cases to {OUT}", flush=True)


def check():
    import hmmlearn
    from hmmlearn.hmm import GaussianHMM as LibraryHMM
    from hmmlearn import _hmmc
    from scipy.special import softmax
    if hmmlearn.__version__ != "0.3.3":
        raise RuntimeError("Reference adapter is audited for hmmlearn 0.3.3 only")

    class AlignedHMM(LibraryHMM):
        """Library numerics with the project's explicit policy choices."""
        def _compute_log_likelihood(self, X):
            good = ~np.isnan(X).any(axis=1)
            result = np.zeros((len(X), self.n_components))
            if good.any():
                result[good] = super()._compute_log_likelihood(X[good])
            return result

        def _accumulate_sufficient_statistics(self, stats, X, lattice, posteriors, fwdlattice, bwdlattice):
            bad = np.isnan(X).any(axis=1)
            clean = X.copy()
            clean[bad] = 0.
            super()._accumulate_sufficient_statistics(stats, clean, lattice, posteriors, fwdlattice, bwdlattice)
            # Missing emissions still contribute initial/transition statistics,
            # but not Gaussian observation counts, sums, or outer products.
            stats["post"] -= posteriors[bad].sum(axis=0)

        def _do_mstep(self, stats):
            super()._do_mstep(stats)
            self._covars_ += self.aligned_reg * np.eye(self.n_features)
            for k in np.flatnonzero(stats["post"] < self.n_features + 1):
                self.resets += 1
                rng = np.random.default_rng(self.aligned_seed + self.aligned_iteration + int(k))
                self.means_[k] = self.observed[rng.integers(len(self.observed))]
                self._covars_[k] = np.cov(self.observed.T).reshape(self.n_features, self.n_features) + self.aligned_reg * np.eye(self.n_features)

    def library(params, adapter=False, reg=0., seed=0, Xs=None):
        pi, A, means, covs = params
        cls = AlignedHMM if adapter else LibraryHMM
        model = cls(n_components=len(pi), covariance_type="full", init_params="", n_iter=1,
                    startprob_prior=1., transmat_prior=1., means_weight=0., covars_prior=0.,
                    covars_weight=0., min_covar=0., implementation="log", tol=-np.inf)
        model.n_features = means.shape[1]
        model.startprob_, model.transmat_, model.means_, model.covars_ = [p.copy() for p in params]
        if adapter:
            model.aligned_reg, model.aligned_seed, model.aligned_iteration, model.resets = reg, seed, 1, 0
            model.observed = np.concatenate([x[~np.isnan(x).any(axis=1)] for x in Xs]) if Xs is not None else None
        return model

    def params(model):
        return model.startprob_, model.transmat_, model.means_, model.covars_

    def max_error(a, b):
        return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))

    records = []
    def record(case, metric, error, tolerance, detail=""):
        good = bool(np.isfinite(error) and error <= tolerance)
        records.append(dict(case=case, metric=metric, error=float(error) if np.isfinite(error) else None,
                            tolerance=tolerance, passed=good, detail=detail))

    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    for p, h in manifest["protected_hashes"].items():
        if digest(ROOT / p) != h:
            raise RuntimeError(f"Source or research output changed during validation: {p}")
    fit_details, configuration_details, diagnostics = [], [], []
    for rec in manifest["cases"]:
        start = time.perf_counter()
        name = rec["name"]
        path = OUT / rec["file"]
        if digest(path) != rec["sha256"]:
            raise RuntimeError(f"Frozen case modified: {path}")
        with np.load(path) as data:
            Xs = [data[f"train_{j}"] for j in range(rec["n_sequences"])]
            evals = [data[f"eval_{j}"] for j in range(rec["n_sequences"])]
            frozen = [data[f"fit_{k}"] for k in range(4)]
            inits = [[data[f"init_{j}_{k}"] for k in range(4)] for j in range(rec["n_init"])]
            old = set_custom(frozen, rec["K"])
            ref = library(frozen, adapter=True)
            # Frozen-model inference: independently implemented library density
            # and C++ log-space forward/backward versus project scaled recursion.
            for j, X in enumerate(evals):
                logB = ref._compute_log_likelihood(X)
                record(name, "emission_max_abs", max_error(old._log_emission(X), logB), 1e-9)
                ll, fw = _hmmc.forward_log(ref.startprob_, ref.transmat_, logB)
                bw = _hmmc.backward_log(ref.startprob_, ref.transmat_, logB)
                alpha, gamma = softmax(fw, axis=1), softmax(fw + bw, axis=1)
                B, offset = old._scaled_emission(X)
                custom_gamma, custom_xi, scales = _forward_backward(B, old.A, old.pi)
                xi = np.exp(_hmmc.compute_log_xi_sum(fw, ref.transmat_, bw, logB))
                record(name, "loglik_abs_per_row", abs(np.log(scales).sum() + offset - ll) / len(X), 1e-9)
                record(name, "forward_probability_max_abs", max_error(old.filter(X), alpha), 1e-8)
                record(name, "production_probability_max_abs", max_error(data[f"filter_{j}"], alpha), 1e-8)
                record(name, "smoothed_probability_max_abs", max_error(custom_gamma, gamma), 1e-8)
                record(name, "transition_count_abs_per_row", max_error(custom_xi, xi) / len(X), 1e-9)
                # Public-API prefix endpoint is a causal-filter oracle. Do not
                # compare the project's filter to all rows of full-sample smoothing.
                if not np.isnan(X).any():
                    stock = library(frozen)
                    for stop in np.unique(np.linspace(1, len(X), 7, dtype=int)):
                        endpoint = stock.predict_proba(X[:stop])[-1]
                        record(name, "public_prefix_endpoint_max_abs", max_error(data[f"filter_{j}"][stop - 1], endpoint), 1e-8)

            X = np.concatenate(Xs)
            lengths = [len(x) for x in Xs]
            # No adapter in this test when data are complete: stock public fit,
            # neutral priors, one EM update, exactly the same initial parameters.
            if not np.isnan(X).any():
                stock = library(inits[0]).fit(X, lengths)
                expected = [data[f"step_0.0_{k}"] for k in range(4)]
                for label, a, b in zip(["pi", "A", "means", "covs"], expected, params(stock)):
                    record(name, f"stock_ML_one_step_{label}", max_error(a, b), 1e-8)
                diff = data["step_0.001_3"] - stock.covars_
                record(name, "regularisation_difference_is_lambda_I", max_error(diff, .001 * np.eye(rec["D"])), 1e-8)

            for reg in (0., .001):
                aligned = library(inits[0], True, reg, rec["seed"], Xs)
                stats, _ = aligned._do_estep(X, lengths)
                aligned._do_mstep(stats)
                for label, k, b in zip(["pi", "A", "means", "covs"], range(4), params(aligned)):
                    record(name, f"aligned_one_step_{reg}_{label}", max_error(data[f"step_{reg}_{k}"], b), 1e-8)

            best = None
            for init_no, initial in enumerate(inits):
                model = library(initial, True, rec["cov_reg"], rec["seed"], Xs)
                previous, history = -np.inf, []
                converged = False
                for iteration in range(1, rec["max_iter"] + 1):
                    stats, ll = model._do_estep(X, lengths)
                    history.append(float(ll))
                    if (ll - previous) / len(X) < rec["tol"]:
                        converged = True
                        break
                    previous = ll
                    model.aligned_iteration = iteration
                    model._do_mstep(stats)
                if not converged:
                    _, ll = model._do_estep(X, lengths)
                if best is None or ll > best[0]:
                    best = float(ll), params(model), iteration, converged, history, init_no, model.resets
            ll, fitted, iteration, converged, history, init_no, resets = best
            record(name, "fitted_loglik_abs_per_row", abs(ll - rec["loglik"]) / len(X), 1e-7)
            for label, a, b in zip(["pi", "A", "means", "covs"], frozen, fitted):
                record(name, f"fitted_{label}_max_abs", max_error(a, b), 1e-5)
            fit_ref = library(fitted, True)
            quarter_errors = []
            for j, (Xeval, Xtrain) in enumerate(zip(evals, Xs)):
                _, fw = _hmmc.forward_log(fitted[0], fitted[1], fit_ref._compute_log_likelihood(Xeval))
                p = softmax(fw, axis=1)
                quarter_errors.append(max_error(data[f"filter_{j}"][len(Xtrain):], p[len(Xtrain):]))
            record(name, "fitted_next_quarter_probability_max_abs", max(quarter_errors), 1e-5)
            fit_details.append(dict(case=name, custom_loglik=rec["loglik"], reference_loglik=ll,
                                    custom_iterations=rec["n_iter"], reference_iterations=iteration,
                                    custom_converged=rec["converged"], reference_converged=converged,
                                    reference_best_init=init_no, reference_dead_state_resets=resets,
                                    quarter_probability_max_abs=max(quarter_errors), runtime_seconds=time.perf_counter() - start))

            # A few examples quantify the isolated effect of dropping lambda I.
            # Fixed, identical number of updates avoids a stopping-time confound.
            if name in ("synthetic_k2_d3", "market_ES", "macro_ES"):
                models = []
                for reg in (0., .001):
                    model = library(inits[0], True, reg, rec["seed"], Xs)
                    for step in range(1, 51):
                        stats, _ = model._do_estep(X, lengths)
                        model.aligned_iteration = step
                        model._do_mstep(stats)
                    models.append(model)
                ps = []
                for model in models:
                    _, fw = _hmmc.forward_log(model.startprob_, model.transmat_, model._compute_log_likelihood(evals[0]))
                    ps.append(softmax(fw, axis=1)[len(Xs[0]):])
                configuration_details.append(dict(case=name, updates=50,
                    comparison="same start and updates; cov_reg 0 versus 0.001; no state relabelling",
                    next_quarter_probability_max_abs=max_error(*ps)))

            h = np.asarray(rec.get("history", data["history"]))
            diagnostics.append(dict(case=name, production_max_cov_condition=float(max(np.linalg.cond(c) for c in frozen[3])),
                                    production_loglik_min_increment=float(np.diff(h).min()) if len(h) > 1 else None,
                                    production_converged=rec["converged"]))
        failed = sum(not row["passed"] for row in records if row["case"] == name)
        print(f"CHECK {name}: {'PASS' if not failed else str(failed) + ' FAILED'}; fit delta p={max(quarter_errors):.3g}; {time.perf_counter() - start:.2f}s", flush=True)
        pd.DataFrame(records).to_csv(OUT / "checks.csv", index=False)
        pd.DataFrame(fit_details).to_csv(OUT / "fit_comparison.csv", index=False)

    # Exhaustive independent path enumeration: verifies how missing observations
    # are modelled without assuming that stock hmmlearn accepts NaN.
    pi = np.array([.6, .4]); A = np.array([[.85, .15], [.25, .75]])
    ps = (pi, A, np.array([[-.7], [.9]]), np.array([[[.4]], [[.8]]]))
    tiny = np.array([[-.4], [np.nan], [.6], [np.nan], [.3]])
    model = set_custom(ps, 2)
    for stop in range(1, len(tiny) + 1):
        totals = np.zeros(2)
        for path in itertools.product(range(2), repeat=stop):
            weight = pi[path[0]]
            for t, state in enumerate(path):
                if t:
                    weight *= A[path[t - 1], state]
                if np.isfinite(tiny[t, 0]):
                    v = ps[3][state, 0, 0]
                    weight *= np.exp(-.5 * (tiny[t, 0] - ps[2][state, 0]) ** 2 / v) / np.sqrt(2 * np.pi * v)
            totals[path[-1]] += weight
        record("enumeration_missing", "prefix_probability_max_abs", max_error(model.filter(tiny)[stop - 1], totals / totals.sum()), 1e-12)
    # Negative control: future observations change smoothing, not filtering.
    demo = simulate(150, 2, 1, 17)
    stock = library(ps)
    smoothed = stock.predict_proba(demo)
    changed = demo.copy(); changed[75:] = 5.
    smoothing_change = max_error(smoothed[:75], stock.predict_proba(changed)[:75])
    record("future_perturbation", "project_prior_probabilities_unchanged", max_error(model.filter(demo)[:75], model.filter(changed)[:75]), 0.)
    record("future_perturbation", "negative_control_smoothing_changes", float(smoothing_change <= 1e-6), 0., f"Actual smoothed change={smoothing_change}")

    failures = [r for r in records if not r["passed"]]
    by_metric = {}
    for row in records:
        by_metric.setdefault(row["metric"], []).append(row)
    aggregates = {key: dict(checks=len(rows), failed=sum(not r["passed"] for r in rows),
                            max_error=max(r["error"] for r in rows if r["error"] is not None)) for key, rows in by_metric.items()}
    unchanged = {p: digest(ROOT / p) == h for p, h in manifest["protected_hashes"].items()}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), reference_versions=versions(),
                  production_versions=manifest["production_versions"], cases=len(manifest["cases"]),
                  checks=len(records), failures=failures, aggregates=aggregates,
                  negative_control_smoothing_change=smoothing_change,
                  configuration_examples=configuration_details, production_diagnostics=diagnostics,
                  protected_files_unchanged=unchanged,
                  limitations=["Real-data fitting checks use 2023Q1 snapshots, not every historical refit or a new P&L backtest.",
                               "Real snapshots use four frozen starts without prior-quarter warm starts.",
                               "Reference adapter explicitly matches missing rows, regularisation, dead-state policy, and stopping; it does not replace library E/M numerics.",
                               "Production Python 3.14 outputs compared against independent Python 3.12 reference environment."])
    pd.DataFrame(records).to_csv(OUT / "checks.csv", index=False)
    dump(OUT / "results.json", result)
    text = [f"HMM independent validation: {len(manifest['cases'])} cases, {len(records)} checks, {len(failures)} failed.",
            f"Production environment: {manifest['production_versions']}", f"Reference environment: {versions()}",
            "", "Maximum absolute errors (normalisation is included in metric names):"]
    text += [f"{key}: {v['max_error']:.12g} ({v['failed']}/{v['checks']} failed)" for key, v in aggregates.items()]
    text += ["", "Limitations:"] + result["limitations"]
    text += ["", f"Protected source/results/report files unchanged: {all(unchanged.values())}"]
    (OUT / "summary.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text), flush=True)
    return diagnose()


def diagnose():
    """Keep strict flags visible; resolve only independently verified roundoff.

    A long-sequence log-space xi sum can lose more precision than filtering or
    likelihoods. Compare BOTH hmmlearn's separate C++ scaling implementation AND
    a per-transition normalised log-space calculation, without relaxing the
    original threshold or changing any production code.
    """
    from hmmlearn import _hmmc
    from hmmlearn.hmm import GaussianHMM as LibraryHMM
    from scipy.special import logsumexp
    result = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    lookup = {r["name"]: r for r in manifest["cases"]}
    reviews = []
    for name in sorted({r["case"] for r in result["failures"] if r["metric"] == "transition_count_abs_per_row"}):
        rec = lookup[name]
        with np.load(OUT / rec["file"]) as data:
            ps = [data[f"fit_{k}"] for k in range(4)]
            custom = set_custom(ps, rec["K"])
            stock = LibraryHMM(rec["K"], covariance_type="full", init_params="")
            stock.startprob_, stock.transmat_, stock.means_, stock.covars_ = [p.copy() for p in ps]
            for j in range(rec["n_sequences"]):
                X = data[f"eval_{j}"]
                good = ~np.isnan(X).any(axis=1)
                logB = np.zeros((len(X), rec["K"]))
                logB[good] = stock._compute_log_likelihood(X[good])
                _, fw = _hmmc.forward_log(ps[0], ps[1], logB)
                bw = _hmmc.backward_log(ps[0], ps[1], logB)
                log_counts = np.exp(_hmmc.compute_log_xi_sum(fw, ps[1], bw, logB))
                B, _ = custom._scaled_emission(X)
                _, custom_counts, _ = _forward_backward(B, ps[1], ps[0])
                scaled_B = np.exp(logB - logB.max(axis=1, keepdims=True))
                _, sfw, factors = _hmmc.forward_scaling(ps[0], ps[1], scaled_B)
                sbw = _hmmc.backward_scaling(ps[0], ps[1], scaled_B, factors)
                scaled_counts = _hmmc.compute_scaling_xi_sum(sfw, ps[1], sbw, scaled_B)
                with np.errstate(divide="ignore"):
                    pair_log = fw[:-1, :, None] + np.log(ps[1])[None] + logB[1:, None, :] + bw[1:, None, :]
                pair_log -= logsumexp(pair_log, axis=(1, 2), keepdims=True)
                normalised_counts = np.exp(pair_log).sum(axis=0)
                scalar = lambda a, b: float(np.max(np.abs(a - b)))
                review = dict(case=name, sequence=j, rows=len(X),
                    original_count_max_abs=scalar(custom_counts, log_counts),
                    library_scaling_count_max_abs=scalar(custom_counts, scaled_counts),
                    normalised_log_count_max_abs=scalar(custom_counts, normalised_counts),
                    expected_total_transitions=len(X) - 1,
                    custom_total=float(custom_counts.sum()), library_log_total=float(log_counts.sum()),
                    library_scaling_total=float(scaled_counts.sum()), followup_absolute_count_tolerance=1e-8)
                # This absolute-count tolerance is stricter than the original
                # 1e-9 PER ROW threshold for the sequences under examination.
                review["resolved_as_log_accumulation_roundoff"] = bool(
                    review["library_scaling_count_max_abs"] < 1e-8
                    and review["normalised_log_count_max_abs"] < 1e-8
                    and abs(review["library_scaling_total"] - (len(X) - 1)) < 1e-8)
                reviews.append(review)
    resolved = {name for name in {r["case"] for r in reviews}
                if all(r["resolved_as_log_accumulation_roundoff"] for r in reviews if r["case"] == name)}
    unresolved = [r for r in result["failures"]
                  if not (r["metric"] == "transition_count_abs_per_row" and r["case"] in resolved)]
    result["roundoff_followup"] = reviews
    result["unresolved_failures"] = unresolved
    result["status"] = "verified_with_roundoff_note" if not unresolved and reviews else ("verified" if not unresolved else "unresolved_difference")
    # Inspect a separate diagnostic issue rather than describing every stopped
    # fit as a proven maximum of the unpenalised likelihood. Fixed lambda I is
    # not the exact MLE covariance update, so its likelihood need not increase.
    continuation = []
    for diagnostic in result["production_diagnostics"]:
        delta = diagnostic["production_loglik_min_increment"]
        if delta is None or delta >= 0:
            continue
        name = diagnostic["case"]
        rec = lookup[name]
        with np.load(OUT / rec["file"]) as data:
            Xs = [data[f"train_{j}"] for j in range(rec["n_sequences"])]
            frozen = [data[f"fit_{k}"] for k in range(4)]
            cls = GaussianHMM if len(Xs) == 1 else MultiSeqGaussianHMM
            continued = cls(rec["K"], n_init=0, max_iter=50, tol=-np.inf,
                            cov_reg=rec["cov_reg"], seed=rec["seed"])
            if len(Xs) == 1:
                continued.fit(Xs[0], warm_start=frozen)
            else:
                continued.fit_multi(Xs, warm_start=frozen)
            changes = [float(np.max(np.abs(continued.filter(data[f"eval_{j}"])[len(x):]
                                           - data[f"filter_{j}"][len(x):]))) for j, x in enumerate(Xs)]
            detail = dict(case=name, original_negative_increment=delta,
                original_negative_increment_per_row=delta / rec["train_rows"], extra_updates=50,
                continued_loglik_change=float(continued.loglik - rec["loglik"]),
                next_quarter_probability_max_abs=max(changes),
                interpretation="Diagnostic sensitivity only; production stops on a signed likelihood gain, including a decrease. No production parameters or results changed.")
            if "er60" in rec.get("features", []):
                feature_index = rec["features"].index("er60")
                old_top = int(np.argmax(frozen[2][:, feature_index]))
                new_top = int(np.argmax(continued.means[:, feature_index]))
                differences = np.concatenate([continued.filter(data[f"eval_{j}"])[len(x):, new_top]
                                              - data[f"filter_{j}"][len(x):, old_top] for j, x in enumerate(Xs)])
                detail.update(actual_filter_max_abs=float(np.abs(differences).max()),
                              actual_filter_mean_abs=float(np.abs(differences).mean()),
                              original_top_state=old_top, continued_top_state=new_top)
            continuation.append(detail)
    result["stopping_rule_diagnostics"] = continuation
    result["validation_script_sha256"] = digest(Path(__file__))
    result["protected_files_unchanged"] = {p: digest(ROOT / p) == h for p, h in manifest["protected_hashes"].items()}
    dump(OUT / "results.json", result)
    note = ["", "Independent follow-up of strict threshold flags:"]
    for r in reviews:
        note.append(f"{r['case']}: original log-space count difference {r['original_count_max_abs']:.12g}; "
                    f"independent library scaling difference {r['library_scaling_count_max_abs']:.12g}; "
                    f"per-transition normalised log difference {r['normalised_log_count_max_abs']:.12g}; "
                    f"roundoff confirmed: {r['resolved_as_log_accumulation_roundoff']}")
    note += [f"Final status: {result['status']}; unresolved discrepancies: {len(unresolved)}.",
             "The initial strict-threshold flag remains recorded in checks.csv and results.json; its tolerance was not changed."]
    note += ["", "Stopping-rule diagnostic (separate from agreement with the library):",
             f"{len(continuation)} winning fits stopped after a negative likelihood increment; fixed covariance regularisation does not guarantee monotonic unpenalised likelihood."]
    for r in continuation:
        note.append(f"{r['case']}: final negative gain per row {r['original_negative_increment_per_row']:.12g}; "
                    f"50 extra updates change next-quarter state probabilities by at most {r['next_quarter_probability_max_abs']:.12g}.")
        if "actual_filter_max_abs" in r:
            note.append(f"  Actual highest-ER60 filter: max change {r['actual_filter_max_abs']:.12g}; mean change {r['actual_filter_mean_abs']:.12g}.")
    note.append("Recommend distinguishing likelihood-decrease stops from convergence in diagnostics; this audit does not alter the research model.")
    summary = (OUT / "summary.txt").read_text(encoding="utf-8").split("\nIndependent follow-up of strict threshold flags:")[0]
    (OUT / "summary.txt").write_text(summary.rstrip() + "\n" + "\n".join(note) + "\n", encoding="utf-8")
    print("\n".join(note), flush=True)
    return int(bool(unresolved) or not all(result["protected_files_unchanged"].values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["export", "check", "diagnose"])
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        raise SystemExit({"export": export, "check": check, "diagnose": diagnose}[args.mode]())
