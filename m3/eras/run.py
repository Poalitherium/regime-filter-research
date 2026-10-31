"""The eras study's single pre-registered runs (PREREG.md and PREREG_1930.md).

    python -m m3.eras.run filters               # first study: filters on data/eras -> m3_outputs/eras/
    python -m m3.eras.run analyze               # first study: H1/H2 and secondary -> m3/eras/results/
    python -m m3.eras.run filters --study 1930  # addendum: filters on data/eras1930 -> m3_outputs/eras/1930/
    python -m m3.eras.run analyze --study 1930  # addendum: H3/H4 (Holm over H1-H4) -> m3/eras/results_1930/

Each `analyze` is the pre-registered run and has been run once (results/, results_1930/).

Filters (frozen M3/M5 settings, nothing tuned):
- M3 #3 BOCPD, hazard 1/250, per contract;
- M3 #1 HMM, K = 3, quarterly refit, `top`, per contract (one process per contract, which gives
  the same filters as one run because contracts never share anything);
- M5-2 global-pooled HMM, pooled within each pool of the study (m5_seeds.valid_seeds keeps its
  refit seeds valid before 1970 without changing m5/, deviation #4).

Backtests: M1's engine via backtest.m1_bridge.m1_book, research bps by asset class (x1 primary,
x0 and x3 sensitivity), fixed equal weights over the pool, same book for A2 and A3, from the
study's start (later if not every contract of the primary pool is in the book yet) or, for a
secondary pool, from the first day all its contracts are in the book, to 2014-12-31, then
sliced by era.
"""

import argparse
import contextlib
import io
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

END = "2014-12-31"
COSTS = {"x1": 1.0, "x0": 0.0, "x3": 3.0}
METHODS = {"bocpd": "M3 #3 BOCPD", "hmm": "M3 #1 HMM", "pool": "M5-2 global-pooled HMM"}
Z80 = 0.841621


@dataclass(frozen=True)
class Study:
    name: str
    data: Path
    out: Path
    res: Path
    pools: tuple[str, ...]            # the first one is the primary pool
    start: str                        # earliest evaluation day of the primary pool
    eras: dict
    pre: tuple[str, str]
    post: tuple[str, str]
    tests: tuple[str, str]            # names of (increment > 0, decline > 0)
    skip_eras: dict = field(default_factory=dict)     # pool -> eras not reported
    extra_slices: dict = field(default_factory=dict)  # descriptive slices added to eras.csv
    earlier_primary: Path | None = None               # primary tests of an earlier run, same Holm family


FIRST = Study(
    name="1977", data=Path("data/eras"), out=Path("m3_outputs/eras"), res=Path("m3/eras/results"),
    pools=("core", "extended"), start="1977-01-03",
    eras={"I": ("1977-01-01", "1989-12-31"), "II": ("1990-01-01", "1999-12-31"), "III": ("2000-01-01", "2014-12-31")},
    pre=("1977-01-01", "1999-12-31"), post=("2000-01-01", "2014-12-31"), tests=("H1", "H2"),
    skip_eras={"extended": {"I"}})
S1930 = Study(
    name="1930", data=Path("data/eras1930"), out=Path("m3_outputs/eras/1930"), res=Path("m3/eras/results_1930"),
    pools=("industry",), start="1930-01-02",
    eras={"E0": ("1930-01-01", "1959-12-31"), "E1": ("1960-01-01", "1976-12-31"), "I": ("1977-01-01", "1989-12-31"),
          "II": ("1990-01-01", "1999-12-31"), "III": ("2000-01-01", "2014-12-31")},
    pre=("1930-01-01", "1959-12-31"), post=("2000-01-01", "2014-12-31"), tests=("H3", "H4"),
    extra_slices={"I+II": ("1977-01-01", "1999-12-31")}, earlier_primary=Path("m3/eras/results/primary.csv"))
STUDIES = {"1977": FIRST, "1930": S1930}

# first-study names kept for the tests and for readers of the earlier code
DATA, OUT, RES = FIRST.data, FIRST.out, FIRST.res
ERAS, PRE, POST, CORE_START = FIRST.eras, FIRST.pre, FIRST.post, FIRST.start


def universe(pool: str, study: Study = FIRST) -> pd.DataFrame:
    uni = pd.read_csv(study.data / "universe.csv")
    return (uni[uni["pool"] == "core"] if pool == "core" else uni)[["code", "asset_class"]].reset_index(drop=True)


def filter_file(method: str, pool: str, study: Study = FIRST) -> Path:
    return {"bocpd": study.out / "bocpd_h250_filter.csv", "hmm": study.out / "hmm_k3_quarter_top_filter.csv",
            "pool": study.out / f"pool_global_k3_quarter_top_{pool}_filter.csv"}[method]


# ---------------------------------------------------------------- filters

def _hmm_one(args) -> str:
    code, study_name = args
    from m3.adaptor import AdaptorConfig
    from m3.run_real_data import run
    s = STUDIES[study_name]
    with contextlib.redirect_stdout(io.StringIO()):
        run(s.data, s.out / "hmm_parts" / code, AdaptorConfig(n_states=3, refit="quarter", mapping="top"), codes=(code,))
    return code


def _pool_one(args) -> str:
    pool, study_name = args
    from m5.pool import run

    from .m5_seeds import valid_seeds
    s = STUDIES[study_name]
    with contextlib.redirect_stdout(io.StringIO()), valid_seeds():
        run(s.data, s.out, "global", universe=universe(pool, s), tag=pool)
    return pool


def make_filters(study: Study = FIRST, workers: int = 15):
    from m3.bocpd import BOCPDConfig
    from m3.run_bocpd import run as run_bocpd
    codes = tuple(pd.read_csv(study.data / "universe.csv").code)
    study.out.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()):
        run_bocpd(study.data, study.out, BOCPDConfig(), codes)
    print("BOCPD done", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        jobs = ([ex.submit(_pool_one, (p, study.name)) for p in study.pools]
                + [ex.submit(_hmm_one, (c, study.name)) for c in codes])
        for j in jobs:
            print("done", j.result(), flush=True)
    for kind in ("filter", "fits"):
        parts = [pd.read_csv(study.out / "hmm_parts" / c / f"hmm_k3_quarter_top_{kind}.csv") for c in codes]
        pd.concat(parts, ignore_index=True).to_csv(study.out / f"hmm_k3_quarter_top_{kind}.csv", index=False)
    print("HMM filters combined", flush=True)


# ---------------------------------------------------------------- statistics

def delta(t: pd.Series, c: pd.Series, lo: str, hi: str) -> dict:
    """Sharpe(t) - Sharpe(c) on [lo, hi]: circular block bootstrap (20-day blocks, 5,000, seed 0)."""
    from backtest.metrics import block_bootstrap_sharpe_diff, perf
    t, c = t.loc[lo:hi], c.loc[lo:hi]
    r = block_bootstrap_sharpe_diff(t, c, block=20, reps=5000, seed=0, return_boot=True)
    boot = r.pop("boot")
    return {"start": str(t.index[0].date()), "end": str(t.index[-1].date()), "days": len(t),
            "sharpe_control": perf(c)["sharpe"], "sharpe_treatment": perf(t)["sharpe"],
            "d_sharpe": r["d_sharpe"], "ci_low": r["ci_low"], "ci_high": r["ci_high"], "se_boot": r["se_boot"],
            "p_two_sided": r["p_boot"], "p_one_sided": float(np.mean(boot - boot.mean() >= r["d_sharpe"] - 1e-12)),
            "mde_80": (1.959964 + Z80) * r["se_boot"], "corr": float(t.loc[lo:hi].corr(c.loc[lo:hi]))}


def decline(pre: dict, post: dict) -> dict:
    """Decline test: D = d_pre - d_post, independent periods, one-sided normal test."""
    from m3.predictability import norm_sf
    d = pre["d_sharpe"] - post["d_sharpe"]
    se = float(np.hypot(pre["se_boot"], post["se_boot"]))
    return {"D": d, "se": se, "z": d / se, "p_one_sided": norm_sf(d / se), "mde_80": (1.959964 + Z80) * se}


# ---------------------------------------------------------------- analysis

def backtest(method: str, pool: str, mult: float, study: Study = FIRST):
    from backtest.m1_bridge import filter_trial, m1_book, window_start
    uni = universe(pool, study)
    data, asset, targets, book = filter_trial(study.data, filter_file(method, pool, study), universe=uni)
    first_full = window_start(book, len(uni))
    start = max(pd.Timestamp(study.start), first_full) if pool == study.pools[0] else first_full
    ports = {k: m1_book(tg, data, asset, book, start, END, "research", cost_multiplier=mult) for k, tg in targets.items()}
    return pd.DataFrame({k: p.net for k, p in ports.items()}), start, ports


def baseline(study: Study = FIRST, mult: float = 1.0) -> pd.DataFrame:
    """A1 and A2 (M2 combo) per era on the primary book (BOCPD, primary pool), as context."""
    from backtest.engine import in_book
    from backtest.m1_bridge import m1_book, window_start
    from backtest.run import load_targets
    pool = study.pools[0]
    uni = universe(pool, study)
    contracts, _ = load_targets(study.data, filter_file("bocpd", pool, study), uni)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    start = max(pd.Timestamp(study.start), window_start(book, len(uni)))
    data = {k: c["data"] for k, c in contracts.items()}
    asset = {k: c["asset"] for k, c in contracts.items()}
    net = {ap: m1_book({k: c[ap]["combo"] for k, c in contracts.items()}, data, asset, book, start, END,
                       "research", cost_multiplier=mult).net for ap in ("A1", "A2")}
    rows = []
    pre_label = "pre (I+II)" if study is FIRST else "pre (E0)"
    for era, (lo, hi) in {**study.eras, pre_label: study.pre, **study.extra_slices}.items():
        r = delta(net["A2"], net["A1"], lo, hi)
        rows.append({"era": era, **{k: r[k] for k in ("start", "end", "sharpe_control", "sharpe_treatment",
                                                     "d_sharpe", "ci_low", "ci_high", "p_two_sided")}})
    return pd.DataFrame(rows).rename(columns={"sharpe_control": "sharpe_A1", "sharpe_treatment": "sharpe_A2"})


def analyze(study: Study = FIRST):
    from m3.predictability import holm
    study.res.mkdir(parents=True, exist_ok=True)
    t_inc, t_dec = study.tests
    tests, eras, windows, daily = [], [], [], {}
    primary_pool = study.pools[0]
    for pool in study.pools:
        for method in METHODS:
            for cname, mult in COSTS.items():
                nets, start, _ = backtest(method, pool, mult, study)
                daily[f"{method}_{pool}_{cname}"] = nets
                windows.append({"method": method, "pool": pool, "cost": cname, "start": str(start.date())})
                tag = {"method": method, "name": METHODS[method], "pool": pool, "cost": cname}
                pre_lo = study.pre[0] if pool == primary_pool else str(start.date())
                pre = delta(nets.treatment, nets.control, pre_lo, study.pre[1])
                post = delta(nets.treatment, nets.control, *study.post)
                h2 = decline(pre, post)
                tests.append({**tag, "test": t_inc, **pre})
                tests.append({**tag, "test": t_dec, "d_sharpe": h2["D"], "se_boot": h2["se"], "z": h2["z"],
                              "p_one_sided": h2["p_one_sided"], "mde_80": h2["mde_80"]})
                for era, (lo, hi) in {**study.eras, **study.extra_slices}.items():
                    if era in study.skip_eras.get(pool, set()):
                        continue
                    eff_lo = lo if pool == primary_pool or era != "II" else str(start.date())
                    eras.append({**tag, "era": era, **delta(nets.treatment, nets.control, eff_lo, hi)})
                print(f"{method:5s} {pool:8s} {cname}: done", flush=True)
    tests, eras = pd.DataFrame(tests), pd.DataFrame(eras)
    primary = tests[(tests.method == "bocpd") & (tests.pool == primary_pool) & (tests.cost == "x1")].copy()
    if study.earlier_primary is not None:  # one Holm family with the earlier run's primary tests
        earlier = pd.read_csv(study.earlier_primary).drop(columns=["p_holm", "reject_holm_5pct"], errors="ignore")
        primary = pd.concat([earlier.assign(source="first run (m3/eras/results)"), primary.assign(source="this run")],
                            ignore_index=True)
    primary["p_holm"] = holm(primary.p_one_sided.to_numpy())
    primary["reject_holm_5pct"] = primary.p_holm < 0.05
    primary.to_csv(study.res / "primary.csv", index=False)
    tests.to_csv(study.res / "tests.csv", index=False)
    eras.to_csv(study.res / "eras.csv", index=False)
    pd.DataFrame(windows).to_csv(study.res / "windows.csv", index=False)
    baseline(study).to_csv(study.res / "baseline.csv", index=False)
    keep = [f"{m}_{primary_pool}_x1" for m in METHODS]
    pd.concat({k: daily[k] for k in keep}, axis=1).set_axis(
        [f"{k}_{c}" for k in keep for c in ("control", "treatment")], axis=1).rename_axis("date").to_csv(
        study.res / f"daily_{primary_pool}_x1.csv")
    sha = "m3/eras/DATA_SHA256.json" if study is FIRST else "m3/eras/DATA_SHA256_1930.json"
    (study.res / "run_info.json").write_text(json.dumps({
        "data_sha256_file": sha, "python": sys.version.split()[0],
        "filters": {m: str(filter_file(m, primary_pool, study)) for m in METHODS}}, indent=2))
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(primary[["test", "d_sharpe", "ci_low", "ci_high", "se_boot", "p_one_sided", "p_holm", "mde_80"]].round(4).to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("step", choices=["filters", "analyze"])
    p.add_argument("--study", choices=list(STUDIES), default="1977")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()
    if a.workers < 1:
        p.error("--workers must be positive")
    make_filters(STUDIES[a.study], workers=a.workers) if a.step == "filters" else analyze(STUDIES[a.study])
