"""Look-ahead audit of everything the 17-trial family depends on (plan.md asks for one).

    python -m integration.lookahead_audit                     # all checks, ~30 min
    python -m integration.lookahead_audit --only m2 execution # a subset

Truncation test: every input is rebuilt from data cut at a date T (data/final, the off-market
table and M4's implied-vol file, copied into a temporary folder) and must equal the full-data
build on every day up to T. A rule that used anything after T would show up as a difference.
The cut dates are Fridays, so weekly rules (meta-labeling) keep the same weekly grid.

Checked:
- A1/A2 targets and signals (M2's build_m2), every contract and factor;
- the filter or strategy of every trial with a real treatment: M3 #1-#8, M4-4, M5-1, M5-2.
  Trials whose gate failed are the identity filter (A3 = A2) and have nothing to check;
- M1's execution: changing the target of day t changes nothing before day t + 2.

The full-data HMM and pooled-HMM filters are read from m3_outputs/ and m5_outputs/ (build them
with the module commands first); everything else is rebuilt here. No module code is changed.
Writes integration/results/lookahead_audit.csv.
"""

import argparse
import contextlib
import io
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .trial_family import OUT, ROOT

CUTS = ("2008-06-27", "2011-12-30")
DATA = ROOT / "data/final"
EXOG = ROOT / "data/processed/offmarket/offmarket_daily.csv"
IV = ROOT / "m4/data/implied_vol_2014.csv"
TOL = 1e-9


# ---------------------------------------------------------------- helpers

def truncate(cut: str, tmp: Path) -> dict:
    """Copies of the inputs with no row after `cut`."""
    d = tmp / "final"
    d.mkdir()
    for f in DATA.glob("*.csv"):
        t = pd.read_csv(f)
        if "date" in t.columns:
            t = t[pd.to_datetime(t["date"]) <= pd.Timestamp(cut)]
        t.to_csv(d / f.name, index=False)
    paths = {"data": d, "exog": tmp / "offmarket_daily.csv", "iv": tmp / "implied_vol_2014.csv"}
    for src, key in ((EXOG, "exog"), (IV, "iv")):
        t = pd.read_csv(src)
        t[pd.to_datetime(t["date"]) <= pd.Timestamp(cut)].to_csv(paths[key], index=False)
    return paths


def long(series: dict, name: str = "value") -> pd.DataFrame:
    """{contract: Series on dates} -> date, contract, value."""
    return pd.concat([s.rename(name).rename_axis("date").reset_index().assign(contract=c)
                      for c, s in series.items()], ignore_index=True)


def compare(full: pd.DataFrame, cut_df: pd.DataFrame, cut: str, col: str = "filter_close_t") -> dict:
    """Rows dated <= cut: same keys, same missing values, values equal within TOL."""
    key = ["contract", "date"]
    a, b = (x.assign(date=pd.to_datetime(x["date"])) for x in (full, cut_df))
    a, b = (x[x["date"] <= pd.Timestamp(cut)].set_index(key)[col] for x in (a, b))
    m = pd.concat([a.rename("full"), b.rename("cut")], axis=1)
    same_keys = len(a) == len(b) == len(m)
    same_nan = bool((m["full"].isna() == m["cut"].isna()).all())
    diff = (m["full"] - m["cut"]).abs()
    maxdiff = float(diff.max()) if diff.notna().any() else 0.0
    return {"rows": len(m), "non_missing": int(m["full"].notna().sum()), "max_abs_diff": maxdiff,
            "differing_rows": int((diff > TOL).sum()), "same_keys": same_keys, "same_missing": same_nan,
            "passed": same_keys and same_nan and maxdiff <= TOL}


def quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


# ---------------------------------------------------------------- checks (each returns {label: (full, cut, col)})

def check_m2(p: dict) -> dict:
    from m2.pipeline import build_m2
    out = {}
    for f in sorted(DATA.glob("*.csv")):
        if f.name in ("ret.csv", "close_adj.csv", "universe.csv"):
            continue
        frames = []
        for d in (pd.read_csv(f, parse_dates=["date"]), pd.read_csv(p["data"] / f.name, parse_dates=["date"])):
            m = build_m2(pd.DataFrame({"date": d.date, "contract": f.stem, "close": d.close_adj, "return": d.ret}))
            frames.append(m.assign(contract=m.contract + ":" + m.factor))
        for col in ("signal_close_t", "approach1_target_close_t", "approach2_target_close_t"):
            out.setdefault(f"M2 {col}", []).append((frames[0], frames[1], col))
    return {k: (pd.concat([a for a, _, _ in v]), pd.concat([b for _, b, _ in v]), v[0][2]) for k, v in out.items()}


def check_bocpd(p: dict, tmp: Path) -> dict:
    from m3.bocpd import BOCPDConfig
    from m3.run_bocpd import run
    full = quiet(run, DATA, tmp / "full_bocpd", BOCPDConfig())
    cut = quiet(run, p["data"], tmp / "cut_bocpd", BOCPDConfig())
    p["bocpd"] = tmp / "cut_bocpd" / "bocpd_h250_filter.csv"
    return {"M3-3 BOCPD": (full, cut, "filter_close_t")}


def check_hmm(p: dict, tmp: Path) -> dict:
    from m3.adaptor import AdaptorConfig
    from m3.features import DEFAULT_FEATURES, OFFMARKET_FEATURES
    from m3.run_real_data import run
    out = {}
    for label, feats, tag in (("M3-1 HMM", DEFAULT_FEATURES, ""),
                              ("M3-2 HMM + macro/news", DEFAULT_FEATURES + OFFMARKET_FEATURES, "offmkt")):
        cfg = AdaptorConfig(n_states=3, refit="quarter", mapping="top", features=feats)
        cut, _ = quiet(run, p["data"], tmp / "cut_hmm", cfg, p["exog"], tag)
        stem = "hmm_k3_quarter_top" + (f"_{tag}" if tag else "")
        out[label] = (pd.read_csv(ROOT / "m3_outputs" / f"{stem}_filter.csv"), cut, "filter_close_t")
    return out


def check_portvol(p: dict) -> dict:
    from m3.portfolio_vol import compute
    k_full, k_cut = quiet(compute, DATA)[2], quiet(compute, p["data"])[2]
    as_long = lambda k: k.rename("filter_close_t").rename_axis("date").reset_index().assign(contract="portfolio")  # noqa: E731
    return {"M3-4 portfolio vol target": (as_long(k_full), as_long(k_cut), "filter_close_t")}


def check_strategies(p: dict) -> dict:
    from m3 import regime_strategy, turning_points
    out = {}
    for label, mod, keys in (("M3-5 momentum turning points", turning_points, ("dynamic", "static_slow")),
                             ("M3-8 pure HMM strategy", regime_strategy, ("hmm", "tsmom_252"))):
        tf, tc = quiet(mod.build_targets, DATA), quiet(mod.build_targets, p["data"])
        tf = next(x for x in tf if isinstance(x, dict) and keys[0] in x)
        tc = next(x for x in tc if isinstance(x, dict) and keys[0] in x)
        for k in keys:
            out[f"{label}: {k}"] = (long(tf[k]), long(tc[k]), "value")
    return out


def check_meta(p: dict, tmp: Path) -> dict:
    from m3.metalabel import main
    out = {}
    for features, label in (("hand", "M3-6 meta-labeling"), ("sig", "M3-7 signature meta-labeling")):
        quiet(main, features, "gbm", p["data"], p["exog"], p["bocpd"], tmp / "cut_meta")
        cut = pd.read_csv(tmp / "cut_meta" / f"meta_{features}_gbm_filter.csv")
        out[label] = (pd.read_csv(ROOT / "m3_outputs" / f"meta_{features}_gbm_filter.csv"), cut, "filter_close_t")
    return out


def check_impvol(p: dict) -> dict:
    from m3.features import load_exog
    from m4.impvol import impvol_filter, load_iv
    from m5.common import load_all
    full = impvol_filter(load_all(DATA), load_iv(load_exog(EXOG)))[0]
    cut = impvol_filter(load_all(p["data"]), load_iv(load_exog(p["exog"]), p["iv"]))[0]
    return {"M4-4 implied/realised vol sizing": (full, cut, "filter_close_t")}


def check_pools(p: dict) -> dict:
    from m5.pool import run
    out = {}
    for pool, label in (("class", "M5-1 class-pooled HMM"), ("global", "M5-2 global-pooled HMM")):
        cut, _ = quiet(run, p["data"], None, pool)
        full = pd.read_csv(ROOT / "m5_outputs" / f"pool_{pool}_k3_quarter_top_filter.csv")
        out[label] = (full, cut, "filter_close_t")
    return out


def check_execution() -> dict:
    """M1's contract_pnl with delay 1: a target set at close t trades at close t + 1 (its cost is
    booked that day) and first earns on t + 2; nothing on or before day t moves."""
    from m1.backtest import contract_pnl
    idx = pd.bdate_range("2005-01-03", periods=300)
    rng = np.random.default_rng(0)
    ret = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    target = pd.Series(np.sign(rng.normal(size=len(idx))), index=idx)
    base = contract_pnl(target, ret, cost_bps=1.0, delay=1)
    worst, n, bad = 0.0, 0, 0
    for t in range(5, 295, 7):
        bumped = target.copy()
        bumped.iloc[t] = -3.0 * bumped.iloc[t]
        moved = (contract_pnl(bumped, ret, cost_bps=1.0, delay=1) - base).abs()
        worst = max(worst, float(moved.iloc[: t + 1].to_numpy().max()))  # day t and before: untouched
        first_trade = moved["position_close"].iloc[t + 1:].ne(0).idxmax()
        first_gross = moved["gross"].iloc[t + 1:].ne(0).idxmax()
        bad += (first_trade != idx[t + 1]) or (first_gross != idx[t + 2])
        n += 1
    return {"rows": n, "non_missing": n, "max_abs_diff": worst, "differing_rows": bad,
            "same_keys": True, "same_missing": True, "passed": worst == 0.0 and bad == 0}


CHECKS = ("m2", "bocpd", "hmm", "portvol", "strategies", "meta", "impvol", "pools")


def main(cuts=CUTS, only=None) -> pd.DataFrame:
    only = set(only or (*CHECKS, "execution"))
    rows = []
    if "execution" in only:
        rows.append({"cut": "-", "check": "M1 execution: target at t first earns on t + 2", **check_execution()})
        print(rows[-1]["check"], "passed" if rows[-1]["passed"] else "FAILED", flush=True)
    for cut in cuts:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            p = truncate(cut, tmp)
            for name in CHECKS:
                if name not in only and not (name == "bocpd" and "meta" in only):
                    continue
                t0 = time.perf_counter()
                fn = globals()[f"check_{name}"]
                res = fn(p, tmp) if name in ("bocpd", "hmm", "meta") else fn(p)
                for label, (full, cut_df, col) in res.items():
                    r = {"cut": cut, "check": label, **compare(full, cut_df, cut, col)}
                    rows.append(r)
                    print(f"{cut}  {label:45s} rows {r['rows']:7d}  max diff {r['max_abs_diff']:.1e}  "
                          f"{'passed' if r['passed'] else 'FAILED'}  ({time.perf_counter() - t0:.0f}s)", flush=True)
            shutil.rmtree(tmp, ignore_errors=True)
    t = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "lookahead_audit.csv", index=False)
    print(f"\n{int(t.passed.sum())}/{len(t)} checks passed; wrote {OUT / 'lookahead_audit.csv'}")
    return t


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("--cuts", nargs="+", default=list(CUTS))
    a.add_argument("--only", nargs="+", choices=[*CHECKS, "execution"])
    args = a.parse_args()
    main(args.cuts, args.only)
