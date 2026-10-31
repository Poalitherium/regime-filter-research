"""Phase 3: run a frozen M5 method on contracts never used in development.

    python -m m5.confirm --new-dir data/confirm --method pool_class --direction -1

`--new-dir` holds `{CODE}.csv` files with exactly the data/final schema
(date, close, close_adj, ret, volume, roll, kind) and a `universe.csv` with at least
`code` and `asset_class` (one of equity, energy, metal, rates, fx, agri). Never build
these by hand: they must come out of the same roll-adjustment pipeline as data/final
(see m5/DATA_REQUEST.md).

Methods (all frozen; settings exactly as in PREREG):
    pool_class   M5-1: each new contract joins its asset class's pool together with the
                 development contracts of that class (features only; no returns of the
                 development contracts enter the test)
    pool_global  M5-2: one pool with the 13 development contracts and all new ones
    expiry       M5-4 proportional filter, class groups formed by the new contracts
    season       M5-3 proportional filter (only codes listed in m5.calendar.SEASONS)

The gates of M5-3 and M5-4 failed on the development window, so their frozen method is
the identity filter; `expiry` / `season` here are exploratory and cannot confirm anything.

PREREG §6: a run here can confirm an effect only for a trial that was significant on the
development window. None was (M5-1 p = 0.80, M5-2 p = 0.68), so every run is
descriptive: the result file reports the one-sided p-value and the CI, but no
accept/reject decision (DEV_SIGNIFICANT below).

Test: A3 vs A2, combo factor, equal weight over the new contracts, their whole history
(all of it is out of sample). One-sided circular block bootstrap (20-day blocks, 5,000
resamples, seed 0) in the pre-registered `--direction` (+1: A3 better, -1: A3 worse).
The book needs all but one of the new contracts (`--min-contracts` overrides this).
Writes m5_outputs/confirm_<method>.json, the filter file and backtest_outputs/m5_confirm_<method>/.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from backtest.metrics import block_bootstrap_sharpe_diff, perf
from backtest.run import load_targets
from backtest.run import run as backtest_run

from . import calendar as cal
from . import calfilter, pool
from .common import LEAD, load_all

ALPHA = 0.05
FAR_END = "2100-12-31"  # no dev/test split: every day of a new contract is out of sample
# Was the trial behind each method significant on the development window (REPORT §4)?
# PREREG §6: only then can the run on new contracts confirm anything.
DEV_SIGNIFICANT = {"pool_class": False, "pool_global": False, "expiry": False, "season": False}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_universe(new_dir: Path, dev_dir: Path) -> pd.DataFrame:
    new = pd.read_csv(new_dir / "universe.csv")
    if {"code", "asset_class"} - set(new.columns):
        raise ValueError("universe.csv needs code and asset_class columns")
    dev = pd.read_csv(dev_dir / "universe.csv")
    clash = set(new.code) & set(dev.code)
    if clash:
        raise ValueError(f"not new contracts (used in development): {sorted(clash)}")
    bad = set(new.asset_class) - set(COST_BPS)
    if bad:
        raise ValueError(f"unknown asset classes {sorted(bad)}; use one of {sorted(COST_BPS)}")
    return new


def build_filter(method: str, new_dir: Path, dev_dir: Path, cfg=pool.M3_TRIAL1) -> pd.DataFrame:
    new = new_universe(new_dir, dev_dir)
    if method in ("pool_class", "pool_global"):
        dev = pd.read_csv(dev_dir / "universe.csv")
        both = pd.concat([dev[["code", "asset_class"]].assign(data_dir=str(dev_dir)),
                          new[["code", "asset_class"]].assign(data_dir=str(new_dir))], ignore_index=True)
        if method == "pool_class":  # only the classes that have a new contract
            both = both[both.asset_class.isin(set(new.asset_class))]
        filt, _ = pool.run(dev_dir, None, method.split("_")[1], cfg, universe=both)
        return filt[filt.contract.isin(set(new.code))][["date", "contract", "filter_close_t", "vintage"]]
    contracts = load_all(new_dir, new)
    if method == "expiry":
        unknown = set(new.code) - set(cal.SCHEDULES) - set(cal.STANDINS)
        if unknown:
            raise ValueError(f"no contract schedule for {sorted(unknown)}; add it to m5.calendar.SCHEDULES")
        groups = {}
        for code, c in contracts.items():
            if code not in cal.STANDINS:
                groups.setdefault(c["asset"], []).append(code)
        fn = lambda idx, code, lead: cal.roll_window(idx, code, 3, lead)  # noqa: E731
    elif method == "season":
        groups = {}
        for code, c in contracts.items():
            if code in cal.SEASONS:
                groups.setdefault(c["asset"], []).append(code)
        fn = cal.in_season
    else:
        raise ValueError(f"unknown method {method!r}")
    filt, _ = calfilter.build(contracts, groups, fn, lead=LEAD)
    return filt


def test(new_dir: Path, filter_file: Path, direction: int, min_contracts: int) -> dict:
    contracts, _ = load_targets(new_dir, filter_file)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    port = {}
    for ap in ("A2", "A3"):
        pnl = {k: contract_pnl(c[ap]["combo"], c["data"].ret, c["data"].roll, COST_BPS[c["asset"]], 1)
               for k, c in contracts.items()}
        port[ap] = portfolio(pnl, book, min_contracts).net
    a2, a3 = port["A2"], port["A3"]
    res = block_bootstrap_sharpe_diff(a3, a2, block=20, reps=5000, seed=0, return_boot=True)
    boot = res.pop("boot")
    obs = res["d_sharpe"]
    c = boot - boot.mean()
    p_one = float(np.mean(c >= obs - 1e-12)) if direction > 0 else float(np.mean(c <= obs + 1e-12))
    return {"window": [str(a2.index[0].date()), str(a2.index[-1].date())], "days": len(a2),
            "contracts": sorted(contracts), "sharpe_A2": perf(a2)["sharpe"], "sharpe_A3": perf(a3)["sharpe"],
            "d_sharpe": obs, "direction": direction, "p_one_sided": p_one,
            "ci95_two_sided": [res["ci_low"], res["ci_high"]], "se_boot": res["se_boot"],
            "corr": float(a3.corr(a2))}


def main(new_dir: Path, method: str, direction: int, dev_dir: Path = Path("data/final"),
         out_dir: Path = Path("m5_outputs"), min_contracts: int | None = None, cfg=pool.M3_TRIAL1):
    filt = build_filter(method, new_dir, dev_dir, cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    ffile = out_dir / f"confirm_{method}_filter.csv"
    filt.to_csv(ffile, index=False)
    n_new = len(pd.read_csv(new_dir / "universe.csv"))
    k = min_contracts if min_contracts is not None else max(1, n_new - 1)
    backtest_run(new_dir, ffile, delay=1, costs=True, dev_end=FAR_END, include_test=False,
                 min_contracts=k, out_dir=Path("backtest_outputs") / f"m5_confirm_{method}")
    confirmatory = DEV_SIGNIFICANT[method]
    result = {"method": method, "confirmatory": confirmatory,
              "status": "confirmatory" if confirmatory else
              "descriptive only (PREREG §6: the trial was not significant on the development window)",
              "exploratory": method in ("expiry", "season"),
              "filter_sha256": _sha(ffile), "min_contracts": k,
              "data_sha256": {p.name: _sha(p) for p in sorted(new_dir.glob("*.csv"))},
              **test(new_dir, ffile, direction, k)}
    if confirmatory:
        result["reject_H0_at_5pct"] = result["p_one_sided"] < ALPHA
    (out_dir / f"confirm_{method}.json").write_text(json.dumps(result, indent=2, default=str))
    for key, v in result.items():
        if key != "data_sha256":
            print(f"{key:20s} {v}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", type=Path, required=True)
    p.add_argument("--method", choices=["pool_class", "pool_global", "expiry", "season"], required=True)
    p.add_argument("--direction", type=int, choices=[-1, 1], required=True,
                   help="pre-registered sign of the effect (+1: A3 better than A2)")
    p.add_argument("--dev-dir", type=Path, default=Path("data/final"))
    p.add_argument("--out-dir", type=Path, default=Path("m5_outputs"))
    p.add_argument("--min-contracts", type=int, default=None)
    a = p.parse_args()
    main(a.new_dir, a.method, a.direction, a.dev_dir, a.out_dir, a.min_contracts)
