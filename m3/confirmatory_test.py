"""The one pre-registered confirmatory test on the hold-out (see the M3 notes).

    python -m m3.confirmatory_test

A2 vs A3 (= A2 x BOCPD filter, hazard 1/250) for the combo of the three M2 factors, on
2015-01-01 .. end of data. H0: SR(A3) - SR(A2) <= 0, H1: > 0. Circular block bootstrap
(20-day blocks, 5,000 resamples, seed 0), one-sided at 5%. Computes nothing else on the
hold-out, by design.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.engine import COST_BPS, contract_pnl, in_book, portfolio
from backtest.metrics import block_bootstrap_sharpe_diff, perf
from backtest.run import load_targets

FILTER = Path("m3_outputs/bocpd_h250_filter.csv")
EXPECTED_SHA256 = "7256924688ce4b0ae206e7e2118d0fa609c8391114ab658f350aebfd86d7c975"
TEST_START = "2015-01-01"
ALPHA = 0.05


def main(data_dir: Path = Path("data/final"), out: Path = Path("m3_outputs/confirmatory_test.json")):
    sha = hashlib.sha256(FILTER.read_bytes()).hexdigest()
    if sha != EXPECTED_SHA256:
        raise SystemExit(f"filter file changed since pre-registration (sha256 {sha})")
    contracts, _ = load_targets(data_dir, FILTER)
    union = sorted(set().union(*(c["data"].index for c in contracts.values())))
    book = in_book(pd.DataFrame({k: c["A3"]["combo"].notna() for k, c in contracts.items()}).reindex(union))
    port = {}
    for ap in ("A2", "A3"):
        pnl = {k: contract_pnl(c[ap]["combo"], c["data"].ret, c["data"].roll, COST_BPS[c["asset"]], 1)
               for k, c in contracts.items()}
        port[ap] = portfolio(pnl, book, 10).loc[TEST_START:].net
    a2, a3 = port["A2"], port["A3"]
    res = block_bootstrap_sharpe_diff(a3, a2, block=20, reps=5000, seed=0, return_boot=True)
    boot = res.pop("boot")
    obs = res["d_sharpe"]
    p_one = float(np.mean(boot - boot.mean() >= obs - 1e-12))
    lower_95 = float(np.quantile(boot, ALPHA))
    result = {
        "window": [str(a2.index[0].date()), str(a2.index[-1].date())], "days": len(a2),
        "filter_sha256": sha,
        "sharpe_A2": perf(a2)["sharpe"], "sharpe_A3": perf(a3)["sharpe"],
        "d_sharpe": obs, "p_one_sided": p_one, "lower_95_one_sided": lower_95,
        "ci95_two_sided": [res["ci_low"], res["ci_high"]], "se_boot": res["se_boot"],
        "corr": float(a3.corr(a2)), "reject_H0_at_5pct": p_one < ALPHA,
    }
    out.write_text(json.dumps(result, indent=2))
    for k, v in result.items():
        print(f"{k:20s} {v}")
    return result


if __name__ == "__main__":
    main()
