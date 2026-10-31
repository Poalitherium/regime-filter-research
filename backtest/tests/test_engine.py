"""Backtest engine and metrics checks: python -m unittest discover -s backtest/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backtest.engine import contract_pnl, in_book, portfolio  # noqa: E402
from backtest.metrics import (  # noqa: E402
    block_bootstrap_sharpe_diff, max_drawdown, paired, perf, sharpe_diff_test)

IDX = pd.bdate_range("2020-01-01", periods=10)


class EngineTests(unittest.TestCase):
    def test_execution_delay(self):
        ret = pd.Series(0.01, index=IDX)
        target = pd.Series(np.nan, index=IDX)
        target.iloc[2] = 1.0  # known after close of day 2 only
        for delay, first_pnl_day in [(0, 3), (1, 4), (2, 5)]:
            p = contract_pnl(target, ret, delay=delay)
            self.assertEqual(p.gross.ne(0).idxmax(), IDX[first_pnl_day])
            self.assertEqual(p.gross.ne(0).sum(), 1)  # NaN after day 2 -> flat again

    def test_no_lookahead_on_targets(self):
        rng = np.random.default_rng(0)
        ret = pd.Series(rng.normal(0, 0.01, 10), index=IDX)
        t1 = pd.Series(rng.choice([-1.0, 1.0], 10), index=IDX)
        t2 = t1.copy()
        t2.iloc[6:] = -t2.iloc[6:]  # change targets known from day 6 on
        p1, p2 = contract_pnl(t1, ret, delay=1), contract_pnl(t2, ret, delay=1)
        pd.testing.assert_series_equal(p1.net.iloc[:8], p2.net.iloc[:8])  # days 0..7 unaffected

    def test_costs_on_turnover_and_rolls(self):
        ret = pd.Series(0.0, index=IDX)
        target = pd.Series([1, 1, -1, -1, -1, 0, 0, 0, 0, 0], index=IDX, dtype=float)
        roll = pd.Series(False, index=IDX)
        roll.iloc[4] = True
        p = contract_pnl(target, ret, roll=roll, cost_bps=10, delay=0)
        # positions: 0,1,1,-1,-1,-1,0,... turnover 1, 2, 1; roll day 4 holds |-1| -> 2 more
        self.assertAlmostEqual(p.turnover.sum(), 4.0)
        self.assertAlmostEqual(p.cost.sum(), (4.0 + 2.0) * 10 / 1e4)

    def test_portfolio_equal_weight_and_holidays(self):
        a_days = IDX
        b_days = IDX.delete([3, 4])  # market B closed on days 3-4
        pa = contract_pnl(pd.Series(1.0, index=a_days), pd.Series(0.02, index=a_days), delay=0)
        pb = contract_pnl(pd.Series(1.0, index=b_days), pd.Series(-0.02, index=b_days), delay=0)
        valid = pd.DataFrame({"A": True, "B": True}, index=IDX)
        port = portfolio({"A": pa, "B": pb}, in_book(valid))
        self.assertTrue((port.n_contracts == 2).all())
        self.assertAlmostEqual(port.net.iloc[3], 0.01)  # A's +2% / 2, B closed contributes 0
        self.assertAlmostEqual(port.net.iloc[5], 0.0)
        self.assertAlmostEqual(port.gross_exposure.iloc[3], 1.0)  # B's position carried

    def test_book_entry(self):
        valid = pd.DataFrame({"A": [False, True, False, True]}, index=IDX[:4])
        self.assertEqual(in_book(valid).A.tolist(), [False, True, True, True])


class MetricTests(unittest.TestCase):
    def test_max_drawdown(self):
        r = pd.Series([0.1, -0.5, 0.2])  # equity 1.1 -> 0.55 -> 0.66
        self.assertAlmostEqual(max_drawdown(r), -0.5)

    def test_perf_sharpe(self):
        z = np.random.default_rng(1).normal(size=2000)
        z = (z - z.mean()) / z.std(ddof=1)  # exactly mean 0, sd 1
        r = pd.Series(0.0004 + 0.01 * z, index=pd.bdate_range("2000-01-01", periods=2000))
        p = perf(r)
        self.assertAlmostEqual(p["sharpe"], 0.0004 / 0.01 * np.sqrt(252))
        self.assertAlmostEqual(p["ann_vol"], 0.01 * np.sqrt(252))

    def test_identical_series_not_different(self):
        r = pd.Series(np.random.default_rng(2).normal(0, 0.01, 500))
        self.assertEqual(sharpe_diff_test(r, r)["p_two_sided"], 1.0)

    def test_paired_detects_clear_improvement(self):
        rng = np.random.default_rng(3)
        base = pd.Series(rng.normal(0.0002, 0.01, 5000))
        better = base + 0.0004 + rng.normal(0, 0.001, 5000)
        res = paired(better, base, reps=1000)
        self.assertGreater(res["nw_t_vm"], 5)
        self.assertLess(res["p_boot"], 0.01)
        self.assertGreater(res["ci_low"], 0)
        self.assertLess(res["min_gap_5pct"], res["d_sharpe"])

    def test_bootstrap_identical_and_deterministic(self):
        r = pd.Series(np.random.default_rng(4).normal(0, 0.01, 1000))
        res = block_bootstrap_sharpe_diff(r, r, reps=300)
        self.assertEqual((res["d_sharpe"], res["ci_low"], res["ci_high"], res["p_boot"]), (0.0, 0.0, 0.0, 1.0))
        other = r * 0.5 + pd.Series(np.random.default_rng(5).normal(0, 0.01, 1000))
        a = block_bootstrap_sharpe_diff(other, r, reps=300, seed=7)
        b = block_bootstrap_sharpe_diff(other, r, reps=300, seed=7)
        self.assertEqual(a, b)
        self.assertLessEqual(a["ci_low"], a["d_sharpe"])
        self.assertLessEqual(a["d_sharpe"], a["ci_high"])

    def test_vol_matched_mean_diff_ignores_leverage(self):
        rng = np.random.default_rng(6)
        base = pd.Series(rng.normal(0.0003, 0.01, 4000))
        quarter = base * 0.25  # same strategy at a quarter of the exposure
        res = paired(quarter, base, reps=200)
        self.assertAlmostEqual(res["ann_mean_diff_vm"], 0.0, places=10)
        self.assertAlmostEqual(res["d_sharpe"], 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
