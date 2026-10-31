"""Construction rules of the eras proxy data (PREREG section 3.3): python -m unittest discover -s m3/eras/tests -t ."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from m3.eras.build_data import bad_ticks, bond_excess_returns, build_one, par_bond_price, price_returns  # noqa: E402

IDX = pd.bdate_range("1980-01-01", periods=200)


class BondTests(unittest.TestCase):
    def test_par_bond_prices_at_100(self):
        for y in (0.0, 0.03, 0.15):
            self.assertAlmostEqual(float(par_bond_price(np.array([y]), np.array([y]), 10)[0]), 100.0, places=10)

    def test_yield_up_price_down_with_duration(self):
        p = float(par_bond_price(np.array([0.05]), np.array([0.0501]), 10)[0])
        self.assertLess(p, 100)
        self.assertAlmostEqual((100 - p) / 100 / 0.0001, 7.8, delta=0.2)  # 10y par bond at 5%: duration about 7.8

    def test_flat_yields_earn_coupon_minus_bill(self):
        y = pd.Series(6.0, index=IDX)
        tb = pd.Series(4.0, index=IDX)
        r = bond_excess_returns(y, tb, 10)
        days = IDX.to_series().diff().dt.days
        expected = (0.06 - 0.04) * days / 365
        np.testing.assert_allclose(r.iloc[1:].to_numpy(), expected.iloc[1:].to_numpy(), atol=1e-12)
        self.assertTrue(np.isnan(r.iloc[0]))

    def test_bill_is_taken_from_the_previous_observation(self):
        y = pd.Series(5.0, index=IDX)
        tb = pd.Series(np.r_[np.full(100, 3.0), np.full(100, 9.0)], index=IDX)
        r = bond_excess_returns(y, tb, 5)
        d = (IDX[100] - IDX[99]).days
        self.assertAlmostEqual(r.iloc[100], (0.05 - 0.03) * d / 365)  # bill as of day 99, not day 100


class BadTickTests(unittest.TestCase):
    def series(self):
        rng = np.random.default_rng(0)
        return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(IDX)))), index=IDX)

    def test_spike_that_reverts_is_removed(self):
        s = self.series()
        s.iloc[150] *= 1.5  # one bad print: +50%, then back
        self.assertEqual(list(bad_ticks(s, price_returns)), [IDX[150]])

    def test_crash_without_reversal_is_kept(self):
        s = self.series()
        s.iloc[150:] *= 0.8  # -20% that stays
        self.assertEqual(len(bad_ticks(s, price_returns)), 0)

    def test_no_check_before_60_returns(self):
        s = self.series()
        s.iloc[30] *= 1.5
        self.assertEqual(len(bad_ticks(s, price_returns)), 0)

    def test_verified_real_move_is_kept_other_spikes_removed(self):
        # deviation #1: NKY 1987-10-20 is a real crash that rebounded; a spike elsewhere is still removed
        idx = pd.bdate_range("1987-01-02", "1988-06-30")
        rng = np.random.default_rng(1)
        s = pd.Series(20000 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        crash = idx.get_loc(pd.Timestamp("1987-10-20"))
        s.iloc[crash] *= 0.85                 # -15% on the day, back up the next day
        spike = idx.get_loc(pd.Timestamp("1988-03-15"))
        s.iloc[spike] *= 1.5                  # a bad print
        out, info, removed = build_one("NKY", s, None)
        self.assertIn(pd.Timestamp("1987-10-20"), out.index)
        self.assertNotIn(pd.Timestamp("1988-03-15"), out.index)
        self.assertEqual(info["flagged_but_kept"], "1987-10-20")
        self.assertEqual([d for d, _ in removed], [pd.Timestamp("1988-03-15").date()])


if __name__ == "__main__":
    unittest.main()
