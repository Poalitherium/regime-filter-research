"""Causality and rule checks for M4 (python -m unittest m4.tests.test_m4)."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from m4.common import event_days, event_state, fomc_dates, proportional_filter, vix_stress


class Calendar(unittest.TestCase):
    def test_fomc_file(self):
        d = fomc_dates()
        self.assertEqual(len(d), 112)
        self.assertTrue(d.is_unique and d.is_monotonic_increasing)
        self.assertTrue((d.year.value_counts() == 8).all())
        self.assertTrue((d.dayofweek < 5).all())
        for unscheduled in ("2001-01-03", "2001-09-17", "2008-01-22", "2008-10-08"):
            self.assertNotIn(pd.Timestamp(unscheduled), d)

    def test_close_timing(self):
        us, fgbl, fesx = event_days("ES"), event_days("FGBL"), event_days("FESX")
        self.assertIn(pd.Timestamp("2008-12-16"), us)
        self.assertIn(pd.Timestamp("2008-12-17"), fgbl)       # European close: next weekday
        self.assertIn(pd.Timestamp("2006-12-12"), fesx)       # FEZ-based (US close) before 2007-03-30
        self.assertIn(pd.Timestamp("2008-12-17"), fesx)
        self.assertIn(pd.Timestamp("2011-08-10"), fgbl)       # Tue 2011-08-09 -> Wed
        self.assertIn(pd.Timestamp("2012-09-14"), fgbl)       # Thu 2012-09-13 -> Fri

    def test_lead_is_two_weekdays(self):
        idx = pd.bdate_range("2008-12-10", "2008-12-19")
        s = event_state(idx, "ES", lead=2)
        self.assertEqual(list(idx[s.to_numpy()]), [pd.Timestamp("2008-12-12")])  # Fri -> Tue 16th


class Stress(unittest.TestCase):
    def test_expanding_quantile_is_causal(self):
        rng = np.random.default_rng(0)
        idx = pd.bdate_range("2000-01-03", periods=600)
        v = pd.DataFrame({"vix": rng.normal(size=600).cumsum()}, index=idx)
        a = vix_stress(v)
        v2 = v.copy()
        v2.iloc[400:, 0] += 100  # change the future only
        b = vix_stress(v2)
        pd.testing.assert_series_equal(a.iloc[:400], b.iloc[:400])


class Rule(unittest.TestCase):
    def _contracts(self, pnl):
        idx = pnl.index
        return {"X": {"data": pd.DataFrame(index=idx), "live": pd.Series(True, index=idx), "gross": pnl}}

    def test_worse_state_scaled_and_estimates_use_past_only(self):
        idx = pd.bdate_range("2000-01-03", "2006-12-29")
        state = pd.Series(np.arange(len(idx)) % 5 == 0, index=idx)
        pnl = pd.Series(np.where(state, 0.0005, 0.001), index=idx)  # in-state half as good
        f, log = proportional_filter(self._contracts(pnl), ["X"], {"X": state}, {"X": state})
        late = f[f.date >= "2004-01-01"]
        self.assertTrue(np.allclose(late[late.date.isin(idx[state])].filter_close_t, 0.5))
        self.assertTrue(np.allclose(late[~late.date.isin(idx[state])].filter_close_t, 1.0))
        self.assertTrue((f[f.date < "2003-01-01"].filter_close_t == 1.0).all())  # needs 3 years first
        # future P&L must not move earlier vintages
        pnl2 = pnl.copy()
        pnl2.loc["2005-01-01":] *= -10
        f2, _ = proportional_filter(self._contracts(pnl2), ["X"], {"X": state}, {"X": state})
        cut = f.date < "2005-01-01"
        np.testing.assert_allclose(f[cut].filter_close_t, f2[cut].filter_close_t)


class COT(unittest.TestCase):
    @unittest.skipUnless(Path('m4/data/cot_legacy_2014.csv').is_file(), 'CFTC input not downloaded')
    def test_subset_and_codes(self):
        from m4 import cot
        d = cot.load()
        self.assertEqual(set(d.code), set(cot.CFTC_CODES))
        self.assertLessEqual(d.report_date.max(), pd.Timestamp("2014-12-31"))   # hold-out never read
        self.assertGreaterEqual(d[d.code == "6E"].report_date.min(), pd.Timestamp("1999-01-01"))  # no ECU

    @unittest.skipUnless(Path('m4/data/cot_legacy_2014.csv').is_file(), 'CFTC input not downloaded')
    def test_availability_and_shutdown(self):
        from m4 import cot
        f = cot.weekly_features(cot.load())
        self.assertTrue(((f.available - f.report_date).dt.days == 7).all())
        self.assertFalse(f.report_date.between(*cot.SHUTDOWN).any())
        # a report is not usable before its available date
        idx = pd.bdate_range("2010-01-04", "2010-03-31")
        es = f[f.code == "ES"].set_index("available").pct
        p = cot.daily_pct(f, "ES", idx)
        first = es.loc["2010-01-04":].index[0]
        self.assertEqual(p.loc[first], es.loc[first])
        self.assertNotEqual(p.loc[first - pd.offsets.BDay(1)], es.loc[first])

    def test_rolling_pct_and_crowded(self):
        from m4 import cot
        x = pd.Series(np.arange(200, dtype=float))
        p = cot.rolling_pct(x)
        self.assertTrue(p.iloc[:103].isna().all())
        self.assertAlmostEqual(p.iloc[150], 1.0)
        t = pd.Series([1.0, 1.0, -1.0, -1.0, np.nan])
        q = pd.Series([0.95, 0.5, 0.05, 0.95, 0.95])
        np.testing.assert_array_equal(cot.crowded(t, q).to_numpy(), [1.0, 0.0, 1.0, 0.0, np.nan])


class ImpliedVol(unittest.TestCase):
    @unittest.skipUnless(all(Path(p).is_file() for p in (
        'm4/data/implied_vol_2014.csv', 'data/processed/offmarket/offmarket_daily.csv')),
        'Implied-volatility or macro input not downloaded')
    def test_lag_and_range(self):
        from m3.features import load_exog
        from m4 import impvol
        raw = pd.read_csv(impvol.IV_FILE, index_col="date", parse_dates=True)
        self.assertLessEqual(raw.index.max(), pd.Timestamp("2014-12-31"))
        iv = impvol.load_iv(load_exog(impvol.OFFMARKET_FILE))
        # the 2008-10-10 OVX close is first usable on Monday 2008-10-13
        self.assertEqual(iv.loc["2008-10-13", "ovx"], raw.loc["2008-10-10", "ovx"])
        self.assertEqual(iv.loc["2008-10-10", "ovx"], raw.loc["2008-10-09", "ovx"])  # not yet the 10th's close

    def test_filter_is_causal_and_capped(self):
        from m4 import impvol
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2003-01-01", periods=900)
        ret = pd.Series(rng.normal(0, 0.01, 900), index=idx)
        c = {"ES": {"data": pd.DataFrame({"ret": ret}), "live": pd.Series(True, index=idx)}}
        iv = pd.DataFrame({"vix": 20 + rng.normal(0, 2, 900)}, index=idx)
        f, _ = impvol.impvol_filter(c, iv)
        self.assertTrue((f.filter_close_t <= 1).all() and (f.filter_close_t >= 0).all())
        iv2 = iv.copy()
        iv2.iloc[600:, 0] *= 3  # change the future only
        f2, _ = impvol.impvol_filter(c, iv2)
        np.testing.assert_allclose(f.filter_close_t.iloc[:600], f2.filter_close_t.iloc[:600])


if __name__ == "__main__":
    unittest.main()
