"""BOCPD checks on simulated data: python -m unittest discover -s m3/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3.bocpd import BOCPDConfig, contract_features, run_bocpd, standardise  # noqa: E402


class BOCPDTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(11)
        self.x = np.r_[rng.normal(0, 1, 400), rng.normal(2.0, 1, 200)]  # mean break at t = 400

    def test_detects_a_clear_mean_break(self):
        cp, mu, rl = run_bocpd(self.x, BOCPDConfig(hazard=1 / 250))
        self.assertLess(np.median(cp[100:395]), 0.2)        # quiet before the break
        self.assertGreater(cp[400:415].max(), 0.8)          # a break is flagged within ~15 days
        self.assertGreater(mu[450], 1.5)                    # new segment's drift is picked up
        self.assertLess(abs(np.median(mu[200:395])), 0.3)
        self.assertGreater(rl[399], 150)                    # long run before the break

    def test_outputs_are_probabilities(self):
        cp, _, _ = run_bocpd(self.x, BOCPDConfig(k_recent=20))
        self.assertTrue(((cp >= 0) & (cp <= 1 + 1e-12)).all())
        cp_all, _, _ = run_bocpd(self.x, BOCPDConfig(k_recent=10_000))  # every run length
        np.testing.assert_allclose(cp_all, 1.0)

    def test_tiny_hazard_is_one_segment(self):
        _, mu, _ = run_bocpd(self.x[:300], BOCPDConfig(hazard=1e-12, kappa0=1.0, mu0=0.0))
        n = np.arange(1, 301)
        np.testing.assert_allclose(mu, np.cumsum(self.x[:300]) / (1.0 + n), atol=1e-8)

    def test_causal(self):
        x2 = self.x.copy()
        x2[300:] = -x2[300:] * 4
        a, b = run_bocpd(self.x), run_bocpd(x2)
        for u, v in zip(a, b):
            np.testing.assert_array_equal(u[:300], v[:300])

    def test_missing_days_carry_state(self):
        x2 = self.x.copy()
        x2[50:55] = np.nan
        cp, mu, _ = run_bocpd(x2)
        self.assertTrue(np.isfinite(cp[50:55]).all())
        self.assertEqual(cp[52], cp[49])
        self.assertTrue(np.isnan(run_bocpd(np.r_[np.nan, np.nan, 1.0])[0][:2]).all())

    def test_fold_bucket_keeps_mass(self):
        cp, _, rl = run_bocpd(self.x, BOCPDConfig(r_max=50, hazard=1e-6, k_recent=10_000))
        np.testing.assert_allclose(cp, 1.0)
        self.assertLessEqual(rl.max(), 49 + 1e-9)

    def test_standardise_uses_past_vol_only(self):
        idx = pd.bdate_range("2020-01-01", periods=200)
        r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 200), index=idx)
        r2 = r.copy()
        r2.iloc[120] = 0.5  # a shock on day 120 must not scale day 120 itself
        z1, z2 = standardise(r), standardise(r2)
        self.assertAlmostEqual(z2.iloc[120], 0.5 / r.iloc[60:120].std(ddof=1))
        pd.testing.assert_series_equal(z1.iloc[:120], z2.iloc[:120])
        f = contract_features(r)
        self.assertTrue(f.iloc[:60].cp20.isna().all() and f.iloc[61:].cp20.notna().all())


if __name__ == "__main__":
    unittest.main()
