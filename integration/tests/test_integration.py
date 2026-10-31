"""The project trial family against the module tables it reads."""

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from integration.trial_family import TABLES, TRIAL_IDS, Z_80, check_m6, family, load


class TrialFamilyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(p.is_file() for p in TABLES.values()) or not Path('m6/results/all_trials.csv').is_file():
            raise unittest.SkipTest('Legacy result tables not generated')
        cls.f = family(load())

    def test_seventeen_trials_per_cost_model(self):
        for cost, s in self.f.groupby("cost_model"):
            self.assertEqual(list(s["trial"].astype(str)), TRIAL_IDS, cost)
            self.assertFalse(s["p_boot"].isna().any())

    def test_holm_over_12_matches_m6(self):
        self.assertEqual(check_m6(self.f), 24)

    def test_holm_over_17_is_never_below_holm_over_12(self):
        both = self.f.dropna(subset=["p_holm_12"])
        self.assertTrue((both["p_holm_17"] >= both["p_holm_12"] - 1e-12).all())

    def test_failed_gates_are_identity(self):
        ident = self.f[self.f["a3_equals_a2"]]
        self.assertEqual(sorted(set(ident["trial"].astype(str))), ["M4-1", "M4-2", "M4-3", "M4-5", "M5-3", "M5-4"])
        self.assertTrue(np.allclose(ident["d_sharpe"], 0.0))
        self.assertTrue(np.allclose(ident["p_boot"], 1.0))

    def test_static_tsmom_controls(self):
        static = set(self.f.loc[self.f["control"] != "A2", "trial"].astype(str))
        self.assertEqual(static, {"M3-5", "M3-8"})

    def test_mde_matches_m4s_own_column(self):
        m4 = pd.read_csv(TABLES["M4"]).query("trial == 'M4-4' and role == 'primary'").set_index("cost_model")
        mine = self.f[self.f["trial"] == "M4-4"].set_index("cost_model")
        for cost in ("research", "official"):
            self.assertAlmostEqual(mine.loc[cost, "mde_80"], m4.loc[cost, "mde_80"], places=5)
        self.assertAlmostEqual(Z_80, 0.841621, places=6)


class VolSizingTests(unittest.TestCase):
    """M3-5 and M3-8 size their own signals with m3.turning_points.vol_scale; it must equal
    M2's A2 scale (60-day vol, 15% target, 1% floor, cap 2) on every contract and day."""

    @unittest.skipUnless(Path('data/final/universe.csv').is_file(), 'Market data not downloaded')
    def test_m3_strategies_use_m2_scale(self):
        from m2.factors import generate_signals
        from m2.volatility import volatility_target
        from m3.turning_points import vol_scale
        from integration.trial_family import ROOT
        for code in pd.read_csv(ROOT / "data/final/universe.csv").code:
            d = pd.read_csv(ROOT / "data/final" / f"{code}.csv", index_col="date", parse_dates=True)
            m2 = volatility_target(generate_signals(d.close_adj), d.ret)["vol_scale"]
            m3 = vol_scale(d.ret)
            pd.testing.assert_series_equal(m3, m2, check_names=False, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
