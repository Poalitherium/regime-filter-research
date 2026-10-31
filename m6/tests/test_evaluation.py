
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from m6.analysis import COST_MODELS, load_trials
from m6.subperiod_power import Z_80
from backtest.metrics import Z_975


class TestM6Evaluation(unittest.TestCase):

    @unittest.skipUnless(all(Path(p).is_file() for p in (
        'm3/report/tables/trials_m1_engine.csv', 'm5/report/tables/trials_m1_engine.csv')),
        'Legacy result tables not generated')
    def test_holm_family_size(self):
        results = load_trials()

        expected_ids = {
            *(f"M3-{i}" for i in range(1, 9)),
            *(f"M5-{i}" for i in range(1, 5)),
        }

        for cost_model in COST_MODELS:
            sub = results[
                results["cost_model"] == cost_model
            ]

            self.assertEqual(len(sub), 12)
            self.assertEqual(
                set(sub["model_id"]),
                expected_ids,
            )

            self.assertFalse(
                sub["p_boot"].isna().any()
            )

    @unittest.skipUnless(all(Path(p).is_file() for p in (
        'm3/report/tables/trials_m1_engine.csv', 'm5/report/tables/trials_m1_engine.csv')),
        'Legacy result tables not generated')
    def test_failed_gate_identity_filters(self):
        results = load_trials()

        for cost_model in COST_MODELS:
            sub = results[
                (results["cost_model"] == cost_model)
                & (results["model_id"].isin(["M5-3", "M5-4"]))
            ]

            self.assertEqual(len(sub), 2)
            self.assertTrue(
                np.allclose(sub["d_sharpe"], 0.0)
            )
            self.assertTrue(
                np.allclose(sub["p_boot"], 1.0)
            )

    @unittest.skipUnless(all(Path(p).is_file() for p in (
        'm3/report/tables/trials_m1_engine.csv', 'm5/report/tables/trials_m1_engine.csv')),
        'Legacy result tables not generated')
    def test_holm_adjustment(self):
        results = load_trials()

        for cost_model in COST_MODELS:
            sub = results[
                results["cost_model"] == cost_model
            ]

            _, adjusted, _, _ = multipletests(
                sub["p_boot"].to_numpy(),
                alpha=0.05,
                method="holm",
            )

            bocpd_idx = np.where(
                sub["model_id"].to_numpy() == "M3-3"
            )[0][0]

            if cost_model == "research":
                expected = 0.7560
            else:
                expected = 0.5376

            self.assertAlmostEqual(
                adjusted[bocpd_idx],
                expected,
                places=4,
            )

    def test_mde_formula(self):
        se = 0.2
        mde = (Z_975 + Z_80) * se

        self.assertAlmostEqual(
            mde,
            0.5603,
            places=4,
        )


if __name__ == "__main__":
    unittest.main()
