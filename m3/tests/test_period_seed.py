"""Refit seeds (eras study deviation #4): unchanged from 1970 on, valid before 1970."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3.adaptor import period_seed  # noqa: E402


class PeriodSeedTests(unittest.TestCase):
    def test_unchanged_from_1970_on(self):
        for p in pd.period_range("1970Q1", "2026Q4", freq="Q"):
            self.assertEqual(period_seed(0, p), p.ordinal)
        for p in pd.period_range("2000-01", "2001-12", freq="M"):
            self.assertEqual(period_seed(7, p), 7 + p.ordinal)

    def test_before_1970_is_a_valid_distinct_seed(self):
        quarters = pd.period_range("1926Q3", "1969Q4", freq="Q")
        seeds = [period_seed(0, p) for p in quarters]
        self.assertTrue(all(0 <= s < 2**32 for s in seeds))
        self.assertEqual(len(set(seeds)), len(seeds))
        np.random.default_rng(seeds[0])  # accepted by numpy


if __name__ == "__main__":
    unittest.main()
