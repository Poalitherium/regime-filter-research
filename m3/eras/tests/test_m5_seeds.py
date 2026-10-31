"""The M5 seed adapter (deviation #5): same seeds as m3.adaptor.period_seed, m5 left as committed."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import m5.pool  # noqa: E402
from m3.adaptor import period_seed  # noqa: E402
from m3.eras.m5_seeds import ValidSeedHMM, valid_seeds  # noqa: E402
from m5.hmm_pool import MultiSeqGaussianHMM  # noqa: E402


class M5SeedTests(unittest.TestCase):
    def test_seeds_equal_period_seed(self):
        for p in pd.period_range("1926Q3", "2014Q4", freq="Q"):
            hmm = ValidSeedHMM(n_states=3, n_init=4, seed=0 + p.ordinal)
            self.assertEqual(hmm.seed, period_seed(0, p))

    def test_patch_is_scoped(self):
        with valid_seeds():
            self.assertIs(m5.pool.MultiSeqGaussianHMM, ValidSeedHMM)
        self.assertIs(m5.pool.MultiSeqGaussianHMM, MultiSeqGaussianHMM)
        with self.assertRaises(RuntimeError), valid_seeds():
            raise RuntimeError
        self.assertIs(m5.pool.MultiSeqGaussianHMM, MultiSeqGaussianHMM)


if __name__ == "__main__":
    unittest.main()
