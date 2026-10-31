"""Statistics of the eras run (PREREG section 6), on synthetic returns: python -m unittest discover -s m3/eras/tests -t ."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from backtest.metrics import perf  # noqa: E402
from m3.eras.run import decline, delta  # noqa: E402

IDX = pd.bdate_range("1980-01-01", "1999-12-31")


def pair(edge: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    c = pd.Series(rng.normal(0.0002, 0.005, len(IDX)), index=IDX)
    t = c + rng.normal(edge, 0.001, len(IDX))
    return t, c


class StatsTests(unittest.TestCase):
    def test_delta_is_the_sharpe_difference_on_the_slice(self):
        t, c = pair(0.0)
        r = delta(t, c, "1985-01-01", "1989-12-31")
        tt, cc = t.loc["1985":"1989"], c.loc["1985":"1989"]
        self.assertAlmostEqual(r["d_sharpe"], perf(tt)["sharpe"] - perf(cc)["sharpe"], places=12)
        self.assertEqual((r["start"], r["end"]), ("1985-01-01", "1989-12-29"))
        self.assertAlmostEqual(r["mde_80"], 2.801585 * r["se_boot"], places=5)

    def test_one_sided_p_small_for_a_clear_edge_and_large_against_it(self):
        t, c = pair(0.0004)
        self.assertLess(delta(t, c, "1980-01-01", "1999-12-31")["p_one_sided"], 0.01)
        self.assertGreater(delta(c, t, "1980-01-01", "1999-12-31")["p_one_sided"], 0.99)

    def test_decline_combines_independent_standard_errors(self):
        pre = {"d_sharpe": 0.30, "se_boot": 0.03}
        post = {"d_sharpe": 0.10, "se_boot": 0.04}
        h2 = decline(pre, post)
        self.assertAlmostEqual(h2["D"], 0.20)
        self.assertAlmostEqual(h2["se"], 0.05)
        self.assertAlmostEqual(h2["z"], 4.0)
        self.assertLess(h2["p_one_sided"], 1e-4)


if __name__ == "__main__":
    unittest.main()
