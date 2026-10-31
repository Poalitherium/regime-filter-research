"""Phase 3 code path on synthetic new contracts (random walks in a temp dir, never data/)."""
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from m3.adaptor import AdaptorConfig  # noqa: E402
from m5 import confirm  # noqa: E402

DEV = ROOT / "data" / "final"
FAST = AdaptorConfig(n_states=3, refit="year", mapping="top", n_init=1)


def write_contract(folder: Path, code: str, seed: int, start="2003-01-02", n=2600):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = rng.normal(0.0002, 0.012, n)
    ret[0] = np.nan
    close = 100 * np.exp(np.nancumsum(np.r_[0, ret[1:]]))
    pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "close": close, "close_adj": close, "ret": ret,
                  "volume": 1000, "roll": False, "kind": "futures"}).to_csv(folder / f"{code}.csv", index=False)


@unittest.skipUnless((DEV / "universe.csv").is_file(), "data/final not available")
class ConfirmPath(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_contract(cls.dir, "HG", 1)
        write_contract(cls.dir, "ZB", 2)
        pd.DataFrame({"code": ["HG", "ZB"], "asset_class": ["metal", "rates"]}).to_csv(cls.dir / "universe.csv", index=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_rejects_development_contracts(self):
        bad = self.dir / "bad"
        bad.mkdir(exist_ok=True)
        pd.DataFrame({"code": ["GC"], "asset_class": ["metal"]}).to_csv(bad / "universe.csv", index=False)
        with self.assertRaises(ValueError):
            confirm.new_universe(bad, DEV)

    def test_pool_class_filter_and_one_sided_test(self):
        filt = confirm.build_filter("pool_class", self.dir, DEV, FAST)
        self.assertEqual(set(filt.contract), {"HG", "ZB"})
        self.assertTrue(filt.filter_close_t.dropna().between(0, 1).all())
        self.assertGreater(filt.filter_close_t.notna().mean(), 0.5)
        f = self.dir / "f.csv"
        filt.to_csv(f, index=False)
        res = confirm.test(self.dir, f, direction=1, min_contracts=1)
        self.assertEqual(res["contracts"], ["HG", "ZB"])
        self.assertTrue(0 <= res["p_one_sided"] <= 1)
        self.assertNotIn("reject_H0_at_5pct", res)  # the decision is added only for a confirmatory run
        res_neg = confirm.test(self.dir, f, direction=-1, min_contracts=1)
        self.assertAlmostEqual(res["d_sharpe"], res_neg["d_sharpe"])
        self.assertLessEqual(abs(res["p_one_sided"] + res_neg["p_one_sided"] - 1), 0.01)

    def test_every_method_is_descriptive(self):
        # PREREG §6: no M5 trial was significant on the development window
        self.assertEqual(set(confirm.DEV_SIGNIFICANT), {"pool_class", "pool_global", "expiry", "season"})
        self.assertFalse(any(confirm.DEV_SIGNIFICANT.values()))

    def test_expiry_builds_for_scheduled_codes(self):
        filt = confirm.build_filter("expiry", self.dir, DEV)
        self.assertEqual(set(filt.contract), {"HG", "ZB"})
        self.assertTrue(filt.filter_close_t.between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
