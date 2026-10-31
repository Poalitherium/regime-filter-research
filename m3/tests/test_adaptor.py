"""Regime-filter checks: causality, output range, state ordering, real-data smoke test."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from m3.adaptor import AdaptorConfig, run_contract  # noqa: E402
from m3.features import build_features  # noqa: E402


def regime_returns(n=2600, seed=3):
    """Alternating trending (drift, low noise) and choppy (no drift, high noise) blocks."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    trend = (np.arange(n) // 120) % 2 == 0
    r = np.where(trend, 0.002 + rng.normal(0, 0.006, n), rng.normal(0, 0.015, n))
    return pd.Series(r, index=idx), pd.Series(trend, index=idx)


class AdaptorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ret, cls.trend = regime_returns()
        cls.cfg = AdaptorConfig(n_states=2, min_train=500, n_init=2)
        cls.out, cls.fits = run_contract(cls.ret, cls.cfg)

    def test_features_are_causal(self):
        r2 = self.ret.copy()
        r2.iloc[1500:] *= 7
        f1, f2 = build_features(self.ret), build_features(r2)
        pd.testing.assert_frame_equal(f1.iloc[:1500], f2.iloc[:1500])

    def test_filter_is_causal(self):
        cut = self.ret.index[1800]  # mid-year: the next refit sees the change, earlier days must not
        r2 = self.ret.copy()
        r2[r2.index >= cut] = -r2[r2.index >= cut] * 3
        out2, _ = run_contract(r2, self.cfg)
        pd.testing.assert_frame_equal(self.out[self.out.index < cut], out2[out2.index < cut])

    def test_output_range_and_probabilities(self):
        o = self.out.dropna()
        self.assertTrue(o["filter"].between(0, 1).all())
        np.testing.assert_allclose(o[["p_state_0", "p_state_1"]].sum(axis=1), 1.0)
        self.assertTrue(self.out.loc[self.out.index.year < 2012, "filter"].isna().all())

    def test_states_ordered_by_trend_quality(self):
        for f in self.fits:
            self.assertLess(f["er60_0"], f["er60_1"])

    def test_filter_finds_trending_blocks(self):
        o = self.out.dropna()
        self.assertGreater(((o["filter"] > 0.5) == self.trend[o.index]).mean(), 0.8)

    def test_exogenous_features_are_causal_and_carried_forward(self):
        idx = self.ret.index
        exog = pd.DataFrame({"z": np.arange(len(idx), dtype=float)}, index=idx).iloc[::3]  # sparse rows
        f = build_features(self.ret, ("er20", "x:z"), exog)
        # never takes a value dated after t; carries the last known one forward
        known = exog["z"].reindex(idx).ffill(limit=5)
        pd.testing.assert_series_equal(f["x:z"], known.rename("x:z"))
        exog2 = exog.copy()
        exog2.loc[exog2.index >= idx[1500], "z"] = 999.0
        f2 = build_features(self.ret, ("er20", "x:z"), exog2)
        pd.testing.assert_frame_equal(f.iloc[:1500], f2.iloc[:1500])

    def test_adaptor_with_exogenous_feature_is_causal(self):
        idx = self.ret.index
        rng = np.random.default_rng(9)
        exog = pd.DataFrame({"z": rng.normal(size=len(idx)).cumsum()}, index=idx)
        cfg = AdaptorConfig(n_states=2, features=("logvol20", "er60", "x:z"), min_train=500, n_init=1)
        out1, _ = run_contract(self.ret, cfg, exog)
        cut = idx[1800]
        exog2 = exog.copy()
        exog2.loc[exog2.index >= cut, "z"] += 50
        out2, _ = run_contract(self.ret, cfg, exog2)
        pd.testing.assert_frame_equal(out1[out1.index < cut], out2[out2.index < cut])
        self.assertTrue(out1["filter"].dropna().between(0, 1).all())

    def test_missing_exogenous_column_is_an_error(self):
        with self.assertRaises(ValueError):
            build_features(self.ret, ("er20", "x:nope"), pd.DataFrame({"z": [1.0]}, index=self.ret.index[:1]))

    def test_real_contract_smoke(self):
        path = ROOT / "data" / "final" / "ZN.csv"
        if not path.is_file():
            self.skipTest("data/final not available")
        ret = pd.read_csv(path, index_col="date", parse_dates=True)["ret"]
        out, fits = run_contract(ret, AdaptorConfig(n_states=2, n_init=1))
        self.assertGreater(len(fits), 15)
        self.assertTrue(out["filter"].dropna().between(0, 1).all())


if __name__ == "__main__":
    unittest.main()
