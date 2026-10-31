"""Checks for trials 4-7: python -m unittest discover -s m3/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3.portfolio_vol import ewma_portfolio_vol, scale_factor  # noqa: E402
from m3.signature import NAMES, rolling_signature, signature_of_increments  # noqa: E402

IDX = pd.bdate_range("2010-01-01", periods=900)


class PortfolioVolTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rets = pd.DataFrame(rng.normal(0, 0.01, (900, 3)), index=IDX, columns=list("abc"))
        self.w = pd.DataFrame(1 / 3, index=IDX, columns=list("abc"))

    def test_matches_closed_form_for_independent_assets(self):
        s = ewma_portfolio_vol(self.rets, self.w)
        # 3 independent assets, daily sd 1%, weights 1/3: portfolio vol = 0.01*sqrt(3)/3*sqrt(252)
        self.assertAlmostEqual(s.iloc[300:].median(), 0.01 * np.sqrt(3) / 3 * np.sqrt(252), delta=0.015)

    def test_correlation_raises_portfolio_vol(self):
        common = self.rets[["a"]].to_numpy()
        corr_rets = pd.DataFrame(np.repeat(common, 3, axis=1), index=IDX, columns=list("abc"))
        self.assertGreater(ewma_portfolio_vol(corr_rets, self.w).iloc[300:].median(),
                           1.5 * ewma_portfolio_vol(self.rets, self.w).iloc[300:].median())

    def test_causal_and_scale_free(self):
        r2 = self.rets.copy()
        r2.iloc[600:] *= 5
        s1, s2 = ewma_portfolio_vol(self.rets, self.w), ewma_portfolio_vol(r2, self.w)
        pd.testing.assert_series_equal(s1.iloc[:600], s2.iloc[:600])
        k1, k2 = scale_factor(s1), scale_factor(s1 * 2)  # doubling every return leaves k unchanged
        pd.testing.assert_series_equal(k1, k2)
        self.assertTrue(k1.iloc[:250].isna().all() and (k1.dropna() <= 3).all())


class MetaLabelTests(unittest.TestCase):
    def test_walk_forward_purges_and_embargoes(self):
        from m3.metalabel import EMBARGO, walk_forward
        rng = np.random.default_rng(5)
        weeks = pd.date_range("2002-01-04", "2010-12-31", freq="W-FRI")
        rows = []
        for c in ("A", "B", "C"):
            x = rng.normal(size=len(weeks))
            rows.append(pd.DataFrame({"contract": c, "week": weeks, "x": x,
                                      "label_end": weeks + pd.Timedelta(days=30),
                                      "y": (x + rng.normal(size=len(weeks)) > 0).astype(float)}))
        panel = pd.concat(rows)
        seen = []
        import m3.metalabel as ml
        orig = ml.make_model

        class Spy:
            def __init__(self, kind):
                self.m = orig(kind)

            def fit(self, X, y):
                seen.append(X.index)
                self.m.fit(X, y)
                return self

            def predict_proba(self, X):
                return self.m.predict_proba(X)

        ml.make_model = Spy
        try:
            preds, aucs = walk_forward(panel.reset_index(drop=True), ["x"], "logit")
        finally:
            ml.make_model = orig
        self.assertGreater(aucs.auc.mean(), 0.6)  # signal is real, so the pipeline finds it
        for y, idx in zip(aucs.year, seen):
            refit = pd.Timestamp(int(y), 1, 1)
            ends = panel.reset_index(drop=True).loc[idx, "label_end"]
            self.assertTrue((ends < refit - pd.tseries.offsets.BDay(EMBARGO)).all())
        self.assertTrue((preds.week >= pd.Timestamp(2006, 1, 1)).all())


class SignatureTests(unittest.TestCase):
    def sig(self, incs):
        return dict(zip(NAMES, signature_of_increments(np.array(incs, float))))

    def test_straight_line_is_tensor_exponential(self):
        d = np.array([0.3, -1.2])
        s = self.sig([d / 4] * 4)  # four collinear pieces = one segment
        self.assertAlmostEqual(s["S2"], -1.2)
        self.assertAlmostEqual(s["S12"], 0.3 * -1.2 / 2)
        self.assertAlmostEqual(s["S21"], 0.3 * -1.2 / 2)
        self.assertAlmostEqual(s["S122"], 0.3 * 1.44 / 6)
        self.assertAlmostEqual(s["S111"], 0.3 ** 3 / 6)

    def test_levy_area_sign(self):
        late = self.sig([[1, 0], [0, 1]])   # time first, then the rise: accelerating
        early = self.sig([[0, 1], [1, 0]])  # rise first, then flat: decelerating
        self.assertAlmostEqual(late["S12"] - late["S21"], 1.0)
        self.assertAlmostEqual(early["S12"] - early["S21"], -1.0)

    def test_chen_identity_against_brute_force(self):
        rng = np.random.default_rng(3)
        incs = rng.normal(size=(7, 2))
        s = self.sig(incs)
        # level 2 by direct iterated sums over a piecewise-linear path
        x = np.vstack([[0, 0], np.cumsum(incs, axis=0)])
        brute = {}
        for i in (0, 1):
            for j in (0, 1):
                tot = 0.0
                for k in range(len(incs)):
                    tot += (x[k, i] - x[0, i]) * incs[k, j] + incs[k, i] * incs[k, j] / 2
                brute[f"S{i + 1}{j + 1}"] = tot
        for key, v in brute.items():
            self.assertAlmostEqual(s[key], v)

    def test_rolling_is_causal(self):
        z = pd.Series(np.random.default_rng(4).normal(size=300), index=pd.bdate_range("2020-01-01", periods=300))
        z2 = z.copy()
        z2.iloc[200:] *= -3
        a, b = rolling_signature(z, 60), rolling_signature(z2, 60)
        pd.testing.assert_frame_equal(a.iloc[:200], b.iloc[:200])
        self.assertTrue(a.iloc[:59].isna().all().all() and a.iloc[59:].notna().all().all())


if __name__ == "__main__":
    unittest.main()
