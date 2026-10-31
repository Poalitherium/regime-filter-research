"""HMM engine checks on simulated data: python -m unittest discover -s m3/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3.hmm import GaussianHMM  # noqa: E402


def simulate(T=4000, seed=1):
    rng = np.random.default_rng(seed)
    A = np.array([[0.98, 0.02], [0.03, 0.97]])
    means = np.array([[0.0, -1.0], [1.5, 1.0]])
    covs = np.array([[[1.0, 0.3], [0.3, 1.0]], [[0.5, -0.1], [-0.1, 0.8]]])
    s = np.empty(T, dtype=int)
    s[0] = 0
    for t in range(1, T):
        s[t] = rng.choice(2, p=A[s[t - 1]])
    X = np.array([rng.multivariate_normal(means[k], covs[k]) for k in s])
    return X, s, A, means


class HMMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.s, cls.A, cls.means = simulate()
        cls.model = GaussianHMM(n_states=2, n_init=3, seed=0).fit(cls.X)
        order = np.argsort(cls.model.means[:, 0])  # match labels to the simulation
        cls.model.reorder(order)

    def test_recovers_parameters(self):
        np.testing.assert_allclose(self.model.means, self.means, atol=0.15)
        np.testing.assert_allclose(np.diag(self.model.A), np.diag(self.A), atol=0.015)
        self.assertTrue(self.model.converged)

    def test_loglik_never_decreases(self):
        h = np.array(self.model.history)
        self.assertTrue((np.diff(h) > -1e-6).all())

    def test_filter_tracks_true_states(self):
        p = self.model.filter(self.X)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)
        self.assertGreater(((p[:, 1] > 0.5) == (self.s == 1)).mean(), 0.95)

    def test_filter_is_causal(self):
        X2 = self.X.copy()
        X2[3000:] = X2[3000:] * -5 + 10
        np.testing.assert_array_equal(self.model.filter(self.X)[:3000], self.model.filter(X2)[:3000])

    def test_missing_rows_carry_prior(self):
        X2 = self.X.copy()
        X2[100:105] = np.nan
        p = self.model.filter(X2)
        self.assertTrue(np.isfinite(p).all())
        # with no data, the filtered probability drifts toward the stationary mix
        np.testing.assert_allclose(p[100], p[99] @ self.model.A)

    def test_warm_start_converges_fast(self):
        m = GaussianHMM(n_states=2, n_init=0, seed=0).fit(self.X, warm_start=self.model.params())
        self.assertLessEqual(m.n_iter, 3)
        self.assertAlmostEqual(m.loglik, self.model.loglik, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
