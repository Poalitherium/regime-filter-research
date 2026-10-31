"""Multi-sequence HMM and pooled adaptor: python -m unittest discover -s m5/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from m3.adaptor import AdaptorConfig, run_contract  # noqa: E402
from m3.hmm import GaussianHMM  # noqa: E402
from m5.hmm_pool import MultiSeqGaussianHMM  # noqa: E402
from m5.pool import run_pool  # noqa: E402

A_TRUE = np.array([[0.97, 0.02, 0.01], [0.02, 0.96, 0.02], [0.01, 0.03, 0.96]])
MEANS_TRUE = np.array([[-1.5, 0.0], [0.0, 1.0], [1.5, -0.5]])
COVS_TRUE = np.array([[[0.5, 0.1], [0.1, 0.5]], [[0.4, -0.1], [-0.1, 0.6]], [[0.6, 0.0], [0.0, 0.3]]])


def simulate(T, seed, A=A_TRUE, means=MEANS_TRUE, covs=COVS_TRUE):
    rng = np.random.default_rng(seed)
    K = len(A)
    s = np.empty(T, dtype=int)
    s[0] = rng.integers(K)
    for t in range(1, T):
        s[t] = rng.choice(K, p=A[s[t - 1]])
    X = np.array([rng.multivariate_normal(means[k], covs[k]) for k in s])
    return X, s


class SingleSequenceIdentity(unittest.TestCase):
    """With one sequence the pooled fit must be GaussianHMM's fit, bit for bit."""

    def check(self, X, **kw):
        ref = GaussianHMM(**kw).fit(X)
        new = MultiSeqGaussianHMM(**kw).fit_multi([X])
        for a, b in zip(ref.params(), new.params()):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(ref.loglik, new.loglik)
        self.assertEqual(ref.n_iter, new.n_iter)
        self.assertEqual(ref.history, new.history)
        np.testing.assert_array_equal(ref.filter(X), new.filter(X))

    def test_identical_fit(self):
        X, _ = simulate(1500, seed=1)
        self.check(X, n_states=3, n_init=3, seed=7)

    def test_identical_fit_with_missing_rows_and_warm_start(self):
        X, _ = simulate(1200, seed=2)
        X[50:60] = np.nan
        X[700, 1] = np.nan
        warm = GaussianHMM(n_states=3, n_init=1, seed=3).fit(X[:600]).params()
        ref = GaussianHMM(n_states=3, n_init=2, seed=4).fit(X, warm_start=warm)
        new = MultiSeqGaussianHMM(n_states=3, n_init=2, seed=4).fit_multi([X], warm_start=warm)
        for a, b in zip(ref.params(), new.params()):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(ref.loglik, new.loglik)

    def test_identical_when_max_iter_reached(self):
        X, _ = simulate(800, seed=5)
        self.check(X, n_states=3, n_init=2, seed=0, max_iter=3)


class MultiSequenceRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # three sequences from the same HMM, but each one too short to fit well on its own
        cls.seqs = [simulate(700, seed=s)[0] for s in (11, 12, 13)]
        cls.model = MultiSeqGaussianHMM(n_states=3, n_init=4, seed=0).fit_multi(cls.seqs)
        cls.model.reorder(np.argsort(cls.model.means[:, 0]))

    def test_recovers_parameters(self):
        np.testing.assert_allclose(self.model.means, MEANS_TRUE, atol=0.2)
        np.testing.assert_allclose(np.diag(self.model.A), np.diag(A_TRUE), atol=0.03)
        np.testing.assert_allclose(self.model.covs, COVS_TRUE, atol=0.15)

    def test_loglik_is_sum_over_sequences_and_never_decreases(self):
        # cov_reg (added to the M-step covariances, as in m3.hmm) makes EM only nearly monotone;
        # GaussianHMM on the concatenated data shows the same ~1e-8 relative dips
        h = np.array(self.model.history)
        self.assertTrue((np.diff(h) > -1e-7 * np.abs(h[1:])).all())
        total = sum(self.model.score_multi([X]) for X in self.seqs)
        self.assertAlmostEqual(self.model.score_multi(self.seqs), total, places=8)

    def test_no_transition_across_sequence_boundaries(self):
        # concatenating sequences adds one spurious transition per boundary: likelihoods differ
        joined = np.concatenate(self.seqs)
        self.assertNotAlmostEqual(self.model.score_multi([joined]), self.model.score_multi(self.seqs), places=6)

    def test_pooling_beats_single_short_sequence(self):
        err_pool = np.abs(self.model.means - MEANS_TRUE).max()
        solo = MultiSeqGaussianHMM(n_states=3, n_init=4, seed=0).fit_multi(self.seqs[:1])
        solo.reorder(np.argsort(solo.means[:, 0]))
        self.assertLessEqual(err_pool, np.abs(solo.means - MEANS_TRUE).max() + 0.05)


def regime_returns(n=2400, seed=3, shift=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2010-01-01", periods=n)
    trend = ((np.arange(n) + shift) // 120) % 2 == 0
    r = np.where(trend, 0.002 + rng.normal(0, 0.006, n), rng.normal(0, 0.015, n))
    return pd.Series(r, index=idx)


class PooledAdaptor(unittest.TestCase):
    CFG = AdaptorConfig(n_states=2, min_train=500, n_init=2, refit="year")

    def test_single_contract_pool_equals_m3_adaptor(self):
        ret = regime_returns()
        ref, ref_fits = run_contract(ret, self.CFG)
        outs, fits = run_pool({"X": ret}, self.CFG)
        pd.testing.assert_frame_equal(ref, outs["X"])
        self.assertEqual(len(ref_fits), len(fits))
        for a, b in zip(ref_fits, fits):
            for k, v in a.items():
                self.assertEqual(v, b[k], k)

    def test_pool_is_causal_for_every_member(self):
        rets = {"A": regime_returns(seed=3), "B": regime_returns(seed=4, shift=40)}
        out1, _ = run_pool(rets, self.CFG)
        cut = rets["A"].index[1700]  # mid-year
        r2 = dict(rets)
        r2["B"] = rets["B"].where(rets["B"].index < cut, -rets["B"] * 4)  # change only B's future
        out2, _ = run_pool(r2, self.CFG)
        for code in ("A", "B"):
            pd.testing.assert_frame_equal(out1[code][out1[code].index < cut], out2[code][out2[code].index < cut])
        # and the change does reach later output of the other member (parameters are shared)
        later = out1["A"].index >= pd.Timestamp(cut.year + 1, 1, 1)
        self.assertFalse(np.allclose(out1["A"].loc[later, "filter"], out2["A"].loc[later, "filter"]))

    def test_members_join_only_with_enough_history(self):
        a = regime_returns(seed=3)
        b = regime_returns(seed=5).iloc[900:]  # starts later
        outs, fits = run_pool({"A": a, "B": b}, self.CFG)
        first_b = outs["B"]["filter"].first_valid_index()
        n_valid_before = int((b.index < first_b).sum()) - 59  # er60 needs 60 returns
        self.assertGreaterEqual(n_valid_before, self.CFG.min_train)
        sizes = pd.DataFrame(fits).groupby("vintage").pool_size.first()
        self.assertEqual(sizes.iloc[0], 1)
        self.assertEqual(sizes.iloc[-1], 2)


if __name__ == "__main__":
    unittest.main()
