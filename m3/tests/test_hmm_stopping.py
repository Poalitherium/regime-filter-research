"""Behavioural checks for the isolated stopping-rule sensitivity runner."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from m3.hmm import GaussianHMM
from m3.hmm_stopping import StopDecision, StoppingHMM
from m3.adaptor import AdaptorConfig, run_contract
from m3.tests.test_hmm import simulate
from m3.tests.test_adaptor import regime_returns
from m5.hmm_pool import MultiSeqGaussianHMM
from m5.pool import run_pool


class Rules(unittest.TestCase):
    def test_large_negative_gain_is_not_abs_convergence(self):
        original, stable = StopDecision("original", 1e-7), StopDecision("abs3", 1e-7)
        self.assertTrue(original.update(2, -1e-4)[0])
        self.assertFalse(stable.update(2, -1e-4)[0])
        for i, gain in enumerate([-1e-8, 1e-8, -2e-8], 3):
            self.assertEqual(stable.update(i, gain)[0], i == 5)

    def test_extra_updates_are_counted_from_first_trigger(self):
        rule = StopDecision("extra50", 1e-7)
        for i in range(1, 54):
            stop, _, _ = rule.update(i, np.inf if i == 1 else (1e-4 if i == 2 else -1e-6))
            self.assertEqual(stop, i == 53)
        self.assertEqual(rule.trigger, 3)

    def test_original_reproduces_single_and_multi(self):
        X = simulate(T=900)[0]
        for seqs in [[X], [X[:400], X[400:]]]:
            if len(seqs) == 1:
                old = GaussianHMM(2, n_init=2, seed=7).fit(X)
            else:
                old = MultiSeqGaussianHMM(2, n_init=2, seed=7).fit_multi(seqs)
            new = StoppingHMM(2, n_init=2, seed=7, policy="original").fit_multi(seqs)
            for a, b in zip(old.params(), new.params()):
                np.testing.assert_array_equal(a, b)
            self.assertEqual(old.history, new.history)
            self.assertEqual(old.loglik, new.loglik)

    def test_iteration_cap_preserved(self):
        X = simulate(T=600)[0]
        for policy in ["original", "abs3", "extra50"]:
            model = StoppingHMM(2, n_init=1, max_iter=2, policy=policy).fit(X)
            self.assertFalse(model.converged)
            self.assertEqual(model.candidates[0]["reason"], "iteration_cap")

    def test_original_missing_and_warm_start(self):
        X = simulate(T=900)[0]
        X[50:55] = np.nan
        warm = GaussianHMM(2, n_init=1).fit(X[:500]).params()
        old = MultiSeqGaussianHMM(2, n_init=2, seed=3).fit_multi([X], warm)
        new = StoppingHMM(2, n_init=2, seed=3).fit_multi([X], warm)
        for a, b in zip(old.params(), new.params()):
            np.testing.assert_array_equal(a, b)

    def test_alternatives_are_causal_in_single_and_pool(self):
        ret, _ = regime_returns(n=1200)
        changed = ret.copy(); changed.iloc[950:] = -5 * changed.iloc[950:]
        cfg = AdaptorConfig(n_states=2, min_train=350, n_init=1, refit="quarter")
        for policy in ["abs3", "extra50"]:
            factory = lambda *args, **kw: StoppingHMM(*args, policy=policy, **kw)
            with patch("m3.adaptor.GaussianHMM", factory):
                before, _ = run_contract(ret, cfg)
                after, _ = run_contract(changed, cfg)
                pd.testing.assert_frame_equal(before.iloc[:950], after.iloc[:950])
            with patch("m5.pool.MultiSeqGaussianHMM", factory):
                before, _ = run_pool({"a": ret, "b": ret * .9}, cfg)
                after, _ = run_pool({"a": changed, "b": ret * .9}, cfg)
                for code in before:
                    pd.testing.assert_frame_equal(before[code].iloc[:950], after[code].iloc[:950])


if __name__ == "__main__":
    unittest.main()
