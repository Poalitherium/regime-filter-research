"""Isolated stopping-rule sensitivity; production HMM modules are unchanged.

All emissions, initialisations, EM updates, regularisation, state ordering, and
restart selection retain the production implementation. Only stopping changes.
The original-policy branch is checked against both production HMM classes.
"""
import numpy as np

from m3.hmm import _forward_backward
from m5.hmm_pool import MultiSeqGaussianHMM

POLICIES = ("original", "abs3", "extra50")


class StopDecision:
    def __init__(self, policy, tol):
        if policy not in POLICIES:
            raise ValueError(policy)
        self.policy, self.tol = policy, tol
        self.stable = 0
        self.trigger = None

    def update(self, iteration, gain):
        self.stable = self.stable + 1 if np.isfinite(gain) and abs(gain) < self.tol else 0
        if self.trigger is None and gain < self.tol:
            self.trigger = iteration
        if self.policy == "original":
            return gain < self.tol, "legacy_gain", True
        if self.policy == "abs3":
            return self.stable >= 3, "absolute_stable_3", self.stable >= 3
        stop = self.trigger is not None and iteration >= self.trigger + 50
        return stop, "after_50_extra_updates", self.stable >= 3


class StoppingHMM(MultiSeqGaussianHMM):
    def __init__(self, *args, policy="original", callback=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy, self.callback = policy, callback
        self.candidates = []

    def _em_multi(self, Xs, oks, Xo, params):
        # Same arithmetic and ordering as m5.hmm_pool.MultiSeqGaussianHMM.
        # With one sequence this is also identical to m3.hmm.GaussianHMM.
        self.pi, self.A, self.means, self.covs = (p.copy() for p in params)
        K, D = self.n_states, Xo.shape[1]
        T = sum(len(X) for X in Xs)
        history, prev = [], -np.inf
        rule = StopDecision(self.policy, self.tol)
        negative_steps = 0
        minimum_gain = np.inf
        resets = 0
        for it in range(1, self.max_iter + 1):
            ll, g0, xi, gs = 0.0, np.zeros(K), np.zeros((K, K)), []
            for X, ok in zip(Xs, oks):
                B, offset = self._scaled_emission(X)
                gamma, xi_sum, c = _forward_backward(B, self.A, self.pi)
                ll += np.log(c).sum() + offset
                g0 += gamma[0]
                xi += xi_sum
                gs.append(gamma[ok])
            history.append(ll)
            gain = (ll - prev) / T
            if np.isfinite(gain):
                minimum_gain = min(minimum_gain, gain)
                negative_steps += int(gain < 0)
            stop, reason, converged = rule.update(it, gain)
            if stop:
                self._record(ll, it, converged, reason, rule, gain, minimum_gain, negative_steps, resets)
                return ll, it, converged, history
            prev = ll
            self.pi = g0 / g0.sum()
            self.A = xi / xi.sum(axis=1, keepdims=True)
            g = np.concatenate(gs)
            nk = g.sum(axis=0)
            for k in range(K):
                if nk[k] < D + 1:
                    resets += 1
                    rng = np.random.default_rng(self.seed + it + k)
                    self.means[k] = Xo[rng.integers(len(Xo))]
                    self.covs[k] = np.cov(Xo.T).reshape(D, D) + self.cov_reg * np.eye(D)
                    continue
                self.means[k] = g[:, k] @ Xo / nk[k]
                d = Xo - self.means[k]
                self.covs[k] = (g[:, k, None] * d).T @ d / nk[k] + self.cov_reg * np.eye(D)
        final_ll = self._loglik_multi(Xs)
        self._record(final_ll, self.max_iter, False, "iteration_cap", rule,
                     (final_ll - prev) / T, minimum_gain, negative_steps, resets)
        return final_ll, self.max_iter, False, history

    def _record(self, ll, it, converged, reason, rule, gain, minimum_gain, negative_steps, resets):
        self.candidates.append(dict(candidate=len(self.candidates), loglik=float(ll), n_iter=it,
            converged=bool(converged), reason=reason, first_legacy_trigger=rule.trigger,
            final_gain_per_row=float(gain), minimum_gain_per_row=float(minimum_gain),
            negative_steps=negative_steps, dead_state_resets=resets, seed=self.seed,
            policy=self.policy, stable_steps=rule.stable))

    def fit_multi(self, Xs, warm_start=None):
        self.candidates = []
        result = super().fit_multi(Xs, warm_start)
        winner = int(np.argmax([r["loglik"] for r in self.candidates]))
        for i, row in enumerate(self.candidates):
            row.update(selected=i == winner, is_warm_start=warm_start is not None and i == 0,
                       train_rows=sum(map(len, Xs)), n_sequences=len(Xs))
        if self.callback is not None:
            self.callback(self)
        return result
