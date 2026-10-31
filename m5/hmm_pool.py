"""Gaussian HMM fitted on several independent sequences at once (multi-sequence Baum-Welch).

Each sequence gets its own forward-backward pass under shared parameters; the M-step
pools the sufficient statistics over sequences:

    pi    = sum_s gamma_s[0]              (normalised)
    A     = sum_s sum_t xi_s[t]           (row-normalised)
    mu_k  = sum_s sum_t gamma_s[t, k] x_s[t]          / n_k
    S_k   = sum_s sum_t gamma_s[t, k] (x - mu_k)(.)'  / n_k   (+ cov_reg I)

No transition is counted across sequence boundaries. The log-likelihood is the sum
over sequences. With one sequence every step is the same floating-point operation as
m3.hmm.GaussianHMM, so the fit is identical (tested).
"""

import numpy as np

from m3.hmm import GaussianHMM, _forward, _forward_backward


class MultiSeqGaussianHMM(GaussianHMM):
    """K-state full-covariance Gaussian HMM; `fit_multi` takes a list of sequences."""

    def _random_init_obs(self, Xo, rng):
        """GaussianHMM._random_init, given the stacked observed rows."""
        K, D = self.n_states, Xo.shape[1]
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9 if K > 1 else 1.0)
        cov = np.cov(Xo.T).reshape(D, D) + self.cov_reg * np.eye(D)
        return (np.full(K, 1.0 / K), A, Xo[rng.choice(len(Xo), K, replace=False)].copy(),
                np.repeat(cov[None], K, axis=0))

    def _loglik_multi(self, Xs):
        ll = 0.0
        for X in Xs:
            B, offset = self._scaled_emission(X)
            _, c = _forward(B, self.A, self.pi)
            ll += np.log(c).sum() + offset
        return ll

    def _em_multi(self, Xs, oks, Xo, params):
        self.pi, self.A, self.means, self.covs = (p.copy() for p in params)
        K, D = self.n_states, Xo.shape[1]
        T = sum(len(X) for X in Xs)
        history, prev = [], -np.inf
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
            if (ll - prev) / T < self.tol:
                return ll, it, True, history
            prev = ll
            # M-step on the pooled statistics
            self.pi = g0 / g0.sum()
            self.A = xi / xi.sum(axis=1, keepdims=True)
            g = np.concatenate(gs)
            nk = g.sum(axis=0)
            for k in range(K):
                if nk[k] < D + 1:  # state died: restart it at a random observation
                    rng = np.random.default_rng(self.seed + it + k)
                    self.means[k] = Xo[rng.integers(len(Xo))]
                    self.covs[k] = np.cov(Xo.T).reshape(D, D) + self.cov_reg * np.eye(D)
                    continue
                self.means[k] = g[:, k] @ Xo / nk[k]
                d = Xo - self.means[k]
                self.covs[k] = (g[:, k, None] * d).T @ d / nk[k] + self.cov_reg * np.eye(D)
        return self._loglik_multi(Xs), self.max_iter, False, history

    def fit_multi(self, Xs, warm_start: tuple | None = None) -> "MultiSeqGaussianHMM":
        """Baum-Welch over a list of (T_s, D) arrays with `n_init` random restarts
        (+ warm start); keeps the best total log-likelihood."""
        Xs = [np.asarray(X, dtype=float) for X in Xs]
        if not Xs:
            raise ValueError("need at least one sequence")
        oks = [~np.isnan(X).any(axis=1) for X in Xs]
        Xo = np.concatenate([X[ok] for X, ok in zip(Xs, oks)])
        rng = np.random.default_rng(self.seed)
        inits = [self._random_init_obs(Xo, rng) for _ in range(self.n_init)]
        if warm_start is not None:
            inits.insert(0, warm_start)
        best = None
        for params in inits:
            ll, n_iter, conv, hist = self._em_multi(Xs, oks, Xo, params)
            if best is None or ll > best[0]:
                best = (ll, n_iter, conv, hist, self.params())
        self.loglik, self.n_iter, self.converged, self.history, params = best
        self.pi, self.A, self.means, self.covs = params
        return self

    def fit(self, X, warm_start: tuple | None = None) -> "MultiSeqGaussianHMM":
        return self.fit_multi([X], warm_start)

    def score_multi(self, Xs) -> float:
        """Total log-likelihood of the current parameters over a list of sequences."""
        return self._loglik_multi([np.asarray(X, dtype=float) for X in Xs])
