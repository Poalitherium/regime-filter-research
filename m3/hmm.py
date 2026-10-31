"""Gaussian hidden Markov model: Baum-Welch fitting and a causal forward filter.

Written from scratch (numpy + numba) rather than using hmmlearn because the regime
filter must use *filtered* probabilities P(state_t | x_1..x_t). Library predict_proba
returns smoothed probabilities P(state_t | x_1..x_T), which use future data.

Missing observations (any NaN in a row) get emission likelihood 1 in every state, so
the recursion carries the prior forward through them instead of breaking.
"""

from dataclasses import dataclass, field

import numba
import numpy as np

LOG_2PI = np.log(2 * np.pi)


# ---------------------------------------------------------------- recursions

@numba.njit(cache=True)
def _forward(B, A, pi):
    """Scaled forward pass. B[t, k] = emission likelihood (row-rescaled).
    Returns filtered probabilities alpha[t] = P(s_t | x_1..x_t) and scale factors c."""
    T, K = B.shape
    alpha = np.empty((T, K))
    c = np.empty(T)
    s = 0.0
    for k in range(K):
        alpha[0, k] = pi[k] * B[0, k]
        s += alpha[0, k]
    c[0] = s
    for k in range(K):
        alpha[0, k] /= s
    for t in range(1, T):
        s = 0.0
        for j in range(K):
            acc = 0.0
            for i in range(K):
                acc += alpha[t - 1, i] * A[i, j]
            alpha[t, j] = acc * B[t, j]
            s += alpha[t, j]
        c[t] = s
        for j in range(K):
            alpha[t, j] /= s
    return alpha, c


@numba.njit(cache=True)
def _forward_backward(B, A, pi):
    """E-step: smoothed state probabilities gamma, summed pair probabilities xi_sum,
    and the scale factors c (log-likelihood = sum(log c) + emission row offsets)."""
    T, K = B.shape
    alpha, c = _forward(B, A, pi)
    beta = np.empty((T, K))
    for k in range(K):
        beta[T - 1, k] = 1.0
    xi_sum = np.zeros((K, K))
    for t in range(T - 2, -1, -1):
        for i in range(K):
            acc = 0.0
            for j in range(K):
                w = A[i, j] * B[t + 1, j] * beta[t + 1, j] / c[t + 1]
                acc += w
                xi_sum[i, j] += alpha[t, i] * w
            beta[t, i] = acc
    gamma = alpha * beta
    return gamma, xi_sum, c


# ---------------------------------------------------------------- model

@dataclass
class GaussianHMM:
    """K-state HMM with full-covariance Gaussian emissions."""

    n_states: int
    n_init: int = 4  # random restarts (plus an optional warm start)
    max_iter: int = 500
    tol: float = 1e-7  # stop when the per-observation log-likelihood gain is below this
    cov_reg: float = 1e-3  # added to covariance diagonals (features are standardized)
    seed: int = 0

    pi: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    means: np.ndarray = field(default=None, repr=False)
    covs: np.ndarray = field(default=None, repr=False)
    loglik: float = field(default=np.nan)
    n_iter: int = 0
    converged: bool = False
    history: list = field(default_factory=list, repr=False)

    # ---- emissions

    def _log_emission(self, X: np.ndarray) -> np.ndarray:
        T, D = X.shape
        out = np.zeros((T, self.n_states))
        ok = ~np.isnan(X).any(axis=1)
        Xo = X[ok]
        for k in range(self.n_states):
            L = np.linalg.cholesky(self.covs[k])
            z = np.linalg.solve(L, (Xo - self.means[k]).T)
            logdet = 2 * np.log(np.diag(L)).sum()
            out[ok, k] = -0.5 * (D * LOG_2PI + logdet + (z**2).sum(axis=0))
        return out  # missing rows stay 0, i.e. likelihood 1 in every state

    def _scaled_emission(self, X):
        logB = self._log_emission(X)
        m = logB.max(axis=1, keepdims=True)
        return np.exp(logB - m), m.sum()

    # ---- EM

    def _random_init(self, X, rng):
        K, D = self.n_states, X.shape[1]
        Xo = X[~np.isnan(X).any(axis=1)]
        A = np.full((K, K), 0.1 / max(K - 1, 1))
        np.fill_diagonal(A, 0.9 if K > 1 else 1.0)
        cov = np.cov(Xo.T).reshape(D, D) + self.cov_reg * np.eye(D)
        return (np.full(K, 1.0 / K), A, Xo[rng.choice(len(Xo), K, replace=False)].copy(),
                np.repeat(cov[None], K, axis=0))

    def _em(self, X, params):
        self.pi, self.A, self.means, self.covs = (p.copy() for p in params)
        ok = ~np.isnan(X).any(axis=1)
        Xo, T, D = X[ok], len(X), X.shape[1]
        history, prev = [], -np.inf
        for it in range(1, self.max_iter + 1):
            B, offset = self._scaled_emission(X)
            gamma, xi_sum, c = _forward_backward(B, self.A, self.pi)
            ll = np.log(c).sum() + offset
            history.append(ll)
            if (ll - prev) / T < self.tol:
                return ll, it, True, history
            prev = ll
            # M-step
            self.pi = gamma[0] / gamma[0].sum()
            self.A = xi_sum / xi_sum.sum(axis=1, keepdims=True)
            g = gamma[ok]
            nk = g.sum(axis=0)
            for k in range(self.n_states):
                if nk[k] < D + 1:  # state died: restart it at a random observation
                    rng = np.random.default_rng(self.seed + it + k)
                    self.means[k] = Xo[rng.integers(len(Xo))]
                    self.covs[k] = np.cov(Xo.T).reshape(D, D) + self.cov_reg * np.eye(D)
                    continue
                self.means[k] = g[:, k] @ Xo / nk[k]
                d = Xo - self.means[k]
                self.covs[k] = (g[:, k, None] * d).T @ d / nk[k] + self.cov_reg * np.eye(D)
        B, offset = self._scaled_emission(X)  # log-likelihood of the final parameters
        _, c = _forward(B, self.A, self.pi)
        return np.log(c).sum() + offset, self.max_iter, False, history

    def fit(self, X: np.ndarray, warm_start: tuple | None = None) -> "GaussianHMM":
        """Baum-Welch with `n_init` random restarts (+ warm start); keeps the best fit."""
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.seed)
        inits = [self._random_init(X, rng) for _ in range(self.n_init)]
        if warm_start is not None:
            inits.insert(0, warm_start)
        best = None
        for params in inits:
            ll, n_iter, conv, hist = self._em(X, params)
            if best is None or ll > best[0]:
                best = (ll, n_iter, conv, hist, self.params())
        self.loglik, self.n_iter, self.converged, self.history, params = best
        self.pi, self.A, self.means, self.covs = params
        return self

    # ---- inference

    def params(self) -> tuple:
        return tuple(p.copy() for p in (self.pi, self.A, self.means, self.covs))

    def filter(self, X: np.ndarray) -> np.ndarray:
        """Filtered probabilities P(s_t | x_1..x_t). Row t never depends on rows > t."""
        B, _ = self._scaled_emission(np.asarray(X, dtype=float))
        alpha, _ = _forward(B, self.A, self.pi)
        return alpha

    def reorder(self, order: np.ndarray) -> None:
        """Relabel states so that new state i is old state order[i]."""
        self.pi = self.pi[order]
        self.A = self.A[np.ix_(order, order)]
        self.means, self.covs = self.means[order], self.covs[order]

    def n_params(self, n_features: int) -> int:
        K, D = self.n_states, n_features
        return (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2

    def bic(self, n_obs: int, n_features: int) -> float:
        return -2 * self.loglik + self.n_params(n_features) * np.log(n_obs)
