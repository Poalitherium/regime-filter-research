"""Truncated path signatures (depth 3) of rolling 2-D paths, for trial 7.

For each day t, the path is X_j = (j / w, sum of z over the last w days up to step j),
j = 0..w, built from the w most recent vol-standardised returns. Its signature is the
Chen product of the exponentials of the w linear segments:

    exp(D) = (1, D, D(x)D / 2, D(x)D(x)D / 6)
    (a*b)_1 = a1 + b1,  (a*b)_2 = a2 + a1(x)b1 + b2,  (a*b)_3 = a3 + a2(x)b1 + a1(x)b2 + b3

Level 2 holds the Levy area (S^12 - S^21)/2: S^12 = int t dX, S^21 = int X dt, so it is
positive when the cumulative return rose late (an accelerating trend) and negative when it
rose early and flattened (a decelerating trend).
Only z_{t-w+1..t} enter row t.
"""

import numba
import numpy as np
import pandas as pd

D = 2
NAMES = (["S1", "S2"] + [f"S{i}{j}" for i in (1, 2) for j in (1, 2)]
         + [f"S{i}{j}{k}" for i in (1, 2) for j in (1, 2) for k in (1, 2)])
CONSTANT = ("S1", "S11", "S111")  # time channel alone: 1, 1/2, 1/6


@numba.njit(cache=True)
def _signature(incs):
    """Depth-3 signature of a piecewise-linear 2-D path given its (n, 2) increments."""
    s1 = np.zeros(2)
    s2 = np.zeros((2, 2))
    s3 = np.zeros((2, 2, 2))
    for k in range(incs.shape[0]):
        d = incs[k]
        e2 = np.empty((2, 2))
        e3 = np.empty((2, 2, 2))
        for i in range(2):
            for j in range(2):
                e2[i, j] = d[i] * d[j] / 2.0
                for l in range(2):
                    e3[i, j, l] = d[i] * d[j] * d[l] / 6.0
        n3 = np.empty((2, 2, 2))
        for i in range(2):
            for j in range(2):
                for l in range(2):
                    n3[i, j, l] = s3[i, j, l] + s2[i, j] * d[l] + s1[i] * e2[j, l] + e3[i, j, l]
        n2 = np.empty((2, 2))
        for i in range(2):
            for j in range(2):
                n2[i, j] = s2[i, j] + s1[i] * d[j] + e2[i, j]
        s3 = n3
        s2 = n2
        for i in range(2):
            s1[i] += d[i]
    out = np.empty(14)
    out[0], out[1] = s1[0], s1[1]
    c = 2
    for i in range(2):
        for j in range(2):
            out[c] = s2[i, j]
            c += 1
    for i in range(2):
        for j in range(2):
            for l in range(2):
                out[c] = s3[i, j, l]
                c += 1
    return out


@numba.njit(cache=True)
def _rolling(z, w):
    T = len(z)
    out = np.full((T, 14), np.nan)
    incs = np.empty((w, 2))
    for t in range(w - 1, T):
        ok = True
        for j in range(w):
            v = z[t - w + 1 + j]
            if np.isnan(v):
                ok = False
                break
            incs[j, 0] = 1.0 / w
            incs[j, 1] = v
        if ok:
            out[t] = _signature(incs)
    return out


def signature_of_increments(incs: np.ndarray) -> np.ndarray:
    return _signature(np.asarray(incs, dtype=float))


def rolling_signature(z: pd.Series, window: int = 60) -> pd.DataFrame:
    """Signature terms of the trailing `window`-day path at every date; NaN until full."""
    arr = _rolling(z.to_numpy(dtype=float), window)
    return pd.DataFrame(arr, index=z.index, columns=[f"sig_{n}" for n in NAMES])


def features(z: pd.Series, window: int = 60) -> pd.DataFrame:
    """The 11 non-constant terms."""
    sig = rolling_signature(z, window)
    return sig.drop(columns=[f"sig_{c}" for c in CONSTANT])
