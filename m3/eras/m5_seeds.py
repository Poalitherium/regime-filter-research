"""Run M5's pooled HMM on samples that start before 1970, without changing m5/.

m5.pool seeds each quarterly refit with `seed + period.ordinal`. Period ordinals count from
1970, so they are negative before it, and numpy rejects a negative seed. Inside
`valid_seeds()`, m5.pool builds its HMM from a subclass that maps a negative seed s to
s mod 2**32, the rule of m3.adaptor.period_seed (DEVIATIONS.md #4). Seeds from 1970 on are
unchanged, so nothing M5 itself reports can differ.
"""

import contextlib

import m5.pool
from m5.hmm_pool import MultiSeqGaussianHMM


class ValidSeedHMM(MultiSeqGaussianHMM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.seed < 0:
            self.seed %= 2**32


@contextlib.contextmanager
def valid_seeds():
    original = m5.pool.MultiSeqGaussianHMM
    m5.pool.MultiSeqGaussianHMM = ValidSeedHMM
    try:
        yield
    finally:
        m5.pool.MultiSeqGaussianHMM = original
