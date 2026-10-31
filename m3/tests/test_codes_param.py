"""The optional market list of run_bocpd / run_real_data (eras PREREG section 4).

The defaults must stay the 13 data/final contracts (their full outputs were checked byte for
byte against the earlier code, see eras/CHECKS.md); another list runs only those files, and a
contract's output does not depend on which other contracts are run with it.
"""
import contextlib
import inspect
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3 import run_bocpd, run_real_data  # noqa: E402
from m3.adaptor import AdaptorConfig  # noqa: E402
from m3.bocpd import BOCPDConfig  # noqa: E402
from m3.run_real_data import CODES  # noqa: E402


def write_contracts(folder: Path, codes, n=1100):
    idx = pd.bdate_range("1990-01-01", periods=n)
    for k, code in enumerate(codes):
        rng = np.random.default_rng(k)
        ret = rng.normal(0.0002, 0.01, n)
        ret[0] = np.nan
        close = 100 * np.exp(np.nancumsum(np.r_[0, ret[1:]]))
        pd.DataFrame({"date": idx, "close": close, "close_adj": close, "ret": ret, "volume": np.nan,
                      "roll": False, "kind": "proxy"}).to_csv(folder / f"{code}.csv", index=False)


class CodesParameterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_contracts(cls.dir, ("AAA", "BBB", "CCC"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def quiet(self, fn, *args, **kw):
        with contextlib.redirect_stdout(io.StringIO()):
            return fn(*args, **kw)

    def test_defaults_are_the_13_contracts(self):
        self.assertEqual(len(CODES), 13)
        self.assertEqual(inspect.signature(run_bocpd.run).parameters["codes"].default, CODES)
        self.assertEqual(inspect.signature(run_real_data.run).parameters["codes"].default, CODES)

    def test_bocpd_runs_only_the_given_codes_and_each_independently(self):
        both = self.quiet(run_bocpd.run, self.dir, self.dir / "b2", BOCPDConfig(), ("AAA", "BBB"))
        alone = self.quiet(run_bocpd.run, self.dir, self.dir / "b1", BOCPDConfig(), ("AAA",))
        self.assertEqual(set(both.contract), {"AAA", "BBB"})
        a = both[both.contract == "AAA"].reset_index(drop=True)
        pd.testing.assert_frame_equal(a, alone.reset_index(drop=True))

    def test_hmm_runs_only_the_given_codes(self):
        cfg = AdaptorConfig(n_states=2, refit="year", mapping="top", min_train=250, n_init=1)
        filt, fits = self.quiet(run_real_data.run, self.dir, self.dir / "h", cfg, codes=("BBB", "CCC"))
        self.assertEqual(set(filt.contract), {"BBB", "CCC"})
        self.assertEqual(set(fits.contract), {"BBB", "CCC"})


if __name__ == "__main__":
    unittest.main()
