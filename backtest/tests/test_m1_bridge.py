"""Bridge to M1's engine, on synthetic contracts in a temp dir: python -m unittest discover -s backtest/tests"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from backtest.m1_bridge import export_targets, filter_trial, m1_book, window_start  # noqa: E402
from backtest.official_costs import fee_bps, fee_per_contract  # noqa: E402
from m1.run_backtest import run as m1_run  # noqa: E402

IDX = pd.bdate_range("2010-01-01", periods=700)
FILTER_START = {"ES": 200, "ZN": 300, "6E": 400}  # first day each contract has an M3 filter (real codes: fees exist)
DEV_END = str(IDX[600].date())


def write_world(folder: Path):
    rng = np.random.default_rng(0)
    for k, code in enumerate(FILTER_START):
        ret = rng.normal(0.0003, 0.01, len(IDX))
        ret[0] = np.nan
        close = 100 * np.exp(np.nancumsum(np.r_[0, ret[1:]]))
        pd.DataFrame({"date": IDX.strftime("%Y-%m-%d"), "close": close, "close_adj": close, "ret": ret,
                      "volume": 1000, "roll": np.arange(len(IDX)) % 63 == 62, "kind": "futures"}) \
            .to_csv(folder / f"{code}.csv", index=False)
    pd.DataFrame({"code": list(FILTER_START), "asset_class": ["equity", "rates", "fx"]}) \
        .to_csv(folder / "universe.csv", index=False)
    rows = []
    for code, s in FILTER_START.items():
        f = np.full(len(IDX), np.nan)
        f[s:] = 0.5
        rows.append(pd.DataFrame({"date": IDX, "contract": code, "filter_close_t": f}))
    pd.concat(rows).to_csv(folder / "filter.csv", index=False)


class BridgeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        write_world(cls.dir)
        cls.targets, cls.start = export_targets(cls.dir, cls.dir / "filter.csv", min_contracts=2, dev_end=DEV_END)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def entry(self, code, ap):
        t = self.targets[(self.targets.contract == code) & (self.targets.strategy == ap)]
        return pd.Timestamp(t.date[t.target.notna()].iloc[0])

    def test_same_book_for_every_approach(self):
        for code in FILTER_START:
            a3 = self.entry(code, "A3")
            self.assertGreaterEqual(a3, IDX[FILTER_START[code]])
            self.assertEqual(self.entry(code, "A1"), max(a3, self.start))
            self.assertEqual(self.entry(code, "A2"), max(a3, self.start))

    def test_window_starts_with_enough_contracts_and_stops_at_dev_end(self):
        entries = sorted(self.entry(code, "A3") for code in FILTER_START)
        self.assertEqual(self.start, entries[1])  # second contract in the book (min_contracts=2)
        dates = pd.to_datetime(self.targets.date)
        self.assertEqual(dates.min(), self.start)
        self.assertEqual(dates.max(), pd.Timestamp(DEV_END))

    def test_m1_engine_reads_the_export(self):
        tfile = self.dir / "targets.csv"
        self.targets.to_csv(tfile, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m1_run(tfile, self.dir / "out", data_dir=self.dir, delay=1, start=str(self.start.date()), end=DEV_END)
        port = pd.read_csv(self.dir / "out" / "daily_portfolio.csv", parse_dates=["date"])
        self.assertEqual(set(port.strategy), {"A1", "A2", "A3"})
        self.assertEqual(port.date.max(), pd.Timestamp(DEV_END))
        a2, a3 = (port[port.strategy == s].set_index("date") for s in ("A2", "A3"))
        self.assertAlmostEqual(a3.gross.sum(), 0.5 * a2.gross.sum(), places=12)  # filter 0.5 everywhere in the book

    def test_bridge_engine_matches_m1_runner(self):
        tfile = self.dir / "targets_runner.csv"
        self.targets.to_csv(tfile, index=False)
        with contextlib.redirect_stdout(io.StringIO()):
            m1_run(tfile, self.dir / "runner", data_dir=self.dir, delay=1, start=str(self.start.date()), end=DEV_END)
        runner = pd.read_csv(self.dir / "runner" / "daily_portfolio.csv", parse_dates=["date"])
        data, asset, targets, book = filter_trial(self.dir, self.dir / "filter.csv")
        self.assertEqual(window_start(book, 2), self.start)
        for name, strategy in (("control", "A2"), ("treatment", "A3")):
            ours = m1_book(targets[name], data, asset, book, self.start, DEV_END, costs="research")
            theirs = runner[runner.strategy == strategy].set_index("date")
            pd.testing.assert_series_equal(ours.net, theirs.net.rename_axis(ours.index.name), check_freq=False,
                                           check_names=False, atol=1e-15, rtol=0)

    def test_official_costs_are_exact_fees_over_notional(self):
        fee, cur = fee_per_contract("ES")
        self.assertEqual((fee, cur), (2.247, "USD"))  # 1.386 + 0.85 + 0.011
        self.assertAlmostEqual(fee_bps(pd.Series([2000.0]), "ES").iloc[0], 1e4 * 2.247 / (2000 * 50))
        data, asset, targets, book = filter_trial(self.dir, self.dir / "filter.csv")
        p = {c: m1_book(targets["treatment"], data, asset, book, self.start, DEV_END, costs=c)
             for c in ("none", "official")}
        pd.testing.assert_series_equal(p["none"].gross, p["official"].gross)
        self.assertGreater(p["official"].cost.sum(), 0)
        self.assertTrue((p["none"].net - p["official"].net - p["official"].cost).abs().max() < 1e-15)


if __name__ == "__main__":
    unittest.main()
