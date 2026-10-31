"""Parsing and construction of the 1930 study data (PREREG_1930 section 2), on a synthetic French-format file."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from m3.eras.build_1930 import excess_returns, parse_factors, parse_industries  # noqa: E402

INDUSTRY_TEXT = """This file was created using the 202608 CRSP database.
  Average Value Weighted Returns -- Daily
,NoDur,Durbl
19260701,   1.00,   -99.99
19260702,   2.00,    0.50
20150102,   9.00,    9.00

  Average Equal Weighted Returns -- Daily
,NoDur,Durbl
19260701,   7.00,   7.00
"""
FACTORS_TEXT = """This file was created using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,    0.10,   -0.25,   -0.27,    0.01
19260702,    0.45,   -0.33,   -0.06,    0.01
20150102,    0.00,    0.00,    0.00,    0.00

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


class ParseTests(unittest.TestCase):
    def test_takes_the_value_weighted_section_only(self):
        ind = parse_industries(INDUSTRY_TEXT)
        self.assertEqual(list(ind.columns), ["NoDur", "Durbl"])
        self.assertEqual(len(ind), 3)
        self.assertEqual(ind.loc["1926-07-01", "NoDur"], 1.00)

    def test_missing_code_becomes_nan(self):
        self.assertTrue(np.isnan(parse_industries(INDUSTRY_TEXT).loc["1926-07-01", "Durbl"]))

    def test_factors_stop_before_the_copyright_line(self):
        fac = parse_factors(FACTORS_TEXT)
        self.assertEqual(list(fac.columns), ["Mkt-RF", "SMB", "HML", "RF"])
        self.assertEqual(len(fac), 3)

    def test_excess_returns_in_decimals_and_cut_at_2014(self):
        ex = excess_returns(parse_industries(INDUSTRY_TEXT), parse_factors(FACTORS_TEXT)["RF"])
        self.assertEqual(ex.index.max(), pd.Timestamp("1926-07-02"))  # 2015 row dropped
        self.assertAlmostEqual(ex.loc["1926-07-02", "NoDur"], (2.00 - 0.01) / 100)


if __name__ == "__main__":
    unittest.main()
