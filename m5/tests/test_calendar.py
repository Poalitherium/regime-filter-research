"""Calendar states and the calendar filter: python -m unittest discover -s m5/tests"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from m5 import calendar as cal  # noqa: E402
from m5 import calfilter  # noqa: E402
from m5.seasonality import monthly_sample  # noqa: E402


class LastTradingDays(unittest.TestCase):
    KNOWN = [  # (code, contract year, month, exchange last trading day)
        ("ES", 2007, 3, "2007-03-16"), ("ES", 2010, 12, "2010-12-17"),
        ("CL", 2020, 5, "2020-04-21"), ("CL", 2015, 3, "2015-02-20"),
        ("NG", 2021, 2, "2021-01-27"), ("NG", 2014, 1, "2013-12-27"),
        ("ZN", 2010, 3, "2010-03-22"), ("6E", 2012, 3, "2012-03-19"),
        ("GC", 2010, 12, "2010-12-29"), ("ZC", 2006, 7, "2006-07-14"),
        ("NK", 2010, 6, "2010-06-10"), ("SI", 2012, 7, "2012-07-27"),
    ]

    def test_known_expiries(self):
        for code, y, m, ltd in self.KNOWN:
            rule = cal.SCHEDULES[code][1]
            self.assertEqual(rule(y, m), pd.Timestamp(ltd), f"{code} {y}-{m}")

    def test_holidays_are_not_modelled(self):
        # documented limitation (PREREG 4.5): weekdays only, so a date next to an exchange
        # holiday can be one weekday off. CL Jan-2016 really expired on 2015-12-21 because
        # the 25th was Christmas; the weekday rule gives the 22nd.
        self.assertEqual(cal.ltd_cl(2016, 1), pd.Timestamp("2015-12-22"))

    def test_switch_day_is_next_weekday(self):
        sw = cal.roll_days("ES", "2007-03-01", "2007-03-31")
        self.assertEqual(list(sw), [pd.Timestamp("2007-03-19")])  # LTD Fri 16th -> Mon 19th

    def test_cycles(self):
        months = lambda code: sorted(set(cal.last_trading_days(code, "2010-01-01", "2010-12-31").month))  # noqa: E731
        self.assertEqual(months("ES"), [3, 6, 9, 12])
        self.assertEqual(months("ZS"), [1, 3, 5, 7, 8, 9, 11])
        self.assertEqual(len(cal.last_trading_days("CL", "2010-01-01", "2010-12-31")), 12)


class CalendarStates(unittest.TestCase):
    IDX = pd.bdate_range("2005-01-03", "2012-12-31")

    def test_days_to_roll_counts(self):
        idx = pd.bdate_range("2007-03-12", "2007-03-23")
        d = cal.days_to_roll(idx, "ES")
        sw = pd.Timestamp("2007-03-19")
        self.assertEqual(int(d.loc[sw, "d_next"]), 0)
        self.assertEqual(int(d.loc[sw, "d_last"]), 0)
        self.assertEqual(int(d.loc[pd.Timestamp("2007-03-16"), "d_next"]), 1)
        self.assertEqual(int(d.loc[pd.Timestamp("2007-03-22"), "d_last"]), 3)
        w = cal.roll_window(idx, "ES", 3)
        self.assertEqual(list(idx[w.to_numpy()]), list(pd.bdate_range("2007-03-14", "2007-03-22")))

    def test_states_are_pointwise_functions_of_the_date(self):
        """No future (or past) information: the value at t is the same whatever other
        dates are in the index, so truncating or extending the data changes nothing."""
        for fn in (lambda i: cal.roll_window(i, "CL", 3, lead=2), lambda i: cal.in_season(i, "NG", lead=2),
                   lambda i: cal.days_to_roll(i, "ZC")):
            full = fn(self.IDX)
            cut = self.IDX[self.IDX < "2009-06-15"]
            pd.testing.assert_series_equal(fn(cut).iloc[:, 0] if isinstance(full, pd.DataFrame) else fn(cut),
                                           (full.iloc[:, 0] if isinstance(full, pd.DataFrame) else full).loc[cut])
            sparse = self.IDX[::7]
            pd.testing.assert_series_equal(fn(sparse).iloc[:, 0] if isinstance(full, pd.DataFrame) else fn(sparse),
                                           (full.iloc[:, 0] if isinstance(full, pd.DataFrame) else full).loc[sparse])

    def test_lead_shifts_by_weekdays(self):
        fri = pd.DatetimeIndex(["2010-05-28"])  # Friday; +2 weekdays = Tue 1 June (in season)
        self.assertFalse(cal.in_season(fri, "ZC", lead=0).iloc[0])
        self.assertTrue(cal.in_season(fri, "ZC", lead=2).iloc[0])
        sh = cal.shift_weekdays(self.IDX, 2)
        self.assertTrue((sh.weekday < 5).all())
        self.assertTrue(((sh - self.IDX).days >= 2).all())

    def test_seasons(self):
        s = cal.in_season(pd.date_range("2010-01-01", "2010-12-31", freq="MS"), "NG")
        self.assertEqual(list(s.index.month[s.to_numpy()]), [1, 2, 3, 11, 12])


def fake_contract(seed, start="2001-01-01", n=3000, roll_noise=False):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start, periods=n)
    ret = pd.Series(rng.normal(0, 0.01, n), index=idx)
    target = pd.Series(np.sign(rng.normal(size=n)) * 0.5, index=idx)
    roll = pd.Series(rng.random(n) < (0.5 if roll_noise else 0.02), index=idx)
    pos = target.shift(2).fillna(0.0)
    return {"data": pd.DataFrame({"ret": ret, "roll": roll}), "asset": "agri", "target": target,
            "gross": pos * ret, "net": pos * ret, "position": pos, "live": target.shift(2).notna(),
            "logvol20": np.log(ret.rolling(20).std() * np.sqrt(252))}


class CalendarFilter(unittest.TestCase):
    def test_proportional_rule(self):
        self.assertEqual(calfilter.proportional(-0.1, -0.2), (1.0, 1.0))
        self.assertEqual(calfilter.proportional(0.2, 0.1), (1.0, 0.5))
        self.assertEqual(calfilter.proportional(-0.1, 0.2), (0.0, 1.0))
        self.assertEqual(calfilter.proportional(0.0, 0.4), (0.0, 1.0))
        self.assertEqual(calfilter.proportional(np.nan, 0.4), (1.0, 1.0))

    def build(self, contracts):
        state = lambda idx, code, lead: cal.in_season(idx, "ZC", lead)  # noqa: E731
        return calfilter.build(contracts, {"g": ["A", "B"]}, state, lead=2)

    def test_filter_is_causal(self):
        cs = {"A": fake_contract(1), "B": fake_contract(2), "C": fake_contract(3)}
        f1, _ = self.build(cs)
        cut = pd.Timestamp("2008-05-14")  # mid-quarter
        cs2 = {k: dict(v) for k, v in cs.items()}
        for k in ("A", "B"):
            g = cs[k]["gross"].copy()
            g[g.index >= cut] = -10 * g[g.index >= cut] + 0.05
            cs2[k]["gross"] = g
        f2, _ = self.build(cs2)
        before = f1.date < cut
        pd.testing.assert_frame_equal(f1[before], f2[before])
        self.assertFalse(np.allclose(f1[~before].filter_close_t, f2[~before].filter_close_t))

    def test_filter_ignores_roll_column_and_ungrouped_contracts_get_one(self):
        cs = {"A": fake_contract(1), "B": fake_contract(2), "C": fake_contract(3)}
        f1, _ = self.build(cs)
        cs2 = {k: dict(v) for k, v in cs.items()}
        for k in cs2:
            d = cs2[k]["data"].copy()
            d["roll"] = ~d["roll"]
            cs2[k]["data"] = d
        f2, _ = self.build(cs2)
        pd.testing.assert_frame_equal(f1, f2)
        self.assertTrue((f1[f1.contract == "C"].filter_close_t == 1).all())
        self.assertTrue(f1.filter_close_t.between(0, 1).all())

    def test_one_until_three_years_then_state_dependent(self):
        cs = {"A": fake_contract(1), "B": fake_contract(2)}
        for k in cs:  # make in-season P&L clearly worse so the rule bites
            g = cs[k]["gross"]
            s = cal.in_season(g.index, "ZC")
            cs[k]["gross"] = g.where(~s, g - 0.002) + 0.0005
        f, log = self.build(cs)
        f = f.set_index("date")
        early = f[f.index < pd.Timestamp("2004-01-01")]
        self.assertTrue((early.filter_close_t == 1).all())
        late = f[f.index >= pd.Timestamp("2006-01-01")]
        st = cal.in_season(late.index, "ZC", lead=2).to_numpy()
        self.assertTrue((late.filter_close_t[st] < 1).all())
        self.assertTrue((late.filter_close_t[~st] == 1).all())
        self.assertTrue((pd.to_datetime(log.vintage) >= pd.Timestamp("2004-01-01")).all())


class SeasonSample(unittest.TestCase):
    def test_next_month_alignment(self):
        c = fake_contract(4, start="2003-01-01", n=1500)
        c["net"] = pd.Series(1.0, index=c["data"].index)  # 1 per day: y = trading days next month
        df = monthly_sample(c, "ZC", dev_end="2008-12-31")
        for end, row in df.iloc[:-1].iterrows():
            nxt = end.to_period("M") + 1
            days = (c["data"].index.to_period("M") == nxt).sum()
            self.assertEqual(row.y, days)
            self.assertEqual(bool(row.season), nxt.month in (6, 7, 8))
            self.assertEqual(row.logvol, c["logvol20"].loc[end])


if __name__ == "__main__":
    unittest.main()
