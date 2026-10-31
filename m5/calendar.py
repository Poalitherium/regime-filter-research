"""Deterministic contract calendars: scheduled roll days and seasons.

Everything here is a function of the date alone (exchange rules published in advance),
so it is known before any price is seen. It never reads the data's `roll` column:
that column is detected after the fact ("doesn't reverse the next day") and so looks
ahead.

Scheduled roll day ("switch day")
    The first weekday after the expiring contract's last trading day (LTD). From that
    day the continuous series holds the next contract in the cycle. This is the
    convention of the Yahoo `=F` series behind data/final (checked against the
    detected rolls in `m5.calendar_qa`, never used as an input).

Business days are weekdays (Mon-Fri) with no holiday calendar, so a date can be off by
one weekday next to an exchange holiday. Windows are wide enough to absorb that.

Seasons (fixed from agronomic / energy calendars, before looking at returns):
    ZC, ZS : US corn and soybean weather market, June-August
    NG     : US natural-gas storage withdrawal season, November-March
"""

import numpy as np
import pandas as pd

MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
               "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}
ALL_MONTHS = "FGHJKMNQUVXZ"
BD = pd.offsets.BDay()


# ---------------------------------------------------------------- date helpers

def _nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    """n-th (1-based) weekday (Mon=0) of a month."""
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.weekday()) % 7 + 7 * (n - 1))


def _last_bday(year: int, month: int) -> pd.Timestamp:
    return pd.Timestamp(year, month, 1) + pd.offsets.BMonthEnd(0)


def _bday_on_or_after(d: pd.Timestamp) -> pd.Timestamp:
    return d if d.weekday() < 5 else d + BD


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


# ---------------------------------------------------------------- last trading day rules
# Each takes the contract (delivery) year and month and returns the LTD.

def ltd_third_friday(y, m):              # CME equity index (ES, NQ); Eurex FESX
    return _nth_weekday(y, m, 4, 3)


def ltd_bday_before_second_friday(y, m):  # CME NKD
    return _nth_weekday(y, m, 4, 2) - BD


def ltd_cl(y, m):
    """NYMEX WTI: 3 business days before the 25th of the month before delivery
    (if the 25th is not a business day, 3 business days before the last business day
    preceding the 25th)."""
    py, pm = _prev_month(y, m)
    d25 = pd.Timestamp(py, pm, 25)
    anchor = d25 if d25.weekday() < 5 else d25 - BD
    return anchor - 3 * BD


def ltd_ng(y, m):                         # NYMEX Henry Hub: 3rd-last business day of the prior month
    return pd.Timestamp(y, m, 1) - 3 * BD


def ltd_last_bday_prior_month(y, m):      # NYMEX HO, RB
    py, pm = _prev_month(y, m)
    return _last_bday(py, pm)


def ltd_third_last_bday(y, m):            # COMEX GC, SI, HG
    return _last_bday(y, m) - 2 * BD


def ltd_cbot_note(y, m):                  # CBOT ZN, ZB: 7th business day before the last business day
    return _last_bday(y, m) - 7 * BD


def ltd_last_bday(y, m):                  # CBOT ZF
    return _last_bday(y, m)


def ltd_eurex_bond(y, m):                 # Eurex FGBL: 2 exchange days before delivery (10th or next)
    return _bday_on_or_after(pd.Timestamp(y, m, 10)) - 2 * BD


def ltd_cme_fx(y, m):                     # CME 6E, 6J, 6B, 6A: 2 business days before the 3rd Wednesday
    return _nth_weekday(y, m, 2, 3) - 2 * BD


def ltd_cbot_grain(y, m):                 # CBOT ZC, ZS, ZW: business day before the 15th
    return pd.Timestamp(y, m, 15) - BD


def ltd_ice_coffee(y, m):                 # ICE KC: 1 business day before last notice (7 bd before last bd)
    return _last_bday(y, m) - 8 * BD


# code: (contract-month cycle, LTD rule)
SCHEDULES = {
    # data/final contracts (development set)
    "ES": ("HMUZ", ltd_third_friday),
    "FESX": ("HMUZ", ltd_third_friday),      # stand-in index series: no rolls in the data
    "NK": ("HMUZ", ltd_bday_before_second_friday),
    "CL": (ALL_MONTHS, ltd_cl),
    "NG": (ALL_MONTHS, ltd_ng),
    "GC": ("GJMQVZ", ltd_third_last_bday),
    "SI": ("HKNUZ", ltd_third_last_bday),
    "ZN": ("HMUZ", ltd_cbot_note),
    "FGBL": ("HMUZ", ltd_eurex_bond),        # stand-in bond series: no rolls in the data
    "6E": ("HMUZ", ltd_cme_fx),
    "6J": ("HMUZ", ltd_cme_fx),
    "ZC": ("HKNUZ", ltd_cbot_grain),
    "ZS": ("FHKNQUX", ltd_cbot_grain),
    # candidate confirmation contracts (never used in development)
    "NQ": ("HMUZ", ltd_third_friday),
    "HO": (ALL_MONTHS, ltd_last_bday_prior_month),
    "RB": (ALL_MONTHS, ltd_last_bday_prior_month),
    "HG": ("HKNUZ", ltd_third_last_bday),
    "ZB": ("HMUZ", ltd_cbot_note),
    "ZF": ("HMUZ", ltd_last_bday),
    "6B": ("HMUZ", ltd_cme_fx),
    "6A": ("HMUZ", ltd_cme_fx),
    "ZW": ("HKNUZ", ltd_cbot_grain),
    "KC": ("HKNUZ", ltd_ice_coffee),
}
STANDINS = ("FESX", "FGBL")  # their series are not futures: no held-contract switch

SEASONS = {"ZC": (6, 7, 8), "ZS": (6, 7, 8), "NG": (11, 12, 1, 2, 3)}


# ---------------------------------------------------------------- scheduled roll days

def last_trading_days(code: str, start: str | pd.Timestamp, end: str | pd.Timestamp,
                      schedule: tuple | None = None) -> pd.DatetimeIndex:
    """LTDs of every contract in `code`'s cycle expiring between start and end.
    `schedule` = (cycle, rule) overrides SCHEDULES[code]."""
    cycle, rule = schedule or SCHEDULES[code]
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    out = [rule(y, MONTH_CODES[c]) for y in range(start.year - 1, end.year + 2) for c in cycle]
    out = pd.DatetimeIndex(sorted(out))
    return out[(out >= start) & (out <= end)]


def roll_days(code: str, start, end, schedule: tuple | None = None) -> pd.DatetimeIndex:
    """Scheduled switch days: first weekday after each LTD."""
    ltd = last_trading_days(code, pd.Timestamp(start) - pd.Timedelta(days=10), end, schedule)
    sw = pd.DatetimeIndex([d + BD for d in ltd])
    return sw[(sw >= pd.Timestamp(start)) & (sw <= pd.Timestamp(end))]


def _days(index) -> np.ndarray:
    return pd.DatetimeIndex(index).values.astype("datetime64[D]")


def shift_weekdays(index: pd.DatetimeIndex, n: int) -> pd.DatetimeIndex:
    """t + n weekdays (a weekend date first rolls forward to Monday)."""
    return pd.DatetimeIndex(np.busday_offset(_days(index), n, roll="forward"))


def days_to_roll(index: pd.DatetimeIndex, code: str, schedule: tuple | None = None) -> pd.DataFrame:
    """Weekdays to the next scheduled switch day (0 on it) and since the last one (0 on it)."""
    index = pd.DatetimeIndex(index)
    sw = roll_days(code, index.min() - pd.Timedelta(days=200), index.max() + pd.Timedelta(days=200), schedule)
    swd, t = _days(sw), _days(index)
    i_next = np.searchsorted(swd, t, side="left")
    i_last = np.searchsorted(swd, t, side="right") - 1
    d_next = np.busday_count(t, swd[i_next])
    d_last = np.busday_count(swd[i_last], t)
    return pd.DataFrame({"d_next": d_next, "d_last": d_last}, index=index)


def roll_window(index: pd.DatetimeIndex, code: str, half_width: int = 3, lead: int = 0,
                schedule: tuple | None = None) -> pd.Series:
    """True when t + `lead` weekdays is within `half_width` weekdays of a scheduled switch day.

    lead=2 gives the window of the P&L day of a target set at close t under the
    backtest's delay=1 convention (traded at t+1, first earns on t+2).
    """
    index = pd.DatetimeIndex(index)
    d = days_to_roll(shift_weekdays(index, lead), code, schedule)
    return pd.Series((np.minimum(d.d_next, d.d_last) <= half_width).to_numpy(), index=index, name="roll_window")


def in_season(index: pd.DatetimeIndex, code: str, lead: int = 0) -> pd.Series:
    """True when t + `lead` weekdays falls in `code`'s pre-registered season."""
    index = pd.DatetimeIndex(index)
    months = shift_weekdays(index, lead).month
    return pd.Series(np.isin(months, SEASONS[code]), index=index, name="in_season")
