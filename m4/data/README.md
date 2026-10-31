# Historical external inputs

The main evaluation extends these inputs through 2026-10-15 automatically. Their pre-2015 portions must be supplied locally; they are excluded from Git.

## CFTC positioning

Download the Legacy Futures Only historical archive `deafut_xls_1986_2016.zip` from the [CFTC historical data page](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm). Extract `FUT86_06.xls` and `FUT07_14.xls` into a local directory, then run:

```sh
uv run python -m m4.cot --src /path/to/extracted/files
```

This writes `m4/data/cot_legacy_2014.csv`, retaining the eleven supported contracts and dates through 2014-12-31. The mapping is in `m4/cot.py`; the earlier ECU contract is excluded from Euro FX.

## Implied volatility

Prepare `m4/data/implied_vol_2014.csv` with columns `date,vix3m,ovx,gvz,move`, covering available observations through 2014-12-31. Use unlagged daily closing index levels, ISO dates, and empty cells for unavailable values. Availability lags are applied by the code.

| Column | Original source |
|---|---|
| `vix3m` | FRED `VXVCLS` (CBOE three-month volatility) |
| `ovx` | FRED `OVXCLS` (CBOE oil volatility) |
| `gvz` | FRED `GVZCLS` (CBOE gold volatility) |
| `move` | Investing.com ICE BofA MOVE historical daily data, from 2002-11-13 |

There is no automated downloader for the original pre-2015 MOVE snapshot. Supply that series from a permitted data source or an existing research snapshot; substituting a provider may change results. The evaluation checks MOVE units against overlapping Yahoo observations.

## Scheduled events

`fomc_scheduled.csv` is the included event-calendar configuration: 112 scheduled FOMC statement dates for 2001–2014, from Federal Reserve historical calendars. It excludes emergency meetings. Release times are US Eastern time, 14:15 before 2013 and 14:00 thereafter.
