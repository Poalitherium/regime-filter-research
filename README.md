# Regime Filters and Volatility-Targeted Trend Following

Code for comparing trend following with volatility targeting (A2), a regime filter (R), or both (A3).

## Setup

Python 3.14 and [uv](https://docs.astral.sh/uv/) are required. Run from the repository root:

```sh
uv sync --locked
uv run python -m integration.run_tests
```

## Data

```sh
uv run python -m scripts.reproduce data
```

This downloads futures, roll proxies, Bundesbank yields and macro/news inputs, then builds `data/final/` and `data/processed/offmarket/`. Source URLs and transformations are in `data_pipeline/`. Market data are not bundled. Downloads may change with provider revisions; matching an earlier run requires the same input snapshot.

M4 also requires two local pre-2015 inputs. The evaluation downloads their extensions through 2026-10-15.

**CFTC positioning.** Download `deafut_xls_1986_2016.zip` from the [CFTC Legacy Futures Only archive](https://www.cftc.gov/MarketReports/CommitmentsofTraders/HistoricalCompressed/index.htm). Extract `FUT86_06.xls` and `FUT07_14.xls`, then run:

```sh
uv run python -m m4.cot --src /path/to/extracted/files
```

This creates `m4/data/cot_legacy_2014.csv` for the eleven supported contracts through 2014-12-31. Contract mappings and the exclusion of the earlier ECU series are in `m4/cot.py`.

**Implied volatility.** Prepare `m4/data/implied_vol_2014.csv` with columns `date,vix3m,ovx,gvz,move` through 2014-12-31. Use unlagged daily closing levels, ISO dates and empty cells for missing values. The code applies availability lags.

| Column | Original source |
|---|---|
| `vix3m` | FRED `VXVCLS`, CBOE three-month volatility |
| `ovx` | FRED `OVXCLS`, CBOE oil volatility |
| `gvz` | FRED `GVZCLS`, CBOE gold volatility |
| `move` | Investing.com ICE BofA MOVE daily history, from 2002-11-13 |

The original pre-2015 MOVE snapshot has no automated downloader. Supply it from a permitted source or an existing research snapshot; changing providers may change results. The evaluation checks its units against overlapping Yahoo observations.

The included `m4/data/fomc_scheduled.csv` contains 112 scheduled FOMC statement dates for 2001–2014 from Federal Reserve historical calendars, excluding emergency meetings. Release times are US Eastern time: 14:15 before 2013 and 14:00 thereafter.

## Reproduce

```sh
uv run python -m scripts.reproduce models
uv run python -m scripts.reproduce evaluate
```

The common 80/20 split is 2007-07-02–2022-12-05 for development and 2022-12-06–2026-10-15 for testing. Outputs are written to `backtest_outputs/resplit_80_20/`. Model fitting and bootstrap inference can take several hours. Archive the output directory before repeating a completed evaluation.

Historical period tests, after the main evaluation:

```sh
uv run python -m scripts.reproduce historical-data
uv run python -m scripts.reproduce historical --workers 4
```

Historical data use equity portfolios and cross-asset proxies; methods are simplified where their original inputs are unavailable. Outputs are written to `backtest_outputs/historical_sweep_20261031/`.

## Results and checks

| Output directory | Key generated files |
|---|---|
| `backtest_outputs/resplit_80_20/` | `baselines.csv`, `summary.csv`, `comparisons.csv`, `extra_strategy_summary.csv`, `extra_strategy_comparisons.csv` |
| `backtest_outputs/historical_sweep_20261031/` | `primary_filter_tests.csv`, `all_comparisons.csv`, `candidate_block_sensitivity.csv`, `evaluations/` |

The comparison tables contain Sharpe increments, bootstrap confidence intervals and raw and Holm-adjusted p-values. Output directories also retain daily returns, selection decisions, training-date checks and input/code hashes. The main evaluation ends with a saved-result audit; rerun it with:

```sh
uv run python -m scripts.audit_results
```

Append `--dry-run` to any stage to inspect its commands. Tests requiring downloaded inputs or legacy result tables are explicitly skipped when those files are absent.
