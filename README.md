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

This downloads futures, roll proxies, Bundesbank yields and macro/news inputs, then builds `data/final/` and `data/processed/offmarket/`. For M4, also prepare the two historical CSV inputs described in [m4/data/README.md](m4/data/README.md). Market data are not bundled. Downloads may change with provider revisions; matching an earlier run requires the same input snapshot. Source URLs and transformations are in `data_pipeline/`.

## Reproduce

```sh
uv run python -m scripts.reproduce models
uv run python -m scripts.reproduce evaluate
```

The common 80/20 split is 2007-07-02–2022-12-05 for development and 2022-12-06–2026-10-15 for testing. Results, bootstrap intervals, Holm-adjusted p-values and audits are saved under `backtest_outputs/resplit_80_20/`. Model fitting and bootstrap inference can take several hours. Archive the output directory before repeating a completed evaluation.

Historical period tests, after the main evaluation:

```sh
uv run python -m scripts.reproduce historical-data
uv run python -m scripts.reproduce historical --workers 4
```

Historical data use equity portfolios and cross-asset proxies; methods are simplified where their original inputs are unavailable. Results are saved under `backtest_outputs/historical_sweep_20261031/`.

Append `--dry-run` to any stage to inspect its commands. Tests requiring downloaded inputs or legacy result tables are explicitly skipped when those files are absent.
