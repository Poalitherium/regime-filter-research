"""Run the data and analysis stages in dependency order, from any directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CODES = ('ES', 'FESX', 'NK', 'CL', 'NG', 'GC', 'SI', 'ZN', 'FGBL', '6E', '6J', 'ZC', 'ZS')
DATA = [f'data/final/{c}.csv' for c in CODES] + [
    'data/final/universe.csv', 'data/processed/offmarket/offmarket_daily.csv']
LEGACY = ['m4/data/cot_legacy_2014.csv', 'm4/data/implied_vol_2014.csv',
          'm4/data/fomc_scheduled.csv']
FILTERS = ['m3_outputs/hmm_k3_quarter_top_filter.csv',
           'm3_outputs/hmm_k3_quarter_top_offmkt_filter.csv',
           'm3_outputs/bocpd_h250_filter.csv', 'm3_outputs/portvol_filter.csv',
           'm3_outputs/meta_hand_gbm_filter.csv', 'm3_outputs/meta_sig_gbm_filter.csv',
           'm5_outputs/pool_class_k3_quarter_top_filter.csv',
           'm5_outputs/pool_global_k3_quarter_top_filter.csv']


def commands(stage: str, workers: int) -> list[list[str]]:
    if stage == 'data':
        return [[f'data_pipeline/{s}.py'] for s in (
            'pull_yfinance', 'roll_adjust', 'build_standins', 'build_dataset', 'pull_offmarket')]
    if stage == 'historical-data':
        return [['-m', m] for m in ('m3.eras.build_data', 'm3.eras.build_1930')]
    if stage == 'models':
        return [
            ['-m', 'm3.run_real_data', '--n-states', '3', '--refit', 'quarter', '--mapping', 'top'],
            ['-m', 'm3.run_real_data', '--n-states', '3', '--refit', 'quarter', '--mapping', 'top', '--offmarket'],
            ['-m', 'm3.run_bocpd', '--run-length', '250'],
            ['-m', 'm3.portfolio_vol'],
            ['-m', 'm5.pool', '--pool', 'class'], ['-m', 'm5.pool', '--pool', 'global'],
            ['-m', 'm3.metalabel', '--features', 'hand', '--model', 'gbm'],
            ['-m', 'm3.metalabel', '--features', 'sig', '--model', 'gbm'],
        ]
    if stage == 'evaluate':
        return [
            ['-m', 'backtest.resplit_common'], ['-m', 'backtest.resplit_data'],
            ['-m', 'backtest.resplit_data', '--prepare'], ['-m', 'backtest.resplit_models'],
            ['-m', 'backtest.resplit_gates'], ['-m', 'backtest.resplit_models', '--extras'],
            ['-m', 'backtest.resplit_evaluate'], ['-m', 'backtest.resplit_evaluate', '--extras'],
            ['-m', 'scripts.audit_results'],
        ]
    if stage == 'historical':
        return [
            ['-m', 'm3.eras.run', 'filters', '--workers', str(workers)],
            ['-m', 'm3.eras.run', 'filters', '--study', '1930', '--workers', str(workers)],
            ['-m', 'backtest.historical_sweep', 'models', '--workers', str(workers)],
            ['-m', 'backtest.historical_selection'], ['-m', 'backtest.historical_evaluate', 'evaluate'],
        ]
    raise ValueError(stage)


def check_inputs(stage: str) -> None:
    needed = DATA if stage in ('models', 'evaluate', 'historical') else []
    if stage == 'evaluate':
        needed = needed + LEGACY + FILTERS
        if (ROOT/'backtest_outputs/resplit_80_20/summary.csv').exists():
            raise FileExistsError('Completed evaluation exists. Archive backtest_outputs/resplit_80_20 before rerunning.')
    elif stage == 'historical':
        needed = needed + LEGACY + FILTERS + [
            'data/eras/universe.csv', 'data/eras1930/universe.csv',
            'backtest_outputs/resplit_80_20/filters/M3-6.csv',
            'backtest_outputs/resplit_80_20/filters/M3-7.csv',
            'backtest_outputs/resplit_80_20/extra_targets/M3-5_dynamic.csv',
            'backtest_outputs/resplit_80_20/extra_targets/M3-8_hmm.csv',
            'backtest_outputs/resplit_80_20/inputs/cot_legacy.csv',
            'backtest_outputs/resplit_80_20/inputs/implied_vol.csv',
            'backtest_outputs/resplit_80_20/inputs/fomc_scheduled.csv']
    missing = [p for p in needed if not (ROOT/p).is_file()]
    if missing:
        raise FileNotFoundError('Missing inputs:\n  ' + '\n  '.join(missing) +
                                '\nSee the Data section of README.md.')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['data', 'models', 'evaluate', 'historical-data', 'historical'])
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--dry-run', action='store_true', help='Print commands without reading data or running a stage.')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    if not args.dry_run:
        try:
            check_inputs(args.stage)
        except (FileNotFoundError, FileExistsError) as exc:
            parser.exit(2, f'{exc}\n')
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', NUMBA_NUM_THREADS='1')
    for cmd in commands(args.stage, args.workers):
        print('python ' + ' '.join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
