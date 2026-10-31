"""M2 integration: one row per date, contract and factor; NO backtest P&L."""
import numpy as np
import pandas as pd
try:
    from .factors import generate_signals
    from .volatility import volatility_target
except ImportError:
    from factors import generate_signals
    from volatility import volatility_target

REQUIRED = ('date', 'contract', 'close', 'return')

def build_m2(data: pd.DataFrame) -> pd.DataFrame:
    """Return signals and exposures known after date t close, NOT executed positions.

    Input `return` must be supplied/approved by M1; never infer futures returns
    from a potentially back-adjusted continuous price series.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a DataFrame')
    missing = set(REQUIRED).difference(data.columns)
    if missing:
        raise ValueError(f'missing columns: {sorted(missing)}')
    if data.empty:
        raise ValueError('input is empty')
    df = data.loc[:, REQUIRED].copy()
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    if df['date'].isna().any() or df['contract'].isna().any() or df['contract'].astype(str).str.strip().eq('').any():
        raise ValueError('missing date or contract')
    if df.duplicated(['date', 'contract']).any():
        raise ValueError('duplicate date-contract pairs')
    for c in ('close', 'return'):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise TypeError(f'{c} must be numeric')
        if np.isinf(df[c].to_numpy(dtype=float)).any():
            raise ValueError(f'{c} has infinity')
    df = df.sort_values(['contract', 'date'])
    chunks = []
    for contract, group in df.groupby('contract', sort=True):
        group = group.set_index('date').sort_index()
        signals = generate_signals(group['close'])
        exposures = volatility_target(signals, group['return'])
        for factor in signals.columns:
            chunk = pd.DataFrame({
                'date': group.index,
                'contract': contract,
                'factor': factor,
                'signal_close_t': signals[factor].to_numpy(),
                'realized_vol_close_t': exposures['realized_vol'].to_numpy(),
                'vol_scale_close_t': exposures['vol_scale'].to_numpy(),
                'approach1_target_close_t': exposures[f'{factor}_raw'].to_numpy(),
                'approach2_target_close_t': exposures[f'{factor}_vol_target'].to_numpy(),
            })
            chunks.append(chunk)
    out = pd.concat(chunks, ignore_index=True)
    return out.sort_values(['date', 'contract', 'factor']).reset_index(drop=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate M2 signals from M1 daily data')
    parser.add_argument('input_csv')
    parser.add_argument('output_csv')
    args = parser.parse_args()
    build_m2(pd.read_csv(args.input_csv)).to_csv(args.output_csv, index=False)
    print('Saved:', args.output_csv)
