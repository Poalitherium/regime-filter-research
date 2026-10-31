"""Run M2 on the group's data/final/ per-contract files. No P&L is computed."""
from pathlib import Path
import argparse
import pandas as pd
try:
    from .pipeline import build_m2
except ImportError:
    from pipeline import build_m2

CODES = ('ES','FESX','NK','CL','NG','GC','SI','ZN','FGBL','6E','6J','ZC','ZS')

def run(data_dir: Path, output_path: Path):
    outputs = []
    summaries = []
    for code in CODES:
        source = data_dir / f'{code}.csv'
        if not source.is_file():
            raise FileNotFoundError(source)
        df = pd.read_csv(source)
        needed = {'date','close_adj','ret'}
        if not needed.issubset(df.columns):
            raise ValueError(f'{code}: missing {needed-set(df.columns)}')
        df = df[['date','close_adj','ret']].rename(columns={'close_adj':'close','ret':'return'})
        df['contract'] = code
        out = build_m2(df)
        outputs.append(out)
        summaries.append({'contract':code,'days':len(df),'rows':len(out),
                          'valid_a2':int(out['approach2_target_close_t'].notna().sum())})
    result = pd.concat(outputs, ignore_index=True).sort_values(['date','contract','factor'])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    summary = pd.DataFrame(summaries)
    print(summary.to_string(index=False))
    print('TOTAL',len(result),'VALID_A2',int(result.approach2_target_close_t.notna().sum()))
    print('OUTPUT',output_path)
    return result, summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir',type=Path,default=Path('data/final'))
    parser.add_argument('--output',type=Path,default=Path('m2_outputs/signals.csv'))
    args=parser.parse_args()
    run(args.data_dir,args.output)
