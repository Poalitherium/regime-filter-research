"""Independent backtest of externally supplied target exposures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .backtest import COST_BPS, contract_pnl, portfolio, performance


def dates(frame):
    frame=frame.copy()
    frame['date']=pd.to_datetime(frame.date,errors='raise')
    if frame.date.isna().any() or frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
        raise ValueError('unique timezone-free calendar dates required within each product')
    return frame


def fingerprint(path):
    path=Path(path)
    return dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def run(targets_file,out_dir,*,market_file=None,data_dir=None,fee_bps=None,
        fee_multiplier=1.,slippage_bps=0.,delay=1,missing_target='flat',start=None,end=None):
    """Required targets: date,contract,target; optional strategy identifies runs.

    Market table: date,contract,return,roll. Alternatively read existing
    data/final product files plus universe.csv. No signal model is imported.
    A selected sample starts flat, with no pre-sample holdings.
    """
    if (market_file is None)==(data_dir is None):
        raise ValueError('provide exactly one market_file or data_dir')
    for value,name in [(fee_multiplier,'fee_multiplier'),(slippage_bps,'slippage_bps')]:
        if not np.isfinite(value) or value<0: raise ValueError(f'{name} must be finite and nonnegative')
    provenance=[fingerprint(targets_file)]
    asset_map={}
    if market_file:
        market=pd.read_csv(market_file)
        provenance.append(fingerprint(market_file))
    else:
        path=Path(data_dir)
        uni=pd.read_csv(path/'universe.csv')
        if not {'code','asset_class'}.issubset(uni) or uni.code.duplicated().any():
            raise ValueError('invalid universe.csv')
        asset_map=uni.set_index('code').asset_class.to_dict()
        frames=[]
        provenance.append(fingerprint(path/'universe.csv'))
        for code in uni.code:
            file=path/f'{code}.csv'
            d=pd.read_csv(file)
            if not {'date','ret','roll'}.issubset(d): raise ValueError(f'{code}: date,ret,roll required')
            frames.append(d[['date','ret','roll']].rename(columns={'ret':'return'}).assign(contract=code))
            provenance.append(fingerprint(file))
        market=pd.concat(frames,ignore_index=True)
    if not {'date','contract','return','roll'}.issubset(market):
        raise ValueError('market columns: date,contract,return,roll required')
    market=dates(market)
    if market.duplicated(['date','contract']).any() or market.contract.isna().any() or market.contract.astype(str).str.strip().eq('').any():
        raise ValueError('duplicate or missing market keys')
    if market.roll.isna().any() or not market.roll.isin([True,False,0,1]).all():
        raise ValueError('roll must be Boolean or 0/1')
    market['roll']=market.roll.astype(bool)
    market['return']=pd.to_numeric(market['return'],errors='raise')
    if start is not None: market=market.loc[market.date>=pd.Timestamp(start)]
    if end is not None: market=market.loc[market.date<=pd.Timestamp(end)]
    if market.empty: raise ValueError('empty market sample')
    codes=sorted(market.contract.unique())
    if fee_bps is not None:
        if not np.isfinite(fee_bps) or fee_bps<0: raise ValueError('fee_bps must be finite and nonnegative')
        rates={code:float(fee_bps)*fee_multiplier for code in codes}
    else:
        if any(asset_map.get(c) not in COST_BPS for c in codes):
            raise ValueError('explicit fee_bps required without recognized asset classes')
        rates={c:COST_BPS[asset_map[c]]*fee_multiplier for c in codes}
    targets=pd.read_csv(targets_file)
    if not {'date','contract','target'}.issubset(targets):
        raise ValueError('target columns: date,contract,target required')
    if 'strategy' not in targets: targets['strategy']='external_targets'
    targets=dates(targets)
    if targets[['strategy','contract']].isna().any().any() or targets[['strategy','contract']].astype(str).apply(lambda x:x.str.strip().eq('')).any().any():
        raise ValueError('strategy and contract names must be nonempty')
    if targets.duplicated(['strategy','contract','date']).any():
        raise ValueError('duplicate target keys')
    targets['target']=pd.to_numeric(targets.target,errors='raise')
    if np.isinf(targets.target).any(): raise ValueError('infinite target')
    if start is not None: targets=targets.loc[targets.date>=pd.Timestamp(start)]
    if end is not None: targets=targets.loc[targets.date<=pd.Timestamp(end)]
    if targets.empty: raise ValueError('empty target sample')
    data={c:g.set_index('date').sort_index() for c,g in market.groupby('contract')}
    union=pd.DatetimeIndex(sorted(market.date.unique()))
    membership=pd.DataFrame(True,index=union,columns=codes)
    summaries=[];portfolios=[];executions=[]
    for strategy,group in targets.groupby('strategy',sort=True):
        if set(group.contract)!=set(codes):
            raise ValueError(f'{strategy}: targets must identify every market product')
        ledgers={}
        for c,d in data.items():
            target=group.loc[group.contract==c].set_index('date').target.sort_index()
            if len(target.index.difference(d.index)):
                raise ValueError(f'{strategy}/{c}: target date outside observed product calendar')
            ledger=contract_pnl(target,d['return'],d.roll,rates[c],delay,
                                slippage_bps=slippage_bps,missing_target=missing_target)
            ledgers[c]=ledger
            executions.append(ledger.rename_axis('date').reset_index().assign(contract=c,strategy=strategy))
        port=portfolio(ledgers,membership,min_contracts=len(codes))
        if port.empty: raise ValueError('empty portfolio sample')
        row=dict(strategy=strategy,**performance(port.net))
        row.update(sharpe_gross=performance(port.gross)['sharpe'],fee_cost_pa=port.fee_cost.mean()*252,
                   slippage_cost_pa=port.slippage_cost.mean()*252,turnover_pa=port.turnover.mean()*252)
        summaries.append(row)
        portfolios.append(port.rename_axis('date').reset_index().assign(strategy=strategy))
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    summary=pd.DataFrame(summaries)
    summary.to_csv(out/'summary.csv',index=False)
    pd.concat(portfolios,ignore_index=True).to_csv(out/'daily_portfolio.csv',index=False)
    pd.concat(executions,ignore_index=True).to_csv(out/'product_ledger.csv',index=False)
    pd.DataFrame(provenance).to_csv(out/'input_sha256.csv',index=False)
    config=dict(cost_model='exposure_bps_scenario; not sourced cash fees',fee_bps=rates,
        slippage_bps=slippage_bps,slippage_scenario='ZERO_SLIPPAGE_ASSUMPTION' if slippage_bps==0 else 'USER_SPECIFIED_SLIPPAGE_ASSUMPTION',
        delay=delay,missing_target=missing_target,initial_position='flat',terminal_liquidation=False,
        portfolio=f'fixed equal-weight return index across {len(codes)} products',
        sample_start=str(union[0].date()),sample_end=str(union[-1].date()),
        first_union_date='included; costs on that date are retained',
        currency_treatment='dimensionless source-return index; FX available separately for cash fees',
        assumptions_excluded=['financing','margin','taxes','delivery','allocation_transfer_costs'],
        signal_source='externally supplied target CSV; no signal generation')
    (out/'run_config.json').write_text(json.dumps(config,indent=2)+'\n')
    print(summary[['strategy','start','end','days','sharpe','max_drawdown']].to_string(index=False))
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--targets',type=Path,required=True)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--market-data',type=Path)
    source.add_argument('--data-dir',type=Path)
    parser.add_argument('--out-dir',type=Path,default=Path('m1/report/backtest'))
    parser.add_argument('--fee-bps',type=float)
    parser.add_argument('--fee-multiplier',type=float,default=1.)
    parser.add_argument('--slippage-bps',type=float,default=0.)
    parser.add_argument('--delay',type=int,default=1)
    parser.add_argument('--missing-target',choices=['flat','hold'],default='flat')
    parser.add_argument('--start')
    parser.add_argument('--end')
    a=parser.parse_args()
    run(a.targets,a.out_dir,market_file=a.market_data,data_dir=a.data_dir,fee_bps=a.fee_bps,
        fee_multiplier=a.fee_multiplier,slippage_bps=a.slippage_bps,delay=a.delay,
        missing_target=a.missing_target,start=a.start,end=a.end)

if __name__=='__main__':main()
