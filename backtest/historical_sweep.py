"""Historical stage tests requested on 2026-10-31; never modifies report or old runs.

Run: python -m backtest.historical_sweep prepare|models
Evaluate with backtest.historical_evaluate; export with backtest.historical_deliver.
The design is written before new model fits or performance comparisons.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'backtest_outputs/historical_sweep_20261031'
NAMES = {'M3-1':'HMM', 'M3-2':'HMM + macro/news', 'M3-3':'BOCPD',
         'M3-4':'Portfolio volatility targeting', 'M3-5':'Momentum turning points',
         'M3-6':'Meta-labeling', 'M3-7':'Signature meta-labeling', 'M3-8':'Pure HMM strategy',
         'M4-1':'FOMC calendar', 'M4-2':'VIX stress x risk-on', 'M4-3':'COT crowding',
         'M4-4':'Implied / realised volatility', 'M4-5':'VIX curve inversion',
         'M5-1':'Class-pooled HMM', 'M5-2':'Global-pooled HMM',
         'M5-3':'Seasonality', 'M5-4':'Contract expiry'}
PRICE = ['M3-1','M3-3','M3-4','M3-5','M3-7','M3-8','M5-1','M5-2']
ERAS = {
 'industry': [('1930-1959','1930-01-02','1959-12-31'),('1960-1976','1960-01-01','1976-12-31'),
              ('1977-1989','1977-01-01','1989-12-31'),('1990-1999','1990-01-01','1999-12-31'),
              ('2000-2014','2000-01-01','2014-12-31')],
 'core': [('1977-1989','1977-01-03','1989-12-31'),('1990-1999','1990-01-01','1999-12-31'),
          ('2000-2014','2000-01-01','2014-12-31')],
 'extended': [('1990-1999','1990-01-01','1999-12-31'),('2000-2014','2000-01-01','2014-12-31')],
 'futures': [('2007-2014','2007-07-02','2014-12-31'),('2015-2022','2015-01-01','2022-12-31'),
             ('2023-2026','2023-01-01','2026-10-15')],
}
REPS, SEED = 50000, 20261031


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(study):
    return ROOT / ('data/eras1930' if study == 'industry' else 'data/final' if study == 'futures' else 'data/eras')


def universe(study):
    u = pd.read_csv(source(study) / 'universe.csv')
    return u[u.pool.eq('core')].copy() if study == 'core' else u


def methods(study):
    # All industry portfolios are equities: class and global pooling are the same method.
    return list(NAMES) if study == 'futures' else [m for m in PRICE if not (study == 'industry' and m == 'M5-1')]


def path_filter(study, tid):
    if study == 'futures':
        from .resplit_common import REUSED, OUT as OLD
        if tid in REUSED:
            return ROOT / REUSED[tid][1]
        if tid in ('M3-6','M3-7'):
            return OLD / 'filters' / f'{tid}.csv'
    else:
        folder = ROOT / 'm3_outputs/eras' / ('1930' if study == 'industry' else '')
        stems = {'M3-1':'hmm_k3_quarter_top', 'M3-3':'bocpd_h250',
                 'M5-2':f'pool_global_k3_quarter_top_{study}'}
        if tid in stems:
            return folder / f'{stems[tid]}_filter.csv'
    return OUT / 'models' / study / f'{tid}_filter.csv'


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'analysis_spec.json'
    if path.exists():
        return
    files = sorted(set(p for study in ERAS for p in source(study).glob('*.csv')))
    files += [path_filter(s,m) for s in ERAS for m in methods(s)
              if m not in ('M3-5','M3-8') and path_filter(s,m).exists()]
    spec = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        question='Did any existing regime method have a positive annualised net-Sharpe increment in a historical stage?',
        eras=ERAS, methods={s:methods(s) for s in ERAS},
        scope='Existing 1926-2014 industry / 1973-2014 proxy inputs, plus separately evaluated 2000-2026 futures inputs. No cross-universe era contrast or crowding attribution.',
        historical_boundaries='Reuse the old historical studies\' era boundaries; no search for optimal endpoints.',
        modern_boundaries='2007 common warm-up through 2014; 2015-2022; 2023 through latest local data.',
        parameters='Existing model settings unchanged, past-only fits; historical signature first annual refit in 1930/1977; subsequent annual refits retain all earlier labelled data.',
        selection='For externally gated futures filters, original predictability criteria are reassessed each January using only data through the previous December 31. Unavailable/insufficient evidence means inactive, not a positive result.',
        primary='A3-A2, all eligible regime filters x all stages x all four samples, one Holm family; raw and family-adjusted two-sided p-values both reported.',
        other_comparisons='R-A2 and R-A3 form their own joint Holm family; portfolio-VT benchmark and signal-replacement strategies reported separately and also included in a broader family.',
        costs='Original M1 asset-class research bps, 1x primary; 0x and 3x sensitivity for every method and stage; official fees are not meaningful on historical proxies.',
        accounting='Fixed equal weights, identical asset membership/dates for every method per sample, continuous execution over each whole sample; no resets at era boundaries.',
        inference={'annualisation':252,'block':20,'reps':REPS,'seed':SEED,'p':'two-sided null-centred paired circular-block bootstrap, (exceedances+1)/(reps+1)',
                   'ci':'unadjusted 95% percentile intervals','sensitivity_blocks':[10,60],
                   'sensitivity_rule':'All primary-cost nominally positive significant candidates also checked at blocks 10 and 60; same settings, no retuning.'},
        caveats=['Historical equity portfolios and cross-asset proxies are not tradable futures with actual historical execution costs.',
                 'Pre-1952 US daily data include Saturdays; original 252-day annualisation retained for project comparability.',
                 'This is a retrospective survey of fixed recipes, not untouched confirmation of the old descriptive slices.',
                 'No method or stage is dropped because of its outcome.'],
        unavailable={'industry_class_pool':'identical to global pooling because all ten assets are equities',
                     'historical_macro_methods':'original external features unavailable across these full historical samples',
                     'historical_season_expiry':'no agricultural/natural-gas contracts or actual futures-expiry data in the historical proxy universes'},
        input_sha256={str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in files})
    path.write_text(json.dumps(spec, indent=2), encoding='utf-8')
    print('Design fixed:', path, flush=True)


def base(study):
    from m2.pipeline import build_m2
    u = universe(study)
    data, asset, targets = {}, {}, {'A1':{},'A2':{}}
    for row in u.itertuples():
        d = pd.read_csv(source(study)/f'{row.code}.csv', index_col='date', parse_dates=True)
        data[row.code], asset[row.code] = d, row.asset_class
        m = build_m2(pd.DataFrame({'date':d.index,'contract':row.code,'close':d.close_adj.to_numpy(),'return':d.ret.to_numpy()}))
        for ap, col in [('A1','approach1_target_close_t'),('A2','approach2_target_close_t')]:
            w = m.pivot(index='date',columns='factor',values=col)
            targets[ap][row.code] = w[['tsmom','ma_crossover','donchian']].mean(axis=1,skipna=False).reindex(d.index)
    return data, asset, targets


def signature_panel(study):
    """Same selected SIG features and A2 label as M3-7, without irrelevant macro dependencies."""
    from m3 import signature, metalabel as ml
    from m3.bocpd import standardise
    from m3.features import feature
    from m3.predictability import asof
    from backtest.engine import contract_pnl, COST_BPS, in_book
    data, asset, tg = base(study)
    union = pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    book = in_book(pd.DataFrame({c:tg['A2'][c].notna() for c in data}).reindex(union))
    # Original meta panel starts when >=10 contracts are live. Core has only 8;
    # adapt min membership to 8, retaining the same per-contract features/labels.
    days = union[book.sum(axis=1).ge(min(10,len(data)))]
    ends = pd.DatetimeIndex(pd.Series(days,index=days).groupby(days.to_period('W-FRI')).max().values)
    rows = []
    for code,d in data.items():
        net = contract_pnl(tg['A2'][code],d.ret,d.roll,COST_BPS[asset[code]],delay=1).net
        f = pd.DataFrame({'logvol20':feature(d.ret,'logvol20')}).join(signature.features(standardise(d.ret),60))
        fwd = net[::-1].rolling(ml.HORIZON,min_periods=ml.HORIZON).sum()[::-1].shift(-1)
        label_end = pd.Series(net.index,index=net.index).shift(-ml.HORIZON)
        e = ends[book[code].reindex(ends).to_numpy()]
        p = asof(f,e).assign(contract=code,week=e,
            fwd=asof(fwd.to_frame('v'),e).v.to_numpy(),
            label_end=asof(label_end.to_frame('v'),e).v.to_numpy())
        p['y'] = (p.fwd>0).astype(float).where(p.fwd.notna())
        rows.append(p)
    return pd.concat(rows)


def signature_job(study):
    from m3 import metalabel as ml
    dest = OUT/'models'/study
    dest.mkdir(parents=True,exist_ok=True)
    fpath=dest/'M3-7_filter.csv'
    if fpath.exists(): return str(fpath)
    panel=signature_panel(study)
    first = 1930 if study=='industry' else 1977
    audit=[]
    for year in range(first,2015):
        cutoff=pd.Timestamp(year,1,1)-pd.offsets.BDay(ml.EMBARGO)
        tr=panel[(panel.label_end<cutoff)&panel.y.notna()]
        assert tr.empty or tr.label_end.max()<cutoff
        audit.append(dict(refit=f'{year}-01-01',n_train=len(tr),last_label=str(tr.label_end.max()),cutoff=str(cutoff)))
    pd.DataFrame(audit).to_csv(dest/'M3-7_training_audit.csv',index=False)
    with threadpool_limits(limits=1):
        preds,auc=ml.walk_forward(panel,ml.SIG,'gbm',first_refit=first,last_refit=2014)
    ml.write_filter(preds,source(study),fpath)
    auc.to_csv(dest/'M3-7_auc.csv',index=False)
    return str(fpath)


def pool_job(study, group):
    from m5.pool import run
    from m3.eras.m5_seeds import valid_seeds
    dest=OUT/'models'/study/'class_parts'; dest.mkdir(parents=True,exist_ok=True)
    file=dest/f'{group}.csv'
    if file.exists(): return str(file)
    u=universe(study);u=u[u.asset_class.eq(group)]
    with threadpool_limits(limits=1),valid_seeds(),redirect_stdout(io.StringIO()):
        filt,fits=run(source(study),None,'class',universe=u)
    filt.to_csv(file,index=False);fits.to_csv(dest/f'{group}_fits.csv',index=False)
    return str(file)


def direction_job(study,code):
    from m3.regime_strategy import hmm_direction
    from m3.hmm import GaussianHMM
    from m3.turning_points import vol_scale
    dest=OUT/'models'/study/'direction_parts';dest.mkdir(parents=True,exist_ok=True)
    path=dest/f'{code}.csv'
    if path.exists():return str(path)
    class SafeSeed(GaussianHMM):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            if self.seed<0:self.seed%=2**32
    d=pd.read_csv(source(study)/f'{code}.csv',index_col='date',parse_dates=True)
    with threadpool_limits(limits=1),patch('m3.regime_strategy.GaussianHMM',SafeSeed):
        hd,fits=hmm_direction(d.ret)
    target=np.sign(hd.exp_z)*vol_scale(d.ret)
    target.rename('target').rename_axis('date').to_csv(path)
    fits.to_csv(dest/f'{code}_fits.csv',index=False)
    return str(path)


def simple_jobs(study):
    from m3 import portfolio_vol as pv, turning_points as tp
    dest=OUT/'models'/study;dest.mkdir(parents=True,exist_ok=True)
    codes=tuple(universe(study).code)
    if not (dest/'M3-4_filter.csv').exists():
        with patch.object(pv,'CODES',codes):data,_,k=pv.compute(source(study))
        pd.concat([pd.DataFrame({'date':d.index,'contract':c,'filter_close_t':k.reindex(d.index).to_numpy()})
                   for c,d in data.items()],ignore_index=True).to_csv(dest/'M3-4_filter.csv',index=False)
    if not (dest/'M3-5_targets.csv').exists():
        with patch.object(tp,'CODES',codes): result=tp.build_targets(source(study))
        tg=result[4]
        pd.concat([pd.DataFrame({'date':tg['dynamic'][c].index,'contract':c,
                 'target':tg['dynamic'][c].to_numpy(),'original_control':tg['static_slow'][c].to_numpy()})
                 for c in codes],ignore_index=True).to_csv(dest/'M3-5_targets.csv',index=False)
    return study


def models(workers=6):
    prepare()
    jobs=[]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for s in ['industry','core','extended']:
            jobs += [ex.submit(simple_jobs,s),ex.submit(signature_job,s)]
            if s!='industry':
                for group in universe(s).asset_class.unique():
                    jobs.append(ex.submit(pool_job,s,group))
        # Individual direction fits are independent of pool membership; reuse extended for core.
        for s in ['industry','extended']:
            for code in universe(s).code: jobs.append(ex.submit(direction_job,s,code))
        for f in as_completed(jobs):print('Completed',f.result(),flush=True)
    for s in ['core','extended']:
        parts=[pd.read_csv(OUT/'models'/s/'class_parts'/f'{g}.csv') for g in universe(s).asset_class.unique()]
        pd.concat(parts,ignore_index=True).to_csv(OUT/'models'/s/'M5-1_filter.csv',index=False)
    print('Historical models complete',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','models'])
    p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='models':models(a.workers)


if __name__=='__main__':main()
