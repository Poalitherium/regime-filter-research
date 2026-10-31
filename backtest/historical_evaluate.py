"""Matched portfolios and paired era inference for historical_sweep.py."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
from pathlib import Path

import numba
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .historical_sweep import ROOT,OUT,NAMES,ERAS,REPS,SEED,base,methods,path_filter,universe,prepare,digest
from .m1_bridge import m1_book
from m3.predictability import holm


@numba.njit(cache=True)
def _draw_sums(starts,full,last,ncols):
    out=np.zeros((len(starts),ncols))
    for i in range(len(starts)):
        for j in range(starts.shape[1]-1):
            s=starts[i,j]
            for k in range(ncols):out[i,k]+=full[s,k]
        s=starts[i,-1]
        for k in range(ncols):out[i,k]+=last[s,k]
    return out


def bootstrap(a,b,block=20,reps=REPS,seed=SEED,return_boot=False):
    """Same circular-block resampling as project metrics; aggregate sufficient stats.

    O(reps * ceil(T/block)) gathers instead of O(reps * T) daily-return gathers.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    assert a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    n=len(a)
    sr=lambda x:float(np.mean(x)/np.std(x,ddof=1)*np.sqrt(252))
    obs=sr(a)-sr(b)
    if np.array_equal(a,b):
        boot=np.zeros(reps)
    else:
        x=np.column_stack([a,a*a,b,b*b])
        circular=np.vstack([x,x[:block]])
        prefix=np.vstack([np.zeros(4),np.cumsum(circular,axis=0)])
        at=np.arange(n)
        full=prefix[at+block]-prefix[at]
        remainder=n%block or block
        last=prefix[at+remainder]-prefix[at]
        rng=np.random.default_rng(seed);nb=math.ceil(n/block);boot=np.empty(reps)
        for s in range(0,reps,500):
            count=min(500,reps-s)
            starts=rng.integers(0,n,(count,nb))
            z=_draw_sums(starts,full,last,4)
            ma,mb=z[:,0]/n,z[:,2]/n
            va=(z[:,1]-n*ma*ma)/(n-1);vb=(z[:,3]-n*mb*mb)/(n-1)
            if (va<=0).any() or (vb<=0).any():raise ValueError('zero-variance bootstrap sample')
            boot[s:s+count]=(ma/np.sqrt(va)-mb/np.sqrt(vb))*np.sqrt(252)
    lo,hi=np.quantile(boot,[.025,.975])
    exceed=int(np.count_nonzero(np.abs(boot-boot.mean())>=abs(obs)-1e-12))
    out=dict(delta=obs,ci_low=float(lo),ci_high=float(hi),se=float(boot.std(ddof=1)),
             p=(exceed+1)/(reps+1),exceedances=exceed,reps=reps,block=block,
             sharpe_treatment=sr(a),sharpe_control=sr(b))
    if return_boot:out['boot']=boot
    return out


def load_all_targets(study):
    data,asset,tg=base(study)
    meta={};files={}
    for tid in methods(study):
        if tid not in ('M3-5','M3-8'):
            path=path_filter(study,tid)
            f=pd.read_csv(path,parse_dates=['date']).set_index(['contract','date']).filter_close_t
            if f.index.has_duplicates:raise ValueError(f'duplicate filter index {path}')
            for leg,source in [('A3','A2'),('R','A1')]:
                key=f'{tid}_{leg}';tg[key]={}
                for code,d in data.items():
                    ff=f.loc[code].reindex(d.index)
                    if (ff.dropna() < -1e-9).any() or (ff.dropna()> (3 if tid=='M3-4' else 1)+1e-9).any():
                        raise ValueError(f'filter bound {tid}/{code}')
                    tg[key][code]=tg[source][code]*ff
            meta[tid]=dict(kind='volatility_benchmark' if tid=='M3-4' else 'regime_filter',treatment=f'{tid}_A3',control='A2')
            files[str(path.relative_to(ROOT))]=digest(path)
        else:
            if study=='futures':
                folder=ROOT/'backtest_outputs/resplit_80_20/extra_targets'
                treat,control=('dynamic','static_slow') if tid=='M3-5' else ('hmm','tsmom_252')
                for leg,name in [('strategy',treat),('original_control',control)]:
                    p=folder/f'{tid}_{name}.csv'
                    f=pd.read_csv(p,parse_dates=['date']).set_index(['contract','date']).target
                    tg[f'{tid}_{leg}']={c:f.loc[c].reindex(d.index) for c,d in data.items()}
                    files[str(p.relative_to(ROOT))]=digest(p)
            elif tid=='M3-5':
                p=OUT/'models'/study/'M3-5_targets.csv'
                f=pd.read_csv(p,parse_dates=['date']).set_index(['contract','date'])
                tg[f'{tid}_strategy']={c:f.loc[c].target.reindex(d.index) for c,d in data.items()}
                tg[f'{tid}_original_control']={c:f.loc[c].original_control.reindex(d.index) for c,d in data.items()}
                files[str(p.relative_to(ROOT))]=digest(p)
            else:
                from m3.turning_points import vol_scale
                src='extended' if study=='core' else study
                tg[f'{tid}_strategy']={};tg[f'{tid}_original_control']={}
                for c,d in data.items():
                    p=OUT/'models'/src/'direction_parts'/f'{c}.csv'
                    tg[f'{tid}_strategy'][c]=pd.read_csv(p,index_col='date',parse_dates=True).target.reindex(d.index)
                    tg[f'{tid}_original_control'][c]=np.sign(d.close_adj-d.close_adj.shift(252)).where(d.close_adj.shift(252).notna())*vol_scale(d.ret)
                    files[str(p.relative_to(ROOT))]=digest(p)
            meta[tid]=dict(kind='signal_replacement',treatment=f'{tid}_strategy',control='A2')
    # All comparisons use the same warm-up, dates, contracts and fixed weights.
    first=[]
    for key,v in tg.items():
        for c,s in v.items():
            if s.first_valid_index() is None:raise ValueError(f'no targets {study}/{key}/{c}')
            first.append(s.first_valid_index())
    start=max(pd.Timestamp(ERAS[study][0][1]),max(first))
    dates=pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    book=pd.DataFrame(True,index=dates,columns=list(data))
    # Record missing values separately; genuine unavailable observations are flat in M1.
    missing=[]
    for key,v in tg.items():
        for c,s in v.items():
            w=s.loc[start:ERAS[study][-1][2]]
            missing.append(dict(study=study,leg=key,contract=c,rows=len(w),missing=int(w.isna().sum())))
    dest=OUT/'evaluations'/study;dest.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(missing).to_csv(dest/'target_coverage.csv',index=False)
    (dest/'manifest.json').write_text(json.dumps(dict(start=str(start.date()),contracts=list(data),files=files),indent=2))
    return data,asset,tg,meta,start,book


def run_study(study):
    prepare();dest=OUT/'evaluations'/study;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'comparisons.csv').exists():return study+' already complete'
    with threadpool_limits(limits=1):
        data,asset,tg,meta,start,book=load_all_targets(study)
        rows=[];summary=[]
        for cost,mult in [('x1',1.),('x0',0.),('x3',3.)]:
            ports={k:m1_book(v,data,asset,book,start,ERAS[study][-1][2],'research',cost_multiplier=mult) for k,v in tg.items()}
            daily=pd.DataFrame({k:p.net for k,p in ports.items()})
            assert np.isfinite(daily.to_numpy()).all()
            daily.rename_axis('date').to_csv(dest/f'daily_{cost}.csv')
            for era,lo,hi in ERAS[study]:
                d=daily.loc[lo:hi]
                if len(d)<250:raise ValueError(f'short matched sample {study} {era}')
                cache={}
                for key,p in ports.items():
                    w=p.loc[lo:hi];net=w.net
                    summary.append(dict(study=study,era=era,cost=cost,leg=key,start=str(d.index.min().date()),end=str(d.index.max().date()),days=len(d),
                         sharpe=float(net.mean()/net.std(ddof=1)*np.sqrt(252)),ann_ret=float(net.mean()*252),ann_vol=float(net.std(ddof=1)*np.sqrt(252)),
                         cost_pa=float(w.cost.mean()*252),gross_exposure=float(w.gross_exposure.mean())))
                for tid,info in meta.items():
                    pairs=[(info['treatment'],'A2','increment')]
                    if info['kind']=='signal_replacement':pairs.append((info['treatment'],f'{tid}_original_control','original_control'))
                    else:pairs += [(f'{tid}_R','A2','R-A2'),(f'{tid}_R',f'{tid}_A3','R-A3')]
                    for treat,control,comparison in pairs:
                        # Inactive filters repeat the same portfolio pair. Reuse
                        # its identical resampling result, retaining every test.
                        key=(d[treat].to_numpy().tobytes(),d[control].to_numpy().tobytes())
                        if key not in cache:cache[key]=bootstrap(d[treat],d[control])
                        r=cache[key]
                        rows.append(dict(study=study,era=era,cost=cost,trial=tid,name=NAMES[tid],kind=info['kind'],
                            comparison=comparison,treatment=treat,control=control,start=str(d.index.min().date()),end=str(d.index.max().date()),days=len(d),**r))
            print(study,cost,'evaluated',flush=True)
        pd.DataFrame(rows).to_csv(dest/'comparisons.csv',index=False)
        pd.DataFrame(summary).to_csv(dest/'performance.csv',index=False)
    return study+' complete'


def summarize():
    frames=[pd.read_csv(OUT/'evaluations'/s/'comparisons.csv') for s in ERAS]
    d=pd.concat(frames,ignore_index=True)
    for cost in d.cost.unique():
        primary=d.cost.eq(cost)&d.kind.eq('regime_filter')&d.comparison.eq('increment')
        d.loc[primary,'p_holm_filters']=holm(d.loc[primary,'p'].to_numpy())
        for study in ERAS:
            mask=primary&d.study.eq(study)
            d.loc[mask,'p_holm_sample']=holm(d.loc[mask,'p'].to_numpy())
        broad=d.cost.eq(cost)&d.comparison.eq('increment')
        d.loc[broad,'p_holm_all_designs']=holm(d.loc[broad,'p'].to_numpy())
        ablation=d.cost.eq(cost)&d.comparison.isin(['R-A2','R-A3'])
        d.loc[ablation,'p_holm_ablations']=holm(d.loc[ablation,'p'].to_numpy())
        extra=d.cost.eq(cost)&d.kind.eq('signal_replacement')
        d.loc[extra,'p_holm_signal_strategies']=holm(d.loc[extra,'p'].to_numpy())
    d.to_csv(OUT/'all_comparisons.csv',index=False)
    perf=pd.concat([pd.read_csv(OUT/'evaluations'/s/'performance.csv') for s in ERAS],ignore_index=True)
    perf.to_csv(OUT/'all_performance.csv',index=False)
    primary=d.query("cost=='x1' and kind=='regime_filter' and comparison=='increment'").copy()
    primary.to_csv(OUT/'primary_filter_tests.csv',index=False)
    candidates=primary[(primary.delta>0)&(primary.p<.05)]
    candidates.to_csv(OUT/'positive_nominal_candidates.csv',index=False)
    print('Primary filter tests',len(primary),'nominal positive',len(candidates),'Holm positive',int(((primary.delta>0)&(primary.p_holm_filters<.05)).sum()),flush=True)
    print(candidates[['study','era','trial','delta','ci_low','ci_high','p','p_holm_sample','p_holm_filters']].round(6).to_string(index=False),flush=True)
    sensitivity=[]
    for row in candidates.itertuples():
        daily=pd.read_csv(OUT/'evaluations'/row.study/'daily_x1.csv',index_col='date',parse_dates=True).loc[row.start:row.end]
        for block in [10,60]:
            sensitivity.append(dict(study=row.study,era=row.era,trial=row.trial,
                **bootstrap(daily[row.treatment],daily[row.control],block=block)))
    pd.DataFrame(sensitivity).to_csv(OUT/'candidate_block_sensitivity.csv',index=False)
    (OUT/'completion.json').write_text(json.dumps(dict(primary_tests=len(primary),nominal_positive=len(candidates),
         adjusted_positive=int(((primary.delta>0)&(primary.p_holm_filters<.05)).sum()),
         code_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'backtest/historical_sweep.py',ROOT/'backtest/historical_selection.py']}),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['evaluate','summarize']);p.add_argument('--study',choices=list(ERAS));a=p.parse_args()
    if a.action=='summarize':summarize()
    elif a.study:print(run_study(a.study),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=4) as ex:
            jobs=[ex.submit(run_study,s) for s in ERAS]
            for f in as_completed(jobs):print(f.result(),flush=True)
        summarize()


if __name__=='__main__':main()
