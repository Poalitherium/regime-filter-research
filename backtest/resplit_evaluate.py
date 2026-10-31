"""Common-book A2/R/A3 evaluation for the fixed post-hoc 80/20 design."""
import json
import numpy as np
import pandas as pd
from .resplit_common import ROOT,OUT,START,DEV_END,TEST_START,END,REUSED,NAMES,digest,write_spec
from .m1_bridge import m1_book,COST_MODELS
from .metrics import block_bootstrap_sharpe_diff
from m1.backtest import performance
from m2.pipeline import build_m2
from m3.predictability import holm

WINDOWS={'development':(START,DEV_END),'test':(TEST_START,END)}
PAIRS=(('A3','A2'),('R','A2'),('R','A3'))

def load_base():
    uni=pd.read_csv(ROOT/'data/final/universe.csv')
    data={};asset={};targets={'A1':{},'A2':{}}
    for row in uni.itertuples():
        d=pd.read_csv(ROOT/f'data/final/{row.code}.csv',index_col='date',parse_dates=True)
        data[row.code]=d;asset[row.code]=row.asset_class
        m=build_m2(pd.DataFrame({'date':d.index,'contract':row.code,'close':d.close_adj.to_numpy(),'return':d.ret.to_numpy()}))
        for ap,col in [('A1','approach1_target_close_t'),('A2','approach2_target_close_t')]:
            w=m.pivot(index='date',columns='factor',values=col)
            targets[ap][row.code]=w[['tsmom','ma_crossover','donchian']].mean(axis=1,skipna=False).reindex(d.index)
    dates=pd.DatetimeIndex(sorted(set().union(*(d.index for d in data.values()))))
    book=pd.DataFrame(True,index=dates,columns=list(data))
    window=dates[(dates>=pd.Timestamp(START))&(dates<=pd.Timestamp(END))]
    n=int(.8*len(window))
    assert len(window)==4981 and str(window[n-1].date())==DEV_END and str(window[n].date())==TEST_START
    return data,asset,targets,book

def metrics(port):
    return {**performance(port.net),'gross_sharpe':performance(port.gross)['sharpe'],
            'turnover_pa':port.turnover.mean()*252,'cost_pa':port.cost.mean()*252,
            'gross_exposure':port.gross_exposure.mean()}

def run():
    write_spec()
    if (OUT/'summary.csv').exists():raise FileExistsError('Preserve the completed run; choose a new output directory')
    data,asset,base,book=load_base()
    baseline={};base_rows=[]
    for window,(start,end) in WINDOWS.items():
        for cost in COST_MODELS:
            for ap in ['A1','A2']:
                port=m1_book(base[ap],data,asset,book,start,end,cost)
                baseline[window,cost,ap]=port
                base_rows.append(dict(window=window,cost_model=cost,approach=ap,**metrics(port)))
    pd.DataFrame(base_rows).to_csv(OUT/'baselines.csv',index=False)
    decisions={r['trial']:r for r in json.loads((OUT/'gate_decisions.json').read_text())}
    summary=[];comparisons=[];coverage=[];overlap=[];checks=[]
    inputs={}
    for tid in sorted(NAMES):
        path=ROOT/REUSED[tid][1] if tid in REUSED else OUT/'filters'/f'{tid}.csv'
        inputs[tid]={'file':str(path.relative_to(ROOT)).replace('\\','/'),'sha256':digest(path)}
    (OUT/'evaluated_filter_manifest.json').write_text(json.dumps(inputs,indent=2))
    for tid in sorted(NAMES):
        print('Evaluating',tid,NAMES[tid],flush=True)
        f=pd.read_csv(ROOT/inputs[tid]['file'],parse_dates=['date']).set_index(['contract','date']).filter_close_t
        targets={'R':{},'A3':{}}
        identity=not decisions.get(tid,{'passed':True})['passed']
        for code,d in data.items():
            ff=f.loc[code].reindex(d.index)
            w=ff.loc[START:END]
            if w.isna().any() or not np.isfinite(w).all():raise ValueError(f'{tid}/{code}: missing filter')
            if w.min()<0 or (tid!='M3-4' and w.max()>1+1e-12):raise ValueError('Invalid filter bound')
            for ap,source in [('R','A1'),('A3','A2')]:targets[ap][code]=base[source][code]*ff
            coverage.append(dict(trial=tid,contract=code,days=len(w),missing=int(w.isna().sum()),share_filter_below_one=float((w<1-1e-10).mean())))
        daily={}
        for window,(start,end) in WINDOWS.items():
            diag=[]
            for code,d in data.items():
                raw=base['A1'][code].loc[start:end];vt=base['A2'][code].loc[start:end]
                dd=pd.DataFrame({'signal':raw,'v':vt/raw.replace(0,np.nan),'f':f.loc[code].reindex(raw.index)}).dropna()
                diag.append(dd)
            dd=pd.concat(diag);both=(dd.v<1-1e-10)&(dd.f<1-1e-10)
            overlap.append(dict(trial=tid,window=window,both_cut_share=float(both.mean()),
             both_cut_signal_weighted_share=float(dd.signal.abs()[both].sum()/dd.signal.abs().sum()),
             filter_scale_corr=float(dd.f.corr(dd.v)) if dd.f.std()>0 else np.nan))
            for cost in COST_MODELS:
                ports={'A2':baseline[window,cost,'A2']}
                ports.update({ap:m1_book(t,data,asset,book,start,end,cost) for ap,t in targets.items()})
                for ap,p in ports.items():
                    pd.testing.assert_index_equal(p.index,ports['A2'].index)
                    if len(p)!=(3984 if window=='development' else 997):raise ValueError('Window mismatch')
                    if not np.isfinite(p.net).all():raise ValueError('Nonfinite return')
                    summary.append(dict(trial=tid,name=NAMES[tid],kind='volatility_benchmark' if tid=='M3-4' else 'regime_filter',
                       identity=identity,window=window,cost_model=cost,approach=ap,**metrics(p)))
                    for col in ['net','gross','cost','turnover','gross_exposure']:
                        daily[f'{window}_{cost}_{ap}_{col}']=p[col]
                if identity:
                    for ap,bp in [('A3','A2'),('R','A1')]:
                        err=float((ports[ap].net-baseline[window,cost,bp].net).abs().max())
                        np.testing.assert_allclose(ports[ap].net,baseline[window,cost,bp].net,rtol=0,atol=1e-14)
                        checks.append(dict(trial=tid,window=window,cost=cost,check=f'{ap}={bp}',max_abs_error=err))
                for treatment,control in PAIRS:
                    stat=block_bootstrap_sharpe_diff(ports[treatment].net,ports[control].net)
                    delta=metrics(ports[treatment])['sharpe']-metrics(ports[control])['sharpe']
                    if not np.isclose(delta,stat['d_sharpe']):raise ValueError('Sharpe mismatch')
                    comparisons.append(dict(trial=tid,name=NAMES[tid],window=window,cost_model=cost,
                      comparison=f'{treatment}-{control}',identity=identity,**stat))
        pd.DataFrame(daily).rename_axis('date').to_csv(OUT/f'{tid}_daily.csv')
    sm=pd.DataFrame(summary);pr=pd.DataFrame(comparisons)
    for window in WINDOWS:
        mask=pr.window.eq(window)
        assert mask.sum()==90
        pr.loc[mask,'p_holm_joint']=holm(pr.loc[mask,'p_boot'].to_numpy())
        for cost in COST_MODELS:
            m=mask&pr.cost_model.eq(cost);assert m.sum()==45
            pr.loc[m,'p_holm_cost']=holm(pr.loc[m,'p_boot'].to_numpy())
            main=m&pr.comparison.eq('A3-A2')&pr.trial.ne('M3-4')
            pr.loc[main,'p_holm_increment_14']=holm(pr.loc[main,'p_boot'].to_numpy())
    sm.to_csv(OUT/'summary.csv',index=False);pr.to_csv(OUT/'comparisons.csv',index=False)
    pd.DataFrame(overlap).to_csv(OUT/'overlap.csv',index=False)
    pd.DataFrame(coverage).to_csv(OUT/'filter_coverage.csv',index=False)
    pd.DataFrame(checks).to_csv(OUT/'identity_checks.csv',index=False)
    print(sm.query("window=='test' and cost_model=='official'").pivot(index='trial',columns='approach',values='sharpe').round(4).to_string(),flush=True)
    print(pr.query("window=='test' and cost_model=='official'")[[ 'trial','comparison','d_sharpe','ci_low','ci_high','p_boot','p_holm_cost','p_holm_joint']].round(4).to_string(index=False),flush=True)

def extras():
    data,asset,base,book=load_base();rows=[];pairs=[]
    for tid,strategies in [('M3-5',['dynamic','static_slow']),('M3-8',['hmm','tsmom_252'])]:
        target={}
        for st in strategies:
            t=pd.read_csv(OUT/'extra_targets'/f'{tid}_{st}.csv',parse_dates=['date']).set_index(['contract','date']).target
            target[st]={c:t.loc[c].reindex(d.index) for c,d in data.items()}
        target['M2_A2']=base['A2']
        for window,(start,end) in WINDOWS.items():
            for cost in COST_MODELS:
                ports={k:m1_book(v,data,asset,book,start,end,cost) for k,v in target.items()}
                for st,p in ports.items():rows.append(dict(trial=tid,strategy=st,window=window,cost_model=cost,**metrics(p)))
                for ctrl in [strategies[1],'M2_A2']:
                    stat=block_bootstrap_sharpe_diff(ports[strategies[0]].net,ports[ctrl].net)
                    pairs.append(dict(trial=tid,comparison=f'{strategies[0]}-{ctrl}',window=window,cost_model=cost,**stat))
    pd.DataFrame(rows).to_csv(OUT/'extra_strategy_summary.csv',index=False)
    p=pd.DataFrame(pairs)
    for window in WINDOWS:
        m=p.window.eq(window);p.loc[m,'p_holm_extras']=holm(p.loc[m,'p_boot'].to_numpy())
    p.to_csv(OUT/'extra_strategy_comparisons.csv',index=False)
    print(p.query("window=='test'").round(4).to_string(index=False),flush=True)

if __name__=='__main__':
    import sys
    if '--extras' in sys.argv:extras()
    else:run()
