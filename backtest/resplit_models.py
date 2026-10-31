"""Extend frozen model recipes and re-evaluate gates on pre-test data only."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .resplit_common import ROOT,OUT,START,DEV_END,TEST_START,END,REUSED,write_spec,digest

def meta():
    from m3 import metalabel as ml
    write_spec();dest=OUT/'filters';dest.mkdir(parents=True,exist_ok=True)
    panel=ml.build_panel(ROOT/'data/final',ROOT/'data/processed/offmarket/offmarket_daily.csv',
                         ROOT/'m3_outputs/bocpd_h250_filter.csv',end=END)
    audit=[]
    for year in range(2006,2027):
        refit=pd.Timestamp(year,1,1)
        tr=panel[(panel.label_end < refit-pd.offsets.BDay(ml.EMBARGO))&panel.y.notna()]
        audit.append(dict(refit=str(refit.date()),train_rows=len(tr),last_label=str(tr.label_end.max()),
                          embargo_bdays=ml.EMBARGO))
        assert tr.empty or tr.label_end.max()<refit-pd.offsets.BDay(ml.EMBARGO)
    pd.DataFrame(audit).to_csv(OUT/'meta_training_audit.csv',index=False)
    for tid,features,cols in [('M3-6','hand',ml.HAND),('M3-7','sig',ml.SIG)]:
        print('Extending',tid,flush=True)
        with threadpool_limits(limits=4):
            preds,auc=ml.walk_forward(panel,cols,'gbm',last_refit=2026)
        ml.write_filter(preds,ROOT/'data/final',dest/f'{tid}.csv')
        auc.to_csv(OUT/f'{tid}_auc.csv',index=False)
        new=pd.read_csv(dest/f'{tid}.csv',parse_dates=['date']).set_index(['contract','date']).filter_close_t
        old=pd.read_csv(ROOT/f'm3_outputs/meta_{features}_gbm_filter.csv',parse_dates=['date']).set_index(['contract','date']).filter_close_t
        # The old sample treated 2014-12-31 (Wednesday) as its terminal week-end.
        # With a longer sample that week correctly ends in January. Compare completed weeks.
        idx=old.index[old.index.get_level_values('date')<=pd.Timestamp('2014-12-26')]
        err=float((new.reindex(idx)-old.loc[idx]).abs().max())
        np.testing.assert_allclose(new.reindex(idx),old.loc[idx],equal_nan=True,rtol=1e-10,atol=1e-12)
        print(tid,'original development prefix reproduced',err,flush=True)
        (OUT/f'{tid}_reproduction.json').write_text(json.dumps({'max_abs_filter_difference':err,
          'compared_through':'2014-12-26','terminal_partial_week':'Old December 31 partial-week forecast disappears when full week is available'}))

def extra_strategies():
    from m3 import regime_strategy as rs, turning_points as tp
    dest=OUT/'extra_targets';dest.mkdir(parents=True,exist_ok=True)
    for tid,build in [('M3-5',tp.build_targets),('M3-8',rs.build_targets)]:
        with threadpool_limits(limits=4):result=build(ROOT/'data/final')
        data=result[1];tg=result[4] if tid=='M3-5' else result[2]
        for strategy,targets in tg.items():
            rows=[pd.DataFrame({'date':s.index,'contract':code,'target':s.values}) for code,s in targets.items()]
            pd.concat(rows,ignore_index=True).to_csv(dest/f'{tid}_{strategy}.csv',index=False)
        print('Saved extra strategy targets',tid,flush=True)

if __name__=='__main__':
    import sys
    if '--extras' in sys.argv:extra_strategies()
    else:meta()
