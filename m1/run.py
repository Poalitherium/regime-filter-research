"""Audit data and export sourced fee scenarios; never execute a strategy."""
import argparse
from pathlib import Path
import pandas as pd
from .audit import audit
from .exposure_cost_model import BASE_COST_BPS, transaction_costs
from .cost_model import load_schedule, execution_cost, position_legs


def run(data_dir,out_dir):
    out_dir=Path(out_dir)
    a=audit(data_dir,out_dir)
    schedule=load_schedule()
    if set(a.code)!=set(schedule):
        raise ValueError('data universe differs from fee schedule')
    pd.DataFrame(schedule.values()).to_csv(out_dir/'fee_schedule.csv',index=False)
    examples=[]
    for code in schedule:
        row=execution_cost(code,1,'2026-10-19',mode='snapshot_scenario',allow_proxy=True,slippage_per_contract=0.)
        examples.append(dict(example='synthetic_one_contract_one_side',**row))
    for name,before,after in [('open',{}, {'ESZ6':1}),('reverse',{'ESZ6':1},{'ESZ6':-1}),
                             ('roll',{'ESZ6':1},{'ESH7':1}),('reverse_and_roll',{'ESZ6':1},{'ESH7':-1})]:
        for leg in position_legs(before,after):
            row=execution_cost('ES',leg['quantity'],'2026-10-19',mode='snapshot_scenario',slippage_per_contract=0.)
            examples.append(dict(example=name,contract_id=leg['contract_id'],side=leg['side'],**row))
    for example in examples:
        example['slippage_scenario']='ZERO_SLIPPAGE_ASSUMPTION'
    pd.DataFrame(examples).to_csv(out_dir/'cost_examples.csv',index=False)
    pd.DataFrame(columns=['trade_date','code','contract_id','side','quantity','order_type','slippage_per_contract']).to_csv(out_dir/'execution_input_template.csv',index=False)
    assumptions=a[['code','asset_class','kind']].copy()
    assumptions['base_per_side_bps']=assumptions.asset_class.map(BASE_COST_BPS)
    if assumptions.base_per_side_bps.isna().any():
        raise ValueError('asset class has no sensitivity assumption')
    assumptions['status']='unverified research fee allowance; additional slippage specified separately'
    assumptions.to_csv(out_dir/'cost_assumptions.csv',index=False)
    scenarios=pd.concat([assumptions.assign(multiplier=m,per_side_bps=assumptions.base_per_side_bps*m)
                         for m in [0.,.5,1.,2.]])
    scenarios.to_csv(out_dir/'cost_scenarios.csv',index=False)
    names=['open_long','reverse','unchanged_roll','reverse_and_roll','close_on_roll','flat_roll']
    before=pd.Series([0.,1.,1.,1.,1.,0.],index=names)
    after=pd.Series([1.,-1.,1.,-1.,0.,0.],index=names)
    flags=pd.Series([False,False,True,True,True,True],index=names)
    exposure_examples=transaction_costs(before,after,flags,cost_bps=10.)
    exposure_examples['example_bps']=10.
    exposure_examples['status']='synthetic exposure accounting example; not strategy results'
    exposure_examples.rename_axis('example').to_csv(out_dir/'exposure_cost_examples.csv')
    print(a[['code','rows','missing_ret','ret_panel_mismatches','close_adj_panel_mismatches']].to_string(index=False))
    print('Fee examples are synthetic snapshot scenarios, not historical trading results.')
    return a


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path('data/final'))
    p.add_argument('--out-dir',type=Path,default=Path('m1/report/tables'))
    args=p.parse_args()
    run(args.data_dir,args.out_dir)
