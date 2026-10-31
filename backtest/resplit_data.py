"""Download public inputs needed to extend the 80/20 sensitivity study."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import requests
import json
import hashlib
from datetime import datetime, timezone
import re
import zipfile
import pandas as pd
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'backtest_outputs/resplit_80_20'

def probe():
    raw = OUT / 'raw'
    raw.mkdir(parents=True, exist_ok=True)
    urls = {
        'fredcsv': 'https://fred.stlouisfed.org/graph/graph.csv?id=VXVCLS&cosd=2015-01-01&coed=2026-10-15',
        'cftc': 'https://www.cftc.gov/files/dea/history/deacot2025.zip',
        'fed': 'https://www.federalreserve.gov/monetarypolicy/fomchistorical2015.htm',
        'move': 'https://query1.finance.yahoo.com/v8/finance/chart/%5EMOVE?range=max&interval=1d',
    }
    def get(kv):
        k,u=kv
        try:
            r=requests.get(u, timeout=30, headers={'User-Agent':'Mozilla/5.0'})
            (raw/(k+'.raw')).write_bytes(r.content)
            return k,r.status_code,len(r.content),r.text[:60] if k!='cftc' else r.content[:8].hex()
        except Exception as e:
            return k,str(e)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for item in ex.map(get,urls.items()):print(item,flush=True)

def download_inputs():
    raw=OUT/'raw';raw.mkdir(parents=True,exist_ok=True)
    start=int(datetime(2010,1,1,tzinfo=timezone.utc).timestamp())
    end=int(datetime(2026,10,16,tzinfo=timezone.utc).timestamp())
    urls={f'cot_{y}.zip':f'https://www.cftc.gov/files/dea/history/deacot{y}.zip' for y in range(2015,2027)}
    urls.update({f'fed_{y}.html':f'https://www.federalreserve.gov/monetarypolicy/fomchistorical{y}.htm' for y in range(2015,2021)})
    urls['fed_calendar.html']='https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
    for name,ticker in [('vix3m','VIX3M'),('ovx','OVX'),('gvz','GVZ')]:
        urls[name+'.csv']=f'https://cdn.cboe.com/api/global/us_indices/daily_prices/{ticker}_History.csv'
    urls['move_daily.json']=f'https://query1.finance.yahoo.com/v8/finance/chart/%5EMOVE?period1={start}&period2={end}&interval=1d'
    def get(kv):
        k,u=kv; path=raw/k
        if not path.exists():
            r=requests.get(u,timeout=45,headers={'User-Agent':'Mozilla/5.0'})
            r.raise_for_status();path.write_bytes(r.content)
        b=path.read_bytes()
        return dict(file=k,url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    records=[]
    with ThreadPoolExecutor(max_workers=5) as ex:
        for rec in ex.map(get,urls.items()):
            records.append(rec);print(rec['file'],rec['bytes'],flush=True)
    (OUT/'sources.json').write_text(json.dumps({'retrieved_utc':datetime.now(timezone.utc).isoformat(),'sources':records},indent=2))

def prepare_inputs():
    raw=OUT/'raw'; dest=OUT/'inputs';dest.mkdir(parents=True,exist_ok=True)
    end=pd.Timestamp('2026-10-15')
    # Preserve original vintages before 2015, append public daily observations.
    old=pd.read_csv(ROOT/'m4/data/implied_vol_2014.csv',index_col='date',parse_dates=True)
    new={}
    for col in ['vix3m','ovx','gvz']:
        d=pd.read_csv(raw/(col+'.csv'));d['DATE']=pd.to_datetime(d.DATE)
        val='CLOSE' if 'CLOSE' in d else col.upper()
        new[col]=d.set_index('DATE')[val].astype(float)
    j=json.loads((raw/'move_daily.json').read_text())['chart']['result'][0]
    if j['meta']['dataGranularity']!='1d':raise ValueError('MOVE must be daily')
    dates=pd.to_datetime(j['timestamp'],unit='s',utc=True).tz_convert('America/New_York').tz_localize(None).normalize()
    new['move']=pd.Series(j['indicators']['quote'][0]['close'],index=dates)
    recent=pd.DataFrame(new).loc['2015-01-01':end]
    combined=pd.concat([old,recent]).sort_index()
    assert not combined.index.duplicated().any()
    combined.rename_axis('date').to_csv(dest/'implied_vol.csv')
    # Only scheduled, two-day FOMC meetings. Emergency meetings/calls excluded.
    meetings=[]
    for year in range(2015,2021):
        soup=BeautifulSoup((raw/f'fed_{year}.html').read_text(encoding='utf-8-sig'),'html.parser')
        for h in soup.find_all('h5'):
            title=h.get_text(' ',strip=True)
            if not re.search(r'\d\s*[-–]\s*\d.*Meeting',title):continue
            section=[]
            for e in h.next_siblings:
                if getattr(e,'name',None)=='h5':break
                section.append(str(e))
            # Heading lives inside a panel; use its immediate parent if no sibling links.
            text=''.join(section)
            match=re.search(r'monetary(\d{8})a\.htm',text)
            if not match:match=re.search(r'monetary(\d{8})a\.htm',str(h.parent))
            if match:meetings.append(pd.Timestamp(match[1]))
    soup=BeautifulSoup((raw/'fed_calendar.html').read_text(encoding='utf-8-sig'),'html.parser')
    for row in soup.select('.fomc-meeting'):
        day=row.select_one('.fomc-meeting__date')
        if day is None or not re.search(r'\d\s*[-–]\s*\d',day.get_text()):continue
        m=re.search(r'monetary(\d{8})a\.htm',str(row))
        if m:meetings.append(pd.Timestamp(m[1]))
    recent=pd.DataFrame({'statement_date':sorted(set(d for d in meetings if pd.Timestamp('2015-01-01')<=d<=end))})
    recent['release_time_et']='14:00'
    old=pd.read_csv(ROOT/'m4/data/fomc_scheduled.csv',parse_dates=['statement_date'])
    pd.concat([old,recent]).to_csv(dest/'fomc_scheduled.csv',index=False)
    counts=recent.groupby(recent.statement_date.dt.year).size().to_dict()
    expected={**{y:8 for y in range(2015,2026)},2020:7,2026:6}
    if counts!=expected:raise ValueError(f'Check FOMC counts: {counts}')
    # Annual CFTC text files use different headers from the old XLS extract.
    from m4.cot import CFTC_CODES
    cols={'Market and Exchange Names':'market','As of Date in Form YYYY-MM-DD':'report_date',
     'CFTC Contract Market Code':'cftc_code','Open Interest (All)':'oi',
     'Noncommercial Positions-Long (All)':'nc_long','Noncommercial Positions-Short (All)':'nc_short',
     'Commercial Positions-Long (All)':'comm_long','Commercial Positions-Short (All)':'comm_short'}
    parts=[pd.read_csv(ROOT/'m4/data/cot_legacy_2014.csv',dtype={'cftc_code':str},parse_dates=['report_date'])]
    for year in range(2015,2027):
        with zipfile.ZipFile(raw/f'cot_{year}.zip') as z:
            d=pd.read_csv(z.open(z.namelist()[0]),usecols=list(cols),dtype={'CFTC Contract Market Code':str}).rename(columns=cols)
        d.cftc_code=d.cftc_code.str.strip();d=d[d.cftc_code.isin(CFTC_CODES.values())].copy()
        d['code']=d.cftc_code.map({v:k for k,v in CFTC_CODES.items()});d.report_date=pd.to_datetime(d.report_date)
        parts.append(d)
    cot=pd.concat(parts,ignore_index=True).sort_values(['code','report_date'])
    cot=cot[(cot.report_date<=end)&~cot.duplicated(['code','report_date'])]
    cot.to_csv(dest/'cot_legacy.csv',index=False)
    audit={c:{'first':str(combined[c].first_valid_index().date()),'last':str(combined[c].last_valid_index().date()),
      'nonmissing_after_2014':int(recent_c.notna().sum())} for c in combined for recent_c in [combined.loc['2015':,c]]}
    (dest/'coverage.json').write_text(json.dumps({'iv':audit,'fomc_counts':counts,
       'cot_counts':{str(k):int(v) for k,v in cot.groupby(cot.report_date.dt.year).size().items()},
       'move_source_change':'Old investing.com through 2014; Yahoo ICE BofA MOVE from 2015',
       'cboe_source_change':'Old FRED through 2014; underlying CBOE index daily series from 2015'},indent=2))
    print(json.dumps(audit),counts,flush=True)

if __name__=='__main__':
    import sys
    if '--prepare' in sys.argv:prepare_inputs()
    else:download_inputs()
