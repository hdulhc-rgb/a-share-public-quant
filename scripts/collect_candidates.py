"""Public, independent candidate diagnostics; never expands core model inputs."""
import argparse
import json
import time
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from collect_public_snapshot import Transport, standard_curl, dp, write_json, now, verify_identity
from cn_sessions import previous_session

REGISTRY={'159201':('sz159201','华夏国证自由现金流ETF'),
          '159232':('sz159232','南方中证全指自由现金流ETF'),
          '513030':('sh513030','华安德国DAX ETF')}

def assess(code,qfq,raw,secondary,nav_rows,cutoff):
    diagnostics=dp._compare_adjusted_returns(qfq,secondary)
    close_date=raw.date.max().date().isoformat()
    close=float(raw.sort_values('date').iloc[-1]['close'])
    aligned=next((n for n in nav_rows if n.get('FSRQ')==close_date),None)
    nav=float(aligned['DWJZ']) if aligned and aligned.get('DWJZ') else None
    premium=close/nav-1 if nav and nav>0 else None
    valid=diagnostics['secondary_total_return_gate_passed'] and close_date==cutoff and qfq.date.max().date().isoformat()==cutoff and raw.date.equals(qfq.date)
    status='DATA_READY_SHADOW' if valid and len(qfq)>=315 and nav else 'WATCHLIST'
    if not valid:status='NOT_EVALUABLE'
    if code=='513030' and valid:status='WATCHLIST'
    return {'code':code,'status':status,'data_cutoff':close_date,'history_rows':len(qfq),
        'dual_source_diagnostics':diagnostics,'unadjusted_close':close,'aligned_nav':nav,
        'aligned_nav_date':close_date if nav else None,'closing_discount_premium':premium,
        'premium_semantics':'END_OF_DAY_SAME_DATE_ONLY; NOT_EXECUTABLE_INTRADAY_PREMIUM',
        'avg_amount_20d':float(raw.tail(20)['amount'].mean()),
        'core_universe_changed':False,'execution_permission':False,
        'missing':['LIVE_EXECUTION_PREMIUM','CURRENT_FEES_TRACKING_AND_SUBSCRIPTION_REVIEW']+
                  (['DAX_ECONOMIC_FX_WRAPPER_DECOMPOSITION'] if code=='513030' else ['STRATEGIC_LEG_PREREGISTRATION']),
        'nav_state':'ALIGNED' if nav else 'NOT_EVALUABLE'}

def main():
    p=argparse.ArgumentParser();p.add_argument('--as-of',required=True);p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--codes',default=','.join(REGISTRY));args=p.parse_args()
    verify_identity();cutoff=previous_session(args.as_of);start=pd.Timestamp('2014-01-01');end=pd.Timestamp(cutoff)
    out=args.output_dir;out.mkdir(parents=True,exist_ok=True);entries=[];hashes={};began=time.monotonic()
    for code in args.codes.split(','):
        if code not in REGISTRY:raise ValueError('UNREGISTERED_PUBLIC_CANDIDATE')
        transport=Transport(budget_seconds=max(1,min(120,360-(time.monotonic()-began))),curl_runner=standard_curl)
        try:
            symbol,name=REGISTRY[code];proxy=dp.AssetProxy('CANDIDATE_'+code,code,symbol,name)
            with patch.object(dp,'_request_text',transport.request):
                qfq=dp.fetch_eastmoney(proxy,start,end,'qfq');raw=dp.fetch_eastmoney(proxy,start,end,'unadjusted')
                secondary=dp.fetch_tencent(proxy,qfq.date.min(),end)
            frames={'qfq':qfq,'unadjusted':raw,'secondary':secondary}
            for label,frame in frames.items():
                path=out/(code+'_'+label+'.csv');frame.to_csv(path,index=False);hashes[path.name]=dp.sha256_file(path)
            payload=json.loads(transport.request('https://api.fund.eastmoney.com/f10/lsjz',
                     {'fundCode':code,'pageIndex':1,'pageSize':20,'startDate':'','endDate':''}))
            path=out/(code+'_nav.json');write_json(path,payload);hashes[path.name]=dp.sha256_file(path)
            entry=assess(code,qfq,raw,secondary,(payload.get('Data') or {}).get('LSJZList') or [],cutoff)
        except Exception as e:
            entry={'code':code,'status':'NOT_EVALUABLE','failure':type(e).__name__+':'+str(e)[:300],
                   'core_universe_changed':False,'execution_permission':False}
        entry['source_requests']=transport.events;entries.append(entry)
    write_json(out/'candidate_manifest.json',{'as_of':args.as_of,'required_cutoff':cutoff,'generated_at':now(),
        'layer':'INDEPENDENT_CANDIDATES','candidates':entries,'file_hashes':hashes,
        'orders_generated':False,'elapsed_ms':round((time.monotonic()-began)*1000)})
    print(json.dumps([{'code':x['code'],'status':x['status']} for x in entries]))

if __name__=='__main__':main()
