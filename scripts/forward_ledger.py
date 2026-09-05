"""Append-only close-execution shadow accounting, separate from frozen models.

All new snapshots are frozen before their first possible execution. The first
execution close has cost only; its close-to-close return is never earned.
Afterwards positions drift, including the registered cash execution proxy.
"""
import copy
import hashlib
import json
import math
from datetime import date, datetime, timedelta

from cn_sessions import VERSION as CALENDAR_VERSION, TZ, next_execution, session

ENGINE_VERSION = 'FORWARD_CLOSE_V1'
CONTRACT = {'version': ENGINE_VERSION, 'calendar_version': CALENDAR_VERSION,
            'execution': 'FIRST_DOMESTIC_CLOSE_AFTER_SIGNAL_AND_ACTUAL_AVAILABILITY',
            'returns': 'NEXT_CLOSE_AFTER_EXECUTION; BUY_AND_HOLD_DRIFT_BETWEEN_SIGNALS',
            'cost': 'ONE_WAY_TURNOVER_TIMES_10_BPS_AT_REBALANCE_CLOSE',
            'benchmark': 'REGISTERED_STRATEGIC_WEIGHTS; SAME_EXECUTION_AND_COST_RULE',
            'missing_prices': 'FAIL_CLOSED_NO_SKIP_NO_ZERO_FILL', 'orders': False}

def digest(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def weights(w, assets):
    if set(w) != set(assets) or any(not math.isfinite(v) or v < 0 for v in w.values()) or abs(sum(w.values())-1) > 1e-9:
        raise ValueError('INVALID_FORWARD_WEIGHTS')
    return {a:float(w[a]) for a in sorted(assets)}

def seal(record):
    record['record_hash'] = digest({k:v for k,v in record.items() if k != 'record_hash'})
    return record

def check_records(records):
    for r in records:
        if r.get('record_hash') != digest({k:v for k,v in r.items() if k != 'record_hash'}):
            raise ValueError('IMMUTABLE_FORWARD_RECORD_CHANGED')

def register(ledger, decision, available_at):
    """Decision carries model identity, signal, input fingerprint and all targets."""
    result = copy.deepcopy(ledger)
    records = result.setdefault('decision_snapshots', [])
    check_records(records)
    if result.get('forward_contract', CONTRACT) != CONTRACT:
        raise ValueError('FORWARD_CONTRACT_MISMATCH')
    key = digest({k:decision[k] for k in ['model_version','signal_date','input_fingerprint']})
    for old in records:
        if old['decision_id'] == key:
            semantic = lambda x: {k:v for k,v in x.items() if k != 'run_provenance'}
            if semantic(old['decision']) != semantic(decision): raise ValueError('DECISION_ID_CONFLICT')
            return result
    assets = list(decision['benchmark'])
    weights(decision['initial_weights'],assets); weights(decision['benchmark'],assets)
    if not decision['targets']: raise ValueError('NO_FORWARD_TARGETS')
    for w in decision['targets'].values(): weights(w,assets)
    if not session(decision['signal_date']): raise ValueError('SIGNAL_NOT_A_SESSION')
    observed = datetime.fromisoformat(available_at)
    if observed.tzinfo is None: raise ValueError('AVAILABILITY_TIMEZONE_REQUIRED')
    if str(decision['signal_date'])[:10] >= observed.astimezone(TZ).date().isoformat():
        # This weekly contract deliberately accepts only already-closed prior dates.
        raise ValueError('SIGNAL_NOT_PRIOR_TO_REGISTRATION')
    if records:
        prev = records[-1]
        if decision['signal_date'] <= prev['decision']['signal_date']: raise ValueError('BACKFILLED_OR_DUPLICATE_SIGNAL')
        if datetime.fromisoformat(prev['available_at']) > observed: raise ValueError('AVAILABILITY_NOT_MONOTONIC')
        if set(prev['decision']['targets']) != set(decision['targets']) or prev['decision']['benchmark'] != decision['benchmark']:
            raise ValueError('FORWARD_UNIVERSE_OR_BENCHMARK_CHANGED')
    planned = next_execution(decision['signal_date'],available_at)
    if result.get('settlements') and planned <= result['settlements'][-1]['date']:
        raise ValueError('BACKFILLED_EXECUTION')
    records.append(seal({'decision_id':key,'decision':copy.deepcopy(decision),
                         'available_at':available_at,'planned_execution_date':planned,
                         'execution_state_at_registration':'PENDING_FUTURE_CLOSE'}))
    result['forward_contract'] = CONTRACT
    result['engine_state'] = 'ACCEPTED'
    result['decision_count'] = len(records)
    return result


def main():
    import argparse
    from datetime import timezone
    from pathlib import Path
    import pandas as pd
    from collect_public_snapshot import MODEL_COMMIT, verify_identity, validate_packet, write_json
    p=argparse.ArgumentParser(description='Settle existing forward records then freeze a new shadow decision')
    p.add_argument('--ledger',required=True,type=Path)
    p.add_argument('--packet',required=True,type=Path)
    p.add_argument('--base-run',required=True,type=Path)
    p.add_argument('--data2-run',required=True,type=Path)
    p.add_argument('--as-of',required=True)
    p.add_argument('--output',required=True,type=Path)
    args=p.parse_args()
    verify_identity()
    from cn_sessions import previous_session
    cutoff=previous_session(args.as_of)
    validate_packet(args.packet,cutoff)
    snapshots=[]; provenance=[]
    for directory in [args.base_run,args.data2_run]:
        m=json.loads((directory/'run_manifest.json').read_text())
        if m['integrity']!='PASS' or m['orders_generated'] or m['historical_validation_uses_live_current']:
            raise ValueError('FORWARD_REQUIRES_ACCEPTED_SHADOW_RUN')
        for entry in m['artifacts']:
            target=(directory/entry['path']).resolve()
            if not target.is_relative_to(directory.resolve()): raise ValueError('ARTIFACT_PATH_ESCAPE')
            if hashlib.sha256(target.read_bytes()).hexdigest()!=entry['sha256']: raise ValueError('MODEL_ARTIFACT_HASH_MISMATCH')
        s=json.loads((directory/'decision_snapshot.json').read_text())
        if s['signal_date'][:10]!=cutoff: raise ValueError('SIGNAL_CUTOFF_MISMATCH')
        snapshots.append(s)
        provenance.append({'run_id':m['run_id'],'manifest_sha256':hashlib.sha256((directory/'run_manifest.json').read_bytes()).hexdigest()})
    if snapshots[0]['benchmark_weights']!=snapshots[1]['benchmark_weights'] or snapshots[0]['previous_actual_shadow_weights']!=snapshots[1]['previous_actual_shadow_weights']:
        raise ValueError('MODEL_PAIR_INPUT_MISMATCH')
    panel=pd.read_csv(args.packet/'returns.csv',float_precision='round_trip').set_index('date')
    source=json.loads(args.ledger.read_text())
    settled=settle(source,panel.to_dict(orient='index'),cutoff)
    decision={'model_version':'0.6.3+0.6.3-data.2','code_commit':MODEL_COMMIT,'signal_date':cutoff,
              'input_fingerprint':digest({'data':hashlib.sha256((args.packet/'returns.csv').read_bytes()).hexdigest(),
                                          'current':snapshots[0]['previous_actual_shadow_weights'],
                                          'benchmark':snapshots[0]['benchmark_weights']}),
              'initial_weights':snapshots[0]['previous_actual_shadow_weights'],
              'benchmark':snapshots[0]['benchmark_weights'],
              'targets':{s['model_version']+'/'+name:w for s in snapshots for name,w in s['challenger_actual_shadow_targets'].items()},
              'run_provenance':provenance,'data_grade':'B'}
    available=datetime.now(timezone.utc).isoformat()
    result=register(settled,decision,available)
    result['updated_at']=available
    result['last_engine_run']={'version':ENGINE_VERSION,'settled_through':cutoff,
       'new_settlements':len(result['settlements'])-len(source['settlements']),
       'new_decisions':len(result['decision_snapshots'])-len(source['decision_snapshots'])}
    write_json(args.output,result)
    print(json.dumps({'engine_state':result['engine_state'],'decisions':len(result['decision_snapshots']),
        'forward_return_days':result.get('forward_sample_count',0),
        'planned_execution':result['decision_snapshots'][-1]['planned_execution_date']}))
    return 0


def settle(ledger, returns, as_of):
    """Settle old decisions before caller appends this week's new one.

    `returns` maps ISO dates to frozen daily asset returns from an accepted panel.
    Existing daily records must reproduce; revised past returns fail closed.
    """
    result = copy.deepcopy(ledger)
    records = result.setdefault('decision_snapshots',[])
    old = result.setdefault('settlements',[])
    check_records(records); check_records(old)
    if result.get('forward_contract',CONTRACT) != CONTRACT: raise ValueError('FORWARD_CONTRACT_MISMATCH')
    if not records:
        result.update(forward_sample_count=0,decision_count=0,performance_state='NOT_EVALUABLE')
        return result
    assets = sorted(records[0]['decision']['benchmark'])
    by_date = {}
    for r in records:
        day = r['planned_execution_date']
        if day in by_date: raise ValueError('EXECUTION_DATE_COLLISION')
        by_date[day] = r
    start = min(by_date)
    if old and as_of < old[-1]['date']: raise ValueError('FORWARD_CUTOFF_REGRESSION')
    positions, nav, daily = {}, {}, []
    d = date.fromisoformat(start)
    cutoff = date.fromisoformat(as_of)
    while d <= cutoff:
        ds = d.isoformat(); d += timedelta(days=1)
        if not session(ds): continue
        if ds not in returns: raise ValueError('FORWARD_PRICE_GAP:'+ds)
        row = returns[ds]
        if set(row) != set(assets) or any(not math.isfinite(v) or v <= -1 for v in row.values()):
            raise ValueError('INVALID_FORWARD_RETURN')
        trade = by_date.get(ds)
        if not positions:
            dec = trade['decision']
            names = sorted(dec['targets']) + ['BENCHMARK']
            positions = {n:weights(dec['initial_weights'],assets) for n in names}
            nav = {n:1. for n in names}
            first = True
        else: first = False
        results = {}
        for name in positions:
            before = nav[name]
            gross = 0.0 if first else sum(positions[name][a]*row[a] for a in assets)
            nav[name] *= 1+gross
            if not first:
                positions[name] = {a:positions[name][a]*(1+row[a])/(1+gross) for a in assets}
            turnover = 0.
            if trade:
                dec = trade['decision']
                target = dec['benchmark'] if name == 'BENCHMARK' else dec['targets'][name]
                turnover = sum(abs(target[a]-positions[name][a]) for a in assets)/2
                if name != 'BENCHMARK' and turnover > .1+1e-8:
                    raise ValueError('FORWARD_EXECUTION_TURNOVER_EXCEEDS_10_PERCENT')
                nav[name] *= 1-turnover*0.001
                positions[name] = weights(target,assets)
            results[name] = {'gross_return':gross,'cost_rate':turnover*0.001,
                             'turnover':turnover,'net_return':nav[name]/before-1,
                             'nav':nav[name],'end_weights':copy.deepcopy(positions[name])}
        daily.append(seal({'date':ds,'daily_returns_hash':digest(row),
                          'execution_decision_id':trade['decision_id'] if trade else None,
                          'first_execution_no_market_return':first,'portfolios':results}))
    if daily[:len(old)] != old: raise ValueError('FORWARD_HISTORY_RECONCILIATION_FAILED')
    result['settlements'] = daily
    result['decision_count'] = len(records)
    result['executed_decision_count'] = sum(r['execution_decision_id'] is not None for r in daily)
    result['forward_sample_count'] = max(0,len(daily)-1)
    result['performance_state'] = 'EVALUABLE_SHORT_SAMPLE' if result['forward_sample_count'] else 'NOT_EVALUABLE'
    result['performance'] = {}
    if daily:
        for name in daily[-1]['portfolios']:
            peak, dd = 1., 0.
            for r in daily:
                wealth=r['portfolios'][name]['nav'];peak=max(peak,wealth);dd=min(dd,wealth/peak-1)
            end=daily[-1]['portfolios'][name]['nav']
            benchmark=daily[-1]['portfolios']['BENCHMARK']['nav']
            result['performance'][name]={'cumulative_return':end-1,'benchmark_return':benchmark-1,
                                         'excess_return':end-benchmark,'max_drawdown':dd}
    result['engine_state'] = 'ACCEPTED'
    result['forward_contract'] = CONTRACT
    return result


if __name__=='__main__':
    raise SystemExit(main())
