import copy
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import forward_ledger as f
from cn_sessions import previous_session,next_execution

class ForwardTests(unittest.TestCase):
    def decision(self, signal='2026-09-04', fingerprint='one'):
        return {'model_version':'TEST_ONLY','signal_date':signal,'input_fingerprint':fingerprint,
                'initial_weights':{'A':.5,'B':.5},'benchmark':{'A':.5,'B':.5},
                'targets':{'candidate':{'A':.6,'B':.4}}}
    def ledger(self):
        return f.register({'decision_snapshots':[],'settlements':[]},self.decision(),'2026-09-05T14:00:00+08:00')
    def panel(self):
        return {'2026-09-07':{'A':.9,'B':-.5},'2026-09-08':{'A':.1,'B':0},
                '2026-09-09':{'A':0,'B':-.1},'2026-09-10':{'A':0,'B':0},'2026-09-11':{'A':0,'B':0}}
    def test_first_execution_cannot_earn_prior_close_return(self):
        x=f.settle(self.ledger(),self.panel(),'2026-09-07')
        self.assertAlmostEqual(x['performance']['candidate']['cumulative_return'],-.0001)
        self.assertAlmostEqual(x['performance']['candidate']['max_drawdown'],-.0001)
        self.assertEqual(x['performance']['BENCHMARK']['cumulative_return'],0)
        self.assertEqual(x['forward_sample_count'],0)
    def test_buy_and_hold_drifts_instead_of_daily_rebalance(self):
        x=f.settle(self.ledger(),self.panel(),'2026-09-09')
        expected=(1-.0001)*(.6*1.1+.4*.9)
        self.assertAlmostEqual(x['settlements'][-1]['portfolios']['candidate']['nav'],expected)
        self.assertAlmostEqual(x['settlements'][1]['portfolios']['candidate']['end_weights']['A'],.66/1.06)
    def test_incremental_settlement_preserves_existing_bytes(self):
        a=f.settle(self.ledger(),self.panel(),'2026-09-08')
        b=f.settle(a,self.panel(),'2026-09-11')
        self.assertEqual(a['settlements'],b['settlements'][:2])
        self.assertEqual(b,f.settle(b,self.panel(),'2026-09-11'))
    def test_old_signal_settles_even_when_latest_is_immature(self):
        a=f.register(self.ledger(),self.decision('2026-09-11','two'),'2026-09-12T14:00:00+08:00')
        b=f.settle(a,self.panel(),'2026-09-11')
        self.assertEqual(b['forward_sample_count'],4)
        self.assertEqual(b['executed_decision_count'],1)
        self.assertEqual(b['decision_count'],2)
    def test_missing_session_fails_without_filling_zero(self):
        p=self.panel();del p['2026-09-08']
        with self.assertRaisesRegex(ValueError,'PRICE_GAP'):f.settle(self.ledger(),p,'2026-09-09')
    def test_revision_of_already_settled_return_fails(self):
        a=f.settle(self.ledger(),self.panel(),'2026-09-08');p=self.panel();p['2026-09-08']['A']=.11
        with self.assertRaisesRegex(ValueError,'RECONCILIATION'):f.settle(a,p,'2026-09-09')
    def test_decision_tampering_fails(self):
        a=self.ledger();a['decision_snapshots'][0]['decision']['targets']['candidate']['A']=.7
        with self.assertRaisesRegex(ValueError,'IMMUTABLE'):f.settle(a,self.panel(),'2026-09-08')
    def test_duplicate_is_idempotent_but_conflict_rejected(self):
        a=self.ledger();self.assertEqual(a,f.register(a,self.decision(),'2026-09-05T15:00:00+08:00'))
        d=self.decision();d['targets']['candidate']={'A':.7,'B':.3}
        with self.assertRaisesRegex(ValueError,'CONFLICT'):f.register(a,d,'2026-09-05T15:00:00+08:00')
    def test_backfill_signal_rejected(self):
        with self.assertRaisesRegex(ValueError,'BACKFILLED'):f.register(self.ledger(),self.decision('2026-09-03','old'),'2026-09-06T14:00:00+08:00')
    def test_late_registration_cannot_claim_monday_close(self):
        self.assertEqual(next_execution('2026-09-04','2026-09-07T16:00:00+08:00'),'2026-09-08')
    def test_holidays_and_weekend_makeup_days(self):
        self.assertEqual(previous_session('2026-09-26'),'2026-09-24')
        self.assertEqual(previous_session('2026-10-03'),'2026-09-30')
        self.assertEqual(next_execution('2026-09-30','2026-10-01T12:00:00+08:00'),'2026-10-08')
        self.assertEqual(previous_session('2026-02-24'),'2026-02-13')
    def test_unknown_calendar_year_and_bad_weights_fail(self):
        with self.assertRaisesRegex(ValueError,'YEAR'):previous_session('2027-06-01')
        d=self.decision();d['targets']['candidate']['A']=float('nan')
        with self.assertRaises(ValueError):f.register({},d,'2026-09-05T14:00:00+08:00')

    def test_same_signal_new_run_provenance_keeps_first_snapshot(self):
        d=self.decision();d['run_provenance']=[{'run':'first'}]
        a=f.register({},d,'2026-09-05T14:00:00+08:00');d['run_provenance']=[{'run':'second'}]
        self.assertEqual(a,f.register(a,d,'2026-09-05T15:00:00+08:00'))

    def test_turnover_guard_applies_to_simulated_positions(self):
        d=self.decision();d['targets']['candidate']={'A':.8,'B':.2}
        a=f.register({},d,'2026-09-05T14:00:00+08:00')
        with self.assertRaisesRegex(ValueError,'TURNOVER'):f.settle(a,self.panel(),'2026-09-07')

if __name__=='__main__':unittest.main()
