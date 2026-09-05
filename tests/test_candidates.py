import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import pandas as pd
from collect_candidates import assess

class CandidateTests(unittest.TestCase):
    def frames(self):
        d=pd.bdate_range(end='2026-09-04',periods=400)
        q=pd.DataFrame({'date':d,'close':4+np.sin(np.arange(400)/10)/10,'amount':1e8})
        raw=q.copy();raw['close']/=2
        return q,raw,q.copy()
    def test_premium_uses_unadjusted_market_price(self):
        q,r,s=self.frames();nav=[{'FSRQ':'2026-09-04','DWJZ':str(r.iloc[-1]['close'])}]
        x=assess('159201',q,r,s,nav,'2026-09-04')
        self.assertAlmostEqual(x['closing_discount_premium'],0)
        self.assertEqual(x['status'],'DATA_READY_SHADOW')
    def test_lagged_nav_not_used_for_current_premium(self):
        q,r,s=self.frames();x=assess('513030',q,r,s,[{'FSRQ':'2026-09-03','DWJZ':'2'}],'2026-09-04')
        self.assertIsNone(x['closing_discount_premium']);self.assertEqual(x['status'],'WATCHLIST')
    def test_bad_secondary_cannot_promote_candidate(self):
        q,r,s=self.frames();s['close']=np.arange(len(s))+1
        x=assess('159201',q,r,s,[{'FSRQ':'2026-09-04','DWJZ':'2'}],'2026-09-04')
        self.assertEqual(x['status'],'NOT_EVALUABLE')
