# Forward close accounting contract v1

This operational evidence adapter is registered before its first live decision.
The optimizer remains pinned to 6ad8c97af188604cd83f0c5daada15eeda9e6aa7.
No optimizer parameters, core universe, historical targets or model files change.

Run `scripts/forward_ledger.py --ledger PRIVATE_LEDGER --packet VERIFIED_PACKET
--base-run BASE_RUN --data2-run DATA2_RUN --as-of YYYY-MM-DD --output PRIVATE_OUTPUT`.
Keep every input and output of this command private. The command verifies model
identity, packet gates and all model artifact hashes, settles existing records,
then registers the latest decision. Register only after full model gates pass.

Execution is a simulation at the first domestic ETF close strictly after both
the signal date and actual snapshot availability. A planned date is not a fill.
The execution close has costs only; its preceding close-to-close return is not
earned. Positions drift between new decisions. At a later execution close, old
positions earn that day's return before rebalancing and deducting costs.
Candidate turnover against simulated positions must not exceed 10%; otherwise
settlement fails closed rather than clipping or silently changing a frozen target.
The fixed strategic benchmark uses the same dates, drift and 10bp one-way cost
convention; it is an unconstrained reference, not an optimizer candidate.
All candidates continue to use the original signal's 4% tracking-error gate.

One-way turnover is half the L1 weight change. Cost is turnover times 0.001 and
is deducted multiplicatively at the rebalance close. Initial model portfolios
and the benchmark start from the same disclosed operating-sleeve weights.
No borrowing, leverage or negative weights are permitted. Cash uses the existing
511880 execution return proxy; this is not the user's bank cash yield.

Decisions and daily settlements are sealed and append-only. Settlement replays
must reproduce their existing prefix exactly. Missing expected sessions, revised
settled returns, modified records, backwards cutoffs and backfilled signals fail
closed. Re-running identical semantic input preserves the first decision and its
original availability timestamp even when a new model run ID is generated.
The valid decision key is model version + signal date + input fingerprint.

Report frozen decision count, executed decision count and observed forward return
days separately. With no day after execution, performance is NOT_EVALUABLE.
Drawdown includes initial wealth 1, so initial cost/loss cannot vanish. Report
candidate and benchmark cumulative return, excess and drawdown on identical dates.
This is prospective simulation, not a real account return or historical backtest.

`cn_sessions.py` contains a versioned 2026 domestic session baseline sourced from
the official SSE annual closure notice:
https://www.sse.com.cn/disclosure/announcement/general/c/c_20251222_10802507.shtml
It excludes holiday intervals and weekends, including government makeup weekends.
Unknown years fail closed pending a new official calendar. Actual execution still
requires all five instruments to have verified prices on that session; no overseas
calendar or QDII NAV date is substituted for the domestic trading calendar.

Grade B remains the ceiling: official economic total returns, point-in-time local
holdings, QDII FX/premium decomposition and optional external engines are separate
evidence enhancements, not missing prerequisites for this shadow accounting profile.
No strategy promotion, private-data publication or broker action is authorized by
this contract.
