# Public collector operational repair

The consumer referenced a scheduled `public_state` packet that was absent from
the repository. The default branch had only model tests; a historical description
of another collector is not proof of deployment. A local HTTP 502 is not proof
that the origin provider is down: the egress path may also cause the failure.

This change adds an operational adapter and two workflows. It does not change the
model or the Grade B return-proxy data contract. Model code is separately checked
out at `6ad8c97af188604cd83f0c5daada15eeda9e6aa7` and all 28 recorded model files
are verified before collection. The adapter's commit must be recorded separately
from MODEL_COMMIT. Four numerical dependencies are pinned; transitive versions
are recorded in the workflow artifact, not claimed to be fully locked.

## Acceptance and lifecycle

1. Run adapter and fixed-model regression tests. This does not establish live
   source availability.
2. Review/merge this infrastructure PR into the default branch. Scheduled Actions
   do not execute merely because a workflow exists in an unmerged PR.
3. The infrastructure merge triggers an initial `public-collector` run. Manual
   retries use workflow_dispatch. Verify both its job conclusion
   and published `public_state/latest_attempt.json` and `latest_success.json`.
4. Enable consumption only after a complete source packet passes the unchanged
   production validator, manifest hash, model identity and requested cutoff.
   Resolve both pointers at the same public_state branch commit.
5. Keep the existing weekly consumer cadence. Until acceptance succeeds, report
   COLLECTOR_NOT_DEPLOYED or DATA_MISSING, not a recovered data pipeline.

Public collection is scheduled Saturday 02:00 UTC. Its bounded request policy is
three total attempts, at most 20 seconds per request and an eight-minute request
budget. HTTP 403/404 are not retried; 429/5xx and connection failures are. JSON,
identity, hash and data errors close the run. No alternate network route, provider
substitution or single-source downgrade is implemented.

Every completed run has an immutable attempt receipt and terminal checkpoint.
The `latest_attempt` pointer records failure independently of `latest_success`.
Interrupted RUNNING attempts must be treated as INTERRUPTED based on the terminal
Actions conclusion; never interpret a checkpoint alone as a live process. A new
attempt uses a new run id, not an overwritten prior receipt. Publication failure
remains visible in the job and diagnostic artifact; computation success is not
publication success.

Ordinary weeks use the last accepted packet plus a 28-calendar-day overlap and
new dates. Base hashes and the pinned production validator run before reuse.
Any overlap revision (including qfq rescaling or date changes) triggers a full
rebuild in a different packet directory. First runs and first-week-of-month runs
use a full build. This sampling cannot detect historical revisions strictly
outside the overlap until a full rebuild. Source-derived returns are recomputed
from the complete frozen packet; no return history is silently patched in place.

The wrapper refuses a cutoff different from the explicitly requested last complete
session, even if the legacy seven-day freshness gate would pass. The workflow uses
the preceding weekday as a conservative default. Exchange holidays require an
authoritative expected-session update in a future calendar adapter; until then a
holiday mismatch fails closed. Manual weekday runs intentionally fetch through
the previous complete weekday, never the current day's partial bar.

## Consumer and forward evidence

The old `public_state/valuation_history.csv` description is not this ETF panel
contract. Consumers must read `latest_attempt.json` first, then the last successful
pointer as a historical reference. A failed latest attempt cannot become NO_ACTION
merely because the old success fingerprint has not changed. A newly accepted
packet still requires private-side target generation, pinned-model gates and the
existing investment audit. This public workflow never receives private inputs.

Append forward snapshots only when actually emitted. Missing signals remain GAP;
do not import archived August targets retrospectively. Settle existing unexpired
decisions before considering the newest signal; the fact that the newest signal
has no later bar must not block older eligible decisions. This infrastructure
repair does not assert a completed private forward settlement engine or substitute
historical backtests for forward returns. An empty ledger stays NOT_EVALUABLE.

The production strategy, candidate universe, execution permissions, historical
validation policy and private forward accounting are outside this public adapter.
