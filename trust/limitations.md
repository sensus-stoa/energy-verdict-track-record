# Limitations - energy-verdict-track-record (record version 5)

## Sample size (stated next to every number)

- Sensitivity is measured on N=8 planted defective cases (8/8 detected),
  one per defect class. Wilson 95% CI [0.676, 1.000]. Eight detections
  across eight defect classes demonstrate the mechanism per class; they are
  not a reliability estimate for unseen defect variants or for repeated
  runs of the same class.
- Specificity is measured on N=5 clean cases (5/5 passed). Wilson 95% CI
  [0.566, 1.000].
- NC-NULL: N=5, false PASS 0/5, Wilson 95% CI [0.000, 0.434], rule-of-three
  upper bound about 60%. The sets are small by design; the numbers are not
  a power statement.

## Scope of the set

- This record version covers 8 defective + 5 clean cases. Version 4 covered
  6 defective + 5 clean; version 5 adds two defective cases (C-13, C-14)
  that pin down the roles of the claim conditions and of the trial-
  disclosure audit.
- Defect cases are single runs (N=1 each). Repetition across seeds/datasets
  is the next record version.
- The record carries machine verification only; the sign-off status is
  stated in trust/verdict.json (an agent never signs).

## Data

- Synthetic data for a single city profile: daily (425 days, seed 42) and
  hourly (8760 hours of 2023, seed 43, naive UTC index). No holidays, no
  outages, no price effects, one annual temperature cycle. Real datasets
  (ENTSO-E or similar) are planned for a later record version.
- Claim thresholds (daily 3.5% of mean load; hourly 6.0% of mean load) were
  calibrated on these generators BEFORE the criteria lock (pilot:
  src/pilot_calibration.py, committed and disclosed; in the manifest it is
  listed under disclosures, not evidence - it is not evidence). Generator
  parameters were chosen for realism (day-ahead forecast error std about
  1.9 deg C daily, about 1.2 deg C hourly), not to favour any outcome.

## Role of the claim thresholds and baselines

- Daily claim: MAE <= 3.5% of mean test-window load AND better than the
  persistence and seasonal-naive baselines. Hourly claim: MAE <= 6.0% of
  mean hourly load AND better than the 24h/168h baselines. The absolute
  percentage is an UPPER BOUND on error, not a skill criterion: on hourly
  data the seasonal-naive baseline itself sits at 5.52% of mean load, below
  the 6.0% gate (demonstrated by defect_07 / claim C-13). Skill comes from
  the must-beat conditions.
- The must-beat conditions check the two pre-registered cadence baselines
  only. A pipeline built on an arbitrary combination of those baselines
  (for example a fixed mixture of the 24h and 168h lags, which on this data
  lands near 129 MW = 5.2% of mean load and beats both baselines
  individually) is NOT separately baselined in this version: such a
  pipeline would pass the claim conditions. Constraining the baseline set
  to "the best achievable combination of the baselines" is a possible
  future criterion and is not part of this record version.

## Procedure boundaries

- Feature availability at inference time is an AUDIT-SIDE map frozen in
  trust/claims.md, derived from the data generator semantics. In a real
  engagement this map comes from the client intake and is verified by the
  auditor against the data schema. A pipeline could in principle misreport
  semantics; the battery does not defend against a fully adversarial
  submitter, only against the planted defect classes.
- The audit battery re-executes the case's own feature builder
  (feature_frame_fn) and fit/eval functions (retrain_eval_fn): the
  proximity and timezone audits inspect case-built frames through the
  builder, and deployment-gap, drop-test and replication checks re-run the
  pipeline under modified inputs. What entered the fit is read from the
  fitted model (sklearn feature_names_in_), not from self-reports. A
  pipeline that lies INSIDE its builder or eval functions in ways invisible
  through refits could evade the battery; code review of submitted
  pipelines is out of scope for this version.
- The trial-disclosure fixed baseline (the largest feature set common to
  all disclosed trial configurations) is evaluated through the case's own
  retrain-eval function; the audit assumes the disclosed trial log is
  complete and the retrain function is honest. A pipeline that hides part
  of its search or rigs its retrain function cannot be caught by the
  trial-disclosure audit; the claim and deployment audits still apply to
  whatever is submitted.
- The deployment-gap classification is relative:
  relative_gap = (deployment_mae - reported_mae) / reported_mae (denominator:
  the reported MAE on the pre-registered test window). Using an unavailable
  feature is itself the violation and fails the audit regardless of the
  gap; the 20% threshold classifies the deployment impact.
- The DST detection is structural: it flags duplicated wall-clock hours and
  missing local hours in the working frame (the missing-hours check now
  covers both naive and tz-aware indexes). It does not enumerate every
  possible timezone bug; a bug that leaves the frame structurally clean but
  mislabels semantics (for example a wrong fixed UTC offset) is not
  detected in this version.
- The selection-bias cases disclose their trial logs. The trial-disclosure
  replication standard (a material test-window edge must remain strictly
  positive on the confirmation window) is a sign-level rule, not an
  effect-size estimate: a positive-but-shrunk confirmation edge passes it.
- The proximity drop-test and the deployment re-run assume the model family
  can be refit quickly on feature subsets; heavy pipelines would need
  sampling budgets not implemented here.
- NC-NULL scope: the 7-day block permutation destroys exogenous and calendar
  alignment but preserves within-block autocorrelation and the marginal
  distribution of the target. The control verifies that the procedure
  refuses to certify claims on alignment-destroyed data; it is not an
  exhaustive null battery and it does not test pipelines whose claimed skill
  rests only on short-lag autocorrelation.

## Changes in version 5 (honest changelog from version 4)

Version 5 is a new record version with a new criteria lock and a full
re-run; it supersedes V-EVTR-004. It was prompted by an independent review
of version 4, which found real defects. Everything below is fixed in this
version and re-verified on the locked code:

- BUG FOUND IN v4 (claim C-11): the trial-disclosure audit compared the
  best-of-K configuration only against the seasonal-naive baseline. A
  calendar+noise search beats that weak baseline on BOTH windows, so in v4
  the audit returned "ok" for defect_06 and the FAIL came only from the
  claim-conditions audit - while the claim text asserted that the
  trial-disclosure audit classifies the outcome as a selection artifact.
  v5 adds the fixed unsearched baseline (intersection of all disclosed
  trial configurations) as a second replication edge; defect_06's
  trial-disclosure audit now returns fail/NOISE (selection gain 6.1% on
  test, -1.3% on confirmation), and a new case (defect_08, C-14)
  demonstrates audit #8 firing while the pre-registered claim holds.
- BUG FOUND IN v4 (evidence E-24..E-34): five ok-branches of the audit
  battery passed the detail text and the metrics dict positionally into
  AuditResult, so the detail text landed in reason_code and the metrics
  dict landed in detail, corrupting the per-case JSON evidence (the
  `status == "ok"` rows carried text in reason_code and objects in
  detail). v5 passes all AuditResult fields by name; every ok audit now
  has reason_code = "", a string detail and a dict metrics.
- BUG FOUND IN v4 (windows): the declared window row counts (test 30 rows
  = [335, 365), confirmation 60 rows = [365, 425)) did not match the actual
  label bounds, which produced 31 test rows and 59 confirmation rows and
  silently discarded 11 days in the cherry-pick comparison. v5 aligns the
  bounds to exactly 30/60 rows; ALL daily numbers changed accordingly and
  the whole battery was re-run.
- DEFECT FOUND IN v4 (audit #5): LOOKAHEAD_GAP_THRESHOLD was declared but
  never used; the C-01 criterion (gap > 20%) was not enforced anywhere.
  v5 fixes the contract: the structural use of an unavailable feature is
  itself the fail; the relative gap is computed, compared against the 20%
  threshold and reported with the classification in the audit metrics and
  detail.
- FIXED IN v5 (audit #3): the missing wall-clock hours check ran only for
  naive indexes; tz-aware frames skipped it. Now both are checked (tz-aware
  frames are converted to the domain timezone). Also the prediction-
  alignment audit now rejects duplicated prediction timestamps, which a
  set-difference alone did not catch.
- FIXED IN v5 (runner): `--case <id>` silently ran ALL cases because
  `--cases` defaulted to "all", overwriting the full results/track_record.csv
  with a single-case run. Now `--case` (or an explicit `--cases` list) runs
  only the named cases; per-case JSONs are rewritten for the cases actually
  run; the full CSV is rewritten only when the run covers all registered
  cases. The NC-NULL loop also used the last processed case's dataset
  variable instead of the daily dataset (latent bug, same root cause in the
  pilot script) - fixed to the daily data.
- NEW CASES in v5: defect_07_seasonal_naive_only (C-13: the hourly 6% gate
  is an upper bound; a baseline shipped as a forecast passes it and is
  caught by must-beat) and defect_08_search_overfit (C-14: audit #8 fires
  while the claim-conditions audit passes - independence of the two
  audits). Sensitivity references updated from 6/6 to 8/8.
- RECORD-KEEPING changes in v5: the broken external reference
  "V-PTM-001 (claim C-04)" was removed from claims/manifest/verdict (the
  point-shuffle caveat is now stated on its own); the C-01 denominator
  formula is explicit; percentages are rounded uniformly to two decimals;
  pilot_calibration.py moved from the manifest's evidence array to a
  separate disclosures section (it is not evidence); the manifest states
  that E-IDs are bound to a specific record version and content is
  identified by record_version + evidence_id + sha256; the trust check is
  now a committed tool (tools/trust_check.py) and its report carries a
  relative root path and the checked commit instead of a local absolute
  path.
- Non-change note: criteria C-01..C-12 from version 4 are restated with
  updated reference values where the window fix changed numbers; C-11's
  PASS criterion is restated to match the reworked trial-disclosure audit
  (see the bug entry above). All V-runs were re-executed after the v5 lock.

## Status of verification

- Published at https://github.com/sensus-stoa/energy-verdict-track-record
  (branch master); pushed after the record files were committed.
- All runs are author-executed (via an automated agent session) and
  deterministic: daily seed 42, hourly seed 43, null-control seeds 1..5.
  Re-running the committed commands on the committed data must reproduce
  every number.
- No independent reproduction exists yet. Human sign-off has not happened:
  trust/signoff.md is intentionally absent; the machine output in
  trust/verdict.json is verification, not a signed verdict.
- No external dataset has been used yet.
