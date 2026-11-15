# Limitations - energy-verdict-track-record (record version 3)

## Sample size (stated next to every number)

- Sensitivity is measured on N=6 planted defective cases (6/6 detected),
  one per defect class. Wilson 95% CI [0.610, 1.000]. Six detections across
  six defect classes demonstrate the mechanism per class; they are not a
  reliability estimate for unseen defect variants or for repeated runs of
  the same class.
- Specificity is measured on N=5 clean cases (5/5 passed). Wilson 95% CI
  [0.566, 1.000].
- NC-NULL: N=5, false PASS 0/5, Wilson 95% CI [0.000, 0.434], rule-of-three
  upper bound about 60%. The sets are small by design; the numbers are not
  a power statement.

## Scope of the set

- This record version covers 6 defective + 5 clean cases. The Appendix A
  minimum (5 defective + 5 clean) is satisfied; the sixth defective class
  covers every defect class named in the work plan.
- Defect cases are single runs (N=1 each). Repetition across seeds/datasets
  is the next record version.
- Until the record is human-signed and published, no paid Engineering
  Verdict is offered (service brief, section 11).

## Data

- Synthetic data for a single city profile: daily (425 days, seed 42) and
  hourly (8760 hours of 2023, seed 43, naive UTC index). No holidays, no
  outages, no price effects, one annual temperature cycle. Real datasets
  (ENTSO-E or similar) are planned for a later record version.
- Claim thresholds (daily 3.5% of mean load; hourly 6.0% of mean load) were
  calibrated on these generators BEFORE the criteria lock (pilot:
  src/pilot_calibration.py, committed and disclosed). Generator parameters
  were chosen for realism (day-ahead forecast error std about 1.9 deg C
  daily, about 1.2 deg C hourly), not to favour any outcome.

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
- The DST detection is structural: it flags duplicated wall-clock hours and
  missing local hours in the working frame. It does not enumerate every
  possible timezone bug; a bug that leaves the frame structurally clean but
  mislabels semantics (for example a wrong fixed UTC offset) is not
  detected in this version.
- The selection-bias case discloses its trial log. A pipeline that hides
  its search cannot be caught by the trial-disclosure audit; the claim and
  deployment audits still apply to whatever is submitted.
- The proximity drop-test and the deployment re-run assume the model family
  can be refit quickly on feature subsets; heavy pipelines would need
  sampling budgets not implemented here.
- NC-NULL scope: the 7-day block permutation destroys exogenous and calendar
  alignment but preserves within-block autocorrelation and the marginal
  distribution of the target. The control verifies that the procedure
  refuses to certify claims on alignment-destroyed data; it is not an
  exhaustive null battery and it does not test pipelines whose claimed skill
  rests only on short-lag autocorrelation.

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
