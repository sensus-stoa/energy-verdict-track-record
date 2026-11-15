# Limitations - energy-verdict-track-record (record version 2)

## Sample size (stated next to every number)

- Sensitivity is measured on N=3 planted defective cases (3/3 detected).
  Wilson 95% CI [0.439, 1.000]. Three detections across three defect
  classes demonstrate the mechanism per class; they are not a reliability
  estimate for unseen defect variants.
- Specificity is measured on N=3 clean cases (3/3 passed). Wilson 95% CI
  [0.439, 1.000].
- NC-NULL: N=5, false PASS 0/5, Wilson 95% CI [0.000, 0.434], rule-of-three
  upper bound about 60%. The set is small by design; the numbers are not a
  power statement.

## Scope of the set

- This record version covers 3 defective + 3 clean cases of the Appendix A
  start set (5 + 5). Not implemented yet, no claims made: DST / timezone
  handling (ENVIRONMENT; needs hourly data with a timezone), cherry-picked
  metrics (CONFLICT_EVIDENCE), selection bias in best-of-K search (NOISE).
- Until the full set is run and published, no paid Engineering Verdict is
  offered (service brief, section 11).

## Data

- Synthetic daily data for a single city profile (425 days, fixed seed 42).
  No holidays, no outages, no price effects, one annual temperature cycle,
  no hourly structure. Real datasets (ENTSO-E or similar) are planned for a
  later record version; DST cases will need hourly data with a timezone.
- The claim threshold (3.5% of mean load) was calibrated on this generator
  BEFORE the criteria lock (pilot: src/pilot_calibration.py, committed and
  disclosed). Generator parameters were chosen for realism (day-ahead
  forecast error std about 1.9 deg C), not to favour any outcome.

## Procedure boundaries

- Feature availability at inference time is an AUDIT-SIDE map frozen in
  trust/claims.md, derived from the data generator semantics. In a real
  engagement this map comes from the client intake and is verified by the
  auditor against the data schema. A pipeline could in principle misreport
  semantics; the battery does not defend against a fully adversarial
  submitter, only against the planted defect classes.
- The audit battery re-executes the case's own feature builder
  (feature_frame_fn) and fit/eval functions (retrain_eval_fn): the
  proximity audit inspects case-built derived features through the builder,
  and deployment-gap, drop-test and replication checks re-run the pipeline
  under modified inputs. What entered the fit is read from the fitted model
  (sklearn feature_names_in_), not from self-reports. A pipeline that lies
  INSIDE its builder or eval functions in ways invisible through refits
  (for example a rigged eval window) could evade the battery; code review
  of submitted pipelines is out of scope for this version.
- The proximity drop-test and the deployment re-run assume the model family
  can be refit quickly on feature subsets; heavy pipelines would need
  sampling budgets not implemented here.
- NC-NULL scope: the 7-day block permutation destroys exogenous and calendar
  alignment but preserves within-block autocorrelation and the marginal
  distribution of the target. The control verifies that the procedure
  refuses to certify claims on alignment-destroyed data; it is not an
  exhaustive null battery and it does not test pipelines whose claimed skill
  rests only on short-lag autocorrelation.
- The trial-disclosure audit relies on the case disclosing its trial log
  (K configurations tried on the test window). A pipeline that hides its
  search cannot be caught by this version; the planned selection-bias case
  will demonstrate the disclosed variant.

## Status of verification

- The GitHub URL in the manifest is the intended publication target; the
  repository has not been pushed yet at the time of this record version.
- All runs are author-executed (via an automated agent session) and
  deterministic: data seed 42, null-control seeds 1..5. Re-running the
  committed commands on the committed data must reproduce every number.
- No independent reproduction exists yet. Human sign-off has not happened:
  trust/signoff.md is intentionally absent; the machine output in
  trust/verdict.json is verification, not a signed verdict.
- No external dataset has been used yet.
