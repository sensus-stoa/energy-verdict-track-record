# Limitations - energy-verdict-track-record (record version 1)

## Sample size (stated next to every number)

- Sensitivity is measured on N=1 planted defective case. Wilson 95% CI for
  1/1 is [0.207, 1.000]. One detected defect demonstrates the mechanism, not
  a reliability estimate.
- Specificity is measured on N=1 clean case. Wilson 95% CI for 1/1 is
  [0.207, 1.000].
- NC-NULL: N=5, false PASS 0/5, Wilson 95% CI [0.000, 0.434], rule-of-three
  upper bound about 60%. The set is small by design (MVP); the numbers are
  not a power statement.

## Scope of the set

- This record version covers 1 defective + 1 clean case. The full Appendix A
  start set (5 defective + 5 clean) is NOT implemented. Planned defect
  classes and their frozen reason codes: DST / timezone handling
  (ENVIRONMENT), backtest contamination (DATA), target leakage (DATA),
  cherry-picked metrics (CONFLICT_EVIDENCE), selection bias in best-of-K
  search (NOISE).
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
  semantics; the MVP battery does not defend against a fully adversarial
  submitter, only against the planted defect classes.
- The audit battery re-executes the case's own feature builder and
  fit/eval functions (retrain_eval_fn) for deployment-gap, drop-test and
  replication checks. What entered the fit is read from the fitted model
  (sklearn feature_names_in_), not from self-reports.
- NC-NULL scope: the 7-day block permutation destroys exogenous and calendar
  alignment but preserves within-block autocorrelation and the marginal
  distribution of the target. The control verifies that the procedure
  refuses to certify claims on alignment-destroyed data; it is not an
  exhaustive null battery and it does not test pipelines whose claimed skill
  rests only on short-lag autocorrelation.

## Status of verification

- All runs are author-executed (via an automated agent session) and
  deterministic: data seed 42, null-control seeds 1..5. Re-running the
  committed commands on the committed data must reproduce every number.
- No independent reproduction exists yet. Human sign-off has not happened:
  trust/signoff.md is intentionally absent; the machine output in
  trust/verdict.json is verification, not a signed verdict.
- No external dataset has been used yet.
