# Claims - energy-verdict-track-record (public Trust Record)

Protocol: CV->0 Trust Layer v3.5 (spec frozen for the cycle until 2027-02-04)
Object: https://github.com/sensus-stoa/energy-verdict-track-record
Object version: see trust/evidence-manifest.json (object.commit)
Criteria lock: the PASS/FAIL criteria below are fixed in THIS file.
The lock commit hash and timestamp are recorded in
trust/evidence-manifest.json (criteria_lock). Any change to these criteria
after the lock commit is a new version marked post-hoc and requires a
re-run for PASS.

Scope of this record version: 3 planted defective cases (N=1 each: look-ahead
leak, backtest contamination, target leakage), 3 clean cases (N=1 each:
linear, scaled ridge, gradient boosting), and the NC-NULL negative control
(N=5). This is the 3+3 subset of the Appendix A start set (5 defective +
5 clean). DST/timezone handling, cherry-picked metrics and selection bias
cases are not implemented yet; no claim is made about them. A new record
version will extend the set.

Audit-side ground truth used by the procedure (frozen here): feature
availability at inference time for the synthetic energy domain is
temp_actual = NOT available (realised weather is only known after the day
ends), temp_forecast = available at 00:00 of the target day, calendar
features and load lags (yesterday, last week) = available. This map lives in
src/verdict_procedure.py (AUDIT_FEATURE_AVAILABILITY) and is justified by
the data generator semantics (src/data_generator.py).

## C-01 Procedure detects a planted look-ahead leak
Claim: the verdict procedure FAILs a day-ahead load forecasting pipeline
whose backtest features include realised weather (temp_actual) while only a
day-ahead temperature forecast (temp_forecast) exists at inference time.
Case: defect_01_look_ahead (N=1 planted defective case).
PASS criterion: verdict = FAIL with reason_code = DATA, and the deployment
re-run (model refit on features that are available at inference time) shows
a relative MAE gap above 20% against the reported number.
Reference values from the committed run: reported MAE 1387.4 MW, deployment
MAE 1713.6 MW, gap 23.5%.
Verification V-01: python -m src.run_track_record --case defect_01_look_ahead

## C-02 Procedure passes a deployment-consistent pipeline
Claim: the verdict procedure PASSes a clean day-ahead load forecasting
pipeline whose features are all available at inference time (temperature
forecast, calendar, load lags).
Case: clean_01_linear_forecast_features (N=1 clean case).
PASS criterion: verdict = PASS with all audits ok; the recomputed MAE
matches the reported MAE within 1%; the pre-registered case claim holds:
MAE <= 3.5% of mean test-window load AND better than persistence AND better
than seasonal-naive on the same window.
Reference values from the committed run: MAE 1713.6 MW = 3.19% of mean load
53701 MW; persistence 2888 MW; seasonal-naive 3086 MW (N=1 test window,
30 days).
Verification V-01: python -m src.run_track_record --cases all

## C-03 Procedure does not certify on destroyed-alignment data (NC-NULL)
Claim: when the target series is replaced by a block permutation (7-day
blocks) that destroys the alignment between the target and all exogenous
and calendar features, the procedure does not issue PASS for the clean
pipeline's claim.
Negative control NC-NULL: N = 5 independent block-permutation runs (seeds
1..5); false PASS = 0; rate 0.000; Wilson 95% CI [0.000, 0.434]; the
conservative rule-of-three upper bound at 0/5 is about 60%. The bound is
stated next to the zero: a zero without a stated bound is not published.
Method note: naive point-shuffle of labels is NOT used anywhere in this
record; it is invalid on trended data, as measured in V-PTM-001 (claim
C-04). Block permutation is used ONLY as a negative control on deliberately
destroyed data, never as a significance test on real data.
PASS criterion: false PASS = 0 in N = 5 runs.
Verification V-02: python -m src.run_track_record --cases all --with-null-control

## C-04 Procedure detects backtest contamination
Claim: the verdict procedure FAILs a pipeline whose train window overlaps
the test window (15 days of overlap from a self-computed split).
Case: defect_03_backtest_contamination (N=1 planted defective case).
PASS criterion: verdict = FAIL with reason_code = DATA; the train/test
isolation audit flags the window overlap.
Reference values from the committed run: reported MAE 1604.9 MW on a
contaminated backtest.
Verification V-01: python -m src.run_track_record --case defect_03_backtest_contamination

## C-05 Procedure detects target leakage
Claim: the verdict procedure FAILs a pipeline whose feature table contains
a near-copy of the target (load_settlement = same-day realised load plus
reporting noise, |corr| with the target above 0.995).
Case: defect_04_target_leakage (N=1 planted defective case).
PASS criterion: verdict = FAIL with reason_code = DATA; the proximity audit
flags the suspect feature and the drop-test (retrain without it) grows the
MAE at least 2x against the reported number.
Reference values from the committed run: reported MAE 126.0 MW; corr 0.999;
MAE without the suspect feature 1713.6 MW (collapse ratio 13.6).
Verification V-01: python -m src.run_track_record --case defect_04_target_leakage

## C-06 Procedure passes a regularised linear pipeline
Claim: the verdict procedure PASSes a clean scaled-Ridge pipeline (different
model family from C-02, same deployment-consistent features) under the same
claim conditions.
Case: clean_02_ridge_scaled (N=1 clean case).
PASS criterion: verdict = PASS with all audits ok; recomputed MAE matches
reported within 1%; claim holds (MAE <= 3.5% of mean load; better than
persistence and seasonal-naive).
Reference values from the committed run: MAE 1714.3 MW = 3.19% of mean load.
Verification V-01: python -m src.run_track_record --cases all

## C-07 Procedure passes a gradient boosting pipeline
Claim: the verdict procedure PASSes a clean gradient boosting pipeline
(nonlinear family, fixed hyperparameters, single configuration) under the
same claim conditions.
Case: clean_03_gradient_boosting (N=1 clean case).
PASS criterion: verdict = PASS with all audits ok; recomputed MAE matches
reported within 1%; claim holds (MAE <= 3.5% of mean load; better than
persistence and seasonal-naive).
Reference values from the committed run: MAE 1286.3 MW = 2.40% of mean load.
Verification V-01: python -m src.run_track_record --cases all
