# energy-verdict-track-record

Open, deterministic test of the audit procedure used in the Engineering
Verdict method for ML forecasting pipelines (energy domain). The record
answers the question an evaluating engineer asks: what does this procedure
catch, and what does it accuse without cause?

- 8 pipelines with planted defects the procedure must FAIL
- 5 clean pipelines it must PASS
- a negative control on destroyed-alignment data
- sample sizes and uncertainty bounds next to every number

Everything is reproducible: fixed seeds, committed data and results; the
commands below regenerate every number byte-identically.

## Status (record version 5, V-EVTR-005; supersedes V-EVTR-004)

- 8 defective cases (N=1 each), all FAIL: look-ahead leak (DATA),
  DST/timezone handling (ENVIRONMENT), backtest contamination (DATA),
  target leakage (DATA), cherry-picked metric (CONFLICT_EVIDENCE),
  selection bias (NOISE), seasonal-naive baseline shipped as a forecast
  (NOISE), best-of-K search passing the claim (NOISE)
- 5 clean cases (N=1 each), all PASS: linear, scaled ridge, gradient
  boosting, timezone-aware hourly, scaled ElasticNet
- NC-NULL negative control (N=5): false PASS 0/5, Wilson 95% CI
  [0.000, 0.434]

Summary: sensitivity 8/8 (N=8), Wilson 95% CI [0.676, 1.000]; specificity
5/5 (N=5), Wilson 95% CI [0.566, 1.000]; false PASS on defective 0/8 (N=8),
conservative rule-of-three upper bound about 38%.

The record files (criteria, evidence manifest with sha256, machine verdict,
limitations) live in trust/. The machine verdict states the sign-off
status; an agent never signs.

## Layout

    src/verdict_procedure.py     audit battery (8 audits) + verdict logic
    src/case_api.py              contract between a submitted case and the audit
    src/linear_case.py           shared linear pipeline machinery
    src/metrics.py               MAE, cadence-aware baselines, Wilson CI, block permutation
    src/data_generator.py        synthetic daily + hourly energy data (seeds 42/43)
    src/clean_cases/             clean pipelines (C-cases)
    src/defects/                 planted-defect pipelines (D-cases)
    src/run_track_record.py      runner: cases -> verdict -> results CSV
    src/pilot_calibration.py     pre-lock calibration (disclosed, not evidence)
    data/synthetic/              generated datasets (committed)
    results/                     run outputs (committed, deterministic)
    trust/                       the record

## How to run

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m src.run_track_record --cases all --with-null-control
    tc -root .

Mechanical validation of the record uses the canonical protocol checker
`tc` (trust_check v0.2.1) from https://github.com/sensus-stoa/cv0-trust-protocol
(release binary tc-v0.2.1-linux-amd64, sha256
9ad4c66ad70b2903af84348990109afe1003bb26238ea18b734967f52733de48; build
contract in that repository). The committed trust_check_report.json is its
verbatim output: tc records the root as passed on the command line (run as
`tc -root .`, so the report carries no machine-local paths) and carries
checked_commit. The report is committed immediately after the record
commit it verifies, so the checked commit is the report commit's parent.

Single-case probe (runs ONLY the named case; rewrites that case's
results/per_case/<id>.json but never the full-battery
results/track_record.csv):

    python -m src.run_track_record --case defect_01_look_ahead

Deterministic: daily data seed 42, hourly data seed 43, null-control seeds
1..5. Re-running the commands on the committed data reproduces every number
below.

## The audit battery (per submitted case)

1. prediction alignment: predictions must cover the pre-registered test
   window exactly - no missing/extra/duplicated timestamps, no non-finite
   values
2. train/test isolation: train, test and confirmation windows must be
   disjoint and ordered (backtest contamination -> DATA)
3. timezone integrity: the working feature frame (re-executed through the
   case's own builder) must have no duplicated timestamps; hourly frames -
   naive or tz-aware - must have no missing wall-clock hours in the domain
   timezone (DST/timezone defects -> ENVIRONMENT)
4. feature-target proximity: features with |corr| > 0.995 against the
   target trigger a drop-test; if MAE collapses (>= 2x) without them ->
   target leakage (DATA)
5. deployment consistency: features used by the fitted model are checked
   against the audit-side availability map; fitting on an unavailable
   feature is itself the violation (look-ahead leak -> DATA). The impact is
   measured as relative_gap = (deployment_mae - reported_mae) /
   reported_mae and classified against the 20% threshold
   (LOOKAHEAD_GAP_THRESHOLD): above 20% - impact confirmed; at or below -
   the fail stands on the structural violation with the sub-threshold gap
   recorded. Features unknown to the audit map -> inconclusive (recorded,
   never hidden)
6. metric reproduction: the reported number must be recomputable from the
   returned predictions on the full test window within 1%
   (otherwise CONFLICT_EVIDENCE)
7. claim conditions: the pre-registered claim (absolute MAE threshold +
   cadence-aware baseline comparisons) must hold (otherwise NOISE). The
   absolute percentage is an upper bound on error; skill comes from the
   must-beat conditions
8. trial disclosure: if K > 1 configurations were tried on the test window,
   the best one must replicate on the held-out confirmation window on two
   edges - its improvement over the seasonal-naive baseline AND its
   selection gain over the fixed unsearched baseline (the largest feature
   set common to all disclosed trial configurations). An edge is material
   above a 3% test-window magnitude; a material edge that does not stay
   positive on the confirmation window is a multiple-comparison artifact
   (NOISE)

Reason codes are the frozen CV0-TrustLayer-v3.5 vocabulary (9 values).

## Results

Defective cases (N=1 each; all FAIL):

| case | defect | verdict | reason | decisive audit | numbers |
|---|---|---|---|---|---|
| defect_01_look_ahead | look-ahead leak | FAIL | DATA | deployment consistency | reported MAE 1353.9 MW; deployment re-run 1699.3 MW; relative gap 25.5% > 20% threshold |
| defect_02_dst_timezone | DST handling | FAIL | ENVIRONMENT | timezone integrity | 1 duplicated wall-clock hour (2023-10-29 fall-back) in the working frame; reported MAE 107.4 MW barely moves |
| defect_03_backtest_contamination | train/test overlap | FAIL | DATA | train/test isolation | train window overlaps test by 15 days; reported MAE 1592.4 MW on contaminated backtest |
| defect_04_target_leakage | near-copy feature | FAIL | DATA | feature-target proximity | corr(load_settlement, load) = 0.999; reported MAE 129.2 MW; without the feature 1699.3 MW (collapse 13.15x); deployment audit inconclusive (recorded) |
| defect_05_cherry_picked_metric | silent trim | FAIL | CONFLICT_EVIDENCE | metric reproduction | reported MAE 996.5 MW (20 lowest-error days); recomputed full-window 1699.3 MW; diff 70.53% > 1% |
| defect_06_selection_bias | best of K=20 on test | FAIL | NOISE | claim conditions + trial disclosure | best-of-20 MAE 2519.0 MW > claim threshold 1878.3 MW; fixed-baseline selection gain 6.1% test -> -1.3% confirmation (does not replicate) |
| defect_07_seasonal_naive_only | baseline as forecast | FAIL | NOISE | claim conditions (must-beat) | prediction = 168h lag; MAE 135.9 MW = 5.52% of mean load passes the 6.0% gate but is not better than seasonal-naive 135.9 MW |
| defect_08_search_overfit | best of K=20, claim holds | FAIL | NOISE | trial disclosure (only) | winner MAE 1617.4 MW = 3.01% satisfies the claim; fixed-baseline selection gain 4.8% test -> -5.0% confirmation (does not replicate) |

Clean cases (N=1 each; all PASS):

| case | model | data | verdict | MAE |
|---|---|---|---|---|
| clean_01_linear_forecast_features | linear | daily | PASS | 1699.3 MW = 3.17% of mean 53664.9 MW |
| clean_02_ridge_scaled | scaled Ridge | daily | PASS | 1699.7 MW = 3.17% |
| clean_03_gradient_boosting | gradient boosting | daily | PASS | 1318.0 MW = 2.46% |
| clean_04_hourly_tz_aware | linear, UTC index | hourly | PASS | 108.1 MW = 4.39% of mean 2460.7 MW |
| clean_05_elastic_net | scaled ElasticNet | daily | PASS | 1708.4 MW = 3.18% |

All clean cases beat their cadence-aware baselines on the same window
(N=1 window each): daily persistence 2726.9 MW / seasonal-naive 3157.3 MW;
hourly persistence (24h) 183.9 MW / seasonal-naive (168h) 135.9 MW.

NC-NULL control (clean pipeline re-run on block-permuted target data,
7-day blocks, seeds 1..5; the procedure must not certify):

- false PASS: 0/5 (N=5)
- Wilson 95% CI [0.000, 0.434]
- conservative rule-of-three upper bound at 0/5: about 60%
- the bound is stated next to the zero; a zero without a stated bound is
  not published
- MAE on null data: 2232.7 to 2925.2 MW across the 5 runs, all above the
  claim threshold 1878.3 MW (3.5% of mean load)
- naive point-shuffle of labels is NOT used (invalid on trended data: it
  destroys the trend/seasonal structure and produces miscalibrated nulls)

## What the defects are

D1 (look-ahead leak): the pipeline trains and evaluates with realised
weather (temp_actual) among its features. At inference time, a day-ahead
forecast only has the temperature forecast (temp_forecast) available. The
reported backtest MAE is not attainable in deployment: the model
involuntarily looks into the future; the evaluation, not the model, is
defective.

D2 (DST/timezone): an hourly pipeline converts to naive Berlin local wall
time and computes "same hour yesterday" by wall-clock grouping. Across the
October fall-back this duplicates a wall-clock hour and misaligns the lag;
the reported MAE barely moves (107.4 vs 108.1 MW for the clean twin), so
the defect is invisible in the report and only a structural check of the
working frame catches it.

D3 (backtest contamination): the pipeline computed its own split and
extended the train window 15 days into the test window (off-by-N accident
in index arithmetic). Test rows seen during training make the reported MAE
unattainably good.

D4 (target leakage): the feature table contains load_settlement = same-day
realised load plus small reporting noise (a realistic dataset-join
accident: settlement actuals merged into forecast features). The model's
reported MAE collapses to the reporting-noise level (129.2 MW).

D5 (cherry-picked metric): an ordinary clean model whose REPORTED number is
computed on a favorable subset (20 lowest-error days of 30) without
disclosure. The returned predictions cover the full window, so the claimed
number does not reproduce from them.

D6 (selection bias): 20 exchangeable linear configurations (calendar base
+ 20 pure-noise columns each, one noise seed per configuration) tried on
the TEST window; the best one is reported. None of the configurations has
skill beyond the calendar terms; the winner's gain over the fixed
unsearched calendar configuration does not replicate on the confirmation
window. The trial log is disclosed; the undisclosed-search variant is out
of scope for this record version (see limitations).

D7 (baseline as forecast): the hourly "model" IS the seasonal-naive
baseline (168-hour lag), disclosed as such. It passes the absolute 6.0%
gate (5.52% of mean load) - the gate is an upper bound on error, not a
skill criterion - but fails must-beat: it is not better than the baseline
it is identical to.

D8 (search overfit): 20 exchangeable configurations (deployment-consistent
real-feature base + 20 noise columns each) tried on the TEST window. The
winner satisfies the pre-registered claim (3.01% <= 3.5%, beats both
baselines), so the claim-conditions audit passes; the trial-disclosure
audit alone catches the artifact: the winner's selection gain over the
fixed base configuration (4.8% on test) does not replicate on the
confirmation window (-5.0%).

## Changes in version 5 (from version 4)

- Fixed a v4 bug in the trial-disclosure audit: it compared best-of-K only
  against seasonal-naive, so the D6 search artifact was classified "ok"
  and the FAIL came only from claim conditions; the fixed unsearched
  baseline is now the second replication edge (claim C-11 restated
  accordingly).
- Fixed a v4 serialization bug: five ok-branches of the audit battery
  passed AuditResult fields positionally, corrupting reason_code/detail/
  metrics in the per-case JSON evidence.
- Fixed the evaluation windows to the declared row counts (test 30 rows,
  confirmation 60 rows); all daily numbers changed and the full battery
  was re-run on the locked v5 code.
- Enforced the previously declared but unused 20% deployment-gap threshold
  (audit #5) and froze the relative_gap formula.
- The timezone-integrity missing-hours check now covers tz-aware frames;
  the prediction-alignment audit rejects duplicated prediction timestamps.
- Fixed the runner: --case no longer silently runs all cases and partial
  runs never overwrite the full results/track_record.csv; the NC-NULL loop
  always permutes the daily dataset (it previously used the last processed
  case's dataset variable).
- Added two defective cases (D7, D7's claims C-13; D8 -> C-14) pinning the
  roles of the hourly gate and of the trial-disclosure audit.
- Record-keeping: removed a broken external record reference (V-PTM-001),
  made the C-01 denominator formula explicit, unified percentage rounding,
  moved the pilot script out of the manifest evidence into disclosures,
  documented E-ID binding (record_version + evidence_id + sha256), and
  committed the trust check tool so the report is reproducible and carries
  a relative root path plus the checked commit.

## Verification status

- All runs are author-executed (via an automated agent session on
  2026-11-17) and deterministic.
- No independent reproduction exists yet.
- Human sign-off has not happened: `trust/signoff.md` is intentionally
  absent. The machine output (`trust/verdict.json`) is verification, not a
  signed verdict.
- AI assistance in building the procedure is disclosed here and in
  trust/limitations.md.

## Planned next record versions

- Repetition of defective cases (N > 1 per defect class) to narrow the
  Wilson intervals; independent reproduction by a third party.
- An external dataset (ENTSO-E or similar; requires registration).
- Each extension is a new record version with a new criteria lock and a
  full re-run for PASS claims.
