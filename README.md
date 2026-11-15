# energy-verdict-track-record

Verifier Track Record for the Engineering Verdict procedure, energy line
(day-ahead load forecasting). This is the procedure self-test required
before the first paid verdict: planted defective cases must be FAILed, clean
cases must be PASSed, with sample sizes and uncertainty bounds stated next
to every number.

Protocol: CV->0 Trust Layer v3.5. Live record: `trust/` (claims, evidence
manifest, verdict, limitations). Reference record that established the
format: permutation-test-ml (V-PTM-001).

## Status

Record version 3: the full defect-class battery (Appendix A minimum 5+5
satisfied and exceeded by one defect class).

- 6 defective cases (N=1 each): look-ahead leak (DATA), DST/timezone
  handling (ENVIRONMENT), backtest contamination (DATA), target leakage
  (DATA), cherry-picked metric (CONFLICT_EVIDENCE), selection bias (NOISE)
- 5 clean cases (N=1 each): linear, scaled ridge, gradient boosting,
  timezone-aware hourly, scaled ElasticNet
- NC-NULL negative control (N=5)

No paid verdicts are offered until this record is human-signed and
published (service brief, section 11). See trust/limitations.md.

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
    results/                     V-run outputs (committed, deterministic)
    trust/                       the Trust Record

## How to run

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m src.run_track_record --cases all --with-null-control

Deterministic: daily data seed 42, hourly data seed 43, null-control seeds
1..5. Re-running the commands on the committed data reproduces every number
below.

## The audit battery (per submitted case)

1. prediction alignment: predictions must cover the pre-registered test
   window exactly, no non-finite values
2. train/test isolation: train, test and confirmation windows must be
   disjoint and ordered (backtest contamination -> DATA)
3. timezone integrity: the working feature frame (re-executed through the
   case's own builder) must have no duplicated timestamps; hourly frames
   must have no missing wall-clock hours in the domain timezone
   (DST/timezone defects -> ENVIRONMENT)
4. feature-target proximity: features with |corr| > 0.995 against the
   target trigger a drop-test; if MAE collapses (>= 2x) without them ->
   target leakage (DATA)
5. deployment consistency: features used by the fitted model are checked
   against the audit-side availability map; a deployment re-run (refit
   without unavailable features) must reproduce the reported number within
   a 20% relative gap (look-ahead leak -> DATA)
6. metric reproduction: the reported number must be recomputable from the
   returned predictions on the full test window within 1%
   (otherwise CONFLICT_EVIDENCE)
7. claim conditions: the pre-registered claim (absolute MAE threshold +
   cadence-aware baseline comparisons) must hold (otherwise NOISE)
8. trial disclosure: if K > 1 configurations were tried on the test window,
   the best one must replicate its improvement on the held-out confirmation
   window (otherwise NOISE)

Reason codes are the frozen CV0-TrustLayer-v3.5 vocabulary (9 values).

## Results (record version 3)

Defective cases (N=1 each; all must be FAIL):

| case | defect | verdict | reason | numbers |
|---|---|---|---|---|
| defect_01_look_ahead | look-ahead leak | FAIL | DATA | reported MAE 1387.4 MW; deployment re-run 1713.6 MW; gap 23.5% > 20% threshold |
| defect_02_dst_timezone | DST handling | FAIL | ENVIRONMENT | 1 duplicated wall-clock hour (2023-10-29 fall-back) in the working frame; reported MAE 107.4 MW barely moves |
| defect_03_backtest_contamination | train/test overlap | FAIL | DATA | train window overlaps test by 15 days; reported MAE 1604.9 MW on contaminated backtest |
| defect_04_target_leakage | near-copy feature | FAIL | DATA | corr(load_settlement, load) = 0.999; reported MAE 126.0 MW; without the feature 1713.6 MW (collapse 13.6x) |
| defect_05_cherry_picked_metric | silent trim | FAIL | CONFLICT_EVIDENCE | reported MAE 996.5 MW (20 lowest-error days); recomputed full-window 1713.6 MW; diff 72.0% > 1% |
| defect_06_selection_bias | best of K=20 on test | FAIL | NOISE | reported best-of-20 MAE 2607.8 MW > claim threshold 1880 MW; configurations carry no exogenous skill by construction |

Clean cases (N=1 each; all must be PASS):

| case | model | data | verdict | MAE |
|---|---|---|---|---|
| clean_01_linear_forecast_features | linear | daily | PASS | 1713.6 MW = 3.19% of mean 53701 MW |
| clean_02_ridge_scaled | scaled Ridge | daily | PASS | 1714.3 MW = 3.19% |
| clean_03_gradient_boosting | gradient boosting | daily | PASS | 1286.3 MW = 2.40% |
| clean_04_hourly_tz_aware | linear, UTC index | hourly | PASS | 108.1 MW = 4.39% of mean 2461 MW |
| clean_05_elastic_net | scaled ElasticNet | daily | PASS | 1726.7 MW = 3.21% |

All clean cases beat their cadence-aware baselines on the same window
(N=1 window each): daily persistence 2888 MW / seasonal-naive 3086 MW;
hourly persistence (24h) 184 MW / seasonal-naive (168h) 136 MW.

Summary (N is stated next to each number):

- sensitivity (defective detected): 6/6 (N=6), Wilson 95% CI [0.610, 1.000]
- specificity (clean passed): 5/5 (N=5), Wilson 95% CI [0.566, 1.000]
- false PASS on defective: 0/6 (N=6); conservative rule-of-three upper
  bound at 0/6 is about 50%

NC-NULL control (clean pipeline re-run on block-permuted target data,
7-day blocks, seeds 1..5; the procedure must not certify):

- false PASS: 0/5 (N=5)
- Wilson 95% CI [0.000, 0.434]
- conservative rule-of-three upper bound at 0/5: about 60%
- the bound is stated next to the zero; a zero without a stated bound is
  not published
- MAE on null data: 2318.2 to 2960.6 MW across the 5 runs, all above the
  claim threshold 1879.5 MW (3.5% of mean load)
- naive point-shuffle of labels is NOT used (invalid on trended data;
  measured in V-PTM-001, claim C-04)

## What the defects are

D1 (look-ahead leak): the pipeline trains and evaluates with realised
weather (temp_actual) among its features. At inference time, a day-ahead
forecast only has the temperature forecast (temp_forecast) available. The
reported backtest MAE is not attainable in deployment. Mirrors section 2 of
the service brief.

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
reported MAE collapses to the reporting-noise level (126.0 MW).

D5 (cherry-picked metric): an ordinary clean model whose REPORTED number is
computed on a favorable subset (20 lowest-error days of 30) without
disclosure. The returned predictions cover the full window, so the claimed
number does not reproduce from them.

D6 (selection bias): 20 linear configurations (calendar + pure-noise
columns) tried on the TEST window; the best one is reported. The trial log
is disclosed; the undisclosed-search variant is out of scope for this
record version (see limitations).

## Verification status

- All runs are author-executed (via an automated agent session on
  2026-11-15) and deterministic.
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
