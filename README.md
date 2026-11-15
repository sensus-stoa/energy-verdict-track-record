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

MVP subset of the Appendix A start set (record version 1):

- 1 defective case implemented (N=1)
- 1 clean case implemented (N=1)
- NC-NULL negative control implemented (N=5)

The full 5 defective + 5 clean set is NOT implemented yet. No paid verdicts
are offered until the full set is published (service brief, section 11).
See trust/limitations.md.

## Layout

    src/verdict_procedure.py     audit battery (7 audits) + verdict logic
    src/case_api.py              contract between a submitted case and the audit
    src/linear_case.py           shared linear pipeline machinery
    src/metrics.py               MAE, baselines, Wilson CI, block permutation
    src/data_generator.py        synthetic daily energy data (seed 42)
    src/clean_cases/             clean pipelines (C-cases)
    src/defects/                 planted-defect pipelines (D-cases)
    src/run_track_record.py      runner: cases -> verdict -> results CSV
    src/pilot_calibration.py     pre-lock calibration (disclosed, not evidence)
    data/synthetic/              generated dataset (committed)
    results/                     V-run outputs (committed, deterministic)
    trust/                       the Trust Record

## How to run

    python -m venv .venv && .venv/bin/pip install -r requirements.txt
    .venv/bin/python -m src.run_track_record --cases all --with-null-control

Deterministic: data seed 42, null-control seeds 1..5. Re-running the
commands on the committed data reproduces every number below.

## The audit battery (per submitted case)

1. prediction alignment: predictions must cover the pre-registered test
   window exactly, no non-finite values
2. train/test isolation: train, test and confirmation windows must be
   disjoint and ordered (backtest contamination -> DATA)
3. deployment consistency: features used by the fitted model are checked
   against the audit-side availability map; a deployment re-run (refit
   without unavailable features) must reproduce the reported number within
   a 20% relative gap (look-ahead leak -> DATA)
4. feature-target proximity: |corr| > 0.995 features trigger a drop-test;
   if MAE collapses (>= 2x) without them -> target leakage (DATA)
5. metric reproduction: the reported number must be recomputable from the
   returned predictions on the full test window within 1%
   (otherwise CONFLICT_EVIDENCE)
6. claim conditions: the pre-registered claim (absolute MAE threshold +
   baseline comparisons against persistence and seasonal-naive) must hold
   (otherwise NOISE)
7. trial disclosure: if K > 1 configurations were tried on the test window,
   the best one must replicate its improvement on the held-out confirmation
   window (otherwise NOISE)

Reason codes are the frozen CV0-TrustLayer-v3.5 vocabulary (9 values).

## Results (record version 1)

Cases (N=1 per row):

| case | defective | verdict | reason | numbers |
|---|---|---|---|---|
| defect_01_look_ahead | yes | FAIL | DATA | reported MAE 1387.4 MW; deployment re-run 1713.6 MW; gap 23.5% > 20% threshold |
| clean_01_linear_forecast_features | no | PASS | - | MAE 1713.6 MW = 3.19% of mean load 53701 MW; beats persistence 2888 MW and seasonal-naive 3086 MW (N=1 window, 30 days) |

Summary (N is stated next to each number):

- sensitivity (defective detected): 1/1 (N=1), Wilson 95% CI [0.207, 1.000]
- specificity (clean passed): 1/1 (N=1), Wilson 95% CI [0.207, 1.000]
- false PASS on defective: 0/1 (N=1); rule-of-three upper bound at 0/1 is
  about 300%: this single case demonstrates the mechanism and nothing more

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

## What the defect is (case D1)

The defective pipeline trains and evaluates with realised weather
(`temp_actual`) among its features. At inference time, a day-ahead forecast
only has the temperature forecast (`temp_forecast`) available. The reported
backtest MAE (1387.4 MW) is not attainable in deployment: the deployment
re-run gives 1713.6 MW. The model involuntarily looks into the future; the
evaluation, not the model, is defective. This mirrors the failure mode in
section 2 of the service brief.

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

- Defective cases: DST/timezone handling (ENVIRONMENT), backtest
  contamination (DATA), target leakage (DATA), cherry-picked metrics
  (CONFLICT_EVIDENCE), selection bias in best-of-K search (NOISE).
- Clean cases: gradient boosting, autoregressive variants, ridge.
- Hourly data with a timezone for the DST case.
- An external dataset (ENTSO-E or similar; requires registration).
- Each extension is a new record version with a new criteria lock and a
  full re-run for PASS claims.
