"""Verdict procedure for the energy line (Engineering Verdict, miniature).

Machine verification only: the procedure runs an audit battery over a
submitted pipeline artifact and returns PASS / FAIL / INCONCLUSIVE with a
reason_code from the FROZEN vocabulary of CV0-TrustLayer-v3.5:
    DATA, GRAMMAR, NOISE, DEPTH, ACCESS, BUDGET, ENVIRONMENT,
    CONFLICT_EVIDENCE, OUT_OF_SCOPE

The procedure trusts nothing the case self-reports about data semantics:
- what entered the model fit is read from the fitted sklearn model
  (model.feature_names_in_);
- feature availability at inference time comes from AUDIT-side configuration
  (AUDIT_FEATURE_AVAILABILITY below), justified by the data generator's
  semantics and frozen in trust/claims.md before any V-run;
- the reported number must be reproducible from the returned predictions on
  the full pre-registered test window (anti cherry-picking);
- skill claims must hold against pre-registered baselines
  (persistence, seasonal-naive);
- multi-configuration searches (K > 1) must replicate on the held-out
  confirmation window - both the improvement over the seasonal-naive
  baseline and the selection gain over the fixed unsearched baseline (the
  largest feature set common to all disclosed trial configurations) -,
  otherwise the claimed edge is classified as noise;
- a relative deployment MAE gap above LOOKAHEAD_GAP_THRESHOLD (20%),
  relative_gap = (deployment_mae - reported_mae) / reported_mae, documents
  the impact of a look-ahead leak (using an unavailable feature is itself
  the violation and fails the audit regardless of the gap).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, Windows
from src.metrics import baseline_lags, mae, naive_baseline_mae

REASON_CODES = (
    "DATA", "GRAMMAR", "NOISE", "DEPTH", "ACCESS", "BUDGET",
    "ENVIRONMENT", "CONFLICT_EVIDENCE", "OUT_OF_SCOPE",
)

# Audit-side ground truth for the synthetic energy domain (see data_generator
# semantics, frozen in trust/claims.md). Unknown features are treated as
# not-assessable (inconclusive), NOT as available.
AUDIT_FEATURE_AVAILABILITY: dict[str, bool] = {
    "temp_actual": False,    # realised weather: only known after the day ends
    "temp_forecast": True,   # day-ahead forecast: available at 00:00 of target day
    "doy_sin": True,         # calendar
    "doy_cos": True,
    "dow_weekend": True,
    "hour_sin": True,        # local wall-clock hour: known in advance
    "hour_cos": True,
    "load_lag1": True,       # yesterday's actual load: published at day close
    "load_lag7": True,
    "load_lag24": True,      # hourly: same hour yesterday
    "load_lag168": True,     # hourly: same hour last week
}

LOOKAHEAD_GAP_THRESHOLD = 0.20     # relative reported-vs-deployment MAE gap
LEAK_CORR_THRESHOLD = 0.995        # |corr(feature, target)| suspicion level
LEAK_COLLAPSE_RATIO = 2.0          # MAE must grow >= 2x without the suspect feature
METRIC_TOLERANCE = 0.01            # reported vs recomputed primary metric


@dataclass
class AuditConfig:
    availability: dict[str, bool] = field(default_factory=lambda: dict(AUDIT_FEATURE_AVAILABILITY))
    lookahead_gap_threshold: float = LOOKAHEAD_GAP_THRESHOLD
    leak_corr_threshold: float = LEAK_CORR_THRESHOLD
    leak_collapse_ratio: float = LEAK_COLLAPSE_RATIO
    metric_tolerance: float = METRIC_TOLERANCE
    min_improvement_for_replication: float = 0.03
    domain_tz: str = "Europe/Berlin"   # for timezone-integrity checks on hourly data


@dataclass
class AuditResult:
    name: str
    status: str                    # ok | fail | inconclusive | skipped
    reason_code: str = ""
    detail: str = ""
    metrics: dict = field(default_factory=dict)


def _sel(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    return df[(df.index >= window[0]) & (df.index < window[1])]


# ---------------------------------------------------------------- audits

def audit_prediction_alignment(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    expected = _sel(data, art.windows.test).index
    preds = art.predictions
    missing = expected.difference(preds.index)
    extra = preds.index.difference(expected)
    # set difference above does not see duplicated timestamps (a duplicated
    # prediction index still covers the window); check duplicates explicitly
    dup_preds = int(preds.index.duplicated().sum())
    if len(missing) or len(extra) or dup_preds or not np.isfinite(preds.to_numpy()).all():
        return AuditResult(
            name="prediction_alignment", status="inconclusive", reason_code="DATA",
            detail=f"predictions do not cover the test window exactly: missing={len(missing)} "
                   f"extra={len(extra)} duplicated={dup_preds} "
                   f"nonfinite={int((~np.isfinite(preds.to_numpy())).sum())} (N=1 case)",
            metrics={},
        )
    return AuditResult(name="prediction_alignment", status="ok", reason_code="",
                       detail="predictions cover the pre-registered test window exactly",
                       metrics={"n_test_rows": int(len(expected))})


def audit_train_test_isolation(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    tr0, tr1 = art.windows.train
    te0, te1 = art.windows.test
    cf0, cf1 = art.windows.confirmation
    problems = []
    if max(tr0, te0) < min(tr1, te1):
        problems.append(f"train window {art.windows.train} overlaps test window {art.windows.test}")
    if max(te0, cf0) < min(te1, cf1):
        problems.append(f"test window overlaps confirmation window {art.windows.confirmation}")
    if tr1 > te0:
        problems.append("train window ends after test window starts (temporal ordering violated)")
    if problems:
        return AuditResult("train_test_isolation", "fail", "DATA",
                           "backtest contamination: " + "; ".join(problems))
    return AuditResult("train_test_isolation", "ok")


def audit_deployment_consistency(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    used = list(art.feature_names_used)
    unknown = [f for f in used if f not in cfg.availability]
    if unknown:
        return AuditResult(
            name="deployment_consistency", status="inconclusive", reason_code="DATA",
            detail=f"feature availability unknown to the audit for: {unknown} (N=1 case)",
            metrics={},
        )
    unavailable_used = [f for f in used if not cfg.availability[f]]
    if unavailable_used:
        # The structural finding is decisive on its own: fitting on features that
        # are not available at inference time IS the look-ahead violation,
        # whatever the measured gap. The gap against
        # cfg.lookahead_gap_threshold documents the deployment impact:
        # relative_gap = (deployment_mae - reported_mae) / reported_mae
        # (denominator: the reported MAE on the pre-registered test window).
        safe = [f for f in used if cfg.availability[f]]
        metrics = {"unavailable_features_used": unavailable_used,
                   "lookahead_gap_threshold": cfg.lookahead_gap_threshold}
        if safe:
            _, mae_d = art.retrain_eval_fn(data, safe, art.windows.train, art.windows.test)
            gap = (mae_d - art.reported_mae) / art.reported_mae if art.reported_mae else float("inf")
            metrics["deployment_mae"] = round(mae_d, 1)
            metrics["relative_gap"] = round(gap, 4)
            metrics["gap_exceeds_threshold"] = bool(gap > cfg.lookahead_gap_threshold)
            gap_txt = (f"relative gap (deployment_mae - reported_mae) / reported_mae = "
                       f"{gap:.1%} vs threshold {cfg.lookahead_gap_threshold:.0%}")
            if gap > cfg.lookahead_gap_threshold:
                impact = (f"deployment re-run MAE {mae_d:.0f} vs reported "
                          f"{art.reported_mae:.0f}: {gap_txt} - impact above threshold")
            else:
                impact = (f"deployment re-run MAE {mae_d:.0f} vs reported "
                          f"{art.reported_mae:.0f}: {gap_txt}, below threshold; the fail "
                          f"rests on the structural use of unavailable features")
            detail = (f"look-ahead leak: model uses features not available at inference "
                      f"time ({unavailable_used}); {impact} (N=1 retrain)")
        else:
            detail = (f"look-ahead leak: ALL used features are unavailable at inference "
                      f"time ({unavailable_used})")
        return AuditResult(name="deployment_consistency", status="fail", reason_code="DATA",
                           detail=detail, metrics=metrics)
    return AuditResult(name="deployment_consistency", status="ok", reason_code="",
                       detail=f"all {len(used)} used features available at inference time",
                       metrics={"n_features": len(used)})


def audit_feature_target_proximity(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    feat_df = art.feature_frame_fn(data)
    # target taken from the SAME frame the case built: for a case whose frame
    # has a broken index (e.g. duplicated wall hours) x and y stay consistent
    y = _sel(feat_df, art.windows.test)["load"]
    suspects = []
    for f in art.feature_names_used:
        if f not in feat_df.columns:
            continue
        x = _sel(feat_df, art.windows.test)[f]
        ok = x.notna() & y.notna()
        if ok.sum() < 10:
            continue
        corr = float(np.corrcoef(x[ok], y[ok])[0, 1])
        if abs(corr) > cfg.leak_corr_threshold:
            suspects.append((f, corr))
    metrics = {"suspects": [(f, round(c, 4)) for f, c in suspects]}
    if not suspects:
        return AuditResult(name="feature_target_proximity", status="ok", reason_code="",
                           detail="no feature is a near-copy of the target", metrics=metrics)
    # Drop-test: remove ALL suspect features at once, retrain, re-evaluate.
    safe = [f for f in art.feature_names_used if f not in {s for s, _ in suspects}]
    if not safe:
        return AuditResult(name="feature_target_proximity", status="fail", reason_code="DATA",
                           detail=f"target leakage: every used feature is a near-copy of the target "
                                  f"({suspects}); nothing left to retrain with (N=1 drop-test)",
                           metrics=metrics)
    _, mae_drop = art.retrain_eval_fn(data, safe, art.windows.train, art.windows.test)
    ratio = mae_drop / art.reported_mae if art.reported_mae else float("inf")
    metrics["mae_without_suspects"] = round(mae_drop, 1)
    metrics["collapse_ratio"] = round(ratio, 2)
    if ratio >= cfg.leak_collapse_ratio:
        return AuditResult(
            name="feature_target_proximity", status="fail", reason_code="DATA",
            detail=f"target leakage: features {suspects} are near-copies of the target "
                   f"(|corr| > {cfg.leak_corr_threshold}); without them MAE grows "
                   f"{ratio:.1f}x ({art.reported_mae:.0f} -> {mae_drop:.0f}), N=1 drop-test",
            metrics=metrics,
        )
    return AuditResult(
        name="feature_target_proximity", status="ok", reason_code="",
        detail=f"near-target features present but performance does not collapse without them "
               f"({ratio:.2f}x < {cfg.leak_collapse_ratio}x); not classified as leakage",
        metrics=metrics,
    )


def audit_metric_reproduction(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    y = _sel(data, art.windows.test)["load"]
    recomputed = mae(y, art.predictions)
    if art.reported_mae == 0:
        rel = 0.0 if recomputed == 0 else float("inf")
    else:
        rel = abs(recomputed - art.reported_mae) / art.reported_mae
    metrics = {"reported_mae": round(art.reported_mae, 1),
               "recomputed_mae": round(recomputed, 1),
               "relative_diff": round(rel, 5)}
    if rel > cfg.metric_tolerance:
        return AuditResult(
            name="metric_reproduction", status="fail", reason_code="CONFLICT_EVIDENCE",
            detail=f"reported MAE {art.reported_mae:.1f} does not reproduce from returned "
                   f"predictions on the full test window ({recomputed:.1f}, diff {rel:.2%} "
                   f"> {cfg.metric_tolerance:.0%}); reported number not reproducible as stated "
                   f"(N=1 recomputation)",
            metrics=metrics,
        )
    return AuditResult(name="metric_reproduction", status="ok", reason_code="",
                       detail=f"reported MAE reproduces from predictions (diff {rel:.4%})",
                       metrics=metrics)


def audit_claim_conditions(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    y = _sel(data, art.windows.test)["load"]
    recomputed = mae(y, art.predictions)
    threshold = art.claim.max_mae_pct_of_mean / 100.0 * float(y.mean())
    lag_p, lag_s = baseline_lags(data)
    mae_p = naive_baseline_mae(data, art.windows.test, lag_p)
    mae_s = naive_baseline_mae(data, art.windows.test, lag_s)
    metrics = {
        "recomputed_mae": round(recomputed, 1),
        "threshold_mae": round(threshold, 1),
        "persistence_mae": round(mae_p, 1),
        "seasonal_naive_mae": round(mae_s, 1),
        "mean_load_test_window": round(float(y.mean()), 1),
        "baseline_lags": [lag_p, lag_s],
    }
    failures = []
    if recomputed > threshold:
        failures.append(f"MAE {recomputed:.0f} exceeds {art.claim.max_mae_pct_of_mean}% of mean "
                        f"load ({threshold:.0f})")
    if "seasonal_naive" in art.claim.must_beat and not recomputed < mae_s:
        failures.append(f"MAE {recomputed:.0f} not better than seasonal-naive {mae_s:.0f}")
    if "persistence" in art.claim.must_beat and not recomputed < mae_p:
        failures.append(f"MAE {recomputed:.0f} not better than persistence {mae_p:.0f}")
    if failures:
        return AuditResult(
            name="claim_conditions", status="fail", reason_code="NOISE",
            detail="claimed performance does not hold: " + "; ".join(failures) +
                   " (N=1 test window, baselines computed on the same window)",
            metrics=metrics,
        )
    return AuditResult(name="claim_conditions", status="ok", reason_code="",
                       detail=f"claim holds: MAE {recomputed:.0f} <= {threshold:.0f}; "
                              f"beats persistence {mae_p:.0f} and seasonal-naive {mae_s:.0f}",
                       metrics=metrics)


def audit_trial_disclosure(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    log = art.trial_log or []
    k = len(log)
    if k <= 1:
        return AuditResult(
            name="trial_disclosure", status="ok", reason_code="",
            detail="single pre-registered configuration (K=1); no multiplicity correction needed",
            metrics={"K": k},
        )
    best = min(log, key=lambda e: e["test_mae"])
    lag_p, lag_s = baseline_lags(data)
    naive_test = naive_baseline_mae(data, art.windows.test, lag_s)
    naive_conf = naive_baseline_mae(data, art.windows.confirmation, lag_s)
    preds_c, mae_c = art.retrain_eval_fn(data, best["features"],
                                         art.windows.train, art.windows.confirmation)
    imp_test = (naive_test - best["test_mae"]) / naive_test
    imp_conf = (naive_conf - mae_c) / naive_conf
    metrics = {"K": k, "best_test_mae": round(best["test_mae"], 1),
               "confirmation_mae": round(mae_c, 1),
               "seasonal_naive_test_mae": round(naive_test, 1),
               "seasonal_naive_confirmation_mae": round(naive_conf, 1),
               "improvement_test": round(imp_test, 4),
               "improvement_confirmation": round(imp_conf, 4),
               "min_improvement_for_replication": cfg.min_improvement_for_replication}
    failures = []
    if imp_test > cfg.min_improvement_for_replication and imp_conf <= 0:
        failures.append(
            f"improvement over seasonal-naive does not replicate on the confirmation "
            f"window ({imp_test:.1%} test -> {imp_conf:.1%} confirmation <= 0)")

    # Fixed, unsearched baseline: the largest feature set common to ALL disclosed
    # trial configurations (their intersection). Nobody selected it on the test
    # window, so the best-of-K winner's gain over it measures the pure selection
    # (multiple-comparison) effect. Comparing only against seasonal-naive is not
    # enough: a searched configuration can beat a weak cadence baseline on BOTH
    # windows while its edge over the unsearched configuration is pure noise.
    common = set(log[0]["features"])
    for entry in log[1:]:
        common &= set(entry["features"])
    fixed_features = [f for f in log[0]["features"] if f in common]
    if fixed_features:
        _, mae_fixed_test = art.retrain_eval_fn(data, fixed_features,
                                                art.windows.train, art.windows.test)
        _, mae_fixed_conf = art.retrain_eval_fn(data, fixed_features,
                                                art.windows.train, art.windows.confirmation)
        gain_test = ((mae_fixed_test - best["test_mae"]) / mae_fixed_test
                     if mae_fixed_test else float("inf"))
        gain_conf = ((mae_fixed_conf - mae_c) / mae_fixed_conf
                     if mae_fixed_conf else float("inf"))
        metrics["fixed_baseline_features"] = fixed_features
        metrics["fixed_baseline_test_mae"] = round(mae_fixed_test, 1)
        metrics["fixed_baseline_confirmation_mae"] = round(mae_fixed_conf, 1)
        metrics["selection_gain_test"] = round(gain_test, 4)
        metrics["selection_gain_confirmation"] = round(gain_conf, 4)
        if gain_test > cfg.min_improvement_for_replication and gain_conf <= 0:
            failures.append(
                f"selection gain over the fixed unsearched baseline does not replicate "
                f"on the confirmation window ({gain_test:.1%} test -> "
                f"{gain_conf:.1%} confirmation <= 0)")
    else:
        metrics["fixed_baseline_features"] = []
        metrics["fixed_baseline_note"] = ("no common feature subset across disclosed "
                                          "trials; fixed-baseline comparison not assessable")

    if failures:
        return AuditResult(
            name="trial_disclosure", status="fail", reason_code="NOISE",
            detail=f"selection bias: best of K={k} configurations tried on the test window; "
                   + "; ".join(failures)
                   + "; the claimed edge is a multiple-comparison artifact (N=1 replication)",
            metrics=metrics,
        )
    gain_test_m = metrics.get("selection_gain_test", 0.0)
    material = (imp_test > cfg.min_improvement_for_replication
                or gain_test_m > cfg.min_improvement_for_replication)
    edges = (f"seasonal-naive edge {imp_test:.1%} test -> {imp_conf:.1%} confirmation"
             + (f"; fixed-baseline selection gain {gain_test_m:.1%} test -> "
                f"{metrics.get('selection_gain_confirmation', 0.0):.1%} confirmation"
                if "selection_gain_test" in metrics else ""))
    if not material:
        detail = (f"K={k} configurations disclosed; best-of-K shows no material test-window "
                  f"edge (improvement over seasonal-naive {imp_test:.1%}"
                  + (f", over fixed baseline {gain_test_m:.1%}" if "selection_gain_test" in metrics else "")
                  + f"; both <= {cfg.min_improvement_for_replication:.0%} materiality); "
                  f"no replication correction needed")
    else:
        detail = (f"K={k} configurations disclosed; best-config edges replicate on the "
                  f"confirmation window: {edges}")
    return AuditResult(name="trial_disclosure", status="ok", reason_code="",
                       detail=detail, metrics=metrics)


def audit_timezone_integrity(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    """Structural check of the case's working frame: no duplicated timestamps;
    for hourly cadence, no missing wall-clock hours in the domain timezone."""
    fdf = art.feature_frame_fn(data)
    idx = fdf.index
    dups = int(idx.duplicated().sum())
    problems = []
    if dups > 0:
        problems.append(
            f"{dups} duplicated timestamps in the working feature frame "
            f"(naive local time across a DST transition duplicates wall-clock hours)"
        )
    diffs = pd.Series(idx[1:] - idx[:-1]) if len(idx) > 1 else pd.Series(dtype="timedelta64[ns]")
    median_step = diffs.median() if len(diffs) else pd.Timedelta(0)
    if median_step <= pd.Timedelta("1h") and dups == 0:
        # hourly cadence: gap check against the domain-timezone wall calendar.
        # Works for both naive indexes (localize to UTC first) and tz-aware
        # ones (convert to the domain timezone). Only run when the frame has
        # no duplicates: with duplicated wall labels the localize/convert is
        # undefined and the duplicate finding already fails the audit.
        try:
            if idx.tz is None:
                local_idx = idx.tz_localize("UTC").tz_convert(cfg.domain_tz)
            else:
                local_idx = idx.tz_convert(cfg.domain_tz)
            full_local = pd.date_range(local_idx.min(), local_idx.max(), freq="h",
                                       tz=cfg.domain_tz)
            missing = len(full_local.difference(local_idx.unique()))
            if missing > 0:
                problems.append(f"{missing} missing wall-clock hours vs {cfg.domain_tz} calendar")
        except Exception as e:  # ambiguous/nonexistent local times
            problems.append(f"tz localization failed: {type(e).__name__}")
    if problems:
        return AuditResult(
            name="timezone_integrity", status="fail", reason_code="ENVIRONMENT",
            detail="timezone handling defect: " + "; ".join(problems) +
                   " (N=1 case, working frame inspected through the case feature builder)",
            metrics={"duplicated_timestamps": dups},
        )
    cadence = "hourly" if median_step <= pd.Timedelta("1h") else "daily"
    return AuditResult(name="timezone_integrity", status="ok", reason_code="",
                       detail=f"no duplicate/gap anomalies at {cadence} cadence",
                       metrics={"duplicated_timestamps": dups})


AUDITS = (
    audit_prediction_alignment,
    audit_train_test_isolation,
    audit_timezone_integrity,
    audit_feature_target_proximity,
    audit_deployment_consistency,
    audit_metric_reproduction,
    audit_claim_conditions,
    audit_trial_disclosure,
)


def run_verdict(data: pd.DataFrame, art: CaseArtifact, cfg: AuditConfig | None = None) -> dict:
    cfg = cfg or AuditConfig()
    results = [fn(data, art, cfg) for fn in AUDITS]
    fails = [r for r in results if r.status == "fail"]
    inconv = [r for r in results if r.status == "inconclusive"]
    if fails:
        status, reason = "FAIL", fails[0].reason_code
    elif inconv:
        status, reason = "INCONCLUSIVE", inconv[0].reason_code
    else:
        status, reason = "PASS", ""
    return {
        "status": status,
        "reason_code": reason,
        "audits": [
            {"name": r.name, "status": r.status, "reason_code": r.reason_code,
             "detail": r.detail, "metrics": r.metrics}
            for r in results
        ],
    }
