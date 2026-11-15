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
  confirmation window, otherwise the claimed edge is classified as noise.
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
    if len(missing) or len(extra) or not np.isfinite(preds.to_numpy()).all():
        return AuditResult(
            "prediction_alignment", "inconclusive", "DATA",
            f"predictions do not cover the test window exactly: missing={len(missing)} "
            f"extra={len(extra)} nonfinite={int((~np.isfinite(preds.to_numpy())).sum())} (N=1 case)",
        )
    return AuditResult("prediction_alignment", "ok",
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
            "deployment_consistency", "inconclusive", "DATA",
            f"feature availability unknown to the audit for: {unknown} (N=1 case)",
        )
    unavailable_used = [f for f in used if not cfg.availability[f]]
    metrics = {"unavailable_features_used": unavailable_used}
    if unavailable_used:
        # Structural finding already decisive; quantify the impact anyway.
        safe = [f for f in used if cfg.availability[f]]
        if safe:
            _, mae_d = art.retrain_eval_fn(data, safe, art.windows.train, art.windows.test)
            gap = (mae_d - art.reported_mae) / art.reported_mae if art.reported_mae else float("inf")
            metrics["deployment_mae"] = round(mae_d, 1)
            metrics["relative_gap"] = round(gap, 3)
            detail = (
                f"look-ahead leak: model uses features not available at inference time "
                f"({unavailable_used}); re-run with deployment-realistic features gives "
                f"MAE {mae_d:.0f} vs reported {art.reported_mae:.0f} (gap {gap:.1%}, N=1 retrain)"
            )
        else:
            gap = float("inf")
            detail = (f"look-ahead leak: ALL used features are unavailable at inference "
                      f"({unavailable_used})")
        return AuditResult("deployment_consistency", "fail", "DATA", detail, metrics)
    return AuditResult("deployment_consistency", "ok",
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
        return AuditResult("feature_target_proximity", "ok", metrics=metrics)
    # Drop-test: remove ALL suspect features at once, retrain, re-evaluate.
    safe = [f for f in art.feature_names_used if f not in {s for s, _ in suspects}]
    if not safe:
        return AuditResult("feature_target_proximity", "fail", "DATA",
                           f"target leakage: every used feature is a near-copy of the target "
                           f"({suspects}); nothing left to retrain with (N=1 drop-test)", metrics)
    _, mae_drop = art.retrain_eval_fn(data, safe, art.windows.train, art.windows.test)
    ratio = mae_drop / art.reported_mae if art.reported_mae else float("inf")
    metrics["mae_without_suspects"] = round(mae_drop, 1)
    metrics["collapse_ratio"] = round(ratio, 2)
    if ratio >= cfg.leak_collapse_ratio:
        return AuditResult(
            "feature_target_proximity", "fail", "DATA",
            f"target leakage: features {suspects} are near-copies of the target "
            f"(|corr| > {cfg.leak_corr_threshold}); without them MAE grows "
            f"{ratio:.1f}x ({art.reported_mae:.0f} -> {mae_drop:.0f}), N=1 drop-test",
            metrics,
        )
    return AuditResult(
        "feature_target_proximity", "ok",
        f"near-target features present but performance does not collapse without them "
        f"({ratio:.2f}x < {cfg.leak_collapse_ratio}x); not classified as leakage",
        metrics,
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
            "metric_reproduction", "fail", "CONFLICT_EVIDENCE",
            f"reported MAE {art.reported_mae:.1f} does not reproduce from returned "
            f"predictions on the full test window ({recomputed:.1f}, diff {rel:.2%} "
            f"> {cfg.metric_tolerance:.0%}); reported number not reproducible as stated "
            f"(N=1 recomputation)", metrics,
        )
    return AuditResult("metric_reproduction", "ok",
                       f"reported MAE reproduces from predictions (diff {rel:.4%})",
                       metrics)


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
            "claim_conditions", "fail", "NOISE",
            "claimed performance does not hold: " + "; ".join(failures) +
            " (N=1 test window, baselines computed on the same window)", metrics,
        )
    return AuditResult("claim_conditions", "ok",
                       f"claim holds: MAE {recomputed:.0f} <= {threshold:.0f}; "
                       f"beats persistence {mae_p:.0f} and seasonal-naive {mae_s:.0f}", metrics)


def audit_trial_disclosure(data, art: CaseArtifact, cfg: AuditConfig) -> AuditResult:
    log = art.trial_log or []
    k = len(log)
    if k <= 1:
        return AuditResult(
            "trial_disclosure", "ok",
            f"single pre-registered configuration (K=1); no multiplicity correction needed",
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
               "improvement_test": round(imp_test, 4),
               "improvement_confirmation": round(imp_conf, 4)}
    if imp_test > cfg.min_improvement_for_replication and imp_conf <= 0:
        return AuditResult(
            "trial_disclosure", "fail", "NOISE",
            f"selection bias: best of K={k} configurations tried on the test window "
            f"(improvement {imp_test:.1%} over seasonal-naive) does not replicate on the "
            f"held-out confirmation window (improvement {imp_conf:.1%} <= 0); the claimed "
            f"edge is a multiple-comparison artifact, N=1 replication", metrics,
        )
    return AuditResult(
        "trial_disclosure", "ok",
        f"K={k} configurations disclosed; best-config improvement replicates on the "
        f"confirmation window ({imp_test:.1%} test -> {imp_conf:.1%} confirmation)", metrics,
    )


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
    if median_step <= pd.Timedelta("1h"):
        # hourly cadence: gap check via tz-aware local calendar (only if no dups;
        # otherwise localize is undefined and the duplicate finding already fails)
        if dups == 0 and idx.tz is None:
            try:
                localized = idx.tz_localize("UTC").tz_convert(cfg.domain_tz)
                full_local = pd.date_range(localized.min(), localized.max(), freq="h",
                                           tz=cfg.domain_tz)
                missing = len(full_local.difference(localized.unique()))
                if missing > 0:
                    problems.append(f"{missing} missing wall-clock hours vs {cfg.domain_tz} calendar")
            except Exception as e:  # ambiguous/nonexistent local times
                problems.append(f"tz localization failed: {type(e).__name__}")
    if problems:
        return AuditResult(
            "timezone_integrity", "fail", "ENVIRONMENT",
            "timezone handling defect: " + "; ".join(problems) +
            " (N=1 case, working frame inspected through the case feature builder)",
        )
    cadence = "hourly" if median_step <= pd.Timedelta("1h") else "daily"
    return AuditResult("timezone_integrity", "ok",
                       f"no duplicate/gap anomalies at {cadence} cadence")


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
