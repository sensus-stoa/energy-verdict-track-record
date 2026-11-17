"""Audit-battery unit contour (CHECKABLE_FINDINGS.md: F-01..F-05, F-16, F-17).

Numbers pinned here belong to the v5 record (windows 335/30/60, daily seed
42, hourly seed 43); a future record version that changes windows or data
must update these expectations together with the record.
"""
import ast
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.clean_cases import clean_pipeline_01 as c1
from src.clean_cases import clean_pipeline_04_hourly as c4
from src.data_generator import load_or_generate, load_or_generate_hourly, windows
from src.defects import look_ahead_leak as d1
from src.defects import search_overfit as d8
from src.defects import selection_bias as d6
from src.defects import seasonal_naive_only as d7
from src.verdict_procedure import (AUDITS, AuditConfig, REASON_CODES,
                                   audit_deployment_consistency,
                                   audit_prediction_alignment,
                                   audit_timezone_integrity, run_verdict)

W = windows()
WIN = Windows(train=W["train"], test=W["test"], confirmation=W["confirmation"])


def audits(v: dict) -> dict:
    return {a["name"]: a for a in v["audits"]}


def stub_artifact(frame_fn, predictions=None, windows=WIN) -> CaseArtifact:
    """Minimal artifact for audits that only need the frame (timezone)."""
    return CaseArtifact(
        case_id="stub", is_defective=False, defect_type=None,
        claim=CaseClaim(statement="stub", max_mae_pct_of_mean=100.0),
        reported_mae=0.0,
        predictions=predictions if predictions is not None else pd.Series(dtype=float),
        windows=windows, model=None, feature_names_used=[],
        feature_frame_fn=frame_fn,
        retrain_eval_fn=lambda d, f, tr, ev: (pd.Series(dtype=float), 0.0),
        trial_log=[], notes="",
    )


@pytest.fixture(scope="module")
def daily():
    return load_or_generate()


@pytest.fixture(scope="module")
def hourly():
    return load_or_generate_hourly()


# ---- F-01: AuditResult serialization contract -----------------------------

def test_audit_battery_order_and_shape(daily):
    art = c1.run(daily, WIN)
    v = run_verdict(daily, art, AuditConfig())
    assert v["status"] == "PASS" and v["reason_code"] == ""
    expected = [fn.__name__.removeprefix("audit_") for fn in AUDITS]
    assert len(expected) == 8
    assert [a["name"] for a in v["audits"]] == expected
    for a in v["audits"]:
        assert isinstance(a["detail"], str)
        assert isinstance(a["metrics"], dict)
        if a["status"] == "ok":
            assert a["reason_code"] == ""          # no text in reason_code
        else:
            assert a["reason_code"] in REASON_CODES


# ---- F-17: timezone sub-checks run for BOTH naive and tz-aware frames -----

def test_timezone_integrity_runs_and_passes_on_tz_aware_frames(hourly):
    # v4 regression: the missing-hours check silently skipped tz-aware
    # frames; it must now run and NOT accuse the clean UTC-index pipeline.
    art = c4.run(hourly, WIN)          # hourly cases keep their own windows
    v = run_verdict(hourly, art, AuditConfig())
    tz = audits(v)["timezone_integrity"]
    assert tz["status"] == "ok"
    assert tz["metrics"]["duplicated_timestamps"] == 0


def test_timezone_integrity_detects_missing_hour_naive():
    dh = load_or_generate_hourly()
    gapped = dh.drop(dh.index[1000])
    r = audit_timezone_integrity(gapped, stub_artifact(lambda d: d.copy()), AuditConfig())
    assert r.status == "fail" and r.reason_code == "ENVIRONMENT"
    assert "missing wall-clock hours" in r.detail


def test_timezone_integrity_detects_missing_hour_tz_aware():
    dh = load_or_generate_hourly().tz_localize("UTC")
    gapped = dh.drop(dh.index[1000])
    r = audit_timezone_integrity(gapped, stub_artifact(lambda d: d.copy()), AuditConfig())
    assert r.status == "fail" and r.reason_code == "ENVIRONMENT"
    assert "missing wall-clock hours" in r.detail


# ---- F-17: prediction alignment sees duplicated timestamps ----------------

def test_prediction_alignment_rejects_duplicate_timestamps(daily):
    art = c1.run(daily, WIN)
    idx = list(art.predictions.index)
    vals = list(art.predictions.to_numpy())
    # pure duplication: same window coverage, one timestamp twice (len 31)
    dup_idx = pd.DatetimeIndex([idx[0]] + idx)
    art2 = replace(art, predictions=pd.Series([vals[0]] + vals, index=dup_idx, name="pred"))
    r = audit_prediction_alignment(daily, art2, AuditConfig())
    assert r.status == "inconclusive" and r.reason_code == "DATA"
    assert "duplicated=1" in r.detail


# ---- F-02: required-metrics contracts of individual audits ----------------

def test_deployment_audit_reports_gap_against_threshold(daily):
    art = d1.run(daily, WIN)
    v = run_verdict(daily, art, AuditConfig())
    dc = audits(v)["deployment_consistency"]
    assert dc["status"] == "fail" and dc["reason_code"] == "DATA"
    m = dc["metrics"]
    assert m["lookahead_gap_threshold"] == 0.20        # declared AND used
    assert m["gap_exceeds_threshold"] is True
    assert m["relative_gap"] == pytest.approx(0.2551, abs=5e-4)
    assert "(deployment_mae - reported_mae) / reported_mae" in dc["detail"]


def test_trial_disclosure_reports_fixed_baseline_edge(daily):
    art = d6.run(daily, WIN)
    v = run_verdict(daily, art, AuditConfig())
    assert v["status"] == "FAIL" and v["reason_code"] == "NOISE"
    td = audits(v)["trial_disclosure"]
    assert td["status"] == "fail" and td["reason_code"] == "NOISE"
    m = td["metrics"]
    assert m["K"] == 20
    assert m["fixed_baseline_features"] == ["doy_sin", "doy_cos", "dow_weekend"]
    assert m["selection_gain_test"] > m["min_improvement_for_replication"]
    assert m["selection_gain_confirmation"] <= 0       # edge does not replicate


# ---- pinned v5 semantics of the new cases ---------------------------------

def test_search_overfit_is_decided_solely_by_trial_disclosure(daily):
    # C-14: the claim holds on the test window; audit #8 fires alone.
    art = d8.run(daily, WIN)
    v = run_verdict(daily, art, AuditConfig())
    fails = [a["name"] for a in v["audits"] if a["status"] == "fail"]
    assert v["status"] == "FAIL" and v["reason_code"] == "NOISE"
    assert fails == ["trial_disclosure"]
    assert audits(v)["claim_conditions"]["status"] == "ok"


def test_claim_conditions_gate_vs_must_beat(hourly):
    # C-13: the 6.0% absolute gate passes; skill (must-beat) fails.
    art = d7.run(hourly, WIN)
    v = run_verdict(hourly, art, AuditConfig())
    cc = audits(v)["claim_conditions"]
    assert cc["status"] == "fail" and cc["reason_code"] == "NOISE"
    m = cc["metrics"]
    assert m["recomputed_mae"] <= m["threshold_mae"]
    assert "not better than seasonal-naive" in cc["detail"]


# ---- F-16: no dead config — every declared parameter is used --------------

def _procedure_source():
    return (Path(__file__).resolve().parents[1] / "src" / "verdict_procedure.py").read_text()


def test_auditconfig_fields_are_all_used():
    tree = ast.parse(_procedure_source())
    cls = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "AuditConfig")
    fields = [st.target.id for st in cls.body if isinstance(st, ast.AnnAssign)]
    lo, hi = cls.lineno, cls.end_lineno
    used = set()
    for node in ast.walk(tree):
        line = getattr(node, "lineno", None)
        if line is not None and lo <= line <= hi:
            continue                                   # class body = declaration
        if isinstance(node, ast.Attribute):
            used.add(node.attr)
        elif isinstance(node, ast.Name):
            used.add(node.id)
    for f in fields:
        assert f in used, f"AuditConfig.{f} declared but never used"


def test_module_threshold_constants_are_all_used():
    src = _procedure_source()
    for const in ("LOOKAHEAD_GAP_THRESHOLD", "LEAK_CORR_THRESHOLD",
                  "LEAK_COLLAPSE_RATIO", "METRIC_TOLERANCE"):
        assert src.count(const) >= 2, f"{const} declared but never used"
