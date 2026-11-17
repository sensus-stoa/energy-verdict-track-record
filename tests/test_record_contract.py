"""Mechanical contracts of the committed v5 record
(CHECKABLE_FINDINGS.md: F-01 at record level, F-04, F-05, F-06, F-08, F-13).

These tests pin the COMMITTED record files against the committed run
outputs. On a new record version they must be updated together with the
record (that is the point: the text must never drift from the machine).
"""
import json
import re
from pathlib import Path

from src.data_generator import load_or_generate, windows
from src.run_track_record import REGISTRY
from src.verdict_procedure import REASON_CODES

REPO = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((REPO / "trust/evidence-manifest.json").read_text())
VERDICT = json.loads((REPO / "trust/verdict.json").read_text())


def ev_by_id() -> dict:
    return {e["id"]: e for e in MANIFEST["evidence"]}


def case_of(claim: dict) -> str | None:
    m = re.search(r"case ([a-z0-9_]+)", claim["verification"])
    return m.group(1) if m else None


def per_case_path(claim: dict) -> Path:
    ev = ev_by_id()
    case_id = case_of(claim)
    ids = [i for i in claim["evidence"]
           if ev[i]["path"] == f"results/per_case/{case_id}.json"]
    assert len(ids) == 1, f"{claim['id']}: per-case JSON not referenced"
    return REPO / ev[ids[0]]["path"]


# ---- F-13 (in-repo part): manifest hygiene and ID binding -----------------

def test_evidence_ids_unique_and_disjoint_from_disclosures():
    ids = [e["id"] for e in MANIFEST["evidence"]]
    assert len(ids) == len(set(ids))
    disc = [d["id"] for d in MANIFEST.get("disclosures", [])]
    assert not set(ids) & set(disc)


def test_all_hashes_are_sha256_hex():
    for it in MANIFEST["evidence"] + MANIFEST.get("disclosures", []):
        assert re.fullmatch(r"[0-9a-f]{64}", it["sha256"]), it["id"]


def test_evidence_rules_declare_version_binding():
    rules = MANIFEST.get("evidence_rules", "")
    for token in ("record_version", "evidence_id", "sha256"):
        assert token in rules, token


# ---- F-08: every claim carries its case module, dataset and output --------

def test_every_claim_carries_case_dataset_and_case_output():
    ev = ev_by_id()
    dataset_id = {}
    for e in MANIFEST["evidence"]:
        if e["path"].endswith("energy_daily.csv"):
            dataset_id["daily"] = e["id"]
        if e["path"].endswith("energy_hourly.csv"):
            dataset_id["hourly"] = e["id"]
    assert set(dataset_id) == {"daily", "hourly"}
    for c in VERDICT["claims"]:
        if c["id"] == "C-03":                          # NC-NULL: control, not a case
            assert c.get("control") == "NC-NULL"
            assert any(ev[i]["path"] == "results/null_control.csv"
                       for i in c["evidence"])
            continue
        case_id = case_of(c)
        assert case_id, f"{c['id']}: verification does not name a case"
        assert case_id in REGISTRY, case_id
        key = REGISTRY[case_id][2]
        assert dataset_id[key] in c["evidence"], f"{c['id']}: missing {key} dataset"
        assert any(ev[i]["path"] == f"results/per_case/{case_id}.json"
                   for i in c["evidence"]), f"{c['id']}: missing per-case JSON"


# ---- F-04/F-05: verdict claims must match the machine evidence ------------

def test_audit_outcomes_match_verdict_claims():
    for c in VERDICT["claims"]:
        if c["id"] == "C-03":
            continue
        d = json.loads(per_case_path(c).read_text())
        statuses = {a["name"]: a["status"] for a in d["verdict"]["audits"]}
        fails = sorted(n for n, s in statuses.items() if s == "fail")
        inconv = sorted(n for n, s in statuses.items() if s == "inconclusive")
        assert sorted(c["failing_audits"]) == fails, c["id"]
        assert sorted(c["inconclusive_audits"]) == inconv, c["id"]
        assert c["case_verdict"] == d["verdict"]["status"], c["id"]
        assert c["case_reason_code"] == d["verdict"]["reason_code"], c["id"]


def test_promised_audits_fire_in_their_cases():
    # C-11 v4 bug: claims.md PROMISED "the trial-disclosure audit
    # classifies the outcome as a selection artifact" while the evidence
    # JSON had trial_disclosure=ok. Each claim's promised audit must
    # actually fail in its case evidence.
    promised = {
        "C-01": "deployment_consistency",
        "C-04": "train_test_isolation",
        "C-05": "feature_target_proximity",
        "C-08": "timezone_integrity",
        "C-10": "metric_reproduction",
        "C-11": "trial_disclosure",
        "C-13": "claim_conditions",
        "C-14": "trial_disclosure",
    }
    for cid, audit_name in promised.items():
        c = next(x for x in VERDICT["claims"] if x["id"] == cid)
        d = json.loads(per_case_path(c).read_text())
        st = {a["name"]: a["status"] for a in d["verdict"]["audits"]}
        assert st[audit_name] == "fail", (cid, audit_name, st[audit_name])


# ---- F-06: window row counts match the record everywhere ------------------

def test_window_row_counts_match_the_record():
    df = load_or_generate()
    assert len(df) == 425
    w = windows()
    counts = {k: int(((df.index >= w[k][0]) & (df.index < w[k][1])).sum())
              for k in ("train", "test", "confirmation")}
    assert counts == {"train": 335, "test": 30, "confirmation": 60}
    hourly_cases = {cid for cid, spec in REGISTRY.items() if spec[2] == "hourly"}
    for p in sorted((REPO / "results/per_case").glob("*.json")):
        d = json.loads(p.read_text())
        pa = next(a for a in d["verdict"]["audits"]
                  if a["name"] == "prediction_alignment")
        expected = 168 if d["case_id"] in hourly_cases else 30
        assert pa["status"] == "ok" and pa["metrics"]["n_test_rows"] == expected, d["case_id"]


# ---- F-01 at record level: per-case JSON shape contract -------------------

def test_per_case_json_shape_contract():
    for p in sorted((REPO / "results/per_case").glob("*.json")):
        d = json.loads(p.read_text())
        for a in d["verdict"]["audits"]:
            assert a["status"] in {"ok", "fail", "inconclusive", "skipped"}
            assert isinstance(a["detail"], str)
            assert isinstance(a["metrics"], dict)
            if a["status"] == "ok":
                assert a["reason_code"] == "", (d["case_id"], a["name"])
            else:
                assert a["reason_code"] in REASON_CODES, (d["case_id"], a["name"])
