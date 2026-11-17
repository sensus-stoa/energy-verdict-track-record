"""trust_check - consistency checker for the trust record files.

Verifies that trust/claims.md, trust/limitations.md,
trust/evidence-manifest.json and trust/verdict.json are internally
consistent, that every evidence hash matches the committed file, that the
criteria lock and object pin resolve in git history, that no sign-off file
exists (an agent never signs) and that no local machine paths or personal
email addresses leaked into the public record files.

Usage (from the repository root):
    python tools/trust_check.py            # writes trust_check_report.json

The report carries a relative root path and the checked commit hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RECORD_FILES = ["trust/claims.md", "trust/limitations.md",
                "trust/evidence-manifest.json", "trust/verdict.json"]
REPORT_PATH = ROOT / "trust_check_report.json"
TOOL_VERSION = "0.2.0"

# public record files scanned for machine-local paths / personal emails
HYGIENE_FILES = ["README.md", "trust/claims.md", "trust/limitations.md",
                 "trust/evidence-manifest.json", "trust/verdict.json",
                 "results/track_record.csv", "results/null_control.csv"]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
EMAIL_ALLOWLIST = ("@users.noreply.github.com",)
HOME_PATH_RE = re.compile(r"(/home/[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+)")


def git(args: list[str]) -> str:
    out = subprocess.run(["git"] + args, cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout.strip()


def head_commit() -> str:
    return git(["rev-parse", "HEAD"])


def is_ancestor(commit: str) -> bool:
    out = subprocess.run(["git", "merge-base", "--is-ancestor", commit, "HEAD"],
                         cwd=ROOT, capture_output=True, text=True)
    return out.returncode == 0


def blob_at(commit: str, path: str) -> bytes:
    out = subprocess.run(["git", "show", f"{commit}:{path}"],
                         cwd=ROOT, capture_output=True)
    if out.returncode != 0:
        raise RuntimeError(f"blob {path} not found at {commit}: {out.stderr.decode().strip()}")
    return out.stdout


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def check(name: str, ok: bool, detail: str) -> dict:
    return {"id": name, "status": "PASS" if ok else "FAIL", "detail": detail}


def main() -> int:
    checks: list[dict] = []

    # 1. files present
    missing = [p for p in RECORD_FILES if not (ROOT / p).is_file() or (ROOT / p).stat().st_size == 0]
    checks.append(check("FILES_PRESENT", not missing,
                        "all 4 record files present and non-empty" if not missing
                        else f"missing/empty: {missing}"))

    # 2. json valid
    try:
        manifest = json.loads((ROOT / "trust/evidence-manifest.json").read_text())
        verdict = json.loads((ROOT / "trust/verdict.json").read_text())
        need_manifest = ["schema", "object", "criteria_lock", "evidence", "controls"]
        need_verdict = ["verdict_id", "object_version", "claims", "negative_controls"]
        bad = [k for k in need_manifest if k not in manifest] + \
              [k for k in need_verdict if k not in verdict]
        checks.append(check("JSON_VALID", not bad,
                            "manifest and verdict parse; structure complete" if not bad
                            else f"missing keys: {bad}"))
    except Exception as e:  # noqa: BLE001
        checks.append(check("JSON_VALID", False, f"parse error: {type(e).__name__}: {e}"))
        _write_report([], checks)
        return 1

    # 3. hashes match (evidence AND disclosures)
    evidence = manifest.get("evidence", [])
    disclosures = manifest.get("disclosures", [])
    mismatches, missing = [], []
    for item in evidence + disclosures:
        p = ROOT / item["path"]
        if not p.is_file():
            missing.append(item["path"])
            continue
        if sha256_bytes(p.read_bytes()) != item["sha256"]:
            mismatches.append(item["path"])
    hashed = len(evidence) + len(disclosures)
    checks.append(check("HASHES_MATCH", not mismatches and not missing,
                        f"{hashed} hashed artifacts verified "
                        f"({len(evidence)} evidence + {len(disclosures)} disclosures)"
                        if not mismatches and not missing else
                        f"missing: {missing}; hash mismatches: {mismatches}"))

    # 4. claims cross-reference
    claims_md = (ROOT / "trust/claims.md").read_text()
    claim_ids_md = set(re.findall(r"^## (C-\d+)", claims_md, flags=re.M))
    vclaims = verdict.get("claims", [])
    unknown = [c["id"] for c in vclaims if c["id"] not in claim_ids_md]
    checks.append(check("CLAIMS_XREF", not unknown,
                        f"all {len(vclaims)} claim ids present in claims.md" if not unknown
                        else f"claim ids not found in claims.md: {unknown}"))

    # 5. evidence cross-reference
    ev_ids = {e["id"] for e in evidence}
    unresolved = []
    for c in vclaims:
        for ref in c.get("evidence", []):
            if ref not in ev_ids:
                unresolved.append(f"{c['id']}->{ref}")
    checks.append(check("EVIDENCE_XREF", not unresolved,
                        "all evidence references resolve to manifest evidence entries"
                        if not unresolved else f"unresolved references: {unresolved}"))

    # 6. controls
    control_problems = []
    manifest_controls = {c["id"]: c for c in manifest.get("controls", [])}
    for c in vclaims:
        cid = c.get("control")
        if cid and cid not in manifest_controls:
            control_problems.append(f"{c['id']} references unknown control {cid}")
    for nc in verdict.get("negative_controls", []):
        mc = manifest_controls.get(nc["id"])
        if not mc:
            control_problems.append(f"negative control {nc['id']} not in manifest controls")
            continue
        for field in ("N", "rejections", "rate", "wilson_ci_95"):
            if nc.get(field) != mc.get(field):
                control_problems.append(f"{nc['id']}.{field}: verdict {nc.get(field)} != "
                                        f"manifest {mc.get(field)}")
    n_pass = sum(1 for c in vclaims if c.get("status") == "PASS")
    checks.append(check("CONTROLS", not control_problems,
                        f"controls sound for {n_pass} PASS claims" if not control_problems
                        else "; ".join(control_problems)))

    # 7. reason codes on non-PASS claims
    bad_rc = [c["id"] for c in vclaims
              if c.get("status") != "PASS" and not c.get("reason_code")]
    checks.append(check("REASON_CODES", not bad_rc,
                        "every non-PASS claim carries a reason_code" if not bad_rc
                        else f"non-PASS claims without reason_code: {bad_rc}"))

    # 8. valid_until
    vu = verdict.get("valid_until")
    checks.append(check("VALID_UNTIL", vu == "object-change",
                        f"valid_until: {vu}"))

    # 9. lock blob
    lock = manifest.get("criteria_lock", {})
    lock_commit = lock.get("commit", "")
    try:
        blob = blob_at(lock_commit, "trust/claims.md")
        checks.append(check("LOCK_BLOB", len(blob) > 0,
                            f"locked trust/claims.md non-empty ({len(blob)} bytes at "
                            f"{lock_commit[:7]}...)"))
    except Exception as e:  # noqa: BLE001
        checks.append(check("LOCK_BLOB", False, str(e)))

    # 10. lock ancestor
    checks.append(check("LOCK_ANCESTOR", bool(lock_commit) and is_ancestor(lock_commit),
                        f"{lock_commit[:7]}... is an ancestor of HEAD"
                        if lock_commit else "criteria_lock.commit missing"))

    # 11. object pin: object.commit is an ancestor and evidence hashes match its tree
    obj_commit = manifest.get("object", {}).get("commit", "")
    obj_problems = []
    if not obj_commit or not is_ancestor(obj_commit):
        obj_problems.append("object.commit missing or not an ancestor of HEAD")
    else:
        for item in evidence:
            try:
                if sha256_bytes(blob_at(obj_commit, item["path"])) != item["sha256"]:
                    obj_problems.append(f"{item['path']} differs from {obj_commit[:7]}")
            except Exception:  # noqa: BLE001
                obj_problems.append(f"{item['path']} not found at {obj_commit[:7]}")
    checks.append(check("OBJECT_PIN", not obj_problems,
                        f"all evidence hashes match the tree at object.commit "
                        f"{obj_commit[:7]}..." if not obj_problems else "; ".join(obj_problems)))

    # 12. sign-off absent
    signoff = ROOT / "trust/signoff.md"
    checks.append(check("SIGNOFF_ABSENT", not signoff.exists(),
                        "trust/signoff.md does not exist (an agent never signs)"
                        if not signoff.exists() else "trust/signoff.md exists"))

    # 13. path/email hygiene
    leaks = []
    for path in HYGIENE_FILES:
        p = ROOT / path
        if not p.is_file():
            continue
        text = p.read_text(errors="replace")
        for m in HOME_PATH_RE.finditer(text):
            leaks.append(f"{path}: local path {m.group(0)}")
        for m in EMAIL_RE.finditer(text):
            if not any(m.group(0).endswith(a) for a in EMAIL_ALLOWLIST):
                leaks.append(f"{path}: email {m.group(0)}")
    checks.append(check("PATH_HYGIENE", not leaks,
                        "no local machine paths or personal emails in the public record files"
                        if not leaks else "; ".join(leaks[:10])))

    overall = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    _write_report(vclaims, checks, overall)
    print(f"trust_check: {overall} ({sum(1 for c in checks if c['status'] == 'PASS')}"
          f"/{len(checks)} checks) -> {REPORT_PATH.relative_to(ROOT)}")
    return 0 if overall == "PASS" else 1


def _write_report(claims: list[dict], checks: list[dict], overall: str = "FAIL") -> None:
    report = {
        "tool": "trust_check",
        "version": TOOL_VERSION,
        "checked_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "root": ".",
        "checked_commit": head_commit(),
        "overall": overall,
        "claims": [{"id": c.get("id"), "status": c.get("status"),
                    "verification": c.get("verification"),
                    **({"control": c["control"]} if c.get("control") else {})}
                   for c in claims],
        "checks": checks,
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
