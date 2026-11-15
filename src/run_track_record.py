"""Run the verifier track record: submitted cases -> verdict procedure -> results.

Usage (from repo root):
    evtr-venv/bin/python -m src.run_track_record --cases all
    evtr-venv/bin/python -m src.run_track_record --case defect_01_look_ahead
    evtr-venv/bin/python -m src.run_track_record --cases all --with-null-control
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.case_api import Windows                                  # noqa: E402
from src.data_generator import load_or_generate, windows          # noqa: E402
from src.metrics import block_permute_target, rule_of_three_upper, wilson_ci  # noqa: E402
from src.verdict_procedure import AuditConfig, run_verdict        # noqa: E402

CASES_DIR = ROOT / "src"
RESULTS_DIR = ROOT / "results"
PER_CASE_DIR = RESULTS_DIR / "per_case"

REGISTRY = {
    "defect_01_look_ahead": ("src.defects.look_ahead_leak", True),
    "clean_01_linear_forecast_features": ("src.clean_cases.clean_pipeline_01", False),
}

NULL_CONTROL_ID = "NC-NULL"
NULL_CONTROL_N = 5


def run_case(case_id: str, data, win: Windows, cfg: AuditConfig) -> dict:
    module_path, is_defective = REGISTRY[case_id]
    mod = importlib.import_module(module_path)
    artifact = mod.run(data, win)
    verdict = run_verdict(data, artifact, cfg)
    false_pass = 1 if (is_defective and verdict["status"] == "PASS") else 0
    false_fail = 1 if (not is_defective and verdict["status"] != "PASS") else 0
    notes = artifact.notes or "; ".join(
        a["detail"] for a in verdict["audits"] if a["status"] != "ok" and a["detail"]
    )
    return {
        "case_id": case_id,
        "is_defective": is_defective,
        "verdict_status": verdict["status"],
        "reason_code": verdict["reason_code"],
        "false_pass": false_pass,
        "false_fail": false_fail,
        "notes": notes,
        "_verdict": verdict,
        "_reported_mae": artifact.reported_mae,
        "_claim": artifact.claim,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = ["case_id", "is_defective", "verdict_status", "reason_code",
              "false_pass", "false_fail", "notes"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="all", help="comma-separated case ids or 'all'")
    ap.add_argument("--case", action="append", default=[], help="single case id (repeatable)")
    ap.add_argument("--with-null-control", action="store_true",
                    help=f"also run {NULL_CONTROL_ID}: N={NULL_CONTROL_N} block-permuted "
                         f"target runs of the clean pipeline; procedure must not PASS")
    args = ap.parse_args()

    if args.cases == "all":
        ids = list(REGISTRY)
    else:
        ids = [c.strip() for c in args.cases.split(",") if c.strip()]
    ids += [c for c in args.case if c not in ids]
    unknown = [c for c in ids if c not in REGISTRY]
    if unknown:
        print(f"unknown cases: {unknown}; known: {list(REGISTRY)}")
        return 2

    data = load_or_generate()
    w = windows()
    win = Windows(train=w["train"], test=w["test"], confirmation=w["confirmation"])
    cfg = AuditConfig()

    rows = []
    for case_id in ids:
        r = run_case(case_id, data, win, cfg)
        rows.append(r)
        PER_CASE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "case_id": r["case_id"],
            "is_defective": r["is_defective"],
            "reported_mae": round(r["_reported_mae"], 1),
            "claim": {"statement": r["_claim"].statement,
                      "max_mae_pct_of_mean": r["_claim"].max_mae_pct_of_mean,
                      "must_beat": list(r["_claim"].must_beat)},
            "verdict": r["_verdict"],
        }
        (PER_CASE_DIR / f"{case_id}.json").write_text(json.dumps(payload, indent=2) + "\n")
        print(f"[{r['case_id']}] {r['verdict_status']}"
              f"{'/' + r['reason_code'] if r['reason_code'] else ''}"
              f" reported_mae={r['_reported_mae']:.1f}")

    write_csv(RESULTS_DIR / "track_record.csv", rows)

    if args.with_null_control:
        clean_mod = importlib.import_module(REGISTRY["clean_01_linear_forecast_features"][0])
        nc_rows = []
        for seed in range(1, NULL_CONTROL_N + 1):
            dperm = block_permute_target(data, block=7, seed=seed)
            artifact = clean_mod.run(dperm, win)
            v = run_verdict(dperm, artifact, cfg)
            false_pass = 1 if v["status"] == "PASS" else 0
            nc_rows.append({"run": seed, "block_days": 7, "verdict_status": v["status"],
                            "reason_code": v["reason_code"], "false_pass": false_pass,
                            "reported_mae_on_null_data": round(artifact.reported_mae, 1)})
            print(f"[NC-NULL seed={seed}] {v['status']}"
                  f"{'/' + v['reason_code'] if v['reason_code'] else ''}"
                  f" mae_on_null={artifact.reported_mae:.1f}")
        with (RESULTS_DIR / "null_control.csv").open("w", newline="") as f:
            wtr = csv.DictWriter(f, fieldnames=list(nc_rows[0]))
            wtr.writeheader()
            wtr.writerows(nc_rows)
        fp = sum(r["false_pass"] for r in nc_rows)
        n = len(nc_rows)
        lo, hi = wilson_ci(fp, n)
        print(f"[NC-NULL] false PASS: {fp}/{n} (N={n}); "
              f"Wilson 95% CI [{lo:.3f}, {hi:.3f}]; "
              f"rule-of-three upper bound at 0/{n}: ~{rule_of_three_upper(n):.0%}")

    # summary (every number carries its N)
    defective = [r for r in rows if r["is_defective"]]
    clean = [r for r in rows if not r["is_defective"]]
    sens = sum(1 for r in defective if r["verdict_status"] == "FAIL")
    spec = sum(1 for r in clean if r["verdict_status"] == "PASS")
    print("\n== Verifier Track Record Summary ==")
    print(f"sensitivity (defective detected): {sens}/{len(defective)} (N={len(defective)})")
    print(f"specificity (clean passed):       {spec}/{len(clean)} (N={len(clean)})")
    if defective:
        lo, hi = wilson_ci(sens, len(defective))
        print(f"  sensitivity Wilson 95% CI: [{lo:.3f}, {hi:.3f}] (N={len(defective)})")
    if clean:
        lo, hi = wilson_ci(spec, len(clean))
        print(f"  specificity Wilson 95% CI: [{lo:.3f}, {hi:.3f}] (N={len(clean)})")
    fp = sum(r["false_pass"] for r in defective)
    print(f"false PASS on defective: {fp}/{len(defective)} (N={len(defective)})")
    if fp == 0 and defective:
        print(f"  rule-of-three upper bound at 0/{len(defective)}: "
              f"~{rule_of_three_upper(len(defective)):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
