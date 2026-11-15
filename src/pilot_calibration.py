"""Pilot calibration (pre-lock): run every registered case plus the NC-NULL
control and print verdicts, reported numbers and baselines. Used to check
that honest pipelines pass the frozen thresholds with margin and defective
ones are caught, BEFORE a new criteria lock.

Not a V-run.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case_api import Windows                                       # noqa: E402
from src.data_generator import generate_daily, windows                 # noqa: E402
from src.metrics import block_permute_target, persistence_mae, seasonal_naive_mae  # noqa: E402
from src.run_track_record import REGISTRY                              # noqa: E402
from src.verdict_procedure import AuditConfig, run_verdict             # noqa: E402

import importlib


def main() -> None:
    data = generate_daily()
    w = windows()
    win = Windows(train=w["train"], test=w["test"], confirmation=w["confirmation"])
    cfg = AuditConfig()

    y_test = data.loc[(data.index >= w["test"][0]) & (data.index < w["test"][1]), "load"]
    mean_load = float(y_test.mean())
    threshold = 0.035 * mean_load
    print(f"mean test-window load {mean_load:.0f} MW; claim threshold {threshold:.0f} MW; "
          f"persistence {persistence_mae(data, w['test']):.0f}; "
          f"seasonal-naive {seasonal_naive_mae(data, w['test']):.0f}")
    print("-" * 72)

    for case_id, (module_path, is_defective) in REGISTRY.items():
        mod = importlib.import_module(module_path)
        artifact = mod.run(data, win)
        v = run_verdict(data, artifact, cfg)
        expected = "FAIL" if is_defective else "PASS"
        flag = "OK " if v["status"] == expected else "MISMATCH"
        print(f"[{flag}] {case_id}: {v['status']}"
              f"{'/' + v['reason_code'] if v['reason_code'] else ''} "
              f"reported={artifact.reported_mae:.1f}")
        for a in v["audits"]:
            if a["status"] not in ("ok",):
                print(f"       -> {a['name']}: {a['status']}"
                      f"{'/' + a['reason_code'] if a['reason_code'] else ''}: {a['detail'][:110]}")

    print("-" * 72)
    clean_mod = importlib.import_module(REGISTRY["clean_01_linear_forecast_features"][0])
    fp = 0
    for seed in range(1, 6):
        dperm = block_permute_target(data, block=7, seed=seed)
        artifact = clean_mod.run(dperm, win)
        v = run_verdict(dperm, artifact, cfg)
        fp += 1 if v["status"] == "PASS" else 0
        print(f"[NC-NULL seed={seed}] {v['status']} mae_on_null={artifact.reported_mae:.1f}")
    print(f"NC-NULL false PASS: {fp}/5 (N=5)")


if __name__ == "__main__":
    main()
