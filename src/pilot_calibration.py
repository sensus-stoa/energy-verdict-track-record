"""Pilot calibration (pre-lock): measure what honest and defective pipelines
achieve on the current synthetic world, BEFORE the criteria in
trust/claims.md are frozen. Used to set claim thresholds that honest
pipelines pass with margin and the null control does not pass.

Not a V-run. Results of this file are not evidence; the frozen-criteria run
is `python -m src.run_track_record --cases all --with-null-control`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.case_api import Windows                                       # noqa: E402
from src.clean_cases.clean_pipeline_01 import run as clean_run         # noqa: E402
from src.data_generator import generate_daily, windows                 # noqa: E402
from src.defects.look_ahead_leak import run as leak_run                # noqa: E402
from src.metrics import (mae, persistence_mae, seasonal_naive_mae,     # noqa: E402
                         wilson_ci)
from src.verdict_procedure import AuditConfig, run_verdict             # noqa: E402


def main() -> None:
    data = generate_daily()
    w = windows()
    win = Windows(train=w["train"], test=w["test"], confirmation=w["confirmation"])
    cfg = AuditConfig()

    y_test = data.loc[(data.index >= w["test"][0]) & (data.index < w["test"][1]), "load"]
    mean_load = float(y_test.mean())
    threshold = 0.025 * mean_load
    mae_p = persistence_mae(data, w["test"])
    mae_s = seasonal_naive_mae(data, w["test"])

    clean = clean_run(data, win)
    leak = leak_run(data, win)
    _, mae_deploy = leak.retrain_eval_fn(
        data, [f for f in leak.feature_names_used if f != "temp_actual"],
        win.train, win.test)

    print(f"mean test-window load:      {mean_load:8.0f} MW")
    print(f"claim threshold (2.5%):     {threshold:8.0f} MW")
    print(f"persistence MAE:            {mae_p:8.0f} MW")
    print(f"seasonal-naive MAE:         {mae_s:8.0f} MW")
    print("-" * 52)
    print(f"clean reported MAE:         {clean.reported_mae:8.0f} MW"
          f"  ({clean.reported_mae / mean_load:.2%} of mean)")
    print(f"leak reported MAE:          {leak.reported_mae:8.0f} MW"
          f"  ({leak.reported_mae / mean_load:.2%} of mean)")
    print(f"leak deployment MAE:        {mae_deploy:8.0f} MW"
          f"  (re-run without temp_actual)")
    gap = (mae_deploy - leak.reported_mae) / leak.reported_mae
    print(f"look-ahead gap:             {gap:8.1%}  (audit threshold 20%)")
    print("-" * 52)
    print(f"clean beats seasonal-naive: {clean.reported_mae < mae_s}"
          f"  (margin {mae_s - clean.reported_mae:.0f} MW)")
    print(f"clean beats persistence:    {clean.reported_mae < mae_p}"
          f"  (margin {mae_p - clean.reported_mae:.0f} MW)")
    print(f"leak beats naive (looks good): {leak.reported_mae < mae_s}")
    print("-" * 52)
    print("verdict clean:", run_verdict(data, clean, cfg)["status"])
    print("verdict leak: ", run_verdict(data, leak, cfg)["status"])

    lo, hi = wilson_ci(0, 5)
    print(f"reference: Wilson 95% CI for 0/5: [{lo:.3f}, {hi:.3f}] "
          f"(rule of three upper ~{3 / 5:.0%})")


if __name__ == "__main__":
    main()
