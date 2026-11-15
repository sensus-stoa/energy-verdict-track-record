"""D5 - DEFECTIVE case: cherry-picked metric via silent trimming.

The pipeline is an ordinary clean linear model; the DEFECT is in what it
reports: the claimed MAE is computed after silently dropping the 10
highest-error days of the 30-day test window ("anomalous days removed"
without disclosure). The returned predictions cover the full window, so the
reported number must be reproducible from them on the full pre-registered
window - it is not. Mirrors the brief's failure class "reported evidence
contradicts recomputed evidence" (CONFLICT_EVIDENCE).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval
from src.metrics import mae

CASE_ID = "defect_05_cherry_picked_metric"
IS_DEFECTIVE = True
DEFECT_TYPE = "cherry_picking_metric"

FEATURES = ["temp_forecast", "doy_sin", "doy_cos", "dow_weekend", "load_lag1"]

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)

TRIM_KEEP = 20   # report MAE on the 20 lowest-error days of the 30-day window


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return add_case_features(data)


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    model, preds, _ = fit_eval(feat_df, FEATURES, windows.train, windows.test)

    # THE DEFECT: reported number computed on a favorable subset, undisclosed
    y_test = feat_df.loc[(feat_df.index >= windows.test[0]) & (feat_df.index < windows.test[1]), "load"]
    err = (y_test - preds).abs()
    keep = err.nsmallest(TRIM_KEEP).index
    reported = mae(y_test[keep], preds[keep])

    def retrain_eval(d, features, train_w, eval_w):
        fdf = build_features(d)
        _, p, m_ = fit_eval(fdf, features, train_w, eval_w)
        return p, m_

    return CaseArtifact(
        case_id=CASE_ID,
        is_defective=IS_DEFECTIVE,
        defect_type=DEFECT_TYPE,
        claim=CLAIM,
        reported_mae=reported,
        predictions=preds,
        windows=windows,
        model=model,
        feature_names_used=list(model.feature_names_in_),
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "default", "features": list(FEATURES), "test_mae": reported}],
        notes=f"reported MAE computed on the {TRIM_KEEP} lowest-error days of the "
              f"30-day window (silent trim)",
    )
