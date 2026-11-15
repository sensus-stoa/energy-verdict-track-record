"""D4 - DEFECTIVE case: target leakage via a near-copy feature.

The feature load_settlement is the same-day realised load plus small
reporting noise (a realistic dataset-join accident: settlement actuals
merged into the forecast feature table). At inference time this feature
does not exist - the target has not happened yet. The model's reported MAE
collapses to the noise level of the reporting error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval

CASE_ID = "defect_04_target_leakage"
IS_DEFECTIVE = True
DEFECT_TYPE = "target_leakage"

FEATURES = ["temp_forecast", "doy_sin", "doy_cos", "dow_weekend", "load_lag1",
            "load_settlement"]

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = add_case_features(data)
    rng = np.random.default_rng(12345)   # deterministic reporting noise
    df["load_settlement"] = df["load"] + rng.normal(0.0, 150.0, len(df))
    return df


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    model, preds, m = fit_eval(feat_df, FEATURES, windows.train, windows.test)

    def retrain_eval(d, features, train_w, eval_w):
        fdf = build_features(d)
        _, p, m_ = fit_eval(fdf, features, train_w, eval_w)
        return p, m_

    return CaseArtifact(
        case_id=CASE_ID,
        is_defective=IS_DEFECTIVE,
        defect_type=DEFECT_TYPE,
        claim=CLAIM,
        reported_mae=m,
        predictions=preds,
        windows=windows,
        model=model,
        feature_names_used=list(model.feature_names_in_),
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "default", "features": list(FEATURES), "test_mae": m}],
        notes="feature load_settlement = same-day realised load + reporting noise",
    )
