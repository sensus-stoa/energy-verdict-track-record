"""D1 — DEFECTIVE case: look-ahead leak via actual weather in backtest features.

The pipeline trains and evaluates with temp_actual (realised weather) among
its features. At inference time — day-ahead — only temp_forecast exists.
The reported MAE is therefore not attainable in deployment: the model
involuntarily looks into the future. This mirrors the failure mode described
in the service brief, section 2.

Everything else (claim, model family, calendar/lag features) is identical to
clean_01_linear_forecast_features, so the ONLY difference is the leak.
"""

from __future__ import annotations

import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval

CASE_ID = "defect_01_look_ahead"
IS_DEFECTIVE = True
DEFECT_TYPE = "look_ahead_leak"

FEATURES = ["temp_actual", "temp_forecast", "doy_sin", "doy_cos",
            "dow_weekend", "load_lag1"]

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return add_case_features(data)


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    # THE DEFECT: backtest (train AND test) features include temp_actual.
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
        notes="backtest features include realised weather (temp_actual)",
    )
