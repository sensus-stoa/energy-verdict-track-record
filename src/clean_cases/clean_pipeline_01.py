"""C1 — CLEAN linear pipeline: deployment-consistent day-ahead load forecast.

Features use ONLY information available at 00:00 of the target day:
day-ahead temperature forecast, calendar, yesterday's/last-week's actual load.
"""

from __future__ import annotations

import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval

CASE_ID = "clean_01_linear_forecast_features"
IS_DEFECTIVE = False
DEFECT_TYPE = None

FEATURES = ["temp_forecast", "doy_sin", "doy_cos", "dow_weekend", "load_lag1"]

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
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "default", "features": list(FEATURES), "test_mae": m}],
        notes="single pre-registered configuration (K=1)",
    )
