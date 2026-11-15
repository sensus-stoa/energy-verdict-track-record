"""C2 - CLEAN case: regularised linear pipeline (scaled Ridge).

Different model family from C1 (standardised features, L2 penalty); still
deployment-consistent: every feature is available at 00:00 of the target day.
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval

CASE_ID = "clean_02_ridge_scaled"
IS_DEFECTIVE = False
DEFECT_TYPE = None

FEATURES = ["temp_forecast", "doy_sin", "doy_cos", "dow_weekend", "load_lag1"]

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)


def make_model():
    return Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return add_case_features(data)


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    model, preds, m = fit_eval(feat_df, FEATURES, windows.train, windows.test,
                               model=make_model())

    def retrain_eval(d, features, train_w, eval_w):
        fdf = build_features(d)
        _, p, m_ = fit_eval(fdf, features, train_w, eval_w, model=make_model())
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
        trial_log=[{"config_id": "alpha=1.0", "features": list(FEATURES), "test_mae": m}],
        notes="single pre-registered configuration (K=1), scaled Ridge",
    )
