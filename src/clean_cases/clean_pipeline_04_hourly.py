"""C4 - CLEAN case: hourly pipeline, timezone handled correctly.

Twin of defect_02 on the same hourly data: the working frame keeps the raw
UTC index; local-hour features are computed through a proper tz-aware
conversion as COLUMNS; "same hour yesterday" is a positional 24-row lag,
which is exact on a regular UTC grid. Proves the timezone-integrity audit
does not accuse correct hourly pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import fit_eval
from src.metrics import mae

CASE_ID = "clean_04_hourly_tz_aware"
IS_DEFECTIVE = False
DEFECT_TYPE = None

FEATURES = ["temp_forecast", "hour_sin", "hour_cos", "dow_weekend", "load_lag24"]

TRAIN_W = ("2023-01-01", "2023-10-25")
TEST_W = ("2023-10-25", "2023-11-01")
CONF_W = ("2023-11-01", "2023-12-01")

CLAIM = CaseClaim(
    statement=("Hour-ahead (1..24h) city load forecast: MAE <= 6.0% of mean "
               "test-window load, better than hourly persistence (24h) and "
               "seasonal-naive (168h) on the same window."),
    max_mae_pct_of_mean=6.0,
    must_beat=("seasonal_naive", "persistence"),
)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    local = df.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    hod = local.hour.to_numpy(dtype=float)
    dow = local.dayofweek.to_numpy()
    df["hour_sin"] = np.sin(2.0 * np.pi * hod / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hod / 24.0)
    df["dow_weekend"] = (dow >= 5).astype(float)
    df["load_lag24"] = df["load"].shift(24)   # positional: exact on a regular UTC grid
    return df


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    model, preds, m = fit_eval(feat_df, FEATURES, TRAIN_W, TEST_W)

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
        windows=Windows(train=TRAIN_W, test=TEST_W, confirmation=CONF_W),
        model=model,
        feature_names_used=list(model.feature_names_in_),
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "default", "features": list(FEATURES), "test_mae": m}],
        notes="single pre-registered configuration (K=1), UTC index preserved",
    )
