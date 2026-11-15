"""D3 - DEFECTIVE case: backtest contamination via train/test overlap.

The pipeline computed its own split and extended the train window 15 days
into the test window (a realistic off-by-N accident in index arithmetic).
Test rows seen during training make the reported MAE unattainably good.
Mirrors the brief, section 4 item 3 ("не проверяется ли модель на данных,
на которых она училась").
"""

from __future__ import annotations

import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval

CASE_ID = "defect_03_backtest_contamination"
IS_DEFECTIVE = True
DEFECT_TYPE = "backtest_contamination"

FEATURES = ["temp_forecast", "doy_sin", "doy_cos", "dow_weekend", "load_lag1"]

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)

# THE DEFECT: train window extends into the test window
# (test starts 2023-12-02; train runs to 2023-12-17, 15 days of overlap).
CONTAMINATED_TRAIN = ("2023-01-01", "2023-12-17")


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    return add_case_features(data)


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    model, preds, m = fit_eval(feat_df, FEATURES, CONTAMINATED_TRAIN, windows.test)

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
        windows=Windows(train=CONTAMINATED_TRAIN, test=windows.test,
                        confirmation=windows.confirmation),
        model=model,
        feature_names_used=list(model.feature_names_in_),
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "default", "features": list(FEATURES), "test_mae": m}],
        notes="train window overlaps the test window by 15 days (self-computed split)",
    )
