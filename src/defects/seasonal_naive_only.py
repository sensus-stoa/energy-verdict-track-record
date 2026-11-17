"""D7 - DEFECTIVE case: the seasonal-naive baseline shipped as a forecast.

Hourly pipeline whose entire "model" is the 168-hour lag ("same hour last
week"): prediction = load.shift(168), disclosed as such. The claimed skill
is exactly the cadence seasonal-naive baseline that the claim must beat.

Demonstrates the division of labour between the two hourly claim conditions
(see claims C-09/C-13): the 6.0%-of-mean-load threshold is an UPPER BOUND on
absolute error, not a skill criterion - the seasonal-naive baseline itself
satisfies it (MAE about 5.5% of mean load, below 6.0%). The FAIL comes from
the must_beat condition of the claim-conditions audit: a prediction that IS
the seasonal-naive baseline is not better than seasonal-naive. A threshold
that the trivial baseline passes cannot certify skill on its own.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.metrics import mae

CASE_ID = "defect_07_seasonal_naive_only"
IS_DEFECTIVE = True
DEFECT_TYPE = "baseline_as_forecast"

FEATURES = ["load_lag168"]
LAG = 168

# the case computed its own windows (UTC label boundaries, same as clean_04)
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
    """Clean working frame (UTC index preserved, no wall-clock tricks)."""
    df = data.copy()
    df["load_lag168"] = df["load"].shift(LAG)
    return df


def _positions(idx: pd.DatetimeIndex, window: tuple[str, str]) -> np.ndarray:
    return np.where((idx >= window[0]) & (idx < window[1]))[0]


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    pos_te = _positions(data.index, TEST_W)
    preds = feat_df["load_lag168"].iloc[pos_te]
    y = feat_df["load"].iloc[pos_te]
    reported = mae(y, preds)   # = the seasonal-naive baseline MAE, by construction

    def retrain_eval(d, features, train_w, eval_w):
        # the "refit" of this model is the identity: the model IS the lag
        fdf = build_features(d)
        p_ev = _positions(d.index, eval_w)
        pp = fdf["load_lag168"].iloc[p_ev]
        return pp, mae(fdf["load"].iloc[p_ev], pp)

    return CaseArtifact(
        case_id=CASE_ID,
        is_defective=IS_DEFECTIVE,
        defect_type=DEFECT_TYPE,
        claim=CLAIM,
        reported_mae=reported,
        predictions=preds,
        windows=Windows(train=TRAIN_W, test=TEST_W, confirmation=CONF_W),
        model=None,   # no fitted model: the artifact IS the 168h lag
        feature_names_used=list(FEATURES),
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=[{"config_id": "seasonal_naive_168", "features": list(FEATURES),
                    "test_mae": reported}],
        notes="prediction = 168h lag disclosed as the model; passes the 6.0% "
              "absolute gate, fails must-beat seasonal-naive",
    )
