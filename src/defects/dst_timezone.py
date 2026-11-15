"""D2 - DEFECTIVE case: DST / timezone handling bug (hourly data).

Raw data is hourly with a naive UTC index (semantics: UTC). The pipeline
converts to naive Berlin local wall time and computes "same hour yesterday"
by grouping on the wall-clock hour. Across the October fall-back transition
this (a) duplicates a wall-clock hour in the working frame and (b) breaks
the lag alignment (the duplicated hour and the following day get lag values
that are not "yesterday, same hour"). The reported MAE barely moves, so the
defect is invisible in the pipeline's own report - exactly the class of bug
the timezone-integrity audit must catch.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.metrics import mae

CASE_ID = "defect_02_dst_timezone"
IS_DEFECTIVE = True
DEFECT_TYPE = "dst_timezone_bug"

FEATURES = ["temp_forecast", "hour_sin", "hour_cos", "dow_weekend", "load_lag24"]

# the case computed its own windows (UTC label boundaries)
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
    # THE BUG: naive local wall time as the working index
    local = df.index.tz_localize("UTC").tz_convert("Europe/Berlin")
    df.index = local.tz_localize(None)
    hod = df.index.hour.to_numpy(dtype=float)
    dow = df.index.dayofweek.to_numpy()
    df["hour_sin"] = np.sin(2.0 * np.pi * hod / 24.0)
    df["hour_cos"] = np.cos(2.0 * np.pi * hod / 24.0)
    df["dow_weekend"] = (dow >= 5).astype(float)
    # THE BUG: wall-clock lag instead of a positional 24-row lag
    df["load_lag24"] = df.groupby(df.index.hour)["load"].shift(1)
    return df


def _positions(idx: pd.DatetimeIndex, window: tuple[str, str]) -> np.ndarray:
    return np.where((idx >= window[0]) & (idx < window[1]))[0]


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    pos_tr = _positions(data.index, TRAIN_W)
    pos_te = _positions(data.index, TEST_W)
    tr = feat_df.iloc[pos_tr].dropna(subset=FEATURES + ["load"])
    ev = feat_df.iloc[pos_te]
    model = LinearRegression().fit(tr[FEATURES], tr["load"])
    preds = pd.Series(model.predict(ev[FEATURES]), index=data.index[pos_te], name="pred")
    m = mae(ev["load"], preds)

    def retrain_eval(d, features, train_w, eval_w):
        fdf = build_features(d)
        p_tr = _positions(d.index, train_w)
        p_ev = _positions(d.index, eval_w)
        trr = fdf.iloc[p_tr].dropna(subset=list(features) + ["load"])
        evv = fdf.iloc[p_ev]
        mm = LinearRegression().fit(trr[features], trr["load"])
        pp = pd.Series(mm.predict(evv[features]), index=d.index[p_ev])
        return pp, mae(evv["load"], pp)

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
        notes="working frame on naive Berlin wall time; lags grouped by wall-clock hour",
    )
