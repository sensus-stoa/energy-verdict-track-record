"""D6 - DEFECTIVE case: selection bias, best of K configurations on the test
window.

Twenty linear configurations (calendar features + m pure-noise columns,
m = 10..200) are tried on the TEST window and the best one is reported.
None of the configurations has real skill beyond the calendar terms, so the
apparent edge of the winner is a multiple-comparison artifact: it does not
replicate on the held-out confirmation window. The trial log is disclosed
(the undisclosed-search variant is out of scope for this record version,
see limitations).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.case_api import CaseArtifact, CaseClaim, Windows
from src.linear_case import add_case_features, fit_eval
from src.metrics import mae

CASE_ID = "defect_06_selection_bias"
IS_DEFECTIVE = True
DEFECT_TYPE = "selection_bias"

BASE_FEATURES = ["doy_sin", "doy_cos", "dow_weekend"]
N_NOISE_COLS = 200
NOISE_SCALE = 3000.0
N_CONFIGS = 20   # m = 10, 20, ..., 200 noise columns

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    base = add_case_features(data)
    rng = np.random.default_rng(777)   # deterministic noise columns
    noise = pd.DataFrame(
        {f"n{i:03d}": rng.normal(0.0, NOISE_SCALE, len(base))
         for i in range(1, N_NOISE_COLS + 1)},
        index=base.index,
    )
    return pd.concat([base, noise], axis=1)


def config_features(m: int) -> list[str]:
    return BASE_FEATURES + [f"n{i:03d}" for i in range(1, m + 1)]


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    trial_log = []
    best = None
    for k in range(1, N_CONFIGS + 1):
        feats = config_features(10 * k)
        _, preds, m = fit_eval(feat_df, feats, windows.train, windows.test)
        entry = {"config_id": f"m{10 * k:03d}", "features": feats, "test_mae": m}
        trial_log.append(entry)
        if best is None or m < best["test_mae"]:
            best = entry
    reported = best["test_mae"]
    # export the WINNING configuration's artifacts (realistic: the vendor
    # ships the selected model), so the reported number reproduces from the
    # returned predictions - the defect is the selection itself
    _, preds, _ = fit_eval(feat_df, best["features"], windows.train, windows.test)

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
        predictions=preds,   # predictions of the LAST fitted config; reported number = best of 20
        windows=windows,
        model=None,          # no single "the" model: the artifact IS the trial log
        feature_names_used=best["features"],
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=trial_log,
        notes=f"best of K={N_CONFIGS} configurations selected on the test window",
    )
