"""D6 - DEFECTIVE case: selection bias, best of K configurations on the test
window.

Twenty exchangeable linear configurations are tried on the TEST window and
the best one is reported. Each configuration is the calendar base
(doy_sin, doy_cos, dow_weekend) plus 20 pure-noise columns drawn with its
own seed (config k uses noise seed k; noise columns are namespaced
c<k>_n* so no two configurations share a noise column). None of the
configurations has real skill beyond the calendar terms, so the apparent
edge of the winner over the fixed unsearched calendar configuration is a
multiple-comparison artifact: it does not replicate on the held-out
confirmation window. The trial log is disclosed (the undisclosed-search
variant is out of scope for this record version, see limitations).
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
NOISE_COLS_PER_CONFIG = 20
NOISE_SCALE = 3000.0
N_CONFIGS = 20   # config k: calendar base + 20 noise columns, seed k

CLAIM = CaseClaim(
    statement=("Day-ahead daily load forecast: MAE <= 3.5% of mean test-window load, "
               "better than seasonal-naive and persistence baselines on the same window."),
    max_mae_pct_of_mean=3.5,
    must_beat=("seasonal_naive", "persistence"),
)


def build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Calendar base + every configuration's noise block in one frame.

    Config k (k = 1..N_CONFIGS) gets 20 noise columns named c<k>_n001..c<k>_n020,
    drawn from default_rng(k). Column names are unique per configuration, so the
    intersection of all disclosed trial feature sets is exactly the calendar
    base - the audit's fixed unsearched baseline.
    """
    base = add_case_features(data)
    blocks = []
    for k in range(1, N_CONFIGS + 1):
        rng = np.random.default_rng(k)   # deterministic, per-configuration seed
        blocks.append(pd.DataFrame(
            {f"c{k:02d}_n{j:03d}": rng.normal(0.0, NOISE_SCALE, len(base))
             for j in range(1, NOISE_COLS_PER_CONFIG + 1)},
            index=base.index,
        ))
    return pd.concat([base] + blocks, axis=1)


def config_features(k: int) -> list[str]:
    return BASE_FEATURES + [f"c{k:02d}_n{j:03d}"
                            for j in range(1, NOISE_COLS_PER_CONFIG + 1)]


def run(data: pd.DataFrame, windows: Windows) -> CaseArtifact:
    feat_df = build_features(data)
    trial_log = []
    best = None
    for k in range(1, N_CONFIGS + 1):
        feats = config_features(k)
        _, preds, m = fit_eval(feat_df, feats, windows.train, windows.test)
        entry = {"config_id": f"k{k:02d}", "features": feats, "test_mae": m}
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
        predictions=preds,   # predictions of the winning config; reported = best of 20
        windows=windows,
        model=None,          # no single "the" model: the artifact IS the trial log
        feature_names_used=best["features"],
        feature_frame_fn=build_features,
        retrain_eval_fn=retrain_eval,
        trial_log=trial_log,
        notes=f"best of K={N_CONFIGS} configurations selected on the test window",
    )
