"""Shared linear-pipeline machinery for clean and defective cases."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.metrics import mae


def add_case_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar + honest lag features shared by all linear cases."""
    out = df.copy()
    doy = out.index.dayofyear.to_numpy(dtype=float)
    out["doy_sin"] = np.sin(2.0 * np.pi * doy / 365.25)
    out["doy_cos"] = np.cos(2.0 * np.pi * doy / 365.25)
    out["dow_weekend"] = (out.index.dayofweek >= 5).astype(float)
    out["load_lag1"] = out["load"].shift(1)
    out["load_lag7"] = out["load"].shift(7)
    return out


def fit_eval(feat_df: pd.DataFrame, features: list[str],
             train_w: tuple[str, str], eval_w: tuple[str, str],
             target: str = "load", model=None):
    tr = feat_df[(feat_df.index >= train_w[0]) & (feat_df.index < train_w[1])]
    tr = tr.dropna(subset=list(features) + [target])   # lag features are NaN at series head
    ev = feat_df[(feat_df.index >= eval_w[0]) & (feat_df.index < eval_w[1])]
    if model is None:
        model = LinearRegression()
    model.fit(tr[features], tr[target])
    preds = pd.Series(model.predict(ev[features]), index=ev.index, name="pred")
    m = mae(ev[target], preds)
    return model, preds, m
