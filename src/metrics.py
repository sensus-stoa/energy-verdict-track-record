"""Metrics and statistical helpers for the verdict procedure."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def persistence_mae(df: pd.DataFrame, window: tuple[str, str], target: str = "load") -> float:
    """Baseline 'tomorrow = today': y(t-1) evaluated on the window."""
    y = df[target]
    pred = y.shift(1)
    mask = (df.index >= window[0]) & (df.index < window[1]) & pred.notna()
    return float(np.mean(np.abs(y[mask] - pred[mask])))


def seasonal_naive_mae(df: pd.DataFrame, window: tuple[str, str], target: str = "load") -> float:
    """Baseline 'tomorrow = same weekday last week': y(t-7)."""
    y = df[target]
    pred = y.shift(7)
    mask = (df.index >= window[0]) & (df.index < window[1]) & pred.notna()
    return float(np.mean(np.abs(y[mask] - pred[mask])))


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval, 95% by default. Returns (lower, upper)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def rule_of_three_upper(n: int) -> float:
    """95% upper bound on a rate observed as 0/n: approx 3/n (rule of three)."""
    if n == 0:
        return 1.0
    return 3.0 / n


def block_permute_target(df: pd.DataFrame, block: int = 7, seed: int = 0,
                         target: str = "load") -> pd.DataFrame:
    """Null surrogate: shuffle contiguous `block`-day blocks of the target column.

    Destroys the alignment between target and all exogenous/calendar features
    beyond the block scale, while preserving the within-block shape and the
    marginal distribution of the target. This is NOT a naive point shuffle of
    labels (invalid on trended data per V-PTM-001 claim C-04); it is a block
    surrogate used ONLY as a negative control on deliberately destroyed data,
    never as a significance test on real data.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    y = out[target].to_numpy()
    n_blocks = len(y) // block
    blocks = [y[i * block:(i + 1) * block] for i in range(n_blocks)]
    order = rng.permutation(n_blocks)
    permuted = np.concatenate([blocks[i] for i in order])
    tail = len(y) - n_blocks * block
    if tail:
        permuted = np.concatenate([permuted, y[n_blocks * block:]])
    out[target] = permuted
    return out
