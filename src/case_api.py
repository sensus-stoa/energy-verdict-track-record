"""Case contract: what every pipeline case must hand to the verdict procedure.

The case is the "submitted pipeline" in a miniature engagement. It must:

1. build its own feature frame from the raw data (`build_features`),
2. fit on the train window only, evaluate on the test window,
3. report ONE number (the claimed metric) plus everything the audit battery
   needs to re-run it honestly (retrain_eval_fn, model, feature names, claim).

The verdict procedure trusts NOTHING the case says about data semantics:
feature availability comes from the audit-side config (ground truth of the
data generator), and what entered the fit is read from the fitted sklearn
model (model.feature_names_in_).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class Windows:
    train: tuple[str, str]
    test: tuple[str, str]
    confirmation: tuple[str, str]

    @staticmethod
    def select(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
        return df[(df.index >= window[0]) & (df.index < window[1])]


@dataclass(frozen=True)
class CaseClaim:
    """Pre-registered claim the case makes about its own performance."""

    statement: str
    max_mae_pct_of_mean: float                 # (a) MAE <= pct of mean(test-window load)
    must_beat: tuple[str, ...] = ("seasonal_naive", "persistence")


@dataclass
class CaseArtifact:
    case_id: str
    is_defective: bool
    defect_type: str | None
    claim: CaseClaim
    reported_mae: float
    predictions: pd.Series                     # index = test window dates
    windows: Windows
    model: object                              # fitted sklearn model
    feature_names_used: list[str]              # from model.feature_names_in_
    retrain_eval_fn: Callable[[pd.DataFrame, list[str], tuple[str, str], tuple[str, str]],
                              tuple[pd.Series, float]]
    trial_log: list[dict] = field(default_factory=list)  # configs tried on test window
    notes: str = ""


def selected(df: pd.DataFrame, window: tuple[str, str]) -> pd.DataFrame:
    return Windows.select(df, window)
