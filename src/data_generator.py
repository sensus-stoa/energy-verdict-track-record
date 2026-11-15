"""Synthetic energy data generator (deterministic, fixed seed).

Generates data/synthetic/energy_daily.csv:

    date            DatetimeIndex, daily, 2023-01-01 + N_DAYS days
    temp_actual     city-effective temperature, deg C (only known AFTER the day)
    temp_forecast   day-ahead temperature forecast, deg C (available at 00:00 of the day)
    load            daily electricity load, MW

Windows (fixed by design, see README):
    train         [0, 335)   2023-01-01 .. 2023-12-01
    test          [335, 365) 2023-12-02 .. 2024-01-01  (primary evaluation window)
    confirmation  [365, 425) 2024-01-02 .. 2024-03-01  (held out; used only by the
                  selection-bias replication audit, never by pipelines)

Generator semantics are the AUDIT-SIDE ground truth for feature availability:
temp_actual is NOT available at inference time (it materialises at the end of
the day); temp_forecast IS available at 00:00 of the target day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_DAYS = 425
TRAIN_DAYS = 335
TEST_DAYS = 30

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "synthetic"
DEFAULT_CSV = DATA_DIR / "energy_daily.csv"


def _ar1(rng: np.random.Generator, n: int, rho: float, sigma_innov: float) -> np.ndarray:
    x = np.zeros(n)
    x[0] = rng.normal(0.0, sigma_innov)
    innov = rng.normal(0.0, sigma_innov, n)
    for t in range(1, n):
        x[t] = rho * x[t - 1] + innov[t]
    return x


def generate_daily(seed: int = SEED, n_days: int = N_DAYS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.to_numpy(dtype=float)
    dow = dates.dayofweek.to_numpy()

    # Temperature, Berlin-like: annual cycle + persistent weather deviations.
    annual = 10.0 - 10.0 * np.cos(2.0 * np.pi * (doy - 15.0) / 365.25)
    temp_actual = annual + _ar1(rng, n_days, rho=0.85, sigma_innov=3.0)

    # Day-ahead forecast error: slowly drifting bias (AR 0.5) + iid noise.
    # Stationary std ~1.9 deg C - realistic for a 24h-ahead city forecast.
    fc_err = _ar1(rng, n_days, rho=0.5, sigma_innov=1.5) + rng.normal(0.0, 0.7, n_days)
    temp_forecast = temp_actual + fc_err

    # Load (MW): temperature response + weekly profile + persistent residual.
    hdd = 700.0 * np.maximum(0.0, 16.0 - temp_actual)
    cdd = 400.0 * np.maximum(0.0, temp_actual - 22.0)
    weekend = np.where(dow >= 5, -4000.0, 0.0)
    load_ar = _ar1(rng, n_days, rho=0.3, sigma_innov=700.0)
    load = 45000.0 + hdd + cdd + weekend + load_ar

    return pd.DataFrame(
        {"temp_actual": temp_actual, "temp_forecast": temp_forecast, "load": load},
        index=dates,
    )


def load_or_generate(path: Path = DEFAULT_CSV) -> pd.DataFrame:
    """Read the committed CSV; regenerate byte-identically if missing."""
    if path.exists():
        df = pd.read_csv(path, parse_dates=["date"], index_col="date")
        return df
    df = generate_daily()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path)
    return df


def windows() -> dict:
    """Fixed evaluation windows (see module docstring)."""
    return {
        "train": ("2023-01-01", "2023-12-02"),   # [start, end)
        "test": ("2023-12-02", "2024-01-02"),
        "confirmation": ("2024-01-02", "2024-03-02"),
    }


if __name__ == "__main__":
    df = generate_daily()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DEFAULT_CSV)
    print(f"written {DEFAULT_CSV} rows={len(df)}")
    print(df.head(3))
    print(df.tail(3))
