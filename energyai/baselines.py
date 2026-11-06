"""Transparent non-ML baselines and common evaluation metrics."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import pandas as pd

from energyai.features import PARIS, forecast_issue_time


@dataclass(frozen=True)
class MetricSet:
    mae_mw: float
    rmse_mw: float
    wape_percent: float
    observations: int


def calculate_metrics(actual: pd.Series, predicted: pd.Series) -> MetricSet:
    """Calculate MAE, RMSE, and WAPE for aligned prediction series."""
    error = predicted - actual
    denominator = actual.abs().sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined because total absolute actual load is zero")
    return MetricSet(
        mae_mw=float(error.abs().mean()),
        rmse_mw=float(np.sqrt(np.mean(np.square(error)))),
        wape_percent=float(100 * error.abs().sum() / denominator),
        observations=len(actual),
    )


def evaluate_naive_baselines(
    hourly: pd.DataFrame, *, target_local_year: int
) -> tuple[pd.DataFrame, dict[str, MetricSet]]:
    """Evaluate 48h and 168h UTC-lag baselines on identical valid timestamps."""
    required = {"timestamp_utc", "load_mw"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Hourly data is missing columns: {sorted(missing)}")
    frame = hourly[list(required)].copy()
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = (
        frame.drop_duplicates("timestamp_utc").set_index("timestamp_utc").sort_index()
    )
    series = pd.to_numeric(frame["load_mw"], errors="coerce")
    index = series.index
    local_index = index.tz_convert(PARIS)
    local_year = local_index.year
    issued_at = pd.DatetimeIndex(
        [forecast_issue_time(target_date) for target_date in local_index.date]
    )
    predictions = pd.DataFrame(
        {
            "timestamp_utc": index,
            "issued_at_utc": issued_at,
            "actual_mw": series.to_numpy(),
            "naive_48h_mw": series.reindex(index - timedelta(hours=48)).to_numpy(),
            "seasonal_naive_168h_mw": series.reindex(
                index - timedelta(hours=168)
            ).to_numpy(),
        }
    )
    predictions = predictions.loc[local_year == target_local_year].copy()
    predictions["timestamp_local"] = predictions["timestamp_utc"].dt.tz_convert(PARIS)
    common = predictions.dropna(
        subset=["actual_mw", "naive_48h_mw", "seasonal_naive_168h_mw"]
    ).reset_index(drop=True)
    if common.empty:
        raise ValueError(
            "No common valid timestamps are available for baseline evaluation"
        )
    metrics = {
        "naive_48h": calculate_metrics(common["actual_mw"], common["naive_48h_mw"]),
        "seasonal_naive_168h": calculate_metrics(
            common["actual_mw"], common["seasonal_naive_168h_mw"]
        ),
    }
    return common, metrics
