"""Tests for transparent baseline predictions and metrics."""

from datetime import timedelta

import numpy as np
import pandas as pd

from energyai.baselines import evaluate_naive_baselines


def test_baselines_use_timestamp_lags_and_common_rows() -> None:
    timestamps = pd.date_range("2024-01-01", periods=500, freq="1h", tz="UTC")
    load = 7000 + timestamps.hour * 20 + np.arange(len(timestamps)) * 0.1
    hourly = pd.DataFrame({"timestamp_utc": timestamps, "load_mw": load})
    predictions, metrics = evaluate_naive_baselines(hourly, target_local_year=2024)

    assert len(predictions) == 500 - 168
    assert metrics["naive_48h"].observations == len(predictions)
    assert metrics["seasonal_naive_168h"].observations == len(predictions)
    first = predictions.iloc[0]
    assert first["issued_at_utc"] < first["timestamp_utc"]
    assert first["issued_at_utc"].tz_convert("Europe/Paris").hour == 18
    source = hourly.set_index("timestamp_utc").loc[
        first["timestamp_utc"] - timedelta(hours=168), "load_mw"
    ]
    assert first["seasonal_naive_168h_mw"] == source


def test_weekly_baseline_is_exact_for_weekly_pattern() -> None:
    timestamps = pd.date_range("2024-01-01", periods=400, freq="1h", tz="UTC")
    load = 5000 + (np.arange(len(timestamps)) % 168)
    hourly = pd.DataFrame({"timestamp_utc": timestamps, "load_mw": load})
    _, metrics = evaluate_naive_baselines(hourly, target_local_year=2024)
    weekly = metrics["seasonal_naive_168h"]
    assert weekly.mae_mw == 0
    assert weekly.rmse_mw == 0
    assert weekly.wape_percent == 0
