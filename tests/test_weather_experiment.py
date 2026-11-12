from datetime import timedelta

import numpy as np
import pandas as pd

from energyai.features import forecast_issue_time
from energyai.weather_experiment import (
    WEATHER_FEATURES,
    build_weather_experiment_table,
    chronological_day_splits,
    train_weather_comparison,
)


def synthetic_demand() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", "2024-05-10", freq="1h", tz="UTC")
    position = np.arange(len(timestamps))
    load = 7000 + 800 * np.sin(2 * np.pi * timestamps.hour / 24) + position * 0.02
    return pd.DataFrame({"timestamp_utc": timestamps, "load_mw": load})


def synthetic_weather() -> pd.DataFrame:
    timestamps = pd.date_range("2024-03-01", periods=60 * 24, freq="1h", tz="UTC")
    local_dates = timestamps.tz_convert("Europe/Paris").date
    issue = pd.DatetimeIndex([forecast_issue_time(day) for day in local_dates])
    run = pd.DatetimeIndex(
        [pd.Timestamp(day - timedelta(days=1), tz="UTC") for day in local_dates]
    )
    return pd.DataFrame(
        {
            "valid_time_utc": timestamps,
            "weather_run_utc": run,
            "weather_available_at_utc": run + timedelta(hours=6),
            "forecast_issue_utc": issue,
            "temperature_2m": 10 + 5 * np.sin(np.arange(len(timestamps)) / 24),
            "wind_speed_10m": 12.0,
            "precipitation": 0.0,
        }
    )


def test_weather_experiment_has_identical_complete_rows() -> None:
    table = build_weather_experiment_table(synthetic_demand(), synthetic_weather())

    assert len(table) == len(synthetic_weather())
    assert table[WEATHER_FEATURES].notna().all().all()
    assert table["weather_available_at_utc"].le(table["issued_at_utc"]).all()


def test_weather_experiment_splits_and_trains_chronologically() -> None:
    table = build_weather_experiment_table(synthetic_demand(), synthetic_weather())
    train, validation, test = chronological_day_splits(
        table, train_days=36, validation_days=12
    )
    models, metrics, predictions = train_weather_comparison(train, validation, test)

    assert train["timestamp_utc"].max() < validation["timestamp_utc"].min()
    assert validation["timestamp_utc"].max() < test["timestamp_utc"].min()
    assert set(models) == {"no_weather", "with_weather"}
    assert metrics["test"]["no_weather"].observations == len(test)
    assert predictions[["no_weather_mw", "with_weather_mw"]].notna().all().all()
