"""Tests for cutoff-safe calendar and lag features."""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from energyai.features import (
    build_day_ahead_features,
    forecast_issue_time,
    target_hours,
)


def hourly_fixture() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", "2024-04-02", freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "load_mw": 7000 + 500 * np.sin(2 * np.pi * timestamps.hour / 24),
        }
    )


def test_issue_time_respects_paris_timezone() -> None:
    assert forecast_issue_time(date(2024, 1, 15)) == pd.Timestamp(
        "2024-01-14T17:00:00Z"
    )
    assert forecast_issue_time(date(2024, 7, 15)) == pd.Timestamp(
        "2024-07-14T16:00:00Z"
    )


def test_dst_days_have_23_or_25_distinct_utc_hours() -> None:
    assert len(target_hours(date(2024, 3, 31))) == 23
    assert len(target_hours(date(2024, 10, 27))) == 25
    assert target_hours(date(2024, 10, 27)).is_unique


def test_day_ahead_features_include_safe_lags_and_calendar() -> None:
    features = build_day_ahead_features(hourly_fixture(), date(2024, 3, 15))
    assert len(features) == 24
    assert features["timestamp_utc"].dt.tz is not None
    assert features["load_lag_48h_utc"].notna().all()
    assert features["load_lag_168h_utc"].notna().all()
    assert features["load_lag_336h_utc"].notna().all()
    assert set(features["hour_local"]) == set(range(24))


def test_feature_builder_rejects_post_issue_availability() -> None:
    hourly = hourly_fixture()
    hourly["available_at_utc"] = hourly["timestamp_utc"] + timedelta(hours=2)
    target = date(2024, 3, 15)
    source_time = target_hours(target)[0] - timedelta(hours=48)
    hourly.loc[hourly["timestamp_utc"] == source_time, "available_at_utc"] = (
        forecast_issue_time(target) + timedelta(minutes=1)
    )
    with pytest.raises(ValueError, match="available after issuance"):
        build_day_ahead_features(hourly, target)


def test_feature_builder_rejects_wrong_or_naive_issue_time() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        build_day_ahead_features(
            hourly_fixture(),
            date(2024, 3, 15),
            issued_at_utc=pd.Timestamp("2024-03-14 17:00"),
        )
    with pytest.raises(ValueError, match="Issue time must be"):
        build_day_ahead_features(
            hourly_fixture(),
            date(2024, 3, 15),
            issued_at_utc=pd.Timestamp("2024-03-14T15:00:00Z"),
        )
