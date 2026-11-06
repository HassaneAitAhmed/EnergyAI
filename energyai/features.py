"""Cutoff-safe calendar and lag features for day-ahead forecasts."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

import holidays
import numpy as np
import pandas as pd

PARIS = ZoneInfo("Europe/Paris")
UTC = ZoneInfo("UTC")
SAFE_LAGS_HOURS = (48, 168, 336)


def forecast_issue_time(target_local_date: date) -> pd.Timestamp:
    """Return D-1 18:00 Europe/Paris as a UTC timestamp."""
    local_issue = datetime.combine(
        target_local_date - timedelta(days=1), time(18), tzinfo=PARIS
    )
    return pd.Timestamp(local_issue.astimezone(UTC))


def target_hours(target_local_date: date) -> pd.DatetimeIndex:
    """Return every UTC instant in a local day, including 23/25-hour DST days."""
    start_local = datetime.combine(target_local_date, time.min, tzinfo=PARIS)
    end_local = datetime.combine(
        target_local_date + timedelta(days=1), time.min, tzinfo=PARIS
    )
    return pd.date_range(
        pd.Timestamp(start_local).tz_convert("UTC"),
        pd.Timestamp(end_local).tz_convert("UTC"),
        freq="1h",
        inclusive="left",
    )


def _prepare_hourly(hourly: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp_utc", "load_mw"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Hourly data is missing columns: {sorted(missing)}")
    prepared = hourly.copy()
    prepared["timestamp_utc"] = pd.to_datetime(
        prepared["timestamp_utc"], utc=True, errors="coerce"
    )
    if prepared["timestamp_utc"].isna().any():
        raise ValueError("Hourly data contains invalid timestamps")
    if prepared["timestamp_utc"].duplicated().any():
        raise ValueError("Hourly data contains duplicate UTC timestamps")
    prepared["load_mw"] = pd.to_numeric(prepared["load_mw"], errors="coerce")
    return prepared.set_index("timestamp_utc").sort_index()


def build_day_ahead_features(
    hourly: pd.DataFrame,
    target_local_date: date,
    *,
    issued_at_utc: datetime | pd.Timestamp | None = None,
    assumed_publication_delay: timedelta = timedelta(hours=2),
) -> pd.DataFrame:
    """Build calendar and safe lag features for one local target day.

    If ``available_at_utc`` exists in ``hourly``, it is used as the historical
    measurement availability timestamp. Otherwise, availability is conservatively
    approximated as source timestamp plus ``assumed_publication_delay``. Definitive
    RTE archives do not prove their historical live publication time, so this
    assumption must remain visible in evaluation reports.
    """
    source = _prepare_hourly(hourly)
    expected_issue = forecast_issue_time(target_local_date)
    issue = pd.Timestamp(issued_at_utc or expected_issue)
    if issue.tzinfo is None:
        raise ValueError("issued_at_utc must be timezone-aware")
    issue = issue.tz_convert("UTC")
    if issue != expected_issue:
        raise ValueError(
            f"Issue time must be D-1 18:00 Europe/Paris ({expected_issue.isoformat()})"
        )

    targets = target_hours(target_local_date)
    local = targets.tz_convert(PARIS)
    features = pd.DataFrame({"timestamp_utc": targets})
    features["issued_at_utc"] = issue
    features["hour_local"] = local.hour
    features["day_of_week_local"] = local.dayofweek
    features["month_local"] = local.month
    features["is_weekend"] = local.dayofweek >= 5
    french_holidays = holidays.country_holidays("FR", years={target_local_date.year})
    features["is_french_public_holiday"] = target_local_date in french_holidays
    features["hour_sin"] = np.sin(2 * np.pi * features["hour_local"] / 24)
    features["hour_cos"] = np.cos(2 * np.pi * features["hour_local"] / 24)
    day_of_year = local.dayofyear.to_numpy()
    days_in_year = 366 if target_local_date.year % 4 == 0 else 365
    features["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / days_in_year)
    features["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / days_in_year)

    explicit_availability = None
    if "available_at_utc" in source.columns:
        explicit_availability = pd.to_datetime(
            source["available_at_utc"], utc=True, errors="coerce"
        )
        if explicit_availability.isna().any():
            raise ValueError("available_at_utc contains invalid timestamps")

    for lag_hours in SAFE_LAGS_HOURS:
        source_times = targets - timedelta(hours=lag_hours)
        values = source["load_mw"].reindex(source_times).to_numpy()
        if explicit_availability is None:
            available = source_times + assumed_publication_delay
        else:
            available = pd.DatetimeIndex(explicit_availability.reindex(source_times))
        has_value = ~pd.isna(values)
        if ((available > issue) & has_value).any():
            offending = available[(available > issue) & has_value][0]
            raise ValueError(
                f"Lag {lag_hours}h uses a measurement available after issuance: "
                f"{offending.isoformat()} > {issue.isoformat()}"
            )
        features[f"load_lag_{lag_hours}h_utc"] = values

    return features
