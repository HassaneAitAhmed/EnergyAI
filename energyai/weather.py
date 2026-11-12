"""Archived weather forecasts that were available before demand issuance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import pandas as pd

from energyai.features import forecast_issue_time

SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"
PARIS_LATITUDE = 48.8566
PARIS_LONGITUDE = 2.3522
WEATHER_MODEL = "ecmwf_ifs"
WEATHER_VARIABLES = ("temperature_2m", "precipitation", "wind_speed_10m")
CONSERVATIVE_READY_DELAY = timedelta(hours=6)


@dataclass(frozen=True)
class WeatherRun:
    """The archived model run selected for a local target day."""

    target_local_date: date
    weather_run_utc: pd.Timestamp
    forecast_issue_utc: pd.Timestamp
    assumed_available_at_utc: pd.Timestamp


def select_weather_run(target_local_date: date) -> WeatherRun:
    """Select the D-1 00 UTC ECMWF run available before the 18:00 cutoff."""
    issue = forecast_issue_time(target_local_date)
    run = pd.Timestamp(target_local_date - timedelta(days=1), tz="UTC")
    available = run + CONSERVATIVE_READY_DELAY
    if available > issue:
        raise ValueError("Selected weather run was not available by forecast issuance")
    return WeatherRun(target_local_date, run, issue, available)


def build_single_run_params(run: WeatherRun) -> dict[str, object]:
    """Build a bounded Open-Meteo Single Runs API request."""
    return {
        "latitude": PARIS_LATITUDE,
        "longitude": PARIS_LONGITUDE,
        "hourly": ",".join(WEATHER_VARIABLES),
        "models": WEATHER_MODEL,
        "run": run.weather_run_utc.strftime("%Y-%m-%dT%H:%M"),
        "forecast_days": 3,
        "timezone": "UTC",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }
