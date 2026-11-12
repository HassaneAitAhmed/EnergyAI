"""Archived weather forecasts that were available before demand issuance."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from energyai.features import forecast_issue_time, target_hours
from energyai.rte import retrying_session

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


@dataclass(frozen=True)
class WeatherDownload:
    """Cached response and provenance manifest for an archived run."""

    response_path: Path
    manifest_path: Path
    from_cache: bool


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_weather_run(
    run: WeatherRun,
    raw_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    downloaded_at: datetime | None = None,
) -> WeatherDownload:
    """Download one archived run with checksum-verified local caching."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ecmwf_ifs_{run.weather_run_utc.strftime('%Y%m%dT%H%MZ')}"
    response_path = raw_dir / f"{stem}.json"
    manifest_path = raw_dir / f"{stem}.manifest.json"
    if response_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != _sha256(response_path):
            raise ValueError("Cached weather response checksum does not match manifest")
        return WeatherDownload(response_path, manifest_path, True)

    params = build_single_run_params(run)
    client = session if session is not None else retrying_session()
    response = client.get(SINGLE_RUNS_URL, params=params, timeout=(10, 180))
    response.raise_for_status()
    payload: dict[str, Any] = response.json()
    if payload.get("error") or not isinstance(payload.get("hourly"), dict):
        raise ValueError("Open-Meteo returned an invalid hourly forecast response")

    temporary = response_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(response_path)
    timestamp = (downloaded_at or datetime.now(UTC)).astimezone(UTC)
    manifest = {
        "provider": "Open-Meteo",
        "source_url": getattr(response, "url", SINGLE_RUNS_URL),
        "model_name": WEATHER_MODEL,
        "query": params,
        "weather_run_utc": run.weather_run_utc.isoformat(),
        "assumed_available_at_utc": run.assumed_available_at_utc.isoformat(),
        "forecast_issue_utc": run.forecast_issue_utc.isoformat(),
        "downloaded_at_utc": timestamp.isoformat(),
        "http_status": response.status_code,
        "byte_length": response_path.stat().st_size,
        "sha256": _sha256(response_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return WeatherDownload(response_path, manifest_path, False)


def parse_weather_response(
    payload: dict[str, Any],
    run: WeatherRun,
    *,
    downloaded_at: datetime,
) -> pd.DataFrame:
    """Validate an archived response and keep the exact local target-day hours."""
    if run.assumed_available_at_utc > run.forecast_issue_utc:
        raise ValueError("Weather run availability is later than forecast issuance")
    hourly = payload.get("hourly")
    units = payload.get("hourly_units")
    if not isinstance(hourly, dict) or not isinstance(units, dict):
        raise TypeError("Weather response is missing hourly data or units")
    expected_units = {
        "temperature_2m": "°C",
        "precipitation": "mm",
        "wind_speed_10m": "km/h",
    }
    for variable, expected_unit in expected_units.items():
        if units.get(variable) != expected_unit:
            raise ValueError(f"Unexpected unit for {variable}: {units.get(variable)!r}")
    required = {"time", *WEATHER_VARIABLES}
    missing = required.difference(hourly)
    if missing:
        raise ValueError(f"Weather response is missing fields: {sorted(missing)}")
    lengths = {field: len(hourly[field]) for field in required}
    if len(set(lengths.values())) != 1:
        raise ValueError("Weather hourly arrays have different lengths")

    frame = pd.DataFrame({field: hourly[field] for field in required})
    frame["valid_time_utc"] = pd.to_datetime(
        frame.pop("time"), utc=True, errors="coerce"
    )
    if frame["valid_time_utc"].isna().any():
        raise ValueError("Weather response contains invalid timestamps")
    if frame["valid_time_utc"].duplicated().any():
        raise ValueError("Weather response contains duplicate timestamps")
    targets = target_hours(run.target_local_date)
    frame = frame.set_index("valid_time_utc").reindex(targets)
    if frame[list(WEATHER_VARIABLES)].isna().any().any():
        raise ValueError("Weather run does not cover every target hour")
    for variable in WEATHER_VARIABLES:
        frame[variable] = pd.to_numeric(frame[variable], errors="raise")

    timestamp = downloaded_at.astimezone(UTC)
    result = frame.reset_index(names="valid_time_utc")
    result.insert(0, "forecast_issue_utc", run.forecast_issue_utc)
    result.insert(0, "weather_available_at_utc", run.assumed_available_at_utc)
    result.insert(0, "weather_run_utc", run.weather_run_utc)
    result["latitude"] = float(payload["latitude"])
    result["longitude"] = float(payload["longitude"])
    result["provider"] = "Open-Meteo"
    result["model_name"] = WEATHER_MODEL
    result["downloaded_at_utc"] = pd.Timestamp(timestamp)
    return result[
        [
            "weather_run_utc",
            "weather_available_at_utc",
            "forecast_issue_utc",
            "valid_time_utc",
            "latitude",
            "longitude",
            "temperature_2m",
            "wind_speed_10m",
            "precipitation",
            "provider",
            "model_name",
            "downloaded_at_utc",
        ]
    ]


def read_weather_download(download: WeatherDownload, run: WeatherRun) -> pd.DataFrame:
    """Read a cached response and its recorded download timestamp."""
    payload = json.loads(download.response_path.read_text(encoding="utf-8"))
    manifest = json.loads(download.manifest_path.read_text(encoding="utf-8"))
    downloaded_at = datetime.fromisoformat(manifest["downloaded_at_utc"])
    return parse_weather_response(payload, run, downloaded_at=downloaded_at)


def join_weather_features(
    demand_features: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Join weather by valid UTC hour with strict as-of availability guards."""
    demand_required = {"timestamp_utc", "issued_at_utc"}
    weather_required = {
        "valid_time_utc",
        "weather_run_utc",
        "weather_available_at_utc",
        "forecast_issue_utc",
        *WEATHER_VARIABLES,
    }
    missing_demand = demand_required.difference(demand_features.columns)
    missing_weather = weather_required.difference(weather.columns)
    if missing_demand or missing_weather:
        raise ValueError(
            "Missing join columns: "
            f"demand={sorted(missing_demand)}, weather={sorted(missing_weather)}"
        )

    demand = demand_features.copy()
    forecast = weather.copy()
    demand["timestamp_utc"] = pd.to_datetime(demand["timestamp_utc"], utc=True)
    demand["issued_at_utc"] = pd.to_datetime(demand["issued_at_utc"], utc=True)
    for column in (
        "valid_time_utc",
        "weather_run_utc",
        "weather_available_at_utc",
        "forecast_issue_utc",
    ):
        forecast[column] = pd.to_datetime(forecast[column], utc=True)
    if demand["timestamp_utc"].duplicated().any():
        raise ValueError("Demand features contain duplicate target timestamps")
    if forecast["valid_time_utc"].duplicated().any():
        raise ValueError("Weather data contains duplicate valid timestamps")

    joined = demand.merge(
        forecast,
        how="left",
        left_on="timestamp_utc",
        right_on="valid_time_utc",
        validate="one_to_one",
    )
    if joined["valid_time_utc"].isna().any():
        raise ValueError("Weather data does not cover every demand target hour")
    if not joined["issued_at_utc"].eq(joined["forecast_issue_utc"]).all():
        raise ValueError("Weather issue metadata does not match demand issuance")
    if (joined["weather_run_utc"] > joined["issued_at_utc"]).any():
        raise ValueError("Weather run is later than demand forecast issuance")
    if (joined["weather_available_at_utc"] > joined["issued_at_utc"]).any():
        raise ValueError("Weather data became available after forecast issuance")

    joined = joined.drop(columns="valid_time_utc")
    return joined.rename(
        columns={
            "temperature_2m": "weather_temperature_2m_c",
            "wind_speed_10m": "weather_wind_speed_10m_kmh",
            "precipitation": "weather_precipitation_mm",
        }
    )
