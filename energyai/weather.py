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

from energyai.features import forecast_issue_time
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
