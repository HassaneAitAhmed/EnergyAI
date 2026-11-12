"""Build a verified sample of archived, issue-safe weather forecasts."""

from __future__ import annotations

import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from energyai.weather import (
    SINGLE_RUNS_URL,
    download_weather_run,
    read_weather_download,
    select_weather_run,
)


def _dates(start: date, end: date) -> list[date]:
    if end < start:
        raise ValueError("end date must not be earlier than start date")
    return [start + timedelta(days=offset) for offset in range((end - start).days + 1)]


def build_weather_sample(
    start: date,
    end: date,
    *,
    data_dir: Path,
    report_dir: Path,
    force: bool = False,
) -> tuple[Path, Path]:
    """Download, validate, and store archived forecasts for local target days."""
    frames = []
    cache_hits = 0
    for target_date in _dates(start, end):
        run = select_weather_run(target_date)
        download = download_weather_run(run, data_dir / "raw" / "weather", force=force)
        cache_hits += int(download.from_cache)
        frames.append(read_weather_download(download, run))

    weather = pd.concat(frames, ignore_index=True)
    if weather["valid_time_utc"].duplicated().any():
        raise ValueError("Target-day weather sample contains duplicate valid hours")
    if (weather["weather_available_at_utc"] > weather["forecast_issue_utc"]).any():
        raise ValueError("Weather sample contains post-issuance data")

    processed_dir = data_dir / "processed" / "weather"
    processed_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = processed_dir / f"ecmwf_ifs_{start}_{end}.parquet"
    weather.to_parquet(parquet_path, index=False)

    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "weather_sample_summary.json"
    summary = {
        "source": SINGLE_RUNS_URL,
        "provider": "Open-Meteo",
        "model_name": "ecmwf_ifs",
        "location": "Paris representative point",
        "target_local_start": start.isoformat(),
        "target_local_end": end.isoformat(),
        "target_days": len(_dates(start, end)),
        "hourly_rows": len(weather),
        "first_valid_time_utc": weather["valid_time_utc"].min().isoformat(),
        "last_valid_time_utc": weather["valid_time_utc"].max().isoformat(),
        "weather_runs_utc": sorted(
            timestamp.isoformat() for timestamp in weather["weather_run_utc"].unique()
        ),
        "all_weather_available_before_issue": bool(
            weather["weather_available_at_utc"].le(weather["forecast_issue_utc"]).all()
        ),
        "assumed_model_ready_delay_hours": 6,
        "cache_hits": cache_hits,
        "temperature_2m_c_min": float(weather["temperature_2m"].min()),
        "temperature_2m_c_max": float(weather["temperature_2m"].max()),
        "precipitation_mm_total": float(weather["precipitation"].sum()),
        "wind_speed_10m_kmh_max": float(weather["wind_speed_10m"].max()),
        "note": (
            "This is archived forecast output from fixed model runs, not observed "
            "or reanalysis weather. One Paris point does not represent the full region."
        ),
    }
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return parquet_path, report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", type=date.fromisoformat, required=True)
    parser.add_argument("--end-date", type=date.fromisoformat, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase4"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    parquet_path, report_path = build_weather_sample(
        args.start_date,
        args.end_date,
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        force=args.force,
    )
    print(f"Weather sample: {parquet_path}")
    print(f"Verification report: {report_path}")


if __name__ == "__main__":
    main()
