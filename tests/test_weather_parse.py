from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from energyai.weather import parse_weather_response, select_weather_run


def weather_payload(day: date) -> dict[str, object]:
    times = pd.date_range(day - timedelta(days=1), periods=72, freq="1h")
    return {
        "latitude": 48.89,
        "longitude": 2.29,
        "hourly_units": {
            "time": "iso8601",
            "temperature_2m": "°C",
            "precipitation": "mm",
            "wind_speed_10m": "km/h",
        },
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "temperature_2m": list(range(72)),
            "precipitation": [0.0] * 72,
            "wind_speed_10m": [10.0] * 72,
        },
    }


def test_parse_weather_response_keeps_exact_target_hours() -> None:
    run = select_weather_run(date(2024, 6, 2))
    parsed = parse_weather_response(
        weather_payload(date(2024, 6, 2)),
        run,
        downloaded_at=datetime(2024, 6, 5, tzinfo=UTC),
    )

    assert len(parsed) == 24
    assert parsed["valid_time_utc"].iloc[0].isoformat() == "2024-06-01T22:00:00+00:00"
    assert parsed["valid_time_utc"].iloc[-1].isoformat() == "2024-06-02T21:00:00+00:00"
    assert parsed["weather_run_utc"].lt(parsed["forecast_issue_utc"]).all()
    assert set(parsed["model_name"]) == {"ecmwf_ifs"}


def test_parse_weather_response_rejects_wrong_units() -> None:
    payload = weather_payload(date(2024, 6, 2))
    payload["hourly_units"]["temperature_2m"] = "°F"  # type: ignore[index]

    with pytest.raises(ValueError, match="Unexpected unit"):
        parse_weather_response(
            payload,
            select_weather_run(date(2024, 6, 2)),
            downloaded_at=datetime(2024, 6, 5, tzinfo=UTC),
        )


def test_parse_weather_response_rejects_incomplete_target_day() -> None:
    payload = weather_payload(date(2024, 6, 2))
    missing_index = payload["hourly"]["time"].index(  # type: ignore[index,union-attr]
        "2024-06-02T10:00"
    )
    for field in ("time", "temperature_2m", "precipitation", "wind_speed_10m"):
        payload["hourly"][field].pop(missing_index)  # type: ignore[index,union-attr]

    with pytest.raises(ValueError, match="every target hour"):
        parse_weather_response(
            payload,
            select_weather_run(date(2024, 6, 2)),
            downloaded_at=datetime(2024, 6, 5, tzinfo=UTC),
        )
