from datetime import UTC, date, datetime

import pandas as pd
import pytest

from energyai.features import build_day_ahead_features
from energyai.weather import (
    join_weather_features,
    parse_weather_response,
    select_weather_run,
)
from tests.test_weather_parse import weather_payload


def demand_features() -> pd.DataFrame:
    timestamps = pd.date_range("2024-05-15", "2024-06-03", freq="1h", tz="UTC")
    hourly = pd.DataFrame(
        {"timestamp_utc": timestamps, "load_mw": range(len(timestamps))}
    )
    return build_day_ahead_features(hourly, date(2024, 6, 2))


def archived_weather() -> pd.DataFrame:
    return parse_weather_response(
        weather_payload(date(2024, 6, 2)),
        select_weather_run(date(2024, 6, 2)),
        downloaded_at=datetime(2024, 6, 5, tzinfo=UTC),
    )


def test_join_weather_features_matches_all_target_hours() -> None:
    joined = join_weather_features(demand_features(), archived_weather())

    assert len(joined) == 24
    assert joined["weather_temperature_2m_c"].notna().all()
    assert joined["weather_available_at_utc"].le(joined["issued_at_utc"]).all()
    assert joined["weather_precipitation_mm"].eq(0).all()


def test_join_weather_features_rejects_post_issue_availability() -> None:
    weather = archived_weather()
    weather["weather_available_at_utc"] = pd.Timestamp("2024-06-01T18:00:00Z")

    with pytest.raises(ValueError, match="available after"):
        join_weather_features(demand_features(), weather)


def test_join_weather_features_rejects_missing_hour() -> None:
    weather = archived_weather().iloc[:-1]

    with pytest.raises(ValueError, match="every demand target hour"):
        join_weather_features(demand_features(), weather)
