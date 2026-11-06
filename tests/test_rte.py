"""Deterministic Phase 1 tests; external RTE calls are mocked."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pandas as pd
import pytest

from energyai.rte import (
    build_year_params,
    clean_records,
    download_day,
    download_year,
    hourly_averages,
    to_hourly,
    validate_and_clean,
)


def sample_records() -> list[dict]:
    return [
        {
            "code_insee_region": "11",
            "libelle_region": "Île-de-France",
            "nature": "Données définitives",
            "date_heure": "2025-01-14T23:00:00+00:00",
            "consommation": 1000,
        },
        {
            "code_insee_region": "11",
            "libelle_region": "Île-de-France",
            "nature": "Données définitives",
            "date_heure": "2025-01-14T23:30:00+00:00",
            "consommation": 1200,
        },
    ]


def test_api_query_filters_idf_and_date() -> None:
    fake_response = Mock()
    fake_response.json.return_value = {"total_count": 2, "results": sample_records()}
    fake_session = Mock()
    fake_session.get.return_value = fake_response
    records = download_day("2025-01-15", session=fake_session)
    assert len(records) == 2
    where = fake_session.get.call_args.kwargs["params"]["where"]
    assert "code_insee_region = '11'" in where
    assert "date = '2025-01-15'" in where


def test_year_query_is_bounded_by_utc_datetime_not_text_date() -> None:
    params = build_year_params(2024)
    assert "date_heure >= date'2024-01-01T00:00:00Z'" in params["where"]
    assert "date_heure < date'2025-01-01T00:00:00Z'" in params["where"]
    assert params["delimiter"] == ";"


def test_half_hour_readings_average_to_hourly_mw() -> None:
    clean, _ = validate_and_clean(
        sample_records(), ingested_at=datetime(2026, 1, 1, tzinfo=UTC)
    )
    hourly = to_hourly(clean)
    assert len(hourly) == 1
    assert hourly.loc[0, "load_mw"] == 1100
    assert hourly.loc[0, "sample_count"] == 2
    assert str(hourly["timestamp_utc"].dtype) == "datetime64[ns, UTC]"


def test_incomplete_hour_becomes_missing_instead_of_faked() -> None:
    clean, _ = validate_and_clean(sample_records()[:1])
    hourly = to_hourly(clean)
    assert hourly["load_mw"].isna().all()
    assert hourly.loc[0, "sample_count"] == 1


def test_null_load_is_reported_and_makes_hour_incomplete() -> None:
    records = sample_records()
    records[1]["consommation"] = None
    clean, report = validate_and_clean(records)
    hourly = to_hourly(clean)
    assert report.missing_load_values == 1
    assert hourly.loc[0, "sample_count"] == 1
    assert pd.isna(hourly.loc[0, "load_mw"])


def test_gap_count_and_empty_hour_are_preserved() -> None:
    records = sample_records()
    records.append(
        {
            **records[0],
            "date_heure": "2025-01-15T01:00:00+00:00",
            "consommation": 1300,
        }
    )
    clean, report = validate_and_clean(records)
    hourly = to_hourly(clean)
    assert report.missing_half_hours == 2
    assert len(hourly) == 3
    assert hourly["load_mw"].isna().sum() == 2


def test_expected_range_counts_missing_boundary_samples() -> None:
    clean, report = validate_and_clean(
        sample_records()[1:],
        expected_start_utc="2025-01-14T23:00:00Z",
        expected_end_utc="2025-01-15T00:00:00Z",
    )
    assert len(clean) == 1
    assert report.missing_half_hours == 1


def test_duplicate_policy_prefers_definitive_row() -> None:
    duplicate = {
        **sample_records()[0],
        "nature": "Données consolidées",
        "consommation": 9999,
    }
    clean, report = validate_and_clean([duplicate, *sample_records()])
    assert report.duplicate_rows_removed == 1
    assert clean.loc[0, "load_mw"] == 1000
    assert clean.loc[0, "source_nature"] == "Données définitives"


def test_negative_load_is_rejected() -> None:
    records = sample_records()
    records[0]["consommation"] = -1
    with pytest.raises(ValueError, match="negative MW"):
        validate_and_clean(records)


def test_wrong_region_and_bad_timestamp_are_rejected() -> None:
    wrong_region = sample_records()
    wrong_region[0]["code_insee_region"] = "84"
    with pytest.raises(ValueError, match="Unexpected region"):
        validate_and_clean(wrong_region)

    bad_time = sample_records()
    bad_time[0]["date_heure"] = "not-a-date"
    with pytest.raises(ValueError, match="cannot be parsed"):
        validate_and_clean(bad_time)


def test_bad_date_is_rejected() -> None:
    with pytest.raises(ValueError):
        download_day("15/01/2025")


def test_legacy_helpers_still_average_in_utc_then_show_paris() -> None:
    frame = clean_records(sample_records())
    hourly = hourly_averages(frame)
    assert hourly.loc[0, "consumption_mw"] == 1100
    assert str(hourly["hour_paris"].dt.tz) == "Europe/Paris"


def test_year_download_writes_manifest_and_reuses_verified_cache(tmp_path) -> None:
    csv_bytes = (
        "code_insee_region;libelle_region;nature;date_heure;consommation\n"
        "11;Île-de-France;Données définitives;2024-01-01T00:00:00+00:00;1000\n"
    ).encode()
    response = Mock(status_code=200, url="https://example.test/export")
    response.iter_content.return_value = [csv_bytes]
    session = Mock()
    session.get.return_value = response
    downloaded_at = datetime(2026, 1, 2, tzinfo=UTC)

    first = download_year(2024, tmp_path, session=session, downloaded_at=downloaded_at)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert not first.from_cache
    assert manifest["region_code"] == "11"
    assert manifest["byte_length"] == len(csv_bytes)
    assert len(manifest["sha256"]) == 64
    assert "date_heure" in session.get.call_args.kwargs["params"]["where"]

    second = download_year(2024, tmp_path, session=session)
    assert second.from_cache
    assert session.get.call_count == 1


def test_modified_cache_is_rejected(tmp_path) -> None:
    response = Mock(status_code=200, url="https://example.test/export")
    response.iter_content.return_value = [b"original"]
    session = Mock()
    session.get.return_value = response
    result = download_year(2024, tmp_path, session=session)
    result.csv_path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        download_year(2024, tmp_path, session=session)
