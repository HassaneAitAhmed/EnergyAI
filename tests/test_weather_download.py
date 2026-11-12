import json
from datetime import UTC, date, datetime

from energyai.weather import download_weather_run, select_weather_run


class FakeResponse:
    status_code = 200
    url = "https://example.test/weather"

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "latitude": 48.85,
            "longitude": 2.35,
            "hourly": {"time": ["2024-06-02T00:00"], "temperature_2m": [18.0]},
        }


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get(self, *args: object, **kwargs: object) -> FakeResponse:
        self.calls += 1
        return FakeResponse()


def test_download_weather_run_writes_manifest_and_uses_cache(tmp_path) -> None:
    session = FakeSession()
    run = select_weather_run(date(2024, 6, 2))
    downloaded = datetime(2024, 6, 5, tzinfo=UTC)

    first = download_weather_run(
        run, tmp_path, session=session, downloaded_at=downloaded
    )
    second = download_weather_run(run, tmp_path, session=session)
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))

    assert first.from_cache is False
    assert second.from_cache is True
    assert session.calls == 1
    assert manifest["weather_run_utc"] == "2024-06-01T00:00:00+00:00"
    assert manifest["forecast_issue_utc"] == "2024-06-01T16:00:00+00:00"
    assert len(manifest["sha256"]) == 64


def test_cache_rejects_modified_response(tmp_path) -> None:
    run = select_weather_run(date(2024, 6, 2))
    first = download_weather_run(run, tmp_path, session=FakeSession())
    first.response_path.write_text("{}", encoding="utf-8")

    try:
        download_weather_run(run, tmp_path, session=FakeSession())
    except ValueError as error:
        assert "checksum" in str(error)
    else:
        raise AssertionError("Modified cached response should be rejected")
