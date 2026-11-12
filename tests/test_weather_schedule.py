from datetime import date

from energyai.weather import build_single_run_params, select_weather_run


def test_select_weather_run_precedes_summer_issue() -> None:
    run = select_weather_run(date(2024, 6, 2))

    assert run.weather_run_utc.isoformat() == "2024-06-01T00:00:00+00:00"
    assert run.assumed_available_at_utc.isoformat() == "2024-06-01T06:00:00+00:00"
    assert run.forecast_issue_utc.isoformat() == "2024-06-01T16:00:00+00:00"
    assert run.assumed_available_at_utc <= run.forecast_issue_utc


def test_single_run_params_are_fixed_and_utc() -> None:
    params = build_single_run_params(select_weather_run(date(2024, 6, 2)))

    assert params["models"] == "ecmwf_ifs"
    assert params["run"] == "2024-06-01T00:00"
    assert params["timezone"] == "UTC"
    assert "start_date" not in params
