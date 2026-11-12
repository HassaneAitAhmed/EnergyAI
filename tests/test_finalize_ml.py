from energyai.finalize_ml import DEFAULT_END, DEFAULT_START


def test_default_weather_experiment_has_sixty_days() -> None:
    assert (DEFAULT_END - DEFAULT_START).days + 1 == 60
