from energyai.backtesting import aggregate_backtest_metrics, rolling_origin_backtest
from energyai.weather_experiment import build_weather_experiment_table
from tests.test_weather_experiment import synthetic_demand, synthetic_weather


def test_rolling_origin_uses_only_earlier_training_days() -> None:
    table = build_weather_experiment_table(synthetic_demand(), synthetic_weather())
    folds, predictions = rolling_origin_backtest(
        table, initial_train_days=36, fold_days=8
    )

    assert folds["fold"].nunique() == 3
    assert len(predictions) == 24 * 24
    assert (folds["train_end"] < folds["evaluation_start"]).all()
    assert predictions["timestamp_utc"].is_unique


def test_rolling_models_share_every_evaluation_timestamp() -> None:
    table = build_weather_experiment_table(synthetic_demand(), synthetic_weather())
    _, predictions = rolling_origin_backtest(table, initial_train_days=36, fold_days=8)
    metrics = aggregate_backtest_metrics(predictions)

    assert metrics["no_weather"].observations == len(predictions)
    assert metrics["with_weather"].observations == len(predictions)
    assert predictions[["no_weather_mw", "with_weather_mw"]].notna().all().all()
