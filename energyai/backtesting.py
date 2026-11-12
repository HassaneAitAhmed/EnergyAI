"""Expanding-window rolling-origin evaluation without temporal leakage."""

from __future__ import annotations

import pandas as pd

from energyai.baselines import MetricSet, calculate_metrics
from energyai.modeling import FEATURE_COLUMNS
from energyai.weather_experiment import (
    WEATHER_FEATURE_COLUMNS,
    weather_tree_pipeline,
)


def rolling_origin_backtest(
    table: pd.DataFrame,
    *,
    initial_train_days: int = 36,
    fold_days: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain on all prior days and evaluate successive non-overlapping folds."""
    if initial_train_days < 1 or fold_days < 1:
        raise ValueError("initial_train_days and fold_days must be positive")
    dates = sorted(pd.unique(table["target_date_local"]))
    if len(dates) < initial_train_days + fold_days:
        raise ValueError("Not enough target days for one complete rolling fold")

    configurations = {
        "no_weather": FEATURE_COLUMNS,
        "with_weather": WEATHER_FEATURE_COLUMNS,
    }
    metric_rows: list[dict[str, object]] = []
    prediction_frames: list[pd.DataFrame] = []
    for fold_number, start in enumerate(
        range(initial_train_days, len(dates), fold_days), start=1
    ):
        evaluation_dates = dates[start : start + fold_days]
        if len(evaluation_dates) < fold_days:
            break
        training_dates = set(dates[:start])
        evaluation_date_set = set(evaluation_dates)
        train = table.loc[table["target_date_local"].isin(training_dates)]
        evaluation = table.loc[
            table["target_date_local"].isin(evaluation_date_set)
        ].copy()
        if train["timestamp_utc"].max() >= evaluation["timestamp_utc"].min():
            raise ValueError("Rolling fold training data is not strictly earlier")

        fold_predictions = evaluation[
            ["timestamp_utc", "target_date_local", "actual_mw"]
        ].copy()
        fold_predictions["fold"] = fold_number
        for name, columns in configurations.items():
            model = weather_tree_pipeline(columns)
            model.fit(train[columns], train["actual_mw"])
            predicted = pd.Series(
                model.predict(evaluation[columns]), index=evaluation.index
            )
            result = calculate_metrics(evaluation["actual_mw"], predicted)
            metric_rows.append(
                {
                    "fold": fold_number,
                    "model": name,
                    "train_start": min(training_dates).isoformat(),
                    "train_end": max(training_dates).isoformat(),
                    "evaluation_start": min(evaluation_dates).isoformat(),
                    "evaluation_end": max(evaluation_dates).isoformat(),
                    "training_rows": len(train),
                    "evaluation_rows": len(evaluation),
                    "mae_mw": result.mae_mw,
                    "rmse_mw": result.rmse_mw,
                    "wape_percent": result.wape_percent,
                }
            )
            fold_predictions[f"{name}_mw"] = predicted.to_numpy()
        prediction_frames.append(fold_predictions)

    if not metric_rows:
        raise ValueError("No complete rolling folds were produced")
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_frames, ignore_index=True),
    )


def aggregate_backtest_metrics(
    predictions: pd.DataFrame,
) -> dict[str, MetricSet]:
    """Calculate overall metrics across exactly the same rolling predictions."""
    return {
        name: calculate_metrics(predictions["actual_mw"], predictions[f"{name}_mw"])
        for name in ("no_weather", "with_weather")
    }
