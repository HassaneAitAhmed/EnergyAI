"""Small chronological weather comparison for the beginner ML release."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from energyai.baselines import MetricSet, calculate_metrics
from energyai.modeling import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    RANDOM_SEED,
    build_modeling_table,
)
from energyai.weather import join_weather_features

WEATHER_FEATURES = [
    "weather_temperature_2m_c",
    "weather_wind_speed_10m_kmh",
    "weather_precipitation_mm",
]
WEATHER_FEATURE_COLUMNS = FEATURE_COLUMNS + WEATHER_FEATURES


def build_weather_experiment_table(
    hourly: pd.DataFrame, weather: pd.DataFrame
) -> pd.DataFrame:
    """Build identical demand rows for models with and without weather."""
    if "valid_time_utc" not in weather:
        raise ValueError("Weather data is missing valid_time_utc")
    valid_times = pd.to_datetime(weather["valid_time_utc"], utc=True)
    if valid_times.duplicated().any():
        raise ValueError("Weather data contains duplicate valid timestamps")

    demand = build_modeling_table(hourly)
    demand = demand.loc[demand["timestamp_utc"].isin(valid_times)].copy()
    if demand.empty:
        raise ValueError("Demand and weather data do not overlap")
    joined = join_weather_features(demand, weather)
    if joined[WEATHER_FEATURES].isna().any().any():
        raise ValueError("Weather experiment contains missing features")
    return joined.sort_values("timestamp_utc").reset_index(drop=True)


def chronological_day_splits(
    table: pd.DataFrame,
    *,
    train_days: int = 36,
    validation_days: int = 12,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split complete local target days without shuffling timestamps."""
    if train_days < 1 or validation_days < 1:
        raise ValueError("train_days and validation_days must be positive")
    dates = sorted(pd.unique(table["target_date_local"]))
    minimum_days = train_days + validation_days + 1
    if len(dates) < minimum_days:
        raise ValueError(f"At least {minimum_days} complete target days are required")
    train_dates = set(dates[:train_days])
    validation_dates = set(dates[train_days : train_days + validation_days])
    test_dates = set(dates[train_days + validation_days :])
    train = table.loc[table["target_date_local"].isin(train_dates)].copy()
    validation = table.loc[table["target_date_local"].isin(validation_dates)].copy()
    test = table.loc[table["target_date_local"].isin(test_dates)].copy()
    if not (
        train["timestamp_utc"].max() < validation["timestamp_utc"].min()
        and validation["timestamp_utc"].max() < test["timestamp_utc"].min()
    ):
        raise ValueError("Chronological weather split boundaries overlap")
    return train, validation, test


def weather_tree_pipeline(feature_columns: list[str]) -> Pipeline:
    """Return the fixed gradient-boosting pipeline used in weather comparisons."""
    numeric = [
        column for column in feature_columns if column not in CATEGORICAL_FEATURES
    ]
    preprocessor = ColumnTransformer(
        [
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    model = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        max_iter=250,
        early_stopping=False,
        random_state=RANDOM_SEED,
    )
    return Pipeline([("features", preprocessor), ("model", model)])


def train_weather_comparison(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[dict[str, Pipeline], dict[str, dict[str, MetricSet]], pd.DataFrame]:
    """Fit fixed models on train only and evaluate identical validation/test rows."""
    configurations = {
        "no_weather": FEATURE_COLUMNS,
        "with_weather": WEATHER_FEATURE_COLUMNS,
    }
    models: dict[str, Pipeline] = {}
    metrics: dict[str, dict[str, MetricSet]] = {"validation": {}, "test": {}}
    predictions = test[["timestamp_utc", "actual_mw"]].copy()
    for name, columns in configurations.items():
        model = weather_tree_pipeline(columns)
        model.fit(train[columns], train["actual_mw"])
        models[name] = model
        validation_prediction = pd.Series(
            model.predict(validation[columns]), index=validation.index
        )
        test_prediction = pd.Series(model.predict(test[columns]), index=test.index)
        metrics["validation"][name] = calculate_metrics(
            validation["actual_mw"], validation_prediction
        )
        metrics["test"][name] = calculate_metrics(test["actual_mw"], test_prediction)
        predictions[f"{name}_mw"] = test_prediction.to_numpy()
    return models, metrics, predictions.reset_index(drop=True)


def serializable_metrics(
    metrics: dict[str, dict[str, MetricSet]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Convert nested metric dataclasses into JSON-safe dictionaries."""
    return {
        split: {name: asdict(values) for name, values in models.items()}
        for split, models in metrics.items()
    }


def save_weather_models(models: dict[str, Pipeline], directory: Path) -> None:
    """Save both fitted pipelines with their exact feature contracts."""
    directory.mkdir(parents=True, exist_ok=True)
    for name, model in models.items():
        features = FEATURE_COLUMNS if name == "no_weather" else WEATHER_FEATURE_COLUMNS
        joblib.dump(
            {
                "model": model,
                "model_name": f"hist_gradient_boosting_{name}",
                "feature_columns": features,
                "random_seed": RANDOM_SEED,
            },
            directory / f"{name}.joblib",
        )
