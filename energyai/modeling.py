"""Leakage-aware feature table, chronological splits, and Phase 3 models."""

from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from pathlib import Path

import holidays
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from energyai.baselines import MetricSet, calculate_metrics
from energyai.features import PARIS, forecast_issue_time

RANDOM_SEED = 42
CATEGORICAL_FEATURES = ["hour_local", "day_of_week_local", "month_local"]
NUMERIC_FEATURES = [
    "is_weekend",
    "is_french_public_holiday",
    "hour_sin",
    "hour_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "load_lag_48h_utc",
    "load_lag_168h_utc",
    "load_lag_336h_utc",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def load_curated_years(data_dir: Path, years: list[int]) -> pd.DataFrame:
    """Load and combine curated yearly Parquet files."""
    frames = []
    for year in years:
        path = data_dir / "processed" / str(year) / "load_hourly.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python -m energyai.collect --year {year}"
            )
        frames.append(pd.read_parquet(path))
    combined = pd.concat(frames, ignore_index=True)
    combined["timestamp_utc"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    if combined["timestamp_utc"].duplicated().any():
        raise ValueError("Curated yearly files overlap on UTC timestamps")
    return combined.sort_values("timestamp_utc").reset_index(drop=True)


def build_modeling_table(
    hourly: pd.DataFrame,
    *,
    assumed_publication_delay: timedelta = timedelta(hours=2),
) -> pd.DataFrame:
    """Create issue-time-auditable targets and features without row shifts."""
    required = {"timestamp_utc", "load_mw"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Hourly data is missing columns: {sorted(missing)}")
    source = hourly[list(required)].copy()
    source["timestamp_utc"] = pd.to_datetime(source["timestamp_utc"], utc=True)
    if (
        source["timestamp_utc"].isna().any()
        or source["timestamp_utc"].duplicated().any()
    ):
        raise ValueError("Hourly timestamps must be valid and unique")
    source["load_mw"] = pd.to_numeric(source["load_mw"], errors="coerce")
    load = source.set_index("timestamp_utc")["load_mw"].sort_index()

    targets = load.index
    local = targets.tz_convert(PARIS)
    target_dates = local.date
    unique_dates = {day: forecast_issue_time(day) for day in set(target_dates)}
    issued_at = pd.DatetimeIndex([unique_dates[day] for day in target_dates])
    issue_dates = issued_at.tz_convert(PARIS).date
    french_holidays = holidays.country_holidays(
        "FR", years=set(local.year.astype(int).tolist())
    )
    day_of_year = local.dayofyear.to_numpy()
    days_in_year = np.where(local.is_leap_year, 366, 365)

    table = pd.DataFrame(
        {
            "timestamp_utc": targets,
            "timestamp_local": local,
            "target_date_local": target_dates,
            "issued_at_utc": issued_at,
            "issue_date_local": issue_dates,
            "actual_mw": load.to_numpy(),
            "hour_local": local.hour,
            "day_of_week_local": local.dayofweek,
            "month_local": local.month,
            "is_weekend": (local.dayofweek >= 5).astype(int),
            "is_french_public_holiday": [
                int(day in french_holidays) for day in target_dates
            ],
            "hour_sin": np.sin(2 * np.pi * local.hour / 24),
            "hour_cos": np.cos(2 * np.pi * local.hour / 24),
            "day_of_year_sin": np.sin(2 * np.pi * day_of_year / days_in_year),
            "day_of_year_cos": np.cos(2 * np.pi * day_of_year / days_in_year),
        }
    )

    for lag_hours in (48, 168, 336):
        source_times = targets - timedelta(hours=lag_hours)
        values = load.reindex(source_times).to_numpy()
        available_at = source_times + assumed_publication_delay
        has_value = ~pd.isna(values)
        if ((available_at > issued_at) & has_value).any():
            raise ValueError(f"Lag {lag_hours}h violates the forecast issue cutoff")
        table[f"load_lag_{lag_hours}h_utc"] = values

    return table.dropna(subset=["actual_mw", *FEATURE_COLUMNS]).reset_index(drop=True)


def chronological_splits(
    table: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split strictly by local forecast issue year."""
    issue_year = pd.to_datetime(table["issue_date_local"]).dt.year
    train = table.loc[issue_year.isin([2022, 2023])].copy()
    validation = table.loc[issue_year == 2024].copy()
    test = table.loc[issue_year == 2025].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "Train (2022-23), validation (2024), and test (2025) are required"
        )
    if not (
        train["issued_at_utc"].max() < validation["issued_at_utc"].min()
        and validation["issued_at_utc"].max() < test["issued_at_utc"].min()
    ):
        raise ValueError("Chronological split boundaries overlap")
    return train, validation, test


def _preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "calendar",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def ridge_pipeline(alpha: float) -> Pipeline:
    return Pipeline([("features", _preprocessor()), ("model", Ridge(alpha=alpha))])


def tree_pipeline(
    *, learning_rate: float, max_leaf_nodes: int, l2_regularization: float
) -> Pipeline:
    return Pipeline(
        [
            ("features", _preprocessor()),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=learning_rate,
                    max_leaf_nodes=max_leaf_nodes,
                    l2_regularization=l2_regularization,
                    max_iter=250,
                    early_stopping=False,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def tune_models(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, Pipeline], list[dict[str, object]]]:
    """Select hyperparameters on validation data without touching test targets."""
    experiments: list[dict[str, object]] = []
    selected: dict[str, Pipeline] = {}

    ridge_candidates = [0.1, 1.0, 10.0, 100.0]
    best_ridge: tuple[float, Pipeline] | None = None
    for alpha in ridge_candidates:
        model = ridge_pipeline(alpha)
        model.fit(train[FEATURE_COLUMNS], train["actual_mw"])
        prediction = pd.Series(model.predict(validation[FEATURE_COLUMNS]))
        metrics = calculate_metrics(
            validation["actual_mw"].reset_index(drop=True), prediction
        )
        experiments.append(
            {"model": "ridge", "params": {"alpha": alpha}, **asdict(metrics)}
        )
        if best_ridge is None or metrics.mae_mw < best_ridge[0]:
            best_ridge = (metrics.mae_mw, model)
    assert best_ridge is not None
    selected["ridge"] = best_ridge[1]

    tree_candidates = [
        {"learning_rate": 0.05, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.05, "max_leaf_nodes": 31, "l2_regularization": 1.0},
        {"learning_rate": 0.10, "max_leaf_nodes": 15, "l2_regularization": 1.0},
        {"learning_rate": 0.10, "max_leaf_nodes": 31, "l2_regularization": 1.0},
    ]
    best_tree: tuple[float, Pipeline] | None = None
    for params in tree_candidates:
        model = tree_pipeline(**params)
        model.fit(train[FEATURE_COLUMNS], train["actual_mw"])
        prediction = pd.Series(model.predict(validation[FEATURE_COLUMNS]))
        metrics = calculate_metrics(
            validation["actual_mw"].reset_index(drop=True), prediction
        )
        experiments.append(
            {"model": "hist_gradient_boosting", "params": params, **asdict(metrics)}
        )
        if best_tree is None or metrics.mae_mw < best_tree[0]:
            best_tree = (metrics.mae_mw, model)
    assert best_tree is not None
    selected["hist_gradient_boosting"] = best_tree[1]
    return selected, experiments


def evaluate_test_models(
    models: dict[str, Pipeline], test: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, MetricSet]]:
    """Evaluate selected models and both baselines on identical test rows."""
    predictions = test[
        [
            "timestamp_utc",
            "timestamp_local",
            "issued_at_utc",
            "hour_local",
            "month_local",
            "actual_mw",
            "load_lag_48h_utc",
            "load_lag_168h_utc",
        ]
    ].copy()
    predictions = predictions.rename(
        columns={
            "load_lag_48h_utc": "naive_48h_mw",
            "load_lag_168h_utc": "seasonal_naive_168h_mw",
        }
    )
    predictions["ridge_mw"] = models["ridge"].predict(test[FEATURE_COLUMNS])
    predictions["hist_gradient_boosting_mw"] = models["hist_gradient_boosting"].predict(
        test[FEATURE_COLUMNS]
    )
    prediction_columns = {
        "naive_48h": "naive_48h_mw",
        "seasonal_naive_168h": "seasonal_naive_168h_mw",
        "ridge": "ridge_mw",
        "hist_gradient_boosting": "hist_gradient_boosting_mw",
    }
    metrics = {
        name: calculate_metrics(predictions["actual_mw"], predictions[column])
        for name, column in prediction_columns.items()
    }
    return predictions.reset_index(drop=True), metrics


def save_model_bundle(
    model: Pipeline,
    path: Path,
    *,
    model_name: str,
    training_rows: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "model_name": model_name,
            "feature_columns": FEATURE_COLUMNS,
            "training_rows": training_rows,
            "random_seed": RANDOM_SEED,
        },
        path,
    )
