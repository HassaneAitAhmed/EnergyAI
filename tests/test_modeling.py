"""Tests for Phase 3 chronological modeling."""

from datetime import timedelta

import numpy as np
import pandas as pd

from energyai.modeling import (
    FEATURE_COLUMNS,
    build_modeling_table,
    chronological_splits,
    ridge_pipeline,
    tree_pipeline,
)


def multi_year_hourly() -> pd.DataFrame:
    timestamps = pd.date_range(
        "2021-12-15", "2026-01-02", freq="1h", inclusive="left", tz="UTC"
    )
    position = np.arange(len(timestamps))
    load = (
        7500
        + 900 * np.sin(2 * np.pi * timestamps.hour / 24)
        + 300 * np.cos(2 * np.pi * timestamps.dayofweek / 7)
        + 0.01 * position
    )
    return pd.DataFrame({"timestamp_utc": timestamps, "load_mw": load})


def test_modeling_table_has_auditable_safe_features() -> None:
    table = build_modeling_table(multi_year_hourly())
    assert table[FEATURE_COLUMNS].notna().all().all()
    assert (table["issued_at_utc"] < table["timestamp_utc"]).all()
    source_availability = (
        table["timestamp_utc"] - timedelta(hours=48) + timedelta(hours=2)
    )
    assert (source_availability <= table["issued_at_utc"]).all()


def test_splits_are_strictly_chronological_by_issue_date() -> None:
    table = build_modeling_table(multi_year_hourly())
    train, validation, test = chronological_splits(table)
    assert set(pd.to_datetime(train["issue_date_local"]).dt.year) == {2022, 2023}
    assert set(pd.to_datetime(validation["issue_date_local"]).dt.year) == {2024}
    assert set(pd.to_datetime(test["issue_date_local"]).dt.year) == {2025}
    assert train["issued_at_utc"].max() < validation["issued_at_utc"].min()
    assert validation["issued_at_utc"].max() < test["issued_at_utc"].min()


def test_both_model_pipelines_fit_without_future_rows() -> None:
    table = build_modeling_table(multi_year_hourly())
    train, validation, _ = chronological_splits(table)
    train_sample = train.iloc[:2000]
    validation_sample = validation.iloc[:100]
    models = [
        ridge_pipeline(alpha=10.0),
        tree_pipeline(learning_rate=0.1, max_leaf_nodes=15, l2_regularization=1.0),
    ]
    for model in models:
        model.fit(train_sample[FEATURE_COLUMNS], train_sample["actual_mw"])
        prediction = model.predict(validation_sample[FEATURE_COLUMNS])
        assert prediction.shape == (len(validation_sample),)
        assert np.isfinite(prediction).all()
