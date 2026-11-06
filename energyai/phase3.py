"""Train and evaluate the Phase 3 forecasting models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from energyai.modeling import (
    FEATURE_COLUMNS,
    RANDOM_SEED,
    build_modeling_table,
    chronological_splits,
    evaluate_test_models,
    load_curated_years,
    save_model_bundle,
    tune_models,
)


def _plot_predictions(predictions: pd.DataFrame, path: Path) -> None:
    sample = predictions.iloc[: 24 * 14]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample["timestamp_local"], sample["actual_mw"], label="Actual", linewidth=2)
    ax.plot(
        sample["timestamp_local"],
        sample["hist_gradient_boosting_mw"],
        label="Gradient boosting",
        linewidth=1.3,
    )
    ax.plot(
        sample["timestamp_local"],
        sample["naive_48h_mw"],
        label="48-hour baseline",
        linewidth=1,
        alpha=0.8,
    )
    ax.set(
        title="2025 test predictions — first 14 days",
        xlabel="Target time (Europe/Paris)",
        ylabel="Electricity demand (MW)",
    )
    ax.legend(ncols=3)
    ax.grid(alpha=0.2)
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _model_card(
    metrics: dict[str, object],
    split_counts: dict[str, int],
    best_model: str,
) -> str:
    rows = "\n".join(
        f"| {name} | {value.mae_mw:.2f} | {value.rmse_mw:.2f} | "
        f"{value.wape_percent:.2f}% | {value.observations} |"
        for name, value in metrics.items()
    )
    return f"""# EnergyAI Phase 3 model card

## Forecast contract

At 18:00 Europe/Paris on D-1, predict every hourly electricity-demand value for local day D in Île-de-France. Predictions and targets are power in MW.

## Data and split

- Source: RTE/ODRÉ `eco2mix-regional-cons-def`
- Train: issue dates in 2022–2023 ({split_counts["train"]:,} rows)
- Validation: issue dates in 2024 ({split_counts["validation"]:,} rows)
- Test: issue dates in 2025 ({split_counts["test"]:,} rows)
- The test set was evaluated only after hyperparameter selection.

## Features

Known calendar fields plus UTC load lags of 48, 168, and 336 hours. The builder enforces the D-1 18:00 cutoff with a documented two-hour measurement-publication assumption.

## Models

Ridge regression and `HistGradientBoostingRegressor` were tuned on validation MAE. The selected model is **{best_model}**. Both trained pipelines are saved for reproducibility.

## Held-out 2025 results

| Model | MAE (MW) | RMSE (MW) | WAPE | Rows |
|---|---:|---:|---:|---:|
{rows}

All models and baselines use exactly the same valid timestamps.

## Limitations

- This is an educational forecasting system, not grid-operational software.
- RTE definitive archives are revised historical data and do not prove historical publication timing.
- The two-hour measurement availability delay is a conservative modeling assumption, not verified archival availability metadata.
- Weather forecasts are not included.
- UTC weekly lags shift relative to local clock time around DST transitions.
"""


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate EnergyAI Phase 3")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports") / "phase3")
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts") / "models"
    )
    args = parser.parse_args(argv)

    hourly = load_curated_years(args.data_dir, [2022, 2023, 2024, 2025])
    table = build_modeling_table(hourly)
    train, validation, test = chronological_splits(table)
    models, experiments = tune_models(train, validation)
    predictions, test_metrics = evaluate_test_models(models, test)

    best_experiment = min(experiments, key=lambda row: float(row["mae_mw"]))
    best_model = str(best_experiment["model"])
    args.report_dir.mkdir(parents=True, exist_ok=True)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        save_model_bundle(
            model,
            args.artifact_dir / f"{name}.joblib",
            model_name=name,
            training_rows=len(train),
        )
    save_model_bundle(
        models[best_model],
        args.artifact_dir / "best_model.joblib",
        model_name=best_model,
        training_rows=len(train),
    )

    metrics_payload = {name: asdict(value) for name, value in test_metrics.items()}
    (args.report_dir / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "validation_experiments.json").write_text(
        json.dumps(experiments, indent=2) + "\n", encoding="utf-8"
    )
    predictions.to_parquet(
        args.data_dir / "processed" / "phase3_test_predictions.parquet", index=False
    )
    best_prediction_column = f"{best_model}_mw"
    predictions["absolute_error_mw"] = (
        predictions[best_prediction_column] - predictions["actual_mw"]
    ).abs()
    slice_rows: list[dict[str, object]] = []
    for slice_name, values in {
        "hour_local": predictions["timestamp_local"].dt.hour,
        "weekday_local": predictions["timestamp_local"].dt.day_name(),
        "month_local": predictions["timestamp_local"].dt.month,
    }.items():
        sliced = predictions.assign(slice_value=values).groupby("slice_value")[
            "absolute_error_mw"
        ]
        for value, errors in sliced:
            slice_rows.append(
                {
                    "slice": slice_name,
                    "value": value,
                    "mae_mw": float(errors.mean()),
                    "observations": int(errors.count()),
                }
            )
    pd.DataFrame(slice_rows).to_csv(args.report_dir / "error_slices.csv", index=False)
    _plot_predictions(predictions, args.report_dir / "test_predictions.png")

    split_counts = {
        "train": len(train),
        "validation": len(validation),
        "test": len(test),
    }
    (args.report_dir / "model_card.md").write_text(
        _model_card(test_metrics, split_counts, best_model), encoding="utf-8"
    )
    metadata = {
        "random_seed": RANDOM_SEED,
        "feature_columns": FEATURE_COLUMNS,
        "split_counts": split_counts,
        "selected_model": best_model,
        "selected_validation_experiment": best_experiment,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ["pandas", "numpy", "scikit-learn", "joblib"]
        },
    }
    (args.report_dir / "reproducibility.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Train/validation/test rows: {len(train)}/{len(validation)}/{len(test)}")
    print(f"Selected model: {best_model}")
    for name, result in test_metrics.items():
        print(
            f"{name}: MAE={result.mae_mw:.2f} MW, RMSE={result.rmse_mw:.2f} MW, "
            f"WAPE={result.wape_percent:.2f}% (n={result.observations})"
        )


if __name__ == "__main__":
    main()
