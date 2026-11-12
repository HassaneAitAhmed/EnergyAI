"""Run rolling-origin weather comparisons and record them with MLflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from energyai.backtesting import aggregate_backtest_metrics, rolling_origin_backtest
from energyai.modeling import load_curated_years
from energyai.weather_experiment import build_weather_experiment_table


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _plot_fold_metrics(metrics: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for model, values in metrics.groupby("model"):
        label = "With weather" if model == "with_weather" else "Without weather"
        axis.plot(values["fold"], values["mae_mw"], marker="o", label=label)
    axis.set(
        title="Rolling-origin error by evaluation fold",
        xlabel="Chronological fold",
        ylabel="MAE (MW)",
    )
    axis.set_xticks(sorted(metrics["fold"].unique()))
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def run_phase5(
    *,
    data_dir: Path,
    report_dir: Path,
    tracking_db: Path,
) -> dict[str, object]:
    """Evaluate rolling folds, write reports, and record one MLflow run."""
    weather_path = (
        data_dir / "processed" / "weather" / "ecmwf_ifs_2024-05-01_2024-06-29.parquet"
    )
    if not weather_path.exists():
        raise FileNotFoundError(
            f"Missing {weather_path}. Run: python -m energyai.finalize_ml"
        )
    hourly = load_curated_years(data_dir, [2024])
    weather = pd.read_parquet(weather_path)
    table = build_weather_experiment_table(hourly, weather)
    folds, predictions = rolling_origin_backtest(table)
    aggregate = aggregate_backtest_metrics(predictions)

    report_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(report_dir / "rolling_fold_metrics.csv", index=False)
    predictions.to_parquet(
        data_dir / "processed" / "weather" / "rolling_predictions.parquet",
        index=False,
    )
    _plot_fold_metrics(folds, report_dir / "rolling_backtest.png")
    no_weather = aggregate["no_weather"]
    with_weather = aggregate["with_weather"]
    improvement = 100 * (no_weather.mae_mw - with_weather.mae_mw) / no_weather.mae_mw
    summary = {
        "folds": int(folds["fold"].nunique()),
        "initial_train_days": 36,
        "evaluation_days_per_fold": 8,
        "evaluation_rows": len(predictions),
        "same_evaluation_timestamps": True,
        "aggregate_metrics": {
            name: asdict(values) for name, values in aggregate.items()
        },
        "weather_relative_mae_improvement_percent": improvement,
        "dataset_sha256": _sha256(weather_path),
        "git_commit": _git_commit(),
        "conclusion": (
            "Weather improved rolling-origin MAE."
            if improvement > 0
            else "Weather did not improve rolling-origin MAE."
        ),
    }
    summary_path = report_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    tracking_db.parent.mkdir(parents=True, exist_ok=True)
    tracking_uri = f"sqlite:///{tracking_db.resolve().as_posix()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("EnergyAI rolling backtests")
    with mlflow.start_run(run_name="weather-vs-no-weather-rolling") as run:
        mlflow.log_params(
            {
                "initial_train_days": 36,
                "evaluation_days_per_fold": 8,
                "folds": summary["folds"],
                "model": "HistGradientBoostingRegressor",
                "weather_run": "ECMWF D-1 00 UTC",
                "dataset_sha256": summary["dataset_sha256"],
            }
        )
        mlflow.log_metrics(
            {
                "no_weather_mae_mw": no_weather.mae_mw,
                "no_weather_rmse_mw": no_weather.rmse_mw,
                "with_weather_mae_mw": with_weather.mae_mw,
                "with_weather_rmse_mw": with_weather.rmse_mw,
                "weather_mae_improvement_percent": improvement,
            }
        )
        for _, row in folds.iterrows():
            mlflow.log_metric(
                f"{row['model']}_fold_mae_mw",
                float(row["mae_mw"]),
                step=int(row["fold"]),
            )
        mlflow.set_tags(
            {
                "git_commit": summary["git_commit"],
                "forecast_region": "Île-de-France",
                "evaluation_protocol": "expanding-window rolling origin",
            }
        )
        mlflow.log_artifacts(str(report_dir), artifact_path="reports")
        for model_path in sorted(Path("artifacts/weather_comparison").glob("*.joblib")):
            mlflow.log_artifact(str(model_path), artifact_path="models")
        run_id = run.info.run_id

    run_metadata = {
        "run_id": run_id,
        "experiment_name": "EnergyAI rolling backtests",
        "tracking_uri": tracking_uri,
    }
    (report_dir / "mlflow_run.json").write_text(
        json.dumps(run_metadata, indent=2) + "\n", encoding="utf-8"
    )
    return {**summary, "mlflow": run_metadata}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/phase5"))
    parser.add_argument("--tracking-db", type=Path, default=Path("mlflow.db"))
    args = parser.parse_args()
    result = run_phase5(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        tracking_db=args.tracking_db,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
