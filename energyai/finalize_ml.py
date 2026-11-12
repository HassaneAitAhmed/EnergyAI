"""Finish the beginner ML project with an honest weather comparison."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from energyai.modeling import load_curated_years
from energyai.phase4 import build_weather_sample
from energyai.weather_experiment import (
    build_weather_experiment_table,
    chronological_day_splits,
    save_weather_models,
    serializable_metrics,
    train_weather_comparison,
)

DEFAULT_START = date(2024, 5, 1)
DEFAULT_END = date(2024, 6, 29)


def _plot_comparison(metrics: dict[str, dict[str, object]], path: Path) -> None:
    test = metrics["test"]
    labels = ["Without weather", "With archived weather"]
    mae = [test["no_weather"].mae_mw, test["with_weather"].mae_mw]
    figure, axis = plt.subplots(figsize=(7, 4.5))
    bars = axis.bar(labels, mae, color=["#64748b", "#2563eb"])
    axis.bar_label(bars, fmt="%.1f MW", padding=4)
    axis.set_ylabel("Test MAE (MW)")
    axis.set_title("Identical-timestamp weather comparison")
    axis.set_ylim(0, max(mae) * 1.2)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def finish_beginner_ml_project(
    *,
    data_dir: Path,
    report_dir: Path,
    artifact_dir: Path,
    start: date = DEFAULT_START,
    end: date = DEFAULT_END,
) -> dict[str, object]:
    """Collect archived forecasts, compare fixed models, and write final outputs."""
    weather_path = (
        data_dir / "processed" / "weather" / f"ecmwf_ifs_{start}_{end}.parquet"
    )
    if not weather_path.exists():
        weather_path, _ = build_weather_sample(
            start,
            end,
            data_dir=data_dir,
            report_dir=report_dir,
        )
    hourly = load_curated_years(data_dir, [2024])
    weather = pd.read_parquet(weather_path)
    table = build_weather_experiment_table(hourly, weather)
    train, validation, test = chronological_day_splits(table)
    models, metrics, predictions = train_weather_comparison(train, validation, test)

    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_weather_models(models, artifact_dir)
    predictions.to_parquet(
        data_dir / "processed" / "weather" / "comparison_test_predictions.parquet",
        index=False,
    )
    metric_payload = serializable_metrics(metrics)
    (report_dir / "metrics.json").write_text(
        json.dumps(metric_payload, indent=2) + "\n", encoding="utf-8"
    )
    _plot_comparison(metrics, report_dir / "comparison.png")

    no_weather_mae = metrics["test"]["no_weather"].mae_mw
    weather_mae = metrics["test"]["with_weather"].mae_mw
    improvement = 100 * (no_weather_mae - weather_mae) / no_weather_mae
    conclusion = (
        "Archived weather improved test MAE on this limited experiment."
        if improvement > 0
        else "Archived weather did not improve test MAE on this limited experiment."
    )
    comparison = f"""# Final beginner ML comparison

Both histogram-gradient-boosting models use the same timestamps, fixed parameters,
training days, validation days, and untouched test days. The only difference is
the three archived forecast-weather features.

| Test model | MAE (MW) | RMSE (MW) | WAPE | Rows |
|---|---:|---:|---:|---:|
| Without weather | {no_weather_mae:.2f} | {metrics["test"]["no_weather"].rmse_mw:.2f} | {metrics["test"]["no_weather"].wape_percent:.2f}% | {metrics["test"]["no_weather"].observations} |
| With archived weather | {weather_mae:.2f} | {metrics["test"]["with_weather"].rmse_mw:.2f} | {metrics["test"]["with_weather"].wape_percent:.2f}% | {metrics["test"]["with_weather"].observations} |

Relative test MAE change: **{improvement:+.2f}%**.

## Conclusion

{conclusion}

This is a small May–June 2024 learning experiment using one representative Paris
weather point. It does not replace the stronger full-year 2025 no-weather result,
and it is not evidence that weather will always improve forecasting.
"""
    (report_dir / "comparison.md").write_text(comparison, encoding="utf-8")
    reproducibility = {
        "weather_target_start": start.isoformat(),
        "weather_target_end": end.isoformat(),
        "train_target_days": 36,
        "validation_target_days": 12,
        "test_target_days": 12,
        "train_rows": len(train),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "random_seed": 42,
        "split_rule": "chronological local target days; no shuffling",
        "weather_source": "Open-Meteo ECMWF individual runs initialized D-1 00 UTC",
        "weather_availability_rule": "run initialization plus conservative 6 hours",
        "same_test_timestamps": True,
    }
    (report_dir / "reproducibility.json").write_text(
        json.dumps(reproducibility, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "metrics": metric_payload,
        "relative_test_mae_improvement_percent": improvement,
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports/final_ml"))
    parser.add_argument(
        "--artifact-dir", type=Path, default=Path("artifacts/weather_comparison")
    )
    parser.add_argument("--start-date", type=date.fromisoformat, default=DEFAULT_START)
    parser.add_argument("--end-date", type=date.fromisoformat, default=DEFAULT_END)
    args = parser.parse_args()
    result = finish_beginner_ml_project(
        data_dir=args.data_dir,
        report_dir=args.report_dir,
        artifact_dir=args.artifact_dir,
        start=args.start_date,
        end=args.end_date,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
