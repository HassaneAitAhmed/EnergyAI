"""Run Phase 2 EDA and transparent baseline evaluation."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from energyai.baselines import evaluate_naive_baselines
from energyai.eda import generate_eda_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate EnergyAI Phase 2 outputs")
    parser.add_argument("--year", type=int, default=2024, help="Local target year")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--report-dir", type=Path, default=Path("reports") / "phase2")
    args = parser.parse_args(argv)

    hourly_path = args.data_dir / "processed" / str(args.year) / "load_hourly.parquet"
    if not hourly_path.exists():
        raise SystemExit(
            f"Missing {hourly_path}. Run: python -m energyai.collect --year {args.year}"
        )
    hourly = pd.read_parquet(hourly_path)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    figures, summary = generate_eda_report(hourly, args.report_dir / "figures")
    summary_path = args.report_dir / "eda_summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    predictions, metrics = evaluate_naive_baselines(hourly, target_local_year=args.year)
    prediction_path = (
        args.data_dir / "processed" / str(args.year) / "baseline_predictions.parquet"
    )
    predictions.to_parquet(prediction_path, index=False)
    metrics_path = args.report_dir / "baseline_metrics.json"
    metrics_path.write_text(
        json.dumps({name: asdict(value) for name, value in metrics.items()}, indent=2)
        + "\n",
        encoding="utf-8",
    )

    print(f"EDA figures: {len(figures)} in {args.report_dir / 'figures'}")
    print(f"EDA summary: {summary_path}")
    print(f"Baseline predictions: {prediction_path}")
    print(f"Baseline metrics: {metrics_path}")
    for name, result in metrics.items():
        print(
            f"{name}: MAE={result.mae_mw:.2f} MW, RMSE={result.rmse_mw:.2f} MW, "
            f"WAPE={result.wape_percent:.2f}% (n={result.observations})"
        )


if __name__ == "__main__":
    main()
