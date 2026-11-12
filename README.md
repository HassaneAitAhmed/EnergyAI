# EnergyAI

EnergyAI predicts hourly electricity demand for the next day in
**Île-de-France** using official RTE data.

At 18:00 Europe/Paris, the system prepares a forecast for every hour of the
following local day. Demand is expressed in MW. Archived weather inputs come
from forecasts that were available before that issue time.

## Current version

The project currently includes:

- RTE data download, caching, and validation
- UTC hourly demand built from half-hour measurements
- Missing-period, duplicate, and DST handling
- Calendar features and cutoff-safe 48h, 168h, and 336h lags
- Ridge regression and histogram gradient boosting
- Chronological train, validation, and test periods
- MAE, RMSE, and WAPE evaluation against naïve baselines
- Saved model artifacts, charts, metrics, and a model card
- Archived ECMWF forecast-run download and checksum caching
- Temperature, precipitation, and wind aligned to each target UTC hour
- Strict rejection of weather published after the demand forecast cutoff
- A same-timestamp weather-versus-no-weather learning experiment
- Expanding-window rolling backtests recorded with MLflow
- 42 automated tests

## Pipeline

```mermaid
flowchart LR
    A[RTE data] --> B[Validation]
    B --> C[Hourly UTC dataset]
    C --> D[Calendar and lag features]
    W[Archived ECMWF run] --> X[Cutoff-safe weather features]
    X --> D
    D --> E[Train 2022-2023]
    E --> F[Validate 2024]
    F --> G[Test 2025]
    G --> H[Saved model and report]
```

## Held-out 2025 results

All models use the same 8,732 test timestamps.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| 48-hour baseline | 632.72 MW | 846.38 MW | 8.44% |
| Weekly baseline | 615.63 MW | 930.86 MW | 8.21% |
| Ridge | 447.13 MW | 601.51 MW | 5.96% |
| Histogram gradient boosting | **434.18 MW** | 634.00 MW | **5.79%** |

![Actual demand, gradient boosting, and baseline predictions](reports/phase3/test_predictions.png)

## Weather learning experiment

This smaller experiment uses 60 chronological target days from May–June 2024:
36 training days, 12 validation days, and 12 untouched test days. Both models
use the same 288 test timestamps.

| Test model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Without weather | **180.79 MW** | **232.90 MW** | **2.94%** |
| With archived weather | 191.18 MW | 248.15 MW | 3.11% |

Archived weather made test MAE 5.75% worse in this limited experiment. This is
an important ML result: adding features does not automatically improve a model.
The experiment uses one representative Paris weather point and does not replace
the full-year 2025 evaluation above.

![Weather comparison](reports/final_ml/comparison.png)

## Rolling backtest

Three chronological eight-day folds retrain using only earlier target days. On
the same 576 evaluation timestamps, the no-weather model reached 218.33 MW MAE
and the weather model reached 227.82 MW MAE. MLflow records the parameters,
metrics, dataset checksum, Git commit, reports, and model artifacts locally.

![Rolling-origin comparison](reports/phase5/rolling_backtest.png)

## Run locally

```powershell
git clone https://github.com/HassaneAitAhmed/EnergyAI.git
cd EnergyAI

py -3 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

.\.venv\Scripts\python.exe -m energyai.collect --year 2022
.\.venv\Scripts\python.exe -m energyai.collect --year 2023
.\.venv\Scripts\python.exe -m energyai.collect --year 2024
.\.venv\Scripts\python.exe -m energyai.collect --year 2025

.\.venv\Scripts\python.exe -m energyai.phase2 --year 2024
.\.venv\Scripts\python.exe -m energyai.phase3
.\.venv\Scripts\python.exe -m energyai.phase4 --start-date 2024-06-02 --end-date 2024-06-04
.\.venv\Scripts\python.exe -m energyai.finalize_ml
.\.venv\Scripts\python.exe -m energyai.phase5
.\.venv\Scripts\python.exe -m pytest -q
```

## Main outputs

```text
artifacts/models/best_model.joblib
reports/phase3/test_metrics.json
reports/phase3/test_predictions.png
reports/phase3/error_slices.csv
reports/phase3/model_card.md
reports/phase3/reproducibility.json
reports/phase4/weather_sample_summary.json
reports/final_ml/metrics.json
reports/final_ml/comparison.md
reports/final_ml/comparison.png
reports/final_ml/reproducibility.json
artifacts/weather_comparison/no_weather.joblib
artifacts/weather_comparison/with_weather.joblib
reports/phase5/rolling_fold_metrics.csv
reports/phase5/rolling_backtest.png
reports/phase5/summary.json
reports/phase5/mlflow_run.json
```

## Project structure

```text
energyai/       demand, weather, features, baselines, and models
notebooks/      exploratory data analysis
reports/        generated metrics, figures, and model documentation
artifacts/      trained model files
tests/          offline automated tests
```

Data sources:

- [RTE/ODRÉ éCO2mix](https://odre.opendatasoft.com/explore/dataset/eco2mix-regional-cons-def/)
- [Open-Meteo Single Runs API](https://open-meteo.com/en/docs/single-runs-api)
