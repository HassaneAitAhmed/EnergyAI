# EnergyAI Phase 3 model card

## Forecast contract

At 18:00 Europe/Paris on D-1, predict every hourly electricity-demand value for local day D in Île-de-France. Predictions and targets are power in MW.

## Data and split

- Source: RTE/ODRÉ `eco2mix-regional-cons-def`
- Train: issue dates in 2022–2023 (17,199 rows)
- Validation: issue dates in 2024 (8,780 rows)
- Test: issue dates in 2025 (8,732 rows)
- The test set was evaluated only after hyperparameter selection.

## Features

Known calendar fields plus UTC load lags of 48, 168, and 336 hours. The builder enforces the D-1 18:00 cutoff with a documented two-hour measurement-publication assumption.

## Models

Ridge regression and `HistGradientBoostingRegressor` were tuned on validation MAE. The selected model is **hist_gradient_boosting**. Both trained pipelines are saved for reproducibility.

## Held-out 2025 results

| Model | MAE (MW) | RMSE (MW) | WAPE | Rows |
|---|---:|---:|---:|---:|
| naive_48h | 632.72 | 846.38 | 8.44% | 8732 |
| seasonal_naive_168h | 615.63 | 930.86 | 8.21% | 8732 |
| ridge | 447.13 | 601.51 | 5.96% | 8732 |
| hist_gradient_boosting | 434.18 | 634.00 | 5.79% | 8732 |

All models and baselines use exactly the same valid timestamps.

## Limitations

- This is an educational forecasting system, not grid-operational software.
- RTE definitive archives are revised historical data and do not prove historical publication timing.
- The two-hour measurement availability delay is a conservative modeling assumption, not verified archival availability metadata.
- Weather forecasts are not included.
- UTC weekly lags shift relative to local clock time around DST transitions.
