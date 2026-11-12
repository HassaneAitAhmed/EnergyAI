# Final beginner ML comparison

Both histogram-gradient-boosting models use the same timestamps, fixed parameters,
training days, validation days, and untouched test days. The only difference is
the three archived forecast-weather features.

| Test model | MAE (MW) | RMSE (MW) | WAPE | Rows |
|---|---:|---:|---:|---:|
| Without weather | 180.79 | 232.90 | 2.94% | 288 |
| With archived weather | 191.18 | 248.15 | 3.11% | 288 |

Relative test MAE change: **-5.75%**.

## Conclusion

Archived weather did not improve test MAE on this limited experiment.

This is a small May–June 2024 learning experiment using one representative Paris
weather point. It does not replace the stronger full-year 2025 no-weather result,
and it is not evidence that weather will always improve forecasting.
