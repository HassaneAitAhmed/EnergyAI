# Phase 2 exploratory findings

The figures use UTC for the canonical time series and Europe/Paris for calendar interpretation.

1. Average demand peaks around local hour **12:00**, supporting hour-of-day features.
2. Mean weekday load is **7,534 MW**, compared with **6,946 MW** on weekends.
3. Monthly average demand is highest in month **1** and lowest in month **8**.
4. The 168-hour UTC-lag autocorrelation is **0.878**, supporting a weekly naïve baseline.
5. The curated table contains **1 incomplete hourly target(s)**; they remain missing.
6. UTC lags are stable and leakage-safe, but a 168-hour UTC lag is not always the same local clock hour across DST changes.

These observations describe association, not causation. The source is a revised historical archive and does not prove historical publication availability; Phase 2 assumes a conservative two-hour measurement delay for cutoff checks.
