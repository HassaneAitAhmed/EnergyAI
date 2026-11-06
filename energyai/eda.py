"""Compact Phase 2 exploratory report for validated hourly load data."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from energyai.features import PARIS


def prepare_eda_data(hourly: pd.DataFrame) -> pd.DataFrame:
    """Add local-calendar columns without changing the canonical UTC timestamp."""
    required = {"timestamp_utc", "load_mw", "sample_count"}
    missing = required.difference(hourly.columns)
    if missing:
        raise ValueError(f"Hourly data is missing columns: {sorted(missing)}")
    data = hourly.copy()
    data["timestamp_utc"] = pd.to_datetime(data["timestamp_utc"], utc=True)
    data["timestamp_local"] = data["timestamp_utc"].dt.tz_convert(PARIS)
    data["hour_local"] = data["timestamp_local"].dt.hour
    data["weekday_local"] = data["timestamp_local"].dt.day_name()
    data["weekday_number"] = data["timestamp_local"].dt.dayofweek
    data["month_local"] = data["timestamp_local"].dt.month
    data["is_weekend"] = data["weekday_number"] >= 5
    return data.sort_values("timestamp_utc").reset_index(drop=True)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=160, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def generate_eda_report(
    hourly: pd.DataFrame, output_dir: Path
) -> tuple[list[Path], str]:
    """Generate six labeled figures and an evidence-based Markdown summary."""
    data = prepare_eda_data(hourly)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(data["timestamp_utc"], data["load_mw"], linewidth=0.45, color="#2563eb")
    ax.set(
        title="Île-de-France hourly electricity demand",
        xlabel="Time (UTC)",
        ylabel="Load (MW)",
    )
    ax.grid(alpha=0.2)
    path = output_dir / "01_yearly_load.png"
    _save(fig, path)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(data["load_mw"].dropna(), bins=40, color="#2563eb", alpha=0.85)
    ax.set(
        title="Distribution of hourly electricity demand",
        xlabel="Load (MW)",
        ylabel="Hours",
    )
    ax.grid(axis="y", alpha=0.2)
    path = output_dir / "02_load_distribution.png"
    _save(fig, path)
    paths.append(path)

    profile = data.groupby(["is_weekend", "hour_local"])["load_mw"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile.index, profile[False], marker="o", markersize=3, label="Weekday")
    ax.plot(profile.index, profile[True], marker="o", markersize=3, label="Weekend")
    ax.set(
        title="Average daily demand profile",
        xlabel="Local hour (Europe/Paris)",
        ylabel="Average load (MW)",
        xticks=range(0, 24, 2),
    )
    ax.legend()
    ax.grid(alpha=0.2)
    path = output_dir / "03_daily_profile.png"
    _save(fig, path)
    paths.append(path)

    monthly = data.groupby("month_local")["load_mw"].mean()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(monthly.index, monthly.values, color="#2563eb")
    ax.set(
        title="Average demand by local calendar month",
        xlabel="Month",
        ylabel="Average load (MW)",
        xticks=range(1, 13),
    )
    ax.grid(axis="y", alpha=0.2)
    path = output_dir / "04_monthly_average.png"
    _save(fig, path)
    paths.append(path)

    heatmap = data.pivot_table(
        index="weekday_number", columns="hour_local", values="load_mw", aggfunc="mean"
    ).reindex(range(7))
    fig, ax = plt.subplots(figsize=(10, 4))
    image = ax.imshow(heatmap, aspect="auto", cmap="viridis")
    ax.set(
        title="Average load by weekday and local hour",
        xlabel="Local hour (Europe/Paris)",
        ylabel="Weekday",
        xticks=range(0, 24, 2),
        xticklabels=range(0, 24, 2),
        yticks=range(7),
        yticklabels=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
    )
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Average load (MW)")
    path = output_dir / "05_weekday_hour_heatmap.png"
    _save(fig, path)
    paths.append(path)

    complete = data.set_index("timestamp_utc")["load_mw"]
    lag_hours = [1, 24, 48, 168, 336]
    autocorrelation = [complete.autocorr(lag=lag) for lag in lag_hours]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(lag) for lag in lag_hours], autocorrelation, color="#2563eb")
    ax.set(
        title="Load autocorrelation at candidate UTC lags",
        xlabel="Lag (hours)",
        ylabel="Correlation",
        ylim=(0, 1),
    )
    ax.grid(axis="y", alpha=0.2)
    path = output_dir / "06_lag_autocorrelation.png"
    _save(fig, path)
    paths.append(path)

    peak_hour = int(data.groupby("hour_local")["load_mw"].mean().idxmax())
    weekday_mean = float(data.loc[~data["is_weekend"], "load_mw"].mean())
    weekend_mean = float(data.loc[data["is_weekend"], "load_mw"].mean())
    peak_month = int(monthly.idxmax())
    low_month = int(monthly.idxmin())
    weekly_corr = float(complete.autocorr(lag=168))
    summary = f"""# Phase 2 exploratory findings

The figures use UTC for the canonical time series and Europe/Paris for calendar interpretation.

1. Average demand peaks around local hour **{peak_hour:02d}:00**, supporting hour-of-day features.
2. Mean weekday load is **{weekday_mean:,.0f} MW**, compared with **{weekend_mean:,.0f} MW** on weekends.
3. Monthly average demand is highest in month **{peak_month}** and lowest in month **{low_month}**.
4. The 168-hour UTC-lag autocorrelation is **{weekly_corr:.3f}**, supporting a weekly naïve baseline.
5. The curated table contains **{int(data["load_mw"].isna().sum())} incomplete hourly target(s)**; they remain missing.
6. UTC lags are stable and leakage-safe, but a 168-hour UTC lag is not always the same local clock hour across DST changes.

These observations describe association, not causation. The source is a revised historical archive and does not prove historical publication availability; Phase 2 assumes a conservative two-hour measurement delay for cutoff checks.
"""
    return paths, summary
