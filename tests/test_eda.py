"""Smoke tests for the reproducible EDA report."""

import numpy as np
import pandas as pd

from energyai.eda import generate_eda_report


def test_eda_generates_six_explained_figures(tmp_path) -> None:
    timestamps = pd.date_range("2024-01-01", periods=24 * 40, freq="1h", tz="UTC")
    hourly = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "load_mw": 7000 + 800 * np.sin(2 * np.pi * timestamps.hour / 24),
            "sample_count": 2,
        }
    )
    paths, summary = generate_eda_report(hourly, tmp_path)
    assert len(paths) == 6
    assert all(path.exists() and path.stat().st_size > 0 for path in paths)
    assert "five" not in summary.lower()
    assert "UTC" in summary
    assert "Europe/Paris" in summary
