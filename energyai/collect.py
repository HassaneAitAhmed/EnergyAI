"""Phase 1 CLI: download, validate, and aggregate one RTE UTC year.

Usage: python -m energyai.collect --year 2024
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from energyai.rte import (
    download_year,
    probe_schema,
    read_raw_csv,
    to_hourly,
    validate_and_clean,
    write_quality_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare an RTE Île-de-France UTC calendar year"
    )
    parser.add_argument("--year", type=int, help="UTC calendar year, e.g. 2024")
    parser.add_argument(
        "--data-dir", type=Path, default=Path("data"), help="raw/ and processed/ parent"
    )
    parser.add_argument(
        "--force", action="store_true", help="replace an existing raw cache"
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="print a five-row live schema probe and exit",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.probe:
        print(json.dumps(probe_schema(), ensure_ascii=False, indent=2, default=str))
        return
    if args.year is None:
        raise SystemExit("--year is required unless --probe is used")

    raw_dir = args.data_dir / "raw" / "rte"
    processed_dir = args.data_dir / "processed" / str(args.year)
    download = download_year(args.year, raw_dir, force=args.force)
    raw = read_raw_csv(download.csv_path)
    clean, report = validate_and_clean(
        raw,
        expected_start_utc=datetime(args.year, 1, 1, tzinfo=UTC),
        expected_end_utc=datetime(args.year + 1, 1, 1, tzinfo=UTC),
    )
    hourly = to_hourly(clean)

    processed_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = processed_dir / "load_hourly.parquet"
    report_path = processed_dir / "data_quality.json"
    hourly.to_parquet(hourly_path, index=False)
    write_quality_report(report, report_path, hourly)

    cache_note = "cache" if download.from_cache else "network"
    print(f"Raw source: {download.csv_path} ({cache_note})")
    print(f"Manifest: {download.manifest_path}")
    print(f"Hourly data: {hourly_path}")
    print(f"Quality report: {report_path}")
    print(
        f"Rows raw/clean/hourly: {report.raw_rows}/{report.clean_rows}/{len(hourly)}; "
        f"complete hours: {hourly['load_mw'].notna().sum()}"
    )
    print(
        f"UTC range: {hourly['timestamp_utc'].min()} to {hourly['timestamp_utc'].max()}"
    )


if __name__ == "__main__":
    main()
