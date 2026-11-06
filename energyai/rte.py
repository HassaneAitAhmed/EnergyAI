"""RTE historical demand ingestion and UTC-first data preparation.

The source contains half-hour power readings in MW. Hourly load is therefore
the arithmetic mean of two readings, never their sum.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from datetime import date as Date
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DATASET_ID = "eco2mix-regional-cons-def"
API_ROOT = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets"
API_URL = f"{API_ROOT}/{DATASET_ID}/records"
EXPORT_URL = f"{API_ROOT}/{DATASET_ID}/exports/csv"
REGION_CODE = "11"
REGION_NAME = "Île-de-France"
EXPECTED_FIELDS = {
    "code_insee_region",
    "libelle_region",
    "nature",
    "date_heure",
    "consommation",
}
NATURE_PRIORITY = {
    "données définitives": 3,
    "données consolidées": 2,
    "données temps réel": 1,
}


@dataclass(frozen=True)
class DownloadResult:
    """Paths and cache status for a raw yearly download."""

    csv_path: Path
    manifest_path: Path
    from_cache: bool


@dataclass(frozen=True)
class QualityReport:
    """Auditable counts produced while validating a raw extract."""

    raw_rows: int
    clean_rows: int
    duplicate_rows_removed: int
    missing_load_values: int
    negative_load_values: int
    implausible_load_values: int
    missing_half_hours: int
    first_timestamp_utc: str | None
    last_timestamp_utc: str | None


def _utc_now() -> datetime:
    return datetime.now(UTC)


def build_year_params(year: int) -> dict[str, str]:
    """Return an ODSQL query for one bounded UTC calendar year."""
    if year < 2000 or year > 2100:
        raise ValueError("year must be between 2000 and 2100")
    start = f"{year:04d}-01-01T00:00:00Z"
    end = f"{year + 1:04d}-01-01T00:00:00Z"
    return {
        "where": (
            f"code_insee_region = '{REGION_CODE}' "
            f"AND date_heure >= date'{start}' "
            f"AND date_heure < date'{end}'"
        ),
        "delimiter": ";",
        "timezone": "UTC",
    }


def retrying_session() -> requests.Session:
    """Create an HTTP session with bounded retries for transient failures."""
    retry = Retry(
        total=4,
        backoff_factor=0.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset({"GET"}),
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_year(
    year: int,
    raw_dir: Path,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    downloaded_at: datetime | None = None,
) -> DownloadResult:
    """Download and cache one region/year CSV together with a source manifest."""
    params = build_year_params(year)
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_path = raw_dir / f"rte_idf_{year}.csv"
    manifest_path = raw_dir / f"rte_idf_{year}.manifest.json"

    if csv_path.exists() and manifest_path.exists() and not force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != _sha256(csv_path):
            raise ValueError(
                f"Cached file checksum does not match {manifest_path}; "
                "use --force only after inspecting the files."
            )
        return DownloadResult(csv_path, manifest_path, True)

    client = session if session is not None else retrying_session()
    response = client.get(EXPORT_URL, params=params, timeout=(10, 180), stream=True)
    response.raise_for_status()
    temp_path = csv_path.with_suffix(".csv.part")
    with temp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                handle.write(chunk)
    if temp_path.stat().st_size == 0:
        temp_path.unlink()
        raise ValueError("RTE export returned an empty CSV")
    temp_path.replace(csv_path)

    timestamp = (downloaded_at or _utc_now()).astimezone(UTC)
    manifest: dict[str, Any] = {
        "source_dataset": DATASET_ID,
        "source_url": getattr(response, "url", EXPORT_URL),
        "query": params,
        "region_code": REGION_CODE,
        "utc_year": year,
        "downloaded_at_utc": timestamp.isoformat(),
        "http_status": response.status_code,
        "byte_length": csv_path.stat().st_size,
        "sha256": _sha256(csv_path),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return DownloadResult(csv_path, manifest_path, False)


def probe_schema(
    *, session: requests.Session | None = None, limit: int = 5
) -> dict[str, Any]:
    """Fetch a tiny live sample to inspect fields without a large export."""
    if not 1 <= limit <= 20:
        raise ValueError("limit must be between 1 and 20")
    client = session if session is not None else retrying_session()
    response = client.get(
        API_URL,
        params={"where": f"code_insee_region = '{REGION_CODE}'", "limit": limit},
        timeout=(10, 30),
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError("Unexpected API response: no non-empty 'results' list")
    missing = EXPECTED_FIELDS.difference(records[0])
    if missing:
        raise ValueError(f"Required columns missing from API: {sorted(missing)}")
    return {
        "total_count": payload.get("total_count"),
        "field_keys": sorted(records[0]),
        "records": records,
    }


def read_raw_csv(path: Path) -> pd.DataFrame:
    """Read an Opendatasoft semicolon-delimited UTF-8 export."""
    return pd.read_csv(
        path,
        sep=";",
        encoding="utf-8-sig",
        dtype={"code_insee_region": str},
        low_memory=False,
    )


def validate_and_clean(
    records: pd.DataFrame | Iterable[dict[str, Any]],
    *,
    ingested_at: datetime | None = None,
    plausible_max_mw: float = 25_000.0,
    expected_start_utc: datetime | str | None = None,
    expected_end_utc: datetime | str | None = None,
) -> tuple[pd.DataFrame, QualityReport]:
    """Validate source rows and resolve timestamp duplicates by source priority.

    Invalid timestamps, off-grid timestamps, incorrect regions, and negative
    loads are hard errors. Null and unusually high loads remain in the table and
    are reported so that missing targets are never silently fabricated.
    """
    frame = (
        records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    )
    missing = EXPECTED_FIELDS.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    raw_rows = len(frame)
    if raw_rows == 0:
        raise ValueError("RTE extract contains no rows")

    region_codes = frame["code_insee_region"].astype("string")
    if not region_codes.eq(REGION_CODE).all():
        bad = sorted(
            region_codes[~region_codes.eq(REGION_CODE)].dropna().unique().tolist()
        )
        raise ValueError(f"Unexpected region codes in bounded extract: {bad}")
    names = set(frame["libelle_region"].dropna().astype(str).str.strip())
    if names != {REGION_NAME}:
        raise ValueError(f"Unexpected region label(s): {sorted(names)}")

    frame["timestamp_utc"] = pd.to_datetime(
        frame["date_heure"], utc=True, errors="coerce", format="mixed"
    )
    if frame["timestamp_utc"].isna().any():
        raise ValueError(
            "Some date_heure values cannot be parsed as timezone-aware timestamps"
        )
    if ((frame["timestamp_utc"].dt.minute % 30) != 0).any() or (
        frame["timestamp_utc"].dt.second != 0
    ).any():
        raise ValueError("Source timestamps must align to 00 or 30 minutes")

    if (expected_start_utc is None) != (expected_end_utc is None):
        raise ValueError(
            "expected_start_utc and expected_end_utc must be provided together"
        )
    if expected_start_utc is not None and expected_end_utc is not None:
        expected_start = pd.Timestamp(expected_start_utc)
        expected_end = pd.Timestamp(expected_end_utc)
        if expected_start.tzinfo is None or expected_end.tzinfo is None:
            raise ValueError("Expected boundaries must be timezone-aware")
        expected_start = expected_start.tz_convert("UTC")
        expected_end = expected_end.tz_convert("UTC")
        outside = ~frame["timestamp_utc"].between(
            expected_start, expected_end, inclusive="left"
        )
        if outside.any():
            raise ValueError(
                "Extract contains timestamps outside the requested UTC range"
            )

    frame["load_mw"] = pd.to_numeric(frame["consommation"], errors="coerce")
    negative_count = int(frame["load_mw"].lt(0).sum())
    if negative_count:
        raise ValueError(f"Found {negative_count} negative MW reading(s)")
    implausible = frame["load_mw"].gt(plausible_max_mw)
    frame["is_plausible_load"] = frame["load_mw"].notna() & ~implausible
    frame["_priority"] = (
        frame["nature"]
        .astype(str)
        .str.casefold()
        .map(NATURE_PRIORITY)
        .fillna(0)
        .astype(int)
    )
    frame["_source_order"] = range(len(frame))
    frame = frame.sort_values(
        ["timestamp_utc", "_priority", "_source_order"],
        ascending=[True, False, False],
    )
    duplicate_rows_removed = int(frame.duplicated("timestamp_utc", keep="first").sum())
    frame = frame.drop_duplicates("timestamp_utc", keep="first").sort_values(
        "timestamp_utc"
    )

    if expected_start_utc is not None and expected_end_utc is not None:
        expected = pd.date_range(
            expected_start, expected_end, freq="30min", inclusive="left"
        )
    else:
        expected = pd.date_range(
            frame["timestamp_utc"].min(), frame["timestamp_utc"].max(), freq="30min"
        )
    missing_half_hours = len(
        expected.difference(pd.DatetimeIndex(frame["timestamp_utc"]))
    )
    timestamp = (ingested_at or _utc_now()).astimezone(UTC)
    clean = pd.DataFrame(
        {
            "region_code": REGION_CODE,
            "timestamp_utc": frame["timestamp_utc"],
            "load_mw": frame["load_mw"].astype(float),
            "source_dataset": DATASET_ID,
            "source_nature": frame["nature"].astype("string"),
            "ingested_at_utc": pd.Timestamp(timestamp),
            "is_plausible_load": frame["is_plausible_load"].astype(bool),
        }
    ).reset_index(drop=True)
    report = QualityReport(
        raw_rows=raw_rows,
        clean_rows=len(clean),
        duplicate_rows_removed=duplicate_rows_removed,
        missing_load_values=int(clean["load_mw"].isna().sum()),
        negative_load_values=negative_count,
        implausible_load_values=int(implausible.sum()),
        missing_half_hours=missing_half_hours,
        first_timestamp_utc=clean["timestamp_utc"].min().isoformat(),
        last_timestamp_utc=clean["timestamp_utc"].max().isoformat(),
    )
    return clean, report


def to_hourly(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Aggregate half-hour MW readings into complete UTC hourly means."""
    required = {
        "region_code",
        "timestamp_utc",
        "load_mw",
        "source_dataset",
        "source_nature",
        "ingested_at_utc",
    }
    missing = required.difference(cleaned.columns)
    if missing:
        raise ValueError(f"Cleaned data is missing columns: {sorted(missing)}")
    if cleaned.empty:
        raise ValueError("Cannot aggregate an empty table")
    indexed = cleaned.set_index("timestamp_utc").sort_index()
    load = indexed["load_mw"].resample("1h").agg(["mean", "count"])
    load = load.rename(columns={"mean": "load_mw", "count": "sample_count"})
    load.loc[load["sample_count"] != 2, "load_mw"] = float("nan")
    nature = (
        indexed["source_nature"]
        .resample("1h")
        .agg(lambda values: " | ".join(sorted(set(values.dropna().astype(str)))))
    )
    result = load.join(nature.rename("source_nature")).reset_index()
    result.insert(0, "region_code", REGION_CODE)
    result["source_dataset"] = DATASET_ID
    result["ingested_at_utc"] = indexed["ingested_at_utc"].max()
    return result[
        [
            "region_code",
            "timestamp_utc",
            "load_mw",
            "sample_count",
            "source_dataset",
            "source_nature",
            "ingested_at_utc",
        ]
    ]


def write_quality_report(
    report: QualityReport, path: Path, hourly: pd.DataFrame
) -> None:
    """Write machine-readable validation and hourly completeness counts."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(report)
    payload.update(
        {
            "hourly_rows": len(hourly),
            "complete_hours": int(hourly["load_mw"].notna().sum()),
            "incomplete_hours": int(hourly["load_mw"].isna().sum()),
            "minimum_load_mw": (
                None
                if hourly["load_mw"].dropna().empty
                else float(hourly["load_mw"].min())
            ),
            "maximum_load_mw": (
                None
                if hourly["load_mw"].dropna().empty
                else float(hourly["load_mw"].max())
            ),
        }
    )
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# Compatibility helpers retained for the original one-day prototype API.
def download_day(
    day: str, session: requests.Session | None = None
) -> list[dict[str, Any]]:
    try:
        Date.fromisoformat(day)
    except ValueError as exc:
        raise ValueError("Use date format YYYY-MM-DD, e.g. 2025-01-15") from exc
    client = session if session is not None else retrying_session()
    response = client.get(
        API_URL,
        params={
            "where": f"code_insee_region = '{REGION_CODE}' AND date = '{day}'",
            "order_by": "date_heure ASC",
            "limit": 100,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    records = payload.get("results")
    if not isinstance(records, list) or not records:
        raise ValueError(
            f"No RTE records for {day}. Check availability or the API query."
        )
    if payload.get("total_count", len(records)) > len(records):
        raise ValueError("Pagination is needed; refusing to use incomplete data")
    return records


def clean_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    clean, _ = validate_and_clean(records)
    result = clean.rename(columns={"load_mw": "consumption_mw"})
    result["timestamp_paris"] = result["timestamp_utc"].dt.tz_convert("Europe/Paris")
    return result[["timestamp_utc", "timestamp_paris", "consumption_mw"]]


def hourly_averages(cleaned: pd.DataFrame) -> pd.DataFrame:
    temp = cleaned.set_index("timestamp_utc").sort_index()
    grouped = temp["consumption_mw"].resample("1h").agg(["mean", "count"])
    grouped = grouped.rename(
        columns={"mean": "consumption_mw", "count": "valid_samples"}
    )
    grouped.loc[grouped["valid_samples"] != 2, "consumption_mw"] = float("nan")
    grouped.index = grouped.index.tz_convert("Europe/Paris")
    return grouped.reset_index().rename(columns={"timestamp_utc": "hour_paris"})
