"""Download provenance tracking: base class and manifest schema.

Every official dataset download must record:
- dataset name
- canonical source URL used by the dataset access helper
- resolved URL after redirects
- retrieval timestamp (UTC)
- HTTP status
- content length
- SHA-256 hash
- file name
- file size
- official split
- expected fields
- observed fields
- expected row count
- observed row count
- license/access notes
- code commit

This module is I/O-agnostic: it provides the manifest schema and helper functions;
actual HTTP fetching is delegated to dataset-specific downloaders.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["DownloadManifest", "create_manifest", "write_manifest"]


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


@dataclass
class DownloadManifest:
    """Provenance record for a dataset download.

    All fields are required unless marked optional.
    """

    dataset: str
    canonical_source_url: str
    resolved_url: str
    retrieved_at: str  # ISO 8601 UTC
    http_status: int
    sha256: str
    file_name: str
    file_size: int
    official_split: str
    expected_fields: list[str]
    license_access_note: str
    code_commit: str
    content_length: int | None = None
    observed_fields: list[str] | None = None  # None if download failed
    expected_row_count: int | None = None
    observed_row_count: int | None = None  # None if parsing failed
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "canonical_source_url": self.canonical_source_url,
            "resolved_url": self.resolved_url,
            "retrieved_at": self.retrieved_at,
            "http_status": self.http_status,
            "content_length": self.content_length,
            "sha256": self.sha256,
            "file_name": self.file_name,
            "file_size": self.file_size,
            "official_split": self.official_split,
            "expected_fields": self.expected_fields,
            "observed_fields": self.observed_fields,
            "expected_row_count": self.expected_row_count,
            "observed_row_count": self.observed_row_count,
            "license_access_note": self.license_access_note,
            "code_commit": self.code_commit,
            "notes": self.notes,
        }


def create_manifest(
    dataset: str,
    canonical_source_url: str,
    resolved_url: str,
    http_status: int,
    sha256: str,
    file_name: str,
    file_size: int,
    official_split: str,
    expected_fields: list[str],
    expected_row_count: int | None,
    license_access_note: str,
    code_commit: str,
    content_length: int | None = None,
    observed_fields: list[str] | None = None,
    observed_row_count: int | None = None,
    notes: str = "",
) -> DownloadManifest:
    """Create a DownloadManifest with timestamp auto-filled."""
    return DownloadManifest(
        dataset=dataset,
        canonical_source_url=canonical_source_url,
        resolved_url=resolved_url,
        retrieved_at=_utc_now().isoformat(),
        http_status=http_status,
        content_length=content_length,
        sha256=sha256,
        file_name=file_name,
        file_size=file_size,
        official_split=official_split,
        expected_fields=expected_fields,
        observed_fields=observed_fields,
        expected_row_count=expected_row_count,
        observed_row_count=observed_row_count,
        license_access_note=license_access_note,
        code_commit=code_commit,
        notes=notes,
    )


def write_manifest(manifest: DownloadManifest, path: Path) -> None:
    """Write manifest to JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2, ensure_ascii=False)
