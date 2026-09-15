"""Deterministic JSON/JSONL serialization for :class:`~ruchi_bench.schema.sample.Sample`.

Provides two levels:

- **Object round-trip**: :func:`sample_to_json` / :func:`sample_from_json` — a single
  :class:`Sample` to a UTF-8 JSON string and back. Uses ``model_dump(mode="json")``
  and ``model_validate``. Tuple → list, enum → value, datetime → UTC ISO-8601.
- **File I/O**: :func:`write_jsonl` / :func:`read_jsonl` — write or read a JSONL file
  containing zero or more :class:`Sample` records, one JSON object per line.

Error wrapping follows the "classify then elevate" pattern:

- File I/O errors (``OSError``, ``UnicodeDecodeError``) → ``SerializationError``
  with the file path.
- JSON syntax errors (``json.JSONDecodeError``) → ``SerializationError`` with the
  **1-based line number** and the raw error message.
- Pydantic ``ValidationError`` from ``model_validate`` → wrapped in
  ``SerializationError`` with the **1-based line number** and the field-level
  detail from the original exception.

No silent skipping of empty lines (an empty line in a JSONL file is a data-malformation
error, not whitespace). ``write_jsonl`` flushes each record immediately before returning,
counts accurately, and does not falsely report success if a record could not be written.

No model-API checkpoint logic (resume, skip-completed, retry-failed — those belong in
the Phase 06 inference pipeline).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from json import JSONDecodeError
from pathlib import Path

from pydantic import ValidationError

from ruchi_bench.schema.errors import SerializationError
from ruchi_bench.schema.sample import Sample

__all__ = [
    "WriteResult",
    "read_jsonl",
    "sample_from_json",
    "sample_to_json",
    "write_jsonl",
]


def sample_to_json(sample: Sample, *, ensure_ascii: bool = False) -> str:
    """Serialize one ``Sample`` to a UTF-8 JSON string.

    ``ensure_ascii=True`` escapes all non-ASCII characters (``\\uXXXX``); the default
    preserves Chinese characters verbatim. Enum members serialize as their string values;
    ``datetime`` → UTC ISO-8601; tuple → list.
    """
    # ``ensure_ascii`` is a json.dumps option, not a supported argument on every
    # Pydantic 2.x ``model_dump_json`` release. Dump in JSON mode first, then use the
    # stdlib encoder for stable control over Unicode escaping across environments.
    return json.dumps(
        sample.model_dump(mode="json", by_alias=True),
        ensure_ascii=ensure_ascii,
        separators=(",", ":"),
    )


def sample_from_json(value: str) -> Sample:
    """Parse one JSON string into a :class:`Sample`.

    Raises :class:`SerializationError` (wrapping ``JSONDecodeError`` or
    ``ValidationError``) on malformed input.
    """
    # Check JSON syntax explicitly first so we catch json.JSONDecodeError directly
    # (Pydantic v2 catches it internally and re-raises ValidationError type="json_invalid").
    try:
        json.loads(value)
    except JSONDecodeError as exc:
        raise SerializationError(f"JSON decode error: {exc}") from exc
    try:
        return Sample.model_validate_json(value)
    except ValidationError as exc:
        # Pydantic v2 re-raises JSONDecodeError internally as ValidationError
        # with type="json_invalid" — surface it as "JSON decode error" too.
        for err in exc.errors():
            if err["type"] == "json_invalid":
                raise SerializationError(f"JSON decode error: {err['msg']}") from exc
        raise SerializationError(f"JSON parsed but failed Pydantic validation: {exc}") from exc


def _wrap_io(exc: BaseException, path: Path) -> SerializationError:
    error = SerializationError(f"File I/O error on {path}: {exc}")
    error.__cause__ = exc
    return error


class WriteResult:
    """Result of :func:`write_jsonl`."""

    __slots__ = ("count", "written")

    def __init__(self, count: int, written: list[Sample]) -> None:
        self.count = count
        self.written = written


def write_jsonl(
    samples: Iterator[Sample] | list[Sample],
    path: Path,
    *,
    ensure_ascii: bool = False,
) -> WriteResult:
    """Write an iterable of :class:`Sample` records to a UTF-8 JSONL file, one per line.

    Each record is serialized with ``sample_to_json`` and written with an immediate
    ``flush()`` before returning, so data is durable on disk even if the process
    crashes before the file handle is closed.

    Returns a :class:`WriteResult` carrying the number of records written and the
    list of successfully serialized objects.

    Raises :class:`SerializationError` if any record fails to serialize or if a
    file I/O error occurs. On error the file is left in an undefined state; callers
    that need atomicity should write to a temporary path and rename on success.
    """
    written: list[Sample] = []
    count = 0
    try:
        with path.open("w", encoding="utf-8") as fh:
            for sample in samples:
                try:
                    line = sample_to_json(sample, ensure_ascii=ensure_ascii)
                except (TypeError, ValueError, UnicodeEncodeError) as exc:
                    raise SerializationError(
                        f"Sample {sample.sample_id!r} failed JSON serialization: {exc}"
                    ) from exc
                fh.write(line)
                fh.write("\n")
                fh.flush()
                written.append(sample)
                count += 1
    except OSError as exc:
        raise _wrap_io(exc, path) from exc
    except UnicodeDecodeError as exc:
        raise _wrap_io(exc, path) from exc
    return WriteResult(count=count, written=written)


def read_jsonl(path: Path) -> Iterator[Sample]:
    """Yield :class:`Sample` records from a UTF-8 JSONL file, one at a time.

    Lines are parsed lazily; memory usage is O(1) per record.

    Raises :class:`SerializationError` (never silently skips errors):

    - ``UnicodeDecodeError`` → wraps the error with the file path.
    - ``JSONDecodeError`` → wraps with the **1-based line number** and raw message so
      the caller can locate and inspect the offending line.
    - ``ValidationError`` → wraps with the **1-based line number** and the field-level
      detail so the caller can identify which record and which field failed.
    - ``OSError`` (e.g. file not found) → wraps with the file path.

    Empty lines are an error: a valid JSONL file contains no blank lines.
    """
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                line = raw.rstrip("\n\r")
                if not line or line.isspace():
                    raise SerializationError(
                        f"Line {line_no}: empty line (valid JSONL has no blank lines)"
                    )
                try:
                    sample = sample_from_json(line)
                except SerializationError as exc:
                    raise SerializationError(f"Line {line_no}: {exc}") from exc
                except Exception as exc:
                    raise SerializationError(
                        f"Line {line_no}: unexpected parse error: {exc}"
                    ) from exc
                yield sample
    except OSError as exc:
        raise _wrap_io(exc, path) from exc
    except UnicodeDecodeError as exc:
        raise _wrap_io(exc, path) from exc
