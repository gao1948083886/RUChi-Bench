"""Mock OpenAI-compatible API client for Phase 05 dry-run testing.

Does NOT call any real API. Returns deterministic, pre-configured responses.
Supports the same interface as the real client so the inference pipeline is interchangeable.

Pipeline-level features (resume, JSONL write) are in pipeline.py.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from ruchi_bench.inference.request_key import (
    read_request_keys_from_jsonl as _read_jsonl_request_keys,
)
from ruchi_bench.schema.errors import SerializationError

logger = logging.getLogger(__name__)

__all__ = ["MockAPIResponse", "MockAPIClient", "InferenceLogger"]


@dataclass
class MockAPIResponse:
    """Simulated API response matching the OpenAI /chat/completions shape.

    Only the fields needed by the inference pipeline are populated.
    """

    sample_id: str
    model: str
    content: str
    finish_reason: str = "stop"
    prompt_tokens: int = 10
    completion_tokens: int = 3
    total_tokens: int = 13
    latency_ms: float = 50.0

    def to_openai_dict(self) -> dict[str, object]:
        return {
            "id": f"mock-{self.sample_id}",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }


@dataclass
class MockAPIClient:
    """Mock OpenAI-compatible client returning pre-configured responses.

    Parameters
    ----------
    expected_responses :
        Mapping from ``sample_id`` to the exact model response string to return.
        Any sample_id not in this dict gets ``"__UNKNOWN__"``.
    latency_ms :
        Simulated per-request latency in milliseconds. Set to 0 for instant.
    model_name :
        Fake model name placed in the response ``model`` field.
    """

    expected_responses: dict[str, str]
    latency_ms: float = 50.0
    model_name: str = "mock-model/dummy"

    def call(
        self,
        messages: list[dict[str, object]],
        sample_id: str,
        temperature: float = 0.0,
        max_tokens: int = 8,
        top_p: float = 1.0,
        top_k: int = 0,
    ) -> MockAPIResponse:
        """Return a deterministic mock response.

        Parameters
        ----------
        messages :
            OpenAI-style chat messages. Not used for response content — only for
            logging. The content comes from ``expected_responses[sample_id]``.
        sample_id :
            Key into ``expected_responses``.
        temperature, max_tokens, top_p, top_k :
            Accepted for interface compatibility; ignored.

        Returns
        -------
        MockAPIResponse
        """
        content = self.expected_responses.get(sample_id, "__UNKNOWN__")
        start = time.monotonic()
        if self.latency_ms > 0:
            time.sleep(self.latency_ms / 1000.0)
        elapsed = (time.monotonic() - start) * 1000.0
        logger.debug(
            "MockAPI call sample_id=%r -> %r (elapsed=%.1fms)",
            sample_id,
            content,
            elapsed,
        )
        return MockAPIResponse(
            sample_id=sample_id,
            model=self.model_name,
            content=content,
            finish_reason="stop",
            latency_ms=elapsed,
        )


@dataclass
class InferenceLogger:
    """Writes each API call result to a JSONL file immediately (flush_each=True).

    Supports resume: reads the file first to find already-logged request_keys,
    then skips those on subsequent calls.

    Parameters
    ----------
    output_path :
        Path to the output JSONL file. Created / appended as needed.
    flush_each :
        Flush after each write. Always True (non-negotiable for durability).
    """

    output_path: Path
    flush_each: bool = True
    _done: set[str] = field(default_factory=set)
    _legacy_count: int = field(default=0)
    _lock_file: Path = field(default_factory=lambda: Path(".lock"), init=False, repr=False)

    def __post_init__(self) -> None:
        self._load_existing()

    def _load_existing(self) -> None:
        """Read existing file to build the skip set from request_keys."""
        if not self.output_path.exists():
            return
        try:
            request_keys, legacy_sample_ids = _read_jsonl_request_keys(self.output_path)
            self._done = request_keys
            self._legacy_count = len(legacy_sample_ids)
            if self._legacy_count:
                logger.warning(
                    "InferenceLogger resume: %d records with legacy format (no request_key). "
                    "These records will NOT be used for resume; a fresh run is required.",
                    self._legacy_count,
                )
            logger.info(
                "InferenceLogger resume: %d records already in %s (schema_version >= 1)",
                len(self._done),
                self.output_path,
            )
        except OSError as exc:
            logger.warning(
                "Could not read existing output file %s (%s); starting fresh",
                self.output_path,
                exc,
            )

    @property
    def legacy_count(self) -> int:
        """Number of legacy records (no request_key) found in the output file."""
        return self._legacy_count

    def is_done(self, request_key: str) -> bool:
        """Return True if this request_key has already been logged."""
        return request_key in self._done

    def log(
        self,
        request_key: str,
        sample_id: str,
        model: str,
        messages: list[dict[str, object]],
        raw_response: dict[str, object],
        latency_ms: float,
        error: str | None = None,
        *,
        dataset_name: str | None = None,
        corruption: str | None = None,
        level: str | None = None,
        corruption_status: str | None = None,
        sample_uid: str | None = None,
    ) -> None:
        """Append one record and flush immediately.

        Parameters
        ----------
        request_key :
            Stable unique key for this request. Used for deduplication and resume.
        sample_id :
            Original publisher sample_id (for human readability only; NOT used
            for dedup or resume).
        dataset_name :
            Dataset name (for audit).
        corruption :
            Corruption name or 'clean'.
        level :
            Corruption level or 'n/a'.
        corruption_status :
            APPLIED / NOT_APPLIED / INVALID / None.
        sample_uid :
            Sample UID for cross-dataset matching.
        """
        record: dict[str, object] = {
            "request_key": request_key,
            "sample_id": sample_id,
            "model": model,
            "messages": messages,
            "raw_response": raw_response,
            "latency_ms": latency_ms,
        }
        if dataset_name is not None:
            record["dataset_name"] = dataset_name
        if corruption is not None:
            record["corruption"] = corruption
        if level is not None:
            record["level"] = level
        if corruption_status is not None:
            record["corruption_status"] = corruption_status
        if sample_uid is not None:
            record["sample_uid"] = sample_uid
        if error:
            record["error"] = error
        try:
            with self.output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False))
                fh.write("\n")
                if self.flush_each:
                    fh.flush()
            self._done.add(request_key)
            logger.debug("Logged request_key=%r to %s", request_key, self.output_path)
        except OSError as exc:
            raise SerializationError(
                f"Failed to write inference record for request_key={request_key!r}: {exc}"
            ) from exc
