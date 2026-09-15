"""Phase 05 inference pipeline: run model calls on clean + corrupted samples.

Interfaces with either a real API client or a MockAPIClient (swap-in replacement).
Writes one JSONL record per request_key, flushed immediately. Supports resume.

No real API calls are made here unless real=True is passed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ruchi_bench.inference.prompts import prompt_for_sample
from ruchi_bench.inference.request_key import (
    RequestKeyComponents,
    make_request_key,
    sample_uid,
)
from ruchi_bench.schema.sample import Sample

if TYPE_CHECKING:
    from ruchi_bench.inference.client import RealAPIClient
    from ruchi_bench.inference.mock_client import InferenceLogger, MockAPIClient

logger = logging.getLogger(__name__)

__all__ = [
    "InferenceResult",
    "PipelineConfig",
    "run_inference",
    "create_api_client",
    "_build_request_key",
    "_build_label_only_messages",
    "_messages_hash",
]


def create_api_client(
    *, real: bool = False, api_base: str | None = None, api_key: str | None = None
) -> MockAPIClient | RealAPIClient:
    """Factory: build a real or mock API client from environment / explicit args."""
    if real:
        from ruchi_bench.inference.client import RealAPIClient

        base = api_base if api_base is not None else os.getenv("API_BASE", "")
        key = api_key if api_key is not None else os.getenv("API_KEY", "")
        model = os.getenv("API_MODEL", "unknown")
        return RealAPIClient(api_base=base, api_key=key, model_name=model)
    else:
        from ruchi_bench.inference.mock_client import MockAPIClient

        return MockAPIClient(expected_responses={}, model_name=os.getenv("API_MODEL", "mock"))


@dataclass
class InferenceResult:
    """Result of processing one sample through the inference pipeline."""

    sample_id: str
    sample_uid: str
    request_key: str
    model: str
    raw_content: str
    finish_reason: str
    latency_ms: float
    error: str | None = None

    def parsed_label(self) -> str | None:
        """Parse the label from raw model content.

        Label-only protocol: the model returns the label text directly.
        Returns None if parsing fails or content is empty.
        """
        text = self.raw_content.strip()
        if not text:
            return None
        return text


@dataclass
class PipelineConfig:
    """Configuration for one inference run."""

    output_path: Path
    model_name: str
    temperature: float = 0.0
    max_new_tokens: int | None = None
    seed: int = 0
    api_base: str = "https://api.openai.com/v1"
    prompt_template_version: int | str | None = None


def _messages_hash(messages: list[dict[str, object]]) -> str:
    """Stable SHA-256 of the canonical JSON of a messages list."""
    canonical = json.dumps(
        messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _extract_sample_metadata(sample: Sample) -> tuple[str, str, str, str | None, str]:
    """Extract audit-relevant metadata from a Sample.

    Returns (dataset_name, corruption, level, corruption_status, sample_uid_str).
    """
    dataset_name_str = sample.dataset_name.value
    if sample.corruption_name is None:
        corruption_str = "clean"
        level_str = "n/a"
        corruption_status_str: str | None = None
    else:
        corruption_str = sample.corruption_name.value
        level_str = sample.corruption_level or "n/a"
        if sample.corrupted_payload is not None and sample.change_count > 0:
            corruption_status_str = "APPLIED"
        elif sample.failure_reason is not None:
            corruption_status_str = sample.failure_reason
        else:
            corruption_status_str = None

    su = sample_uid(
        sample_id=sample.sample_id,
        dataset_name=dataset_name_str,
        split=sample.split.value,
    )
    return dataset_name_str, corruption_str, level_str, corruption_status_str, su


def _build_request_key(
    sample: Sample,
    config: PipelineConfig,
    is_clean: bool,
    messages: list[dict[str, object]],
) -> str:
    """Build a stable request_key for one sample + condition."""
    uid = sample_uid(
        sample_id=sample.sample_id,
        dataset_name=sample.dataset_name.value,
        split=sample.split.value,
    )
    # Derive the v2 task-specific defaults unless a caller explicitly pins values.
    _, prompt_version, task_max_new_tokens = prompt_for_sample(
        sample, is_clean=is_clean
    )
    prompt_template_version = (
        config.prompt_template_version
        if config.prompt_template_version is not None
        else prompt_version
    )
    max_new_tokens = (
        config.max_new_tokens
        if config.max_new_tokens is not None
        else task_max_new_tokens
    )

    # Include actual corruption parameters so clean and corrupted produce distinct keys.
    corruption_name = sample.corruption_name.value if sample.corruption_name else None
    corruption_level = sample.corruption_level if sample.corruption_level else None
    corruption_seed = (
        sample.corruption_seed if sample.corruption_seed is not None else None
    )
    components = RequestKeyComponents(
        sample_uid=uid,
        is_clean=is_clean,
        corruption_name=corruption_name,
        corruption_level=corruption_level,
        corruption_seed=corruption_seed,
        target_field=None,
        model_uid=config.model_name,
        prompt_template_version=prompt_template_version,
        temperature=config.temperature,
        max_new_tokens=max_new_tokens,
        messages_hash=_messages_hash(messages),
    )
    return make_request_key(components)


def _build_label_only_messages(
    sample: Sample, *, is_clean: bool = True
) -> list[dict[str, object]]:
    """Build label-only zero-shot prompt messages for one sample.

    Routing is based on ``benchmark_task_type``. Payload selection is based on the
    requested condition: clean uses ``clean_payload`` and corrupted uses
    ``corrupted_payload``. A corrupted request without a corrupted payload fails
    closed instead of silently falling back to clean text.
    """
    messages, _, _ = prompt_for_sample(sample, is_clean=is_clean)
    return [dict(message) for message in messages]


def run_inference(
    samples: list[Sample],
    config: PipelineConfig,
    client: MockAPIClient | RealAPIClient,
    logger_instance: InferenceLogger,
    *,
    is_clean: bool = True,
    dry_run: bool = True,
) -> list[InferenceResult]:
    """Run inference on a list of samples using the provided client.

    Parameters
    ----------
    samples :
        All samples to process. Each sample gets exactly one API call.
    config :
        Run configuration (output path, model name, temperature, etc.).
    client :
        API client (real or mock).
    logger_instance :
        InferenceLogger that handles JSONL writing and resume logic.
    is_clean :
        Whether this run is for clean (True) or corrupted (False) samples.
        Included in the request_key so clean/corrupted are distinct.
    dry_run :
        If True, log the call but do not call ``client.call()``.

    Returns
    -------
    list[InferenceResult]
        One result per sample that was actually processed (skips not included).
    """
    results: list[InferenceResult] = []
    skip_count = 0
    error_count = 0

    for sample in samples:
        messages = _build_label_only_messages(sample, is_clean=is_clean)
        request_key = _build_request_key(sample, config, is_clean, messages)
        dataset_name, corruption, level, corruption_status, su = _extract_sample_metadata(
            sample
        )
        effective_max_new_tokens = config.max_new_tokens
        if effective_max_new_tokens is None:
            _, _, effective_max_new_tokens = prompt_for_sample(
                sample, is_clean=is_clean
            )

        if logger_instance.is_done(request_key):
            skip_count += 1
            logger.debug(
                "Skipping already-done request_key=%r (sample_id=%r)",
                request_key,
                sample.sample_id,
            )
            continue

        try:
            if dry_run:
                result = InferenceResult(
                    sample_id=sample.sample_id,
                    sample_uid=su,
                    request_key=request_key,
                    model=config.model_name,
                    raw_content=f"[dry-run] {sample.sample_id}",
                    finish_reason="dry_run",
                    latency_ms=0.0,
                )
                logger_instance.log(
                    request_key=request_key,
                    sample_id=sample.sample_id,
                    model=config.model_name,
                    messages=messages,
                    raw_response={"content": result.raw_content},
                    latency_ms=result.latency_ms,
                    dataset_name=dataset_name,
                    corruption=corruption,
                    level=level,
                    corruption_status=corruption_status,
                    sample_uid=su,
                )
            else:
                api_resp = client.call(
                    messages=messages,
                    sample_id=sample.sample_id,
                    temperature=config.temperature,
                    max_tokens=effective_max_new_tokens,
                )
                content = getattr(api_resp, "content", "")
                finish_reason = getattr(api_resp, "finish_reason", "unknown")
                latency = getattr(api_resp, "latency_ms", 0.0)
                model_name = getattr(api_resp, "model", config.model_name)
                raw_dict = (
                    api_resp.to_openai_dict()
                    if hasattr(api_resp, "to_openai_dict")
                    else {"content": content}
                )
                result = InferenceResult(
                    sample_id=sample.sample_id,
                    sample_uid=su,
                    request_key=request_key,
                    model=model_name,
                    raw_content=content,
                    finish_reason=finish_reason,
                    latency_ms=latency,
                )
                logger_instance.log(
                    request_key=request_key,
                    sample_id=sample.sample_id,
                    model=model_name,
                    messages=messages,
                    raw_response=raw_dict,
                    latency_ms=latency,
                    dataset_name=dataset_name,
                    corruption=corruption,
                    level=level,
                    corruption_status=corruption_status,
                    sample_uid=su,
                )

            results.append(result)

        except Exception as exc:  # noqa: BLE001
            error_msg = f"{exc.__class__.__name__}: {exc}"
            logger.error(
                "Inference failed for request_key=%r (sample_id=%r): %s",
                request_key,
                sample.sample_id,
                error_msg,
            )
            error_count += 1
            logger_instance.log(
                request_key=request_key,
                sample_id=sample.sample_id,
                model=config.model_name,
                messages=messages,
                raw_response={},
                latency_ms=0.0,
                error=error_msg,
                dataset_name=dataset_name,
                corruption=corruption,
                level=level,
                corruption_status=corruption_status,
                sample_uid=su,
            )
            results.append(
                InferenceResult(
                    sample_id=sample.sample_id,
                    sample_uid=su,
                    request_key=request_key,
                    model=config.model_name,
                    raw_content="",
                    finish_reason="error",
                    latency_ms=0.0,
                    error=error_msg,
                )
            )

    logger.info(
        "Inference run complete: total=%d done=%d skipped=%d errors=%d",
        len(samples),
        len(results),
        skip_count,
        error_count,
    )
    return results
