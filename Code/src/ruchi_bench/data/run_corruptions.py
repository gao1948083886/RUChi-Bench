"""Run every registered corruption on a corpus of clean samples.

Produces corrupted JSONL files (one per corruption) alongside a manifest
recording counts, seeds, and parameters.
"""

from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ruchi_bench.corruptions.base import BaseCorruptor
from ruchi_bench.corruptions.registry import CORRUPTOR_REGISTRY
from ruchi_bench.corruptions.result import (
    APPLIED,
    CorruptionResult,
)
from ruchi_bench.corruptions.utils import build_trace_ops
from ruchi_bench.schema.enums import CorruptionName
from ruchi_bench.schema.sample import Sample
from ruchi_bench.schema.serialization import write_jsonl

__all__ = ["run_all_corruptions"]


# Level → parameter name passed to each corruptor.
_LEVELS = ("low", "medium", "high")

# Registry of corruptors, keyed by name.
_CORRUPTORS: dict[CorruptionName, BaseCorruptor] = CORRUPTOR_REGISTRY


def _corrupt_sample(
    sample: Sample,
    corruption_name: CorruptionName,
    seed: int,
    level: str,
) -> tuple[Sample | None, str]:
    """Apply one corruption to one sample.

    Returns (corrupted_sample, status_str).
    ``status_str`` is one of ``APPLIED``, ``NOT_APPLIED``, ``INVALID``.
    On APPLIED, ``corrupted_sample`` is the corrupted Sample.
    On NOT_APPLIED or INVALID, ``corrupted_sample`` is None.
    """
    corruptor = _CORRUPTORS[corruption_name]
    result: CorruptionResult = corruptor.corrupt(sample, seed=seed, level=level)

    if result.status is APPLIED:
        assert result.corrupted_payload is not None
        # Build traces from ops using the shared utility
        internal_trace, public_trace = build_trace_ops(
            corruption_name=result.corruption_name,
            corruption_level=level,
            seed=seed,
            ops=result.ops,
            clean_payload=sample.clean_payload,
        )
        # Build the corrupted Sample from the original + corruption result
        corrupted = Sample(
            sample_id=sample.sample_id,
            source_sample_id=sample.source_sample_id,
            dataset_name=sample.dataset_name,
            dataset_version=sample.dataset_version,
            split=sample.split,
            benchmark_task_type=sample.benchmark_task_type,
            language=sample.language,
            clean_payload=sample.clean_payload,
            source_gold_label=sample.source_gold_label,
            gold_label=sample.gold_label,
            gold_label_text=sample.gold_label_text,
            target_fields=sample.target_fields,
            label_projection_name=sample.label_projection_name,
            label_projection_version=sample.label_projection_version,
            label_projection_status=sample.label_projection_status,
            corrupted_payload=result.corrupted_payload,
            corruption_name=result.corruption_name,
            corruption_level=level,
            corruption_seed=seed,
            corruption_applied=True,
            change_count=result.change_count,
            internal_change_trace=internal_trace,
            public_redacted_trace=public_trace,
            created_at=sample.created_at,
        )
        return corrupted, "APPLIED"
    else:
        return None, str(result.status)


def run_all_corruptions(
    clean_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    levels: tuple[str, ...] = _LEVELS,
) -> dict[str, object]:
    """Apply all seven corruptions to all clean samples.

    For each corruption, creates ``corrupted/<corruption_name>/<level>.jsonl``
    containing all successfully corrupted samples.

    Parameters
    ----------
    clean_dir:
        Directory containing ``clean/<dataset>.jsonl`` files (one file per dataset,
        one JSON object per line). Also expects ``clean/all_clean.jsonl``.
    output_dir:
        Root output directory. Created if it does not exist.
    seed:
        Base seed for deterministic corruption. Each corruption/dataset/level
        combination uses ``seed + hash(combo)`` to derive a stable but distinct seed.
    levels:
        Tuple of corruption levels to apply. Defaults to ``("low", "medium", "high")``.

    Returns
    -------
    dict
        Run manifest with per-corruption and per-level counts.
    """
    manifest: dict[str, Any] = {
        "started_at": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "levels": list(levels),
        "corruptions": {},
    }

    for corruption_name in CorruptionName:
        corruption_dir = output_dir / "corrupted" / corruption_name.value
        corruption_dir.mkdir(parents=True, exist_ok=True)
        manifest["corruptions"][corruption_name.value] = {}

        for level in levels:
            rng = random.Random(seed)
            # Determine a stable seed offset for this (corruption, level) pair
            combo_seed_offset = rng.randint(0, 2**31 - 1)

            corrupted_samples: list[Sample] = []
            applied_count = 0
            not_applied_count = 0
            invalid_count = 0

            # Read all clean samples (from all datasets combined)
            all_clean_path = clean_dir / "all_clean.jsonl"
            if all_clean_path.exists():
                from ruchi_bench.schema.serialization import read_jsonl

                for clean_sample in read_jsonl(all_clean_path):
                    # Derive a stable per-sample seed
                    s_seed = combo_seed_offset + hash((clean_sample.sample_id,)) % (2**31)

                    corrupted, status_str = _corrupt_sample(
                        clean_sample, corruption_name, s_seed, level
                    )
                    if status_str == "APPLIED":
                        assert corrupted is not None
                        corrupted_samples.append(corrupted)
                        applied_count += 1
                    elif status_str == "NOT_APPLIED":
                        not_applied_count += 1
                    else:
                        invalid_count += 1

            # Write corrupted samples to JSONL
            level_path = corruption_dir / f"{level}.jsonl"
            if corrupted_samples:
                result = write_jsonl(iter(corrupted_samples), level_path)
                manifest["corruptions"][corruption_name.value][level] = {
                    "path": str(level_path),
                    "applied": applied_count,
                    "not_applied": not_applied_count,
                    "invalid": invalid_count,
                    "written": result.count,
                }
            else:
                manifest["corruptions"][corruption_name.value][level] = {
                    "path": str(level_path),
                    "applied": applied_count,
                    "not_applied": not_applied_count,
                    "invalid": invalid_count,
                    "written": 0,
                }

    manifest["finished_at"] = datetime.now(tz=UTC).isoformat()
    manifest_path = output_dir / "corruption_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
