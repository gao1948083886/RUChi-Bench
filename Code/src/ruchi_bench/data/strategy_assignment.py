"""Generate the decided pilot layout: one strategy per sample, three clean-based levels."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from ruchi_bench.corruptions.registry import CORRUPTOR_REGISTRY
from ruchi_bench.corruptions.result import APPLIED
from ruchi_bench.data.run_corruptions import _corrupt_sample
from ruchi_bench.schema.enums import CorruptionName, DatasetName
from ruchi_bench.schema.sample import Sample
from ruchi_bench.schema.serialization import read_jsonl, write_jsonl

__all__ = ["assign_strategies", "strategy_assignment"]

LEVELS = ("low", "medium", "high")


def _sample_seed(base_seed: int, sample: Sample, name: CorruptionName, level: str) -> int:
    """Derive a process-stable non-negative seed (never Python's salted hash())."""
    key = f"{base_seed}|{sample.dataset_name.value}|{sample.sample_id}|{name.value}|{level}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**31)


def _eligible_names(sample: Sample, *, seed: int) -> list[CorruptionName]:
    """Preflight every strategy at all three levels and keep only fully usable ones."""
    eligible: list[CorruptionName] = []
    for name, corruptor in CORRUPTOR_REGISTRY.items():
        results = [
            corruptor.corrupt(
                sample,
                seed=_sample_seed(seed, sample, name, level),
                level=level,
            )
            for level in LEVELS
        ]
        if all(result.status is APPLIED for result in results):
            eligible.append(name)
    return eligible


def assign_strategies(samples: Iterable[Sample], *, seed: int = 42) -> dict[str, CorruptionName]:
    """Assign one eligible strategy per sample with approximately balanced counts.

    Balancing is performed independently within each dataset. A sample with no
    eligible strategy raises immediately; no unrelated strategy is silently
    substituted. The returned mapping is keyed by ``sample_id``.
    """
    grouped: dict[DatasetName, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.dataset_name].append(sample)

    assignments: dict[str, CorruptionName] = {}
    for dataset, dataset_samples in grouped.items():
        rng = random.Random(seed + list(DatasetName).index(dataset))
        order = list(dataset_samples)
        rng.shuffle(order)
        eligible_by_id = {sample.sample_id: _eligible_names(sample, seed=seed) for sample in order}
        unavailable = [sample.sample_id for sample in order if not eligible_by_id[sample.sample_id]]
        if unavailable:
            raise ValueError(
                f"no eligible corruption for {dataset.value} sample(s): {unavailable[:5]}"
            )

        counts: Counter[CorruptionName] = Counter()
        for sample in order:
            candidates = eligible_by_id[sample.sample_id]
            minimum = min(counts[name] for name in candidates)
            tied = [name for name in candidates if counts[name] == minimum]
            chosen = rng.choice(tied)
            assignments[sample.sample_id] = chosen
            counts[chosen] += 1
    return assignments


def strategy_assignment(
    clean_path: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    levels: tuple[str, ...] = LEVELS,
) -> dict[str, object]:
    """Write assigned pilot corruptions under ``corrupted/<dataset>/<strategy>/``.

    Each level invokes the corruptor with the original clean :class:`Sample`, never
    with another level's output. The data set directories remain separate.
    """
    samples = list(read_jsonl(clean_path))
    assignments = assign_strategies(samples, seed=seed)
    manifest: dict[str, object] = {
        "started_at": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "levels": list(levels),
        "assignment_counts": Counter(name.value for name in assignments.values()),
        "assignments": {sample_id: name.value for sample_id, name in assignments.items()},
        "files": [],
    }
    buckets: dict[tuple[DatasetName, CorruptionName, str], list[Sample]] = defaultdict(list)
    statuses: Counter[str] = Counter()
    for sample in samples:
        name = assignments[sample.sample_id]
        for level in levels:
            corrupted, status = _corrupt_sample(
                sample,
                name,
                _sample_seed(seed, sample, name, level),
                level,
            )
            statuses[status] += 1
            if corrupted is not None:
                buckets[(sample.dataset_name, name, level)].append(corrupted)

    for (dataset, name, level), records in sorted(
        buckets.items(), key=lambda item: tuple(str(x) for x in item[0])
    ):
        path = output_dir / "corrupted" / dataset.value / name.value / f"{level}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(records, path)
        files = manifest["files"]
        assert isinstance(files, list)
        files.append({"path": str(path), "written": len(records)})
    manifest["statuses"] = dict(statuses)
    manifest["finished_at"] = datetime.now(tz=UTC).isoformat()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pilot_corruption_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    return manifest
