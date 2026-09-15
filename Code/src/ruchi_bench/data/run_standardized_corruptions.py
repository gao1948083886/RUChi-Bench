"""Generate the released perturbation files from standardized JSONL inputs.

This runner deliberately does not convert records into the stricter internal
``Sample`` label model: the current pilot includes the original ASAP 3-star label,
which must remain untouched. A minimal, label-independent proxy is used only to
invoke the corruption engine; output rows preserve every clean field verbatim and
add corruption fields alongside it.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ruchi_bench.corruptions.registry import CORRUPTOR_REGISTRY
from ruchi_bench.corruptions.result import APPLIED, CorruptionResult
from ruchi_bench.corruptions.utils import build_trace_ops
from ruchi_bench.corruptions.vis import is_visual_source_eligible
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import MRCPayload, PairPayload, SingleTextPayload
from ruchi_bench.schema.sample import Sample

__all__ = ["run_standardized_corruptions", "run_standardized_pilot_corruptions"]

LEVELS = ("low", "medium", "high")
DATASET_FILES = {
    DatasetName.PAWSX_ZH: "pawsx_zh_clean.jsonl",
    DatasetName.XNLI_ZH: "xnli_zh_clean.jsonl",
    DatasetName.C3: "c3_clean.jsonl",
    DatasetName.LCQMC: "lcqmc_clean.jsonl",
    DatasetName.ASAP: "asap_clean.jsonl",
}


def _seed(base_seed: int, row: dict[str, Any], name: CorruptionName, level: str) -> int:
    key = f"{base_seed}|{row['dataset_name']}|{row['sample_id']}|{name.value}|{level}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") % (2**31)


def _proxy(row: dict[str, Any]) -> Sample:
    """Build a label-independent valid Sample solely for the corruption engine."""
    dataset = DatasetName(row["dataset_name"])
    payload = row["clean_payload"]
    kind = row["payload_kind"]
    if kind == "pair":
        clean_payload = PairPayload(**payload)
        task = {
            DatasetName.PAWSX_ZH: BenchmarkTaskType.PAIR_PARAPHRASE,
            DatasetName.XNLI_ZH: BenchmarkTaskType.NLI,
            DatasetName.LCQMC: BenchmarkTaskType.QUESTION_MATCHING,
        }[dataset]
        gold_label: int | str = 0 if dataset is not DatasetName.XNLI_ZH else "neutral"
        gold_text = "different_meaning" if dataset is not DatasetName.XNLI_ZH else "neutral"
        return Sample(
            sample_id=str(row["sample_id"]),
            source_sample_id=str(row["source_sample_id"]),
            dataset_name=dataset,
            dataset_version=str(row["dataset_version"]),
            split=SplitName.TEST,
            benchmark_task_type=task,
            language=Language.ZH,
            clean_payload=clean_payload,
            gold_label=gold_label,
            gold_label_text=gold_text,
            target_fields=tuple(PayloadField(field) for field in row["target_fields"]),
            label_projection_status=ProjectionStatus.NOT_APPLICABLE,
            corruption_applied=False,
            change_count=0,
            created_at=datetime.now(tz=UTC),
        )
    if kind == "single_text":
        return Sample(
            sample_id=str(row["sample_id"]),
            source_sample_id=str(row["source_sample_id"]),
            dataset_name=DatasetName.ASAP,
            dataset_version=str(row["dataset_version"]),
            split=SplitName.TEST,
            benchmark_task_type=BenchmarkTaskType.SENTIMENT_POLARITY,
            language=Language.ZH,
            clean_payload=SingleTextPayload(**payload),
            source_gold_label=1,
            gold_label="negative",
            gold_label_text="negative",
            target_fields=(PayloadField.TEXT_A,),
            label_projection_name="asap_polarity",
            label_projection_version="1.0.0",
            label_projection_status=ProjectionStatus.ACCEPTED,
            corruption_applied=False,
            change_count=0,
            created_at=datetime.now(tz=UTC),
        )
    if kind == "mrc":
        mrc_payload = MRCPayload(**payload)
        answer_index = 0
        return Sample(
            sample_id=str(row["sample_id"]),
            source_sample_id=str(row["source_sample_id"]),
            dataset_name=DatasetName.C3,
            dataset_version=str(row["dataset_version"]),
            split=SplitName.TEST,
            benchmark_task_type=BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
            language=Language.ZH,
            clean_payload=mrc_payload,
            gold_label=answer_index,
            gold_label_text=mrc_payload.options[answer_index],
            target_fields=(PayloadField.CONTEXT,),
            label_projection_status=ProjectionStatus.NOT_APPLICABLE,
            corruption_applied=False,
            change_count=0,
            created_at=datetime.now(tz=UTC),
        )
    raise ValueError(f"unsupported pilot payload_kind: {kind!r}")


def _cjk(text: str) -> bool:
    return any(0x3400 <= ord(char) <= 0x9FFF for char in text)


def _static_eligible(row: dict[str, Any]) -> list[CorruptionName]:
    """Fast candidate screening that does not initialize third-party augmenters."""
    payload = row["clean_payload"]
    texts = [str(payload[field]) for field in row["target_fields"]]
    has_text = any(text.strip() for text in texts)
    has_cjk = any(_cjk(text) for text in texts)
    has_visual = any(is_visual_source_eligible(char) for text in texts for char in text)
    names: list[CorruptionName] = []
    if has_cjk:
        names.extend(
            [
                CorruptionName.HOMO,
                CorruptionName.DEL,
                CorruptionName.SWAP,
                CorruptionName.RED_CHAR,
                CorruptionName.RED_WORD,
            ]
        )
    if has_visual:
        names.append(CorruptionName.VIS)
    if has_text:
        names.append(CorruptionName.ADD_NOISE)
    return names


def _stable_assignment(
    rows: list[dict[str, Any]],
    proxies: dict[str, Sample],
    *,
    seed: int,
    strict_preflight: bool = True,
) -> dict[str, CorruptionName]:
    """Balance assignments among strategies that pass the full three-level check.

    Static screening alone is insufficient for stochastic third-party augmenters:
    a strategy can be theoretically applicable yet fail to produce a valid edit at
    one level.  We therefore assign from the actual eligible set, which prevents
    the old post-hoc fallback from silently skewing the strategy counts.
    """
    eligible: dict[str, list[CorruptionName]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        static_names = _static_eligible(row)
        names = (
            [
                name
                for name in static_names
                if _corrupt_levels(row, proxies[sample_id], name, seed=seed) is not None
            ]
            if strict_preflight
            else static_names
        )
        eligible[sample_id] = names
    unavailable = [sample_id for sample_id, names in eligible.items() if not names]
    if unavailable:
        raise RuntimeError(f"no candidate strategy for sample(s): {unavailable[:5]}")

    rng = random.Random(seed)
    # Constrained rows are assigned first.  Random tie-breaking keeps the result
    # reproducible without allowing a rare strategy to be starved at the end.
    order = sorted(rows, key=lambda row: (len(eligible[str(row["sample_id"])]), rng.random()))
    counts: Counter[CorruptionName] = Counter()
    assignment: dict[str, CorruptionName] = {}
    for row in order:
        names = eligible[str(row["sample_id"])]
        minimum = min(counts[name] for name in names)
        tied = [name for name in names if counts[name] == minimum]
        chosen = rng.choice(tied)
        assignment[str(row["sample_id"])] = chosen
        counts[chosen] += 1
    return assignment


def _corrupt_row(
    row: dict[str, Any],
    proxy: Sample,
    name: CorruptionName,
    *,
    level: str,
    seed: int,
) -> dict[str, Any]:
    result: CorruptionResult = CORRUPTOR_REGISTRY[name].corrupt(proxy, seed=seed, level=level)
    if result.status is not APPLIED or result.corrupted_payload is None:
        raise RuntimeError(
            f"assigned strategy failed for {row['dataset_name']}/{row['sample_id']} "
            f"at {level}: {result.status} ({result.failure_reason})"
        )
    internal, public = build_trace_ops(name, level, seed, result.ops, proxy.clean_payload)
    output = dict(row)
    output.update(
        {
            "corrupted_payload": result.corrupted_payload.model_dump(mode="json"),
            "corruption_name": name.value,
            "corruption_level": level,
            "corruption_seed": seed,
            "corruption_applied": True,
            "change_count": result.change_count,
            "internal_change_trace": internal.model_dump(mode="json"),
            "public_redacted_trace": public.model_dump(mode="json"),
        }
    )
    return output


def _corrupt_levels(
    row: dict[str, Any],
    proxy: Sample,
    name: CorruptionName,
    *,
    seed: int,
) -> list[dict[str, Any]] | None:
    """Try all levels for one strategy; return nothing unless all three succeed."""
    records: list[dict[str, Any]] = []
    try:
        for level in LEVELS:
            records.append(
                _corrupt_row(
                    row,
                    proxy,
                    name,
                    level=level,
                    seed=_seed(seed, row, name, level),
                )
            )
    except RuntimeError:
        return None
    return records


def _read_rows(path: Path, *, expected_count: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if expected_count is not None and len(rows) != expected_count:
        raise ValueError(f"expected {expected_count} rows in {path}, found {len(rows)}")
    return rows


def run_standardized_corruptions(
    standardized_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
    schema_version: str = "standardized_perturbed",
    expected_counts: dict[str, int] | None = None,
    strict_preflight: bool = True,
    dataset_names: tuple[str, ...] | None = None,
    write_manifest: bool = True,
) -> dict[str, Any]:
    """Generate three independent corruption levels for every clean record.

    Each record receives one strategy, and that same strategy is applied at all
    three levels. Every level is generated directly from the clean payload.
    ``expected_counts`` is optional so the same implementation can validate the
    300-row pilot and the full official test sets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "started_at": datetime.now(tz=UTC).isoformat(),
        "seed": seed,
        "levels": list(LEVELS),
        "corruption_policy": {
            "description": (
                "Human-like calibrated corruption; all levels are generated "
                "independently from clean input."
            ),
            "preserve_clean_pilot": True,
            "labels_unchanged": True,
            "visual_function_words_excluded": True,
            "homophone_mispronunciation_disabled": True,
        },
        "datasets": {},
    }
    if dataset_names is None:
        selected_datasets: tuple[DatasetName, ...] = tuple(DATASET_FILES)
    else:
        selected_datasets = tuple(DatasetName(name) for name in dataset_names)
    for dataset in selected_datasets:
        filename = DATASET_FILES[dataset]
        expected_count = None if expected_counts is None else expected_counts[dataset.value]
        rows = _read_rows(standardized_dir / filename, expected_count=expected_count)
        proxies = {str(row["sample_id"]): _proxy(row) for row in rows}
        assignment = _stable_assignment(
            rows, proxies, seed=seed, strict_preflight=strict_preflight
        )
        dataset_manifest: dict[str, Any] = {
            "clean_count": len(rows),
            "assignment_counts": {},
            "files": [],
        }
        buckets: dict[tuple[CorruptionName, str], list[dict[str, Any]]] = defaultdict(list)
        final_assignment: dict[str, CorruptionName] = {}
        counts: Counter[CorruptionName] = Counter()
        for row in rows:
            sample_id = str(row["sample_id"])
            selected_name = assignment[sample_id]
            records = _corrupt_levels(
                row,
                proxies[sample_id],
                selected_name,
                seed=seed,
            )
            if records is None:
                # Fast full-test mode assigns from static eligibility first. If
                # a stochastic augmenter fails for this particular row, try
                # the remaining static candidates and keep one strategy across
                # all three levels. Strict pilot mode never reaches this path.
                alternatives = [
                    name
                    for name in _static_eligible(row)
                    if name is not selected_name
                ]
                alternatives.sort(key=lambda name: (counts[name], name.value))
                for candidate in alternatives:
                    candidate_records = _corrupt_levels(
                        row, proxies[sample_id], candidate, seed=seed
                    )
                    if candidate_records is not None:
                        selected_name = candidate
                        records = candidate_records
                        break
                if records is None:
                    raise RuntimeError(
                        f"all candidate strategies failed for "
                        f"{dataset.value}/{sample_id}"
                    )
            final_assignment[sample_id] = selected_name
            counts[selected_name] += 1
            for level, record in zip(LEVELS, records, strict=True):
                buckets[(selected_name, level)].append(record)
        dataset_manifest["assignment_counts"] = Counter(
            name.value for name in final_assignment.values()
        )
        for (name, level), records in sorted(
            buckets.items(), key=lambda item: (item[0][0].value, item[0][1])
        ):
            path = output_dir / dataset.value / name.value / f"{level}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
                    )
            dataset_manifest["files"].append({"path": str(path), "count": len(records)})
        manifest["datasets"][dataset.value] = dataset_manifest
    manifest["finished_at"] = datetime.now(tz=UTC).isoformat()
    if write_manifest:
        (output_dir / "corruption_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
        )
    return manifest


def run_standardized_pilot_corruptions(
    standardized_dir: Path,
    output_dir: Path,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """Backward-compatible 300-row pilot wrapper."""
    return run_standardized_corruptions(
        standardized_dir,
        output_dir,
        seed=seed,
        schema_version="pilot_perturbed",
        expected_counts={dataset.value: 300 for dataset in DATASET_FILES},
    )
