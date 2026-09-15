"""Strict Phase 09 metrics helpers for RUChi-Bench.

This module deliberately requires explicit sample identity on inference records.
It does not infer datasets from row order, filename position, or bare sample_id.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ruchi_bench.metrics.parsers import parse_label

__all__ = [
    "CorruptionResultSummary",
    "DatasetMetrics",
    "MetricResult",
    "SampleCounts",
    "_load_samples_from_jsonl",
    "compute_metrics",
    "format_metrics_table",
    "load_inference_records",
    "load_pilot_samples",
    "normalize_label",
    "parse_model_label",
]

logger = logging.getLogger(__name__)

SampleKey = tuple[str, str, str | None, str | None]


@dataclass
class SampleCounts:
    """Counts of loaded pilot samples."""

    clean: int = 0
    corrupted: int = 0

    @property
    def total(self) -> int:
        return self.clean + self.corrupted


@dataclass
class CorruptionResultSummary:
    """Per-(dataset, corruption, level) bucket of corrupted results."""

    dataset: str
    corruption: str
    level: str
    eligible: int = 0
    applied: int = 0
    not_applied: int = 0
    invalid: int = 0
    responses: int = 0
    parse_failures: int = 0
    correct: int = 0
    api_errors: int = 0
    crr_denominator: int = 0
    crr_numerator: int = 0

    @property
    def corruption_success_rate(self) -> float:
        return self.applied / self.eligible if self.eligible else 0.0

    @property
    def parse_failure_rate(self) -> float:
        return self.parse_failures / self.responses if self.responses else 0.0

    @property
    def crr(self) -> float:
        if self.crr_denominator == 0:
            return float("nan")
        return self.crr_numerator / self.crr_denominator


@dataclass
class DatasetMetrics:
    """Metrics aggregated per dataset."""

    dataset: str
    clean_total: int = 0
    clean_correct: int = 0
    clean_parse_fail: int = 0
    corruption_summaries: dict[str, CorruptionResultSummary] = field(
        default_factory=dict
    )

    @property
    def clean_accuracy(self) -> float:
        return self.clean_correct / self.clean_total if self.clean_total else 0.0

    @property
    def clean_parse_failure_rate(self) -> float:
        return self.clean_parse_fail / self.clean_total if self.clean_total else 0.0


@dataclass
class MetricResult:
    """Top-level metrics result for one validated pilot run."""

    run_id: str
    model: str
    total_samples: int
    clean_samples: int
    corrupted_samples: int
    per_dataset: dict[str, DatasetMetrics] = field(default_factory=dict)
    inference_total: int = 0
    inference_clean: int = 0
    inference_corrupted: int = 0
    overall_clean_accuracy: float = 0.0
    overall_parse_failure_rate: float = 0.0
    clean_parse_failure_rate: float = 0.0
    corrupted_parse_failure_rate: float = 0.0
    crr: dict[str, float] = field(default_factory=dict)
    macro_crr: dict[str, float] = field(default_factory=dict)
    corruption_crr: dict[str, float] = field(default_factory=dict)
    corruption_success_rate: dict[str, float] = field(default_factory=dict)
    corruption_drop: dict[str, float] = field(default_factory=dict)


def _load_samples_from_jsonl(
    path: Path,
    *,
    corruption_name: str | None = None,
    corruption_level: str | None = None,
) -> dict[SampleKey, dict[str, Any]]:
    """Load JSONL samples by explicit composite identity."""

    samples: dict[SampleKey, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            if not raw.strip():
                continue
            sample = json.loads(raw)
            try:
                key = (
                    str(sample["dataset_name"]),
                    str(sample["sample_id"]),
                    corruption_name,
                    corruption_level,
                )
            except KeyError as exc:
                raise ValueError(f"{path}: line {line_no} missing {exc.args[0]!r}") from exc
            if key in samples:
                raise ValueError(f"{path}: duplicate sample identity {key!r}")
            samples[key] = sample
    return samples


def load_pilot_samples(
    clean_path: Path, corrupted_dir: Path
) -> tuple[dict[SampleKey, dict[str, Any]], SampleCounts]:
    """Load clean and perturbed samples without identity guessing.

    The loader accepts both the legacy flat pilot layout and the released full-test
    layout (one clean file per dataset and dataset/strategy/level perturbation files).
    The public function name is retained for compatibility with earlier callers.
    """

    all_samples: dict[SampleKey, dict[str, Any]] = {}
    clean_count = 0
    corrupted_count = 0

    if clean_path.is_dir():
        combined_clean = clean_path / "all_clean.jsonl"
        clean_files = (
            [combined_clean]
            if combined_clean.exists()
            else sorted(clean_path.glob("*_clean.jsonl"))
        )
    else:
        clean_files = [clean_path]

    for clean_file in clean_files:
        if not clean_file.exists():
            continue
        loaded = _load_samples_from_jsonl(clean_file)
        overlap = set(all_samples).intersection(loaded)
        if overlap:
            raise ValueError(
                f"{clean_file}: duplicate sample identities {sorted(overlap)[:3]!r}"
            )
        all_samples.update(loaded)
        clean_count += len(loaded)
    if not clean_files or clean_count == 0:
        logger.warning("No clean JSONL files found at: %s", clean_path)

    if corrupted_dir.exists():
        top_level_dirs = [
            path for path in sorted(corrupted_dir.iterdir()) if path.is_dir()
        ]
        nested_layout = any(
            any(strategy_dir.glob("*.jsonl"))
            for dataset_dir in top_level_dirs
            for strategy_dir in dataset_dir.iterdir()
            if strategy_dir.is_dir()
        )

        if nested_layout:
            level_files = (
                (dataset_dir, strategy_dir, level_file)
                for dataset_dir in top_level_dirs
                for strategy_dir in sorted(dataset_dir.iterdir())
                if strategy_dir.is_dir()
                for level_file in sorted(strategy_dir.glob("*.jsonl"))
            )
        else:
            level_files = (
                (None, strategy_dir, level_file)
                for strategy_dir in top_level_dirs
                for level_file in sorted(strategy_dir.glob("*.jsonl"))
            )

        for _dataset_dir, strategy_dir, level_file in level_files:
            loaded = _load_samples_from_jsonl(
                level_file,
                corruption_name=strategy_dir.name,
                corruption_level=level_file.stem,
            )
            overlap = set(all_samples).intersection(loaded)
            if overlap:
                raise ValueError(
                    f"{level_file}: duplicate sample identities {sorted(overlap)[:3]!r}"
                )
            all_samples.update(loaded)
            corrupted_count += len(loaded)

    return all_samples, SampleCounts(clean=clean_count, corrupted=corrupted_count)


def _parse_file_bucket(path: Path, manifest_path: Path | None) -> tuple[str | None, str | None]:
    run_id = None
    if manifest_path is not None and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            run_id = json.load(fh).get("run_id")

    key = path.stem
    if run_id and key.startswith(f"{run_id}_"):
        key = key[len(run_id) + 1 :]
    if key == "clean":
        return None, None
    if key.endswith("_clean"):
        return None, None

    for corruption in (
        "add_noise", "red_char", "red_word", "homo", "swap", "del", "vis",
        "tpwr", "vscr", "cr", "wr", "muni",
    ):
        marker = f"_{corruption}_"
        if key.startswith(f"{corruption}_"):
            return corruption, key[len(corruption) + 1 :]
        if marker in key:
            return corruption, key.rsplit(marker, 1)[1]
    raise ValueError(f"Cannot parse inference file bucket from {path.name!r}")


def load_inference_records(
    jsonl_paths: list[Path],
    *,
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load inference records and require explicit dataset identity.

    Old Phase 08 records without ``dataset_name`` are intentionally rejected
    because row-order reconstruction is not a valid metrics join.
    """

    records: list[dict[str, Any]] = []
    for path in jsonl_paths:
        corruption_name, corruption_level = _parse_file_bucket(path, manifest_path)
        with path.open(encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, start=1):
                if not raw.strip():
                    continue
                record = json.loads(raw)
                if "dataset_name" not in record:
                    raise ValueError(
                        f"{path}: line {line_no} missing dataset_name; "
                        "refusing row-order dataset inference"
                    )
                if "sample_id" not in record:
                    raise ValueError(f"{path}: line {line_no} missing sample_id")
                record["corruption_name"] = corruption_name
                record["corruption_level"] = corruption_level
                records.append(record)
    return records


def normalize_label(text: str) -> str:
    """Compatibility helper for legacy tests."""

    return str(text).strip().lower()


def _extract_response_text(raw_response: Any) -> str:
    if isinstance(raw_response, dict):
        choices = raw_response.get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content")
            if content is not None:
                return str(content)
        content = raw_response.get("content")
        if content is not None:
            return str(content)
    if isinstance(raw_response, str):
        return raw_response
    return ""


def _allowed_c3_letters(sample: dict[str, Any]) -> str:
    payload = sample.get("clean_payload", {})
    options = payload.get("options", ())
    return "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(options) or 4]


def parse_model_label(record: dict[str, Any]) -> str | None:
    """Parse a model label with strict-v2 if task metadata is available."""

    text = _extract_response_text(record.get("raw_response", {}))
    task_type = record.get("benchmark_task_type")
    if task_type is None:
        return None
    parsed = parse_label(
        text,
        str(task_type),
        c3_allowed_letters=str(record.get("c3_allowed_letters", "ABCD")),
    )
    return parsed.label


def _gold_label(sample: dict[str, Any]) -> str:
    task_type = str(sample.get("benchmark_task_type", ""))
    raw = sample.get("gold_label")
    if task_type == "multiple_choice_mrc":
        if raw is None:
            raise ValueError("C3 sample missing gold_label")
        return chr(ord("A") + int(raw))
    return str(raw)


def _is_applied(sample: dict[str, Any]) -> bool:
    return bool(sample.get("corruption_applied", True))


def _record_with_sample_metadata(
    record: dict[str, Any], sample: dict[str, Any]
) -> dict[str, Any]:
    merged = dict(record)
    task_type = sample.get("benchmark_task_type", sample.get("task_type"))
    merged["benchmark_task_type"] = task_type
    if str(task_type) == "multiple_choice_mrc":
        merged["c3_allowed_letters"] = _allowed_c3_letters(sample)
    return merged


def _is_correct(record: dict[str, Any], sample: dict[str, Any]) -> tuple[bool, bool]:
    enriched = _record_with_sample_metadata(record, sample)
    parsed = parse_model_label(enriched)
    if parsed is None:
        return False, True
    return parsed == _gold_label(sample), False


def compute_metrics(
    samples: dict[SampleKey, dict[str, Any]],
    sample_counts: SampleCounts,
    records: list[dict[str, Any]],
    model_name: str = "unknown",
    run_id: str = "unknown",
    inference_clean: int = 0,
    inference_corrupted: int = 0,
) -> MetricResult:
    """Compute strict metrics after explicit identity has been validated."""

    records_by_key: dict[SampleKey, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["dataset_name"]),
            str(record["sample_id"]),
            record.get("corruption_name"),
            record.get("corruption_level"),
        )
        records_by_key[key].append(record)

    per_dataset = {
        dataset: DatasetMetrics(dataset=dataset)
        for dataset in sorted({key[0] for key in samples})
    }
    clean_correct_by_sample: dict[tuple[str, str], bool] = {}

    for key, sample in samples.items():
        dataset, sample_id, corruption_name, corruption_level = key
        if corruption_name is not None or corruption_level is not None:
            continue
        metrics = per_dataset[dataset]
        metrics.clean_total += 1
        recs = records_by_key.get(key, [])
        if not recs:
            metrics.clean_parse_fail += 1
            clean_correct_by_sample[(dataset, sample_id)] = False
            continue
        correct, parse_failed = _is_correct(recs[0], sample)
        metrics.clean_correct += int(correct)
        metrics.clean_parse_fail += int(parse_failed)
        clean_correct_by_sample[(dataset, sample_id)] = correct

    for key, sample in samples.items():
        dataset, sample_id, corruption_name, corruption_level = key
        if corruption_name is None:
            continue
        bucket = f"{corruption_name}_{corruption_level}"
        metrics = per_dataset[dataset]
        summary = metrics.corruption_summaries.setdefault(
            bucket,
            CorruptionResultSummary(
                dataset=dataset,
                corruption=corruption_name,
                level=corruption_level or "n/a",
            ),
        )
        summary.eligible += 1
        applied = _is_applied(sample)
        summary.applied += int(applied)
        summary.not_applied += int(not applied)
        if not applied:
            continue

        recs = records_by_key.get(key, [])
        if not recs:
            summary.api_errors += 1
            continue
        summary.responses += 1
        correct, parse_failed = _is_correct(recs[0], sample)
        summary.correct += int(correct)
        summary.parse_failures += int(parse_failed)

        if clean_correct_by_sample.get((dataset, sample_id), False):
            summary.crr_denominator += 1
            summary.crr_numerator += int(correct)

    total_clean_correct = sum(m.clean_correct for m in per_dataset.values())
    total_clean_parse_fail = sum(m.clean_parse_fail for m in per_dataset.values())
    total_corrupted_parse_fail = sum(
        s.parse_failures
        for metrics in per_dataset.values()
        for s in metrics.corruption_summaries.values()
    )
    total_records = len(records)
    if inference_clean == 0:
        inference_clean = sum(1 for r in records if r.get("corruption_name") is None)
    if inference_corrupted == 0:
        inference_corrupted = total_records - inference_clean

    crr: dict[str, float] = {}
    macro_by_corruption: dict[str, list[float]] = defaultdict(list)
    success_counts: dict[str, tuple[int, int]] = {}
    drop_counts: dict[str, tuple[int, int]] = {}

    for dataset, metrics in per_dataset.items():
        for bucket, summary in metrics.corruption_summaries.items():
            crr_key = f"{dataset}:{bucket}"
            crr[crr_key] = summary.crr
            if not math.isnan(summary.crr):
                macro_by_corruption[summary.corruption].append(summary.crr)

            applied_count, eligible_count = success_counts.get(summary.corruption, (0, 0))
            success_counts[summary.corruption] = (
                applied_count + summary.applied,
                eligible_count + summary.eligible,
            )
            correct_count, response_count = drop_counts.get(summary.corruption, (0, 0))
            drop_counts[summary.corruption] = (
                correct_count + summary.correct,
                response_count + summary.responses,
            )

    macro_crr = {
        corruption: sum(values) / len(values) * 100
        for corruption, values in macro_by_corruption.items()
        if values
    }
    success_rate = {
        corruption: (applied / eligible * 100 if eligible else 0.0)
        for corruption, (applied, eligible) in success_counts.items()
    }
    clean_accuracy_pct = (
        total_clean_correct / sample_counts.clean * 100 if sample_counts.clean else 0.0
    )
    corruption_drop = {
        corruption: clean_accuracy_pct - (correct / responses * 100)
        for corruption, (correct, responses) in drop_counts.items()
        if responses
    }

    return MetricResult(
        run_id=run_id,
        model=model_name,
        total_samples=sample_counts.total,
        clean_samples=sample_counts.clean,
        corrupted_samples=sample_counts.corrupted,
        per_dataset=per_dataset,
        inference_total=total_records,
        inference_clean=inference_clean,
        inference_corrupted=inference_corrupted,
        overall_clean_accuracy=clean_accuracy_pct,
        overall_parse_failure_rate=(
            (total_clean_parse_fail + total_corrupted_parse_fail) / total_records * 100
            if total_records
            else 0.0
        ),
        clean_parse_failure_rate=(
            total_clean_parse_fail / inference_clean * 100 if inference_clean else 0.0
        ),
        corrupted_parse_failure_rate=(
            total_corrupted_parse_fail / inference_corrupted * 100
            if inference_corrupted
            else 0.0
        ),
        crr=crr,
        macro_crr=macro_crr,
        corruption_crr=macro_crr,
        corruption_success_rate=success_rate,
        corruption_drop=corruption_drop,
    )


def format_metrics_table(result: MetricResult) -> str:
    """Format strict metrics as markdown."""

    lines = [
        "# Phase 09 Metrics",
        "",
        f"Run ID: `{result.run_id}`",
        f"Model: `{result.model}`",
        "",
        "## Input Reconciliation",
        "| Field | Count |",
        "|---|---:|",
        f"| Inference records | {result.inference_total} |",
        f"| Clean inference records | {result.inference_clean} |",
        f"| Corrupted inference records | {result.inference_corrupted} |",
        f"| Clean pilot samples | {result.clean_samples} |",
        f"| Corrupted pilot samples | {result.corrupted_samples} |",
        "",
        "## Clean Accuracy",
        "| Dataset | Total | Correct | Parse Fail | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for dataset, metrics in sorted(result.per_dataset.items()):
        lines.append(
            f"| {dataset} | {metrics.clean_total} | {metrics.clean_correct} | "
            f"{metrics.clean_parse_fail} | {metrics.clean_accuracy * 100:.1f}% |"
        )
    lines.extend(
        [
            f"| Overall | {result.clean_samples} | "
            f"{sum(m.clean_correct for m in result.per_dataset.values())} | "
            f"{sum(m.clean_parse_fail for m in result.per_dataset.values())} | "
            f"{result.overall_clean_accuracy:.1f}% |",
            "",
            "## CRR By Bucket",
            "| Dataset | Corruption | Level | Denominator | Numerator | CRR |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for dataset, metrics in sorted(result.per_dataset.items()):
        for summary in sorted(
            metrics.corruption_summaries.values(),
            key=lambda s: (s.corruption, s.level),
        ):
            value = "nan" if math.isnan(summary.crr) else f"{summary.crr * 100:.1f}%"
            lines.append(
                f"| {dataset} | {summary.corruption} | {summary.level} | "
                f"{summary.crr_denominator} | {summary.crr_numerator} | {value} |"
            )
    return "\n".join(lines)
