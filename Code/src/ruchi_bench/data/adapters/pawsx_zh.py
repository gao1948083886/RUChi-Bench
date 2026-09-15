"""PAWS-X zh adapter.

PAWS-X zh is a paraphrase-identification dataset from Google (Yang et al., 2019).
The Chinese (zh) test set is machine-translated from English.
https://github.com/google-research-datasets/paws

Labels (official): 0 = not paraphrase, 1 = paraphrase.
Labels are preserved as-is: gold_label=0/1, gold_label_text="different_meaning"/"paraphrase".
No label inversion is applied.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ruchi_bench.data.adapters.base import AdaptationResult, AdaptationStatus, BaseAdapter
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.labels import PAWSX_LABEL_DECODE
from ruchi_bench.schema.payloads import PairPayload
from ruchi_bench.schema.sample import Sample

# PAWS-X zh field names — from the official CSV headers.
_FIELD_SENTENCE1 = "sentence1"
_FIELD_SENTENCE2 = "sentence2"
_FIELD_LABEL = "label"


@dataclass(frozen=True, slots=True)
class PAWSXAdapter(BaseAdapter):
    """Adapt PAWS-X zh records to :class:`Sample`."""

    dataset_name: str = "pawsx_zh"
    dataset_version: str = "final"
    split: str = "test"

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        try:
            s1 = self._text(raw, _FIELD_SENTENCE1)
            s2 = self._text(raw, _FIELD_SENTENCE2)
            label_val = self._int(raw, _FIELD_LABEL)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=str(exc),
            )

        if label_val not in PAWSX_LABEL_DECODE:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"PAWS-X label {label_val!r} not in {set(PAWSX_LABEL_DECODE)}",
            )

        gold_text = PAWSX_LABEL_DECODE[label_val]
        return AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=Sample(
                sample_id=self._id(raw),
                source_sample_id=self._id(raw),
                dataset_name=DatasetName.PAWSX_ZH,
                dataset_version=self.dataset_version,
                split=SplitName.TEST,
                benchmark_task_type=BenchmarkTaskType.PAIR_PARAPHRASE,
                language=Language.ZH,
                clean_payload=PairPayload(text_a=s1, text_b=s2),
                gold_label=label_val,
                gold_label_text=gold_text,
                target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_B),
                label_projection_status=ProjectionStatus.NOT_APPLICABLE,
                corruption_applied=False,
                change_count=0,
                created_at=self._now(),
            ),
        )

    # ------------------------------------------------------------------
    # Shared helpers (same pattern across all five adapters)
    # ------------------------------------------------------------------
    def _id(self, raw: dict[str, object]) -> str:
        val = raw.get("id")
        if val is None:
            raise ValueError("missing 'id' field")
        s = str(val).strip()
        if not s:
            raise ValueError("blank 'id' field")
        return s

    def _text(self, raw: dict[str, object], field: str) -> str:
        val = raw.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"missing or blank {field!r}")
        return val.strip()

    def _int(self, raw: dict[str, object], field: str) -> int:
        val = raw.get(field)
        if val is None:
            raise ValueError(f"missing {field!r}")
        if isinstance(val, int | float):
            return int(val)
        if isinstance(val, str):
            try:
                return int(float(val.strip()))
            except ValueError:
                pass
        raise ValueError(f"{field!r} value {val!r} is not a valid integer label")

    def _now(self) -> datetime:
        return datetime.now(tz=UTC)
