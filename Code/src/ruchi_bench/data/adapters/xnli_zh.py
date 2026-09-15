"""XNLI zh adapter.

XNLI is from Facebook AI Research (Conneau et al., 2018).
https://github.com/facebookresearch/XNLI

Labels (official): entailment, neutral, contradiction
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
from ruchi_bench.schema.labels import XNLI_LABELS
from ruchi_bench.schema.payloads import PairPayload
from ruchi_bench.schema.sample import Sample

# XNLI field names — from the official JSON format.
_FIELD_SENTENCE1 = "sentence1"
_FIELD_SENTENCE2 = "sentence2"
_FIELD_GOLD_LABEL = "gold_label"


@dataclass(frozen=True, slots=True)
class XNLIAdapter(BaseAdapter):
    """Adapt XNLI zh records to :class:`Sample`."""

    dataset_name: str = "xnli_zh"
    dataset_version: str = "1.0"
    split: str = "test"

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        try:
            s1 = self._text(raw, _FIELD_SENTENCE1)
            s2 = self._text(raw, _FIELD_SENTENCE2)
            gold_label = self._label(raw, _FIELD_GOLD_LABEL)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=str(exc),
            )

        if gold_label not in XNLI_LABELS:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"XNLI gold_label {gold_label!r} not in {XNLI_LABELS}",
            )

        return AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=Sample(
                sample_id=self._id(raw),
                source_sample_id=self._id(raw),
                dataset_name=DatasetName.XNLI_ZH,
                dataset_version=self.dataset_version,
                split=SplitName.TEST,
                benchmark_task_type=BenchmarkTaskType.NLI,
                language=Language.ZH,
                clean_payload=PairPayload(text_a=s1, text_b=s2),
                gold_label=gold_label,
                gold_label_text=gold_label,
                target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_B),
                label_projection_status=ProjectionStatus.NOT_APPLICABLE,
                corruption_applied=False,
                change_count=0,
                created_at=self._now(),
            ),
        )

    def _id(self, raw: dict[str, object]) -> str:
        val = raw.get("pairID")
        if val is None:
            raise ValueError("missing 'pairID' field")
        s = str(val).strip()
        if not s:
            raise ValueError("blank 'pairID' field")
        return s

    def _text(self, raw: dict[str, object], field: str) -> str:
        val = raw.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"missing or blank {field!r}")
        return val.strip()

    def _label(self, raw: dict[str, object], field: str) -> str:
        val = raw.get(field)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"missing or blank {field!r}")
        return val.strip()

    def _now(self) -> datetime:
        return datetime.now(tz=UTC)
