"""LCQMC question-matching adapter.

LCQMC is from Harbin Institute of Technology (Shao et al., 2018).
https://github.com/P01son4745/lcqmc
http://icrc.hitsz.edu.cn/Article/show/171.html

LCQMC is application-gated. The fixture in this package uses only synthetic
text that is entirely self-authored; no real LCQMC text is included.

Labels (official): 0 = not similar, 1 = similar
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
from ruchi_bench.schema.labels import LCQMC_LABEL_DECODE
from ruchi_bench.schema.payloads import PairPayload
from ruchi_bench.schema.sample import Sample

# LCQMC field names — from the official format.
# Both ``sentence1`` and ``question1`` variants appear in the wild; try both.
_FIELD_SENTENCE1 = "sentence1"
_FIELD_SENTENCE2 = "sentence2"
_FIELD_QUESTION1 = "question1"
_FIELD_QUESTION2 = "question2"
_FIELD_LABEL = "label"


@dataclass(frozen=True, slots=True)
class LCQMCAdapter(BaseAdapter):
    """Adapt LCQMC records to :class:`Sample`."""

    dataset_name: str = "lcqmc"
    dataset_version: str = "v1"
    split: str = "test"

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        try:
            s1 = self._text_either(raw, _FIELD_SENTENCE1, _FIELD_QUESTION1)
            s2 = self._text_either(raw, _FIELD_SENTENCE2, _FIELD_QUESTION2)
            label_val = self._int(raw, _FIELD_LABEL)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=str(exc),
            )

        if label_val not in LCQMC_LABEL_DECODE:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"LCQMC label {label_val!r} not in {set(LCQMC_LABEL_DECODE)}",
            )

        gold_text = LCQMC_LABEL_DECODE[label_val]
        return AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=Sample(
                sample_id=self._id(raw),
                source_sample_id=self._id(raw),
                dataset_name=DatasetName.LCQMC,
                dataset_version=self.dataset_version,
                split=SplitName.TEST,
                benchmark_task_type=BenchmarkTaskType.QUESTION_MATCHING,
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

    def _id(self, raw: dict[str, object]) -> str:
        return str(raw.get("id", "") or "")

    def _text_either(self, raw: dict[str, object], field_a: str, field_b: str) -> str:
        val = raw.get(field_a)
        if isinstance(val, str) and val.strip():
            return val.strip()
        val = raw.get(field_b)
        if isinstance(val, str) and val.strip():
            return val.strip()
        raise ValueError(f"neither {field_a!r} nor {field_b!r} is a non-empty string")

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
