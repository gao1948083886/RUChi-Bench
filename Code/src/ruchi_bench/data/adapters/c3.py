"""C3 multiple-choice machine-reading-comprehension adapter.

C3 is from nlpdata/c3 (Chung et al., 2019).
https://github.com/nlpdata/c3
https://aclanthology.org/D19-1653/

The Chinese C3 dataset contains dialogues with a question and multiple-choice options.
Only ``context`` is ever corrupted (frozen research rule); ``question``, ``options``,
and the gold ``answer`` are stored as independent fields and are never modified.
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
from ruchi_bench.schema.payloads import MRCPayload
from ruchi_bench.schema.sample import Sample

# C3 field names — from the official JSON format.
_FIELD_CONTEXT = "context"
_FIELD_QUESTION = "question"
_FIELD_OPTIONS = "options"
_FIELD_ANSWER = "answer"


@dataclass(frozen=True, slots=True)
class C3Adapter(BaseAdapter):
    """Adapt C3 records to :class:`Sample`.

    ``context`` is the corruptible field; ``question`` and ``options`` are
    kept separate and are never targets of corruption.
    """

    dataset_name: str = "c3"
    dataset_version: str = "master"
    split: str = "test"

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        try:
            context = self._text(raw, _FIELD_CONTEXT)
            question = self._text(raw, _FIELD_QUESTION)
            options = self._options(raw, _FIELD_OPTIONS)
            answer = self._text(raw, _FIELD_ANSWER)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=str(exc),
            )

        if answer not in options:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"answer {answer!r} not found in options {options}",
            )

        try:
            answer_index = options.index(answer)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"answer {answer!r} not in options: {exc}",
            )

        return AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=Sample(
                sample_id=self._id(raw),
                source_sample_id=self._id(raw),
                dataset_name=DatasetName.C3,
                dataset_version=self.dataset_version,
                split=SplitName.TEST,
                benchmark_task_type=BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
                language=Language.ZH,
                clean_payload=MRCPayload(context=context, question=question, options=options),
                gold_label=answer_index,
                gold_label_text=answer,
                target_fields=(PayloadField.CONTEXT,),
                label_projection_status=ProjectionStatus.NOT_APPLICABLE,
                corruption_applied=False,
                change_count=0,
                created_at=self._now(),
            ),
        )

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

    def _options(self, raw: dict[str, object], field: str) -> tuple[str, ...]:
        val = raw.get(field)
        if not isinstance(val, list):
            raise ValueError(f"{field!r} is not a list")
        if len(val) == 0:
            raise ValueError(f"{field!r} is empty")
        result: list[str] = []
        for item in val:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"blank or non-string item in {field!r}")
            result.append(item.strip())
        return tuple(result)

    def _now(self) -> datetime:
        return datetime.now(tz=UTC)
