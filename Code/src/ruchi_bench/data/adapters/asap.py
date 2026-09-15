"""ASAP single-text sentiment adapter.

ASAP (Aspect-Sentiment-Pairs) is from Meituan-Dianping (Bu et al., NAACL 2021).
https://github.com/unilightwind/ASAP
https://aclanthology.org/2021.naacl-main.167/

ASAP is a **derived-project task**, NOT an official ASAP task. ASAP has no official
binary sentiment task; ASAP-Polarity is a RUChi-Bench projection derived from the
official ``star`` field of the official ``test.csv``.

Projection rules (DEC-008, design-frozen, final only after Phase 04 count check):
  - star ∈ {1, 2}  → gold_label = "negative"
  - star ∈ {4, 5}  → gold_label = "positive"
  - star = 3        → EXCLUDED (no binary label)
  - star ∉ {1–5}    → INVALID

The projection name is :const:`ASAP_PROJECTION_NAME`, version
:const:`ASAP_PROJECTION_VERSION`, and status is always ``PROVISIONAL``
(:const:`ASAP_POLARITY_STATUS`) until Phase 04 confirms ≥150 samples per class.
No full official test.csv is read in Phase 02.
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
    SplitName,
)
from ruchi_bench.schema.labels import (
    ASAP_EXCLUDED_STARS,
    ASAP_POLARITY_LABELS,
    ASAP_POLARITY_STATUS,
    ASAP_PROJECTION_NAME,
    ASAP_PROJECTION_VERSION,
    ASAP_VALID_STARS,
    polarity_for_star,
)
from ruchi_bench.schema.payloads import SingleTextPayload
from ruchi_bench.schema.sample import Sample

# ASAP field names — from the official CSV format.
_FIELD_REVIEW = "review"
_FIELD_STAR = "star"


@dataclass(frozen=True, slots=True)
class ASAPAdapter(BaseAdapter):
    """Adapt ASAP records to the derived ASAP-Polarity :class:`Sample`.

    The ``star`` field is projected to a binary "positive"/"negative" label.
    Records with ``star == 3`` (neutral) are returned as ``EXCLUDED``.
    Records with ``star`` outside the range 1–5 are returned as ``INVALID``.
    """

    dataset_name: str = "asap"
    dataset_version: str = "master"
    split: str = "test"

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        try:
            review = self._text(raw, _FIELD_REVIEW)
            star = self._star(raw, _FIELD_STAR)
        except ValueError as exc:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=str(exc),
            )

        if star in ASAP_EXCLUDED_STARS:
            return AdaptationResult(
                status=AdaptationStatus.EXCLUDED,
                exclusion_reason=(
                    f"star={star} (neutral); ASAP-Polarity has no binary label for 3-star reviews"
                ),
            )

        if star not in ASAP_VALID_STARS:
            return AdaptationResult(
                status=AdaptationStatus.INVALID,
                invalid_reason=f"star={star!r} is not in the valid range {ASAP_VALID_STARS}",
            )

        polarity = polarity_for_star(star)
        assert polarity in ASAP_POLARITY_LABELS, f"{polarity!r} not in {ASAP_POLARITY_LABELS}"

        return AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=Sample(
                sample_id=self._id(raw),
                source_sample_id=self._id(raw),
                dataset_name=DatasetName.ASAP,
                dataset_version=self.dataset_version,
                split=SplitName.TEST,
                benchmark_task_type=BenchmarkTaskType.SENTIMENT_POLARITY,
                language=Language.ZH,
                clean_payload=SingleTextPayload(text_a=review),
                source_gold_label=star,
                gold_label=polarity,
                gold_label_text=polarity,
                target_fields=(PayloadField.TEXT_A,),
                label_projection_name=ASAP_PROJECTION_NAME,
                label_projection_version=ASAP_PROJECTION_VERSION,
                label_projection_status=ASAP_POLARITY_STATUS,
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

    def _star(self, raw: dict[str, object], field: str) -> int:
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
        raise ValueError(f"{field!r} value {val!r} is not a valid integer star rating")

    def _now(self) -> datetime:
        return datetime.now(tz=UTC)
