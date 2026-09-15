"""The unified :class:`Sample` model and its cross-field invariants (WP2).

A :class:`Sample` is one benchmark item: a clean payload, optional corrupted payload,
gold label, corruption bookkeeping, and (when corrupted) internal + public change
traces. All research invariants live here as validators so an invalid sample cannot
be constructed:

- pilot samples must come from the official ``test`` split and be Chinese;
- dataset ↔ benchmark task ↔ payload kind must agree
  (see :data:`~ruchi_bench.schema.labels.DATASET_BENCHMARK_TASK` and
  :data:`~ruchi_bench.schema.labels.TASK_TO_PAYLOAD_KIND`);
- labels must lie in the dataset's frozen label space (C3 is validated against its
  own options, not a global space);
- ASAP-Polarity metadata is fixed and ACCEPTED after Phase 04 count check (DEC-008);
- corruption fields obey exactly one of three legal states (clean / success /
  failure); every contradictory combination is rejected;
- ``created_at`` is timezone-aware and normalized to UTC (naive datetimes rejected).

No JSON/JSONL file serialization here (WP3); no adapters / AdaptationResult (WP4). But
every model round-trips via ``model_dump(mode="json")`` / ``model_validate``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.labels import (
    ASAP_PROJECTION_NAME,
    ASAP_PROJECTION_VERSION,
    ASAP_STAR_TO_POLARITY,
    DATASET_BENCHMARK_TASK,
    DATASET_LABEL_SPACE,
    TASK_TO_PAYLOAD_KIND,
)
from ruchi_bench.schema.payloads import MRCPayload, NonBlankStr, Payload
from ruchi_bench.schema.trace import InternalChangeTrace, PublicRedactedTrace

__all__ = ["SCHEMA_VERSION", "Sample"]

#: Current unified-schema version stamp.
SCHEMA_VERSION: Final[str] = "0.2.0"

NonNegInt = Annotated[int, Field(ge=0)]


class Sample(BaseModel):
    """One unified benchmark item. Construction fails unless every invariant holds."""

    model_config = ConfigDict(extra="forbid")

    # --- identity / provenance ---
    sample_id: NonBlankStr
    source_sample_id: NonBlankStr
    dataset_name: DatasetName
    dataset_version: NonBlankStr
    split: SplitName
    benchmark_task_type: BenchmarkTaskType
    language: Language

    # --- payloads ---
    clean_payload: Payload
    corrupted_payload: Payload | None = None

    # --- labels ---
    source_gold_label: int | str | None = None
    gold_label: int | str
    gold_label_text: NonBlankStr

    # --- corruption target ---
    target_fields: tuple[PayloadField, ...]

    # --- label projection metadata ---
    label_projection_name: str | None = None
    label_projection_version: str | None = None
    label_projection_status: ProjectionStatus

    # --- corruption bookkeeping ---
    corruption_name: CorruptionName | None = None
    corruption_level: str | None = None
    corruption_seed: NonNegInt | None = None
    corruption_applied: bool
    change_count: NonNegInt
    internal_change_trace: InternalChangeTrace | None = None
    public_redacted_trace: PublicRedactedTrace | None = None
    failure_reason: str | None = None

    # --- meta ---
    created_at: datetime
    schema_version: NonBlankStr = SCHEMA_VERSION

    @field_validator("created_at")
    @classmethod
    def _require_aware_utc(cls, value: datetime) -> datetime:
        """Reject naive datetimes; normalize aware datetimes to UTC."""
        if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
            raise ValueError("created_at must be timezone-aware (naive rejected)")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def _validate(self) -> Sample:
        self._check_pilot_constraints()
        self._check_task_and_payload()
        self._check_target_fields()
        self._check_labels()
        self._check_projection_metadata()
        self._check_corruption_state()
        self._check_trace_consistency()
        return self

    # --- pilot / language ---
    def _check_pilot_constraints(self) -> None:
        if self.split is not SplitName.TEST:
            raise ValueError(f"pilot Sample requires split == TEST, got {self.split.value!r}")
        if self.language is not Language.ZH:
            raise ValueError(f"pilot Sample requires language == zh, got {self.language.value!r}")

    # --- dataset ↔ task ↔ payload agreement ---
    def _check_task_and_payload(self) -> None:
        expected_task = DATASET_BENCHMARK_TASK[self.dataset_name]
        if self.benchmark_task_type is not expected_task:
            raise ValueError(
                f"dataset {self.dataset_name.value!r} expects task "
                f"{expected_task.value!r}, got {self.benchmark_task_type.value!r}"
            )
        expected_kind = TASK_TO_PAYLOAD_KIND[self.benchmark_task_type]
        if self.clean_payload.payload_kind is not expected_kind:
            raise ValueError(
                f"task {self.benchmark_task_type.value!r} expects payload kind "
                f"{expected_kind.value!r}, got "
                f"{self.clean_payload.payload_kind.value!r}"
            )
        if (
            self.corrupted_payload is not None
            and self.corrupted_payload.payload_kind is not self.clean_payload.payload_kind
        ):
            raise ValueError("clean and corrupted payloads must share the same payload kind")

    # --- target fields ---
    def _check_target_fields(self) -> None:
        if not self.target_fields:
            raise ValueError("target_fields must be non-empty")
        if len(self.target_fields) != len(set(self.target_fields)):
            raise ValueError("target_fields must not contain duplicates")
        corruptible = self.clean_payload.CORRUPTIBLE_FIELDS
        illegal = [f for f in self.target_fields if f not in corruptible]
        if illegal:
            raise ValueError(
                f"target_fields {[f.value for f in illegal]} are not corruptible for "
                f"payload kind {self.clean_payload.payload_kind.value!r} "
                f"(allowed: {sorted(f.value for f in corruptible)})"
            )

    # --- labels ---
    def _check_labels(self) -> None:
        if self.benchmark_task_type is BenchmarkTaskType.MULTIPLE_CHOICE_MRC:
            self._check_c3_labels()
            return
        label_space = DATASET_LABEL_SPACE[self.dataset_name]
        if label_space is None:  # pragma: no cover - only C3 is open, handled above
            raise ValueError(
                f"dataset {self.dataset_name.value!r} has an open label space but is "
                "not the MRC task"
            )
        if self.dataset_name in {DatasetName.PAWSX_ZH, DatasetName.LCQMC}:
            if self.gold_label not in (0, 1) or isinstance(self.gold_label, bool):
                raise ValueError(
                    f"{self.dataset_name.value!r} gold_label must be 0 or 1, got "
                    f"{self.gold_label!r}"
                )
        elif not isinstance(self.gold_label, str) or self.gold_label not in label_space:
            raise ValueError(
                f"{self.dataset_name.value!r} gold_label {self.gold_label!r} not in "
                f"label space {label_space}"
            )

    def _check_c3_labels(self) -> None:
        if not isinstance(self.clean_payload, MRCPayload):
            raise ValueError("MRC task requires an MRCPayload clean_payload")
        if isinstance(self.gold_label, bool) or not isinstance(self.gold_label, int):
            raise ValueError(f"C3 gold_label must be an integer index, got {self.gold_label!r}")
        options = self.clean_payload.options
        if not (0 <= self.gold_label < len(options)):
            raise ValueError(
                f"C3 gold_label index {self.gold_label} out of range [0, {len(options)})"
            )
        if self.gold_label_text != options[self.gold_label]:
            raise ValueError("C3 gold_label_text must equal options[gold_label]")
        if self.target_fields != (PayloadField.CONTEXT,):
            raise ValueError(
                "C3 target_fields must be exactly (CONTEXT,), got "
                f"{[f.value for f in self.target_fields]}"
            )
        if isinstance(self.corrupted_payload, MRCPayload):
            if self.corrupted_payload.question != self.clean_payload.question:
                raise ValueError("C3 corrupted question must equal clean question")
            if self.corrupted_payload.options != self.clean_payload.options:
                raise ValueError("C3 corrupted options must equal clean options")

    # --- label projection metadata (ASAP-Polarity; DEC-008) ---
    def _check_projection_metadata(self) -> None:
        if self.dataset_name is DatasetName.ASAP:
            self._check_asap_projection()
            return
        # Non-projection datasets: no projection identity, NOT_APPLICABLE status.
        if self.label_projection_name is not None:
            raise ValueError(f"{self.dataset_name.value!r} must have label_projection_name=None")
        if self.label_projection_version is not None:
            raise ValueError(f"{self.dataset_name.value!r} must have label_projection_version=None")
        if self.label_projection_status is not ProjectionStatus.NOT_APPLICABLE:
            raise ValueError(
                f"{self.dataset_name.value!r} must have label_projection_status=NOT_APPLICABLE"
            )

    def _check_asap_projection(self) -> None:
        if self.benchmark_task_type is not BenchmarkTaskType.SENTIMENT_POLARITY:
            raise ValueError("ASAP must use the SENTIMENT_POLARITY task")
        if self.label_projection_name != ASAP_PROJECTION_NAME:
            raise ValueError(f"ASAP label_projection_name must be {ASAP_PROJECTION_NAME!r}")
        if self.label_projection_version != ASAP_PROJECTION_VERSION:
            raise ValueError(f"ASAP label_projection_version must be {ASAP_PROJECTION_VERSION!r}")
        # ACCEPTED after Phase 04 count check 2026-07-30 (>=150 per class, DEC-008).
        if self.label_projection_status is not ProjectionStatus.ACCEPTED:
            raise ValueError("ASAP-Polarity label_projection_status must be ACCEPTED")
        if (
            isinstance(self.source_gold_label, bool)
            or not isinstance(self.source_gold_label, int)
            or self.source_gold_label not in ASAP_STAR_TO_POLARITY
        ):
            raise ValueError(
                "ASAP source_gold_label must be a star in {1,2,4,5} (3 excluded → no "
                f"Sample), got {self.source_gold_label!r}"
            )
        expected = ASAP_STAR_TO_POLARITY[self.source_gold_label]
        if self.gold_label != expected:
            raise ValueError(
                f"ASAP projection mismatch: star {self.source_gold_label} maps to "
                f"{expected!r}, but gold_label is {self.gold_label!r}"
            )

    # --- corruption three-state consistency ---
    def _check_corruption_state(self) -> None:
        has_name = self.corruption_name is not None
        if not has_name:
            self._require_clean_state()
        elif self.corruption_applied:
            self._require_success_state()
        else:
            self._require_failure_state()

    def _require_clean_state(self) -> None:
        bad = (
            self.corruption_level is not None
            or self.corruption_seed is not None
            or self.corruption_applied
            or self.corrupted_payload is not None
            or self.change_count != 0
            or self.internal_change_trace is not None
            or self.public_redacted_trace is not None
            or self.failure_reason is not None
        )
        if bad:
            raise ValueError(
                "clean state (corruption_name=None) requires all corruption fields "
                "empty/false/zero and no traces or failure_reason"
            )

    def _require_success_state(self) -> None:
        bad = (
            self.corruption_level is None
            or self.corruption_seed is None
            or self.corrupted_payload is None
            or self.change_count <= 0
            or self.internal_change_trace is None
            or self.public_redacted_trace is None
            or self.failure_reason is not None
        )
        if bad:
            raise ValueError(
                "successful corruption requires level, seed, corrupted_payload, "
                "change_count>0, both traces, and no failure_reason"
            )

    def _require_failure_state(self) -> None:
        bad = (
            self.corruption_level is None
            or self.corruption_seed is None
            or self.corrupted_payload is not None
            or self.change_count != 0
            or self.failure_reason is None
        )
        if bad:
            raise ValueError(
                "failed corruption requires level, seed, no corrupted_payload, "
                "change_count==0, and a failure_reason (traces may be None)"
            )

    # --- trace ↔ sample consistency (only meaningful in success state) ---
    def _check_trace_consistency(self) -> None:
        trace = self.internal_change_trace
        if trace is None:
            return
        if len(trace.ops) != self.change_count:
            raise ValueError(
                f"internal_change_trace has {len(trace.ops)} ops but change_count is "
                f"{self.change_count}"
            )
        target_set = set(self.target_fields)
        for op in trace.ops:
            if op.field not in target_set:
                raise ValueError(
                    f"trace op field {op.field.value!r} not in target_fields "
                    f"{[f.value for f in self.target_fields]}"
                )
        if self.corruption_name is not None and trace.corruption_name is not self.corruption_name:
            raise ValueError("internal trace corruption_name must match the Sample")
        if self.corruption_level is not None and trace.corruption_level != self.corruption_level:
            raise ValueError("internal trace corruption_level must match the Sample")
        if self.corruption_seed is not None and trace.seed != self.corruption_seed:
            raise ValueError("internal trace seed must match the Sample")
        public = self.public_redacted_trace
        if public is not None and len(public.ops) != len(trace.ops):
            raise ValueError("public_redacted_trace op count must match internal_change_trace")
