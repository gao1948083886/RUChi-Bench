"""Unit tests for ruchi_bench.schema.sample. All text is synthetic (never real data)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    ChangeOperation,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import MRCPayload, PairPayload, SingleTextPayload
from ruchi_bench.schema.sample import Sample
from ruchi_bench.schema.trace import InternalChangeOp, InternalChangeTrace, redact_trace

_AWARE = datetime(2026, 7, 26, 1, 0, tzinfo=UTC)


def _base(**overrides: Any) -> dict[str, Any]:
    """A valid clean PAWS-X sample as a kwargs dict; override per test."""
    data: dict[str, Any] = {
        "sample_id": "pawsx-0001",
        "source_sample_id": "0",
        "dataset_name": DatasetName.PAWSX_ZH,
        "dataset_version": "x-final",
        "split": SplitName.TEST,
        "benchmark_task_type": BenchmarkTaskType.PAIR_PARAPHRASE,
        "language": Language.ZH,
        "clean_payload": PairPayload(text_a="第一句", text_b="第二句"),
        "gold_label": 1,
        "gold_label_text": "paraphrase",
        "target_fields": (PayloadField.TEXT_A, PayloadField.TEXT_B),
        "label_projection_status": ProjectionStatus.NOT_APPLICABLE,
        "corruption_applied": False,
        "change_count": 0,
        "created_at": _AWARE,
    }
    data.update(overrides)
    return data


def _mrc_base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sample_id": "c3-0001",
        "source_sample_id": "0",
        "dataset_name": DatasetName.C3,
        "dataset_version": "master",
        "split": SplitName.TEST,
        "benchmark_task_type": BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
        "language": Language.ZH,
        "clean_payload": MRCPayload(context="上下文", question="问?", options=("甲", "乙")),
        "gold_label": 0,
        "gold_label_text": "甲",
        "target_fields": (PayloadField.CONTEXT,),
        "label_projection_status": ProjectionStatus.NOT_APPLICABLE,
        "corruption_applied": False,
        "change_count": 0,
        "created_at": _AWARE,
    }
    data.update(overrides)
    return data


def _success_corruption_kwargs() -> dict[str, Any]:
    """Overrides that put a PAWS-X _base sample into a valid success state."""
    op = InternalChangeOp(
        op_index=0,
        op_type=ChangeOperation.REPLACE,
        field=PayloadField.TEXT_A,
        start=0,
        end=1,
        original="第",
        replacement="苐",
        unit_kind="char",
        metadata={"candidate_count": 2},
    )
    internal = InternalChangeTrace(
        corruption_name=CorruptionName.VSCR, corruption_level="low", seed=7, ops=(op,)
    )
    return {
        "corrupted_payload": PairPayload(text_a="苐一句", text_b="第二句"),
        "corruption_name": CorruptionName.VSCR,
        "corruption_level": "low",
        "corruption_seed": 7,
        "corruption_applied": True,
        "change_count": 1,
        "internal_change_trace": internal,
        "public_redacted_trace": redact_trace(internal),
    }


def _asap_base(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "sample_id": "asap-0001",
        "source_sample_id": "12",
        "dataset_name": DatasetName.ASAP,
        "dataset_version": "master",
        "split": SplitName.TEST,
        "benchmark_task_type": BenchmarkTaskType.SENTIMENT_POLARITY,
        "language": Language.ZH,
        "clean_payload": SingleTextPayload(text_a="这家店不错"),
        "source_gold_label": 5,
        "gold_label": "positive",
        "gold_label_text": "positive",
        "target_fields": (PayloadField.TEXT_A,),
        "label_projection_name": "asap_polarity",
        "label_projection_version": "1.0.0",
        "label_projection_status": ProjectionStatus.ACCEPTED,
        "corruption_applied": False,
        "change_count": 0,
        "created_at": _AWARE,
    }
    data.update(overrides)
    return data


# --- five datasets: one valid clean sample each -----------------------------------
def test_clean_pawsx_valid() -> None:
    s = Sample(**_base())
    assert s.dataset_name is DatasetName.PAWSX_ZH
    assert s.corruption_applied is False


def test_clean_xnli_valid() -> None:
    s = Sample(
        **_base(
            sample_id="xnli-1",
            dataset_name=DatasetName.XNLI_ZH,
            dataset_version="XNLI-1.0",
            benchmark_task_type=BenchmarkTaskType.NLI,
            gold_label="entailment",
            gold_label_text="entailment",
        )
    )
    assert s.benchmark_task_type is BenchmarkTaskType.NLI


def test_clean_lcqmc_valid() -> None:
    s = Sample(
        **_base(
            sample_id="lcqmc-1",
            dataset_name=DatasetName.LCQMC,
            dataset_version="v1",
            benchmark_task_type=BenchmarkTaskType.QUESTION_MATCHING,
            gold_label=0,
            gold_label_text="not_matched",
        )
    )
    assert s.benchmark_task_type is BenchmarkTaskType.QUESTION_MATCHING


def test_clean_c3_valid() -> None:
    s = Sample(**_mrc_base())
    assert s.gold_label == 0
    assert s.gold_label_text == "甲"


def test_clean_asap_valid() -> None:
    s = Sample(**_asap_base())
    assert s.label_projection_status is ProjectionStatus.ACCEPTED
    assert s.gold_label == "positive"


def test_json_round_trip_all_kinds() -> None:
    for builder in (_base, _mrc_base, _asap_base):
        s = Sample(**builder())
        restored = Sample.model_validate(s.model_dump(mode="json"))
        assert restored == s


# --- pilot / split / language -----------------------------------------------------
@pytest.mark.parametrize("split", [SplitName.TRAIN, SplitName.VALIDATION])
def test_non_test_split_rejected(split: SplitName) -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(split=split))


def test_split_enum_still_expresses_all_three() -> None:
    assert {s.value for s in SplitName} == {"train", "validation", "test"}


# --- dataset/task and payload/task mismatches -------------------------------------
def test_dataset_task_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(benchmark_task_type=BenchmarkTaskType.NLI))


def test_payload_task_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(clean_payload=SingleTextPayload(text_a="x")))


def test_corrupted_payload_kind_mismatch_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["corrupted_payload"] = SingleTextPayload(text_a="x")
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


# --- target_fields ----------------------------------------------------------------
def test_target_fields_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(target_fields=()))


def test_target_fields_duplicate_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_A)))


def test_target_fields_not_corruptible_rejected() -> None:
    # CONTEXT is not a field of a PairPayload.
    with pytest.raises(ValidationError):
        Sample(**_base(target_fields=(PayloadField.CONTEXT,)))


# --- datetime ---------------------------------------------------------------------
def test_naive_datetime_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(created_at=datetime(2026, 7, 26, 1, 0)))


def test_aware_datetime_normalized_to_utc() -> None:
    s = Sample(**_base(created_at=datetime(2026, 7, 26, 9, 0, tzinfo=timezone(timedelta(hours=8)))))
    assert s.created_at.tzinfo is UTC
    assert s.created_at == datetime(2026, 7, 26, 1, 0, tzinfo=UTC)


# --- label space ------------------------------------------------------------------
def test_pawsx_label_out_of_space_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(gold_label=2))


def test_xnli_label_out_of_space_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(
            **_base(
                dataset_name=DatasetName.XNLI_ZH,
                benchmark_task_type=BenchmarkTaskType.NLI,
                gold_label="unknown_class",
                gold_label_text="unknown_class",
            )
        )


# --- ASAP-Polarity projection invariants ------------------------------------------
def test_asap_projection_consistent_valid() -> None:
    s = Sample(**_asap_base(source_gold_label=1, gold_label="negative", gold_label_text="negative"))
    assert s.gold_label == "negative"


def test_asap_projection_inconsistent_rejected() -> None:
    # star 5 -> positive, but gold_label says negative
    with pytest.raises(ValidationError):
        Sample(**_asap_base(source_gold_label=5, gold_label="negative", gold_label_text="negative"))


def test_asap_three_star_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_asap_base(source_gold_label=3))


def test_asap_status_accepted_rejected() -> None:
    """ACCEPTED is now valid (DEC-008 Phase 04 count check PASS 2026-07-30)."""
    s = Sample(**_asap_base(label_projection_status=ProjectionStatus.ACCEPTED))
    assert s.label_projection_status is ProjectionStatus.ACCEPTED


def test_asap_status_not_applicable_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_asap_base(label_projection_status=ProjectionStatus.NOT_APPLICABLE))


def test_asap_wrong_projection_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_asap_base(label_projection_name="something_else"))


def test_non_projection_dataset_with_projection_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(label_projection_name="asap_polarity"))


def test_non_projection_dataset_provisional_status_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(label_projection_status=ProjectionStatus.PROVISIONAL))


# --- C3 invariants ----------------------------------------------------------------
def test_c3_gold_index_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_mrc_base(gold_label=5, gold_label_text="甲"))


def test_c3_gold_label_text_mismatch_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_mrc_base(gold_label=0, gold_label_text="乙"))


def test_c3_target_fields_must_be_context_only() -> None:
    with pytest.raises(ValidationError):
        Sample(**_mrc_base(target_fields=(PayloadField.QUESTION,)))


def test_c3_corrupted_question_change_rejected() -> None:
    op = InternalChangeOp(
        op_index=0,
        op_type=ChangeOperation.REPLACE,
        field=PayloadField.CONTEXT,
        start=0,
        end=1,
        original="上",
        replacement="丄",
        unit_kind="char",
        metadata={},
    )
    internal = InternalChangeTrace(
        corruption_name=CorruptionName.VSCR, corruption_level="low", seed=1, ops=(op,)
    )
    with pytest.raises(ValidationError):
        Sample(
            **_mrc_base(
                corrupted_payload=MRCPayload(
                    context="丄下文",
                    question="改了的问?",
                    options=("甲", "乙"),
                ),
                corruption_name=CorruptionName.VSCR,
                corruption_level="low",
                corruption_seed=1,
                corruption_applied=True,
                change_count=1,
                internal_change_trace=internal,
                public_redacted_trace=redact_trace(internal),
            )
        )


def test_c3_corrupted_context_only_valid() -> None:
    op = InternalChangeOp(
        op_index=0,
        op_type=ChangeOperation.REPLACE,
        field=PayloadField.CONTEXT,
        start=0,
        end=1,
        original="上",
        replacement="丄",
        unit_kind="char",
        metadata={},
    )
    internal = InternalChangeTrace(
        corruption_name=CorruptionName.VSCR, corruption_level="low", seed=1, ops=(op,)
    )
    s = Sample(
        **_mrc_base(
            corrupted_payload=MRCPayload(context="丄下文", question="问?", options=("甲", "乙")),
            corruption_name=CorruptionName.VSCR,
            corruption_level="low",
            corruption_seed=1,
            corruption_applied=True,
            change_count=1,
            internal_change_trace=internal,
            public_redacted_trace=redact_trace(internal),
        )
    )
    assert s.corruption_applied is True


# --- corruption three legal states ------------------------------------------------
def test_clean_state_valid() -> None:
    s = Sample(**_base())
    assert s.corruption_name is None
    assert s.internal_change_trace is None


def test_success_state_valid() -> None:
    s = Sample(**_base(**_success_corruption_kwargs()))
    assert s.corruption_applied is True
    assert s.change_count == 1
    assert s.internal_change_trace is not None
    assert s.public_redacted_trace is not None


def test_failure_state_valid() -> None:
    s = Sample(
        **_base(
            corruption_name=CorruptionName.VSCR,
            corruption_level="low",
            corruption_seed=3,
            corruption_applied=False,
            change_count=0,
            failure_reason="no corruptible candidates found",
        )
    )
    assert s.corruption_applied is False
    assert s.failure_reason is not None
    assert s.corrupted_payload is None


# contradictory combinations
def test_clean_with_corrupted_payload_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(corrupted_payload=PairPayload(text_a="x", text_b="y")))


def test_clean_with_change_count_positive_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(change_count=1))


def test_success_without_traces_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["internal_change_trace"] = None
    kw["public_redacted_trace"] = None
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


def test_success_with_zero_change_count_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["change_count"] = 0
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


def test_success_with_failure_reason_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["failure_reason"] = "should not be here"
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


def test_failure_without_reason_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(
            **_base(
                corruption_name=CorruptionName.VSCR,
                corruption_level="low",
                corruption_seed=3,
                corruption_applied=False,
                change_count=0,
            )
        )


def test_failure_with_corrupted_payload_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(
            **_base(
                corruption_name=CorruptionName.VSCR,
                corruption_level="low",
                corruption_seed=3,
                corruption_applied=False,
                change_count=0,
                failure_reason="x",
                corrupted_payload=PairPayload(text_a="x", text_b="y"),
            )
        )


def test_applied_without_corruption_name_rejected() -> None:
    with pytest.raises(ValidationError):
        Sample(**_base(corruption_applied=True))


# --- trace <-> sample consistency -------------------------------------------------
def test_change_count_mismatch_with_trace_ops_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["change_count"] = 2  # trace has 1 op
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


def test_trace_op_field_outside_target_fields_rejected() -> None:
    kw = _success_corruption_kwargs()
    # restrict target_fields to TEXT_B while the op targets TEXT_A
    kw2 = _base(**kw)
    kw2["target_fields"] = (PayloadField.TEXT_B,)
    kw2["corrupted_payload"] = PairPayload(text_a="第一句", text_b="苐二句")
    with pytest.raises(ValidationError):
        Sample(**kw2)


def test_trace_seed_mismatch_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["corruption_seed"] = 999  # trace seed is 7
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))


def test_trace_corruption_name_mismatch_rejected() -> None:
    kw = _success_corruption_kwargs()
    kw["corruption_name"] = CorruptionName.TPWR  # trace says VSCR
    with pytest.raises(ValidationError):
        Sample(**_base(**kw))
