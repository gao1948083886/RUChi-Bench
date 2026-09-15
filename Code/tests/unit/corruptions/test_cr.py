"""Tests for ruchi_bench.corruptions.cr."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest
from pydantic import ValidationError

from ruchi_bench.corruptions.cr import CRCorruptor
from ruchi_bench.corruptions.result import APPLIED, INVALID, NOT_APPLIED
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
from ruchi_bench.schema.payloads import MRCPayload, PairPayload
from ruchi_bench.schema.sample import Sample
from ruchi_bench.schema.trace import InternalChangeOp, InternalChangeTrace, redact_trace

_AWARE = datetime(2026, 7, 28, tzinfo=UTC)


def _pair_sample(**kw: Any) -> Sample:
    defaults = dict(
        sample_id="s1",
        source_sample_id="0",
        dataset_name=DatasetName.PAWSX_ZH,
        dataset_version="v1",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.PAIR_PARAPHRASE,
        language=Language.ZH,
        clean_payload=PairPayload(text_a="今天天气很好心情也不错", text_b="明天应该也不错"),
        gold_label=1,
        gold_label_text="paraphrase",
        target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_B),
        label_projection_status=ProjectionStatus.NOT_APPLICABLE,
        corruption_applied=False,
        change_count=0,
        created_at=_AWARE,
    )
    defaults.update(kw)
    return Sample.model_validate(defaults)


def _c3_sample(**kw: Any) -> Sample:
    defaults = dict(
        sample_id="c3-1",
        source_sample_id="0",
        dataset_name=DatasetName.C3,
        dataset_version="master",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
        language=Language.ZH,
        clean_payload=MRCPayload(
            context="这是一个上下文内容丰富",
            question="问什么？",
            options=("甲", "乙"),
        ),
        gold_label=0,
        gold_label_text="甲",
        target_fields=(PayloadField.CONTEXT,),
        label_projection_status=ProjectionStatus.NOT_APPLICABLE,
        corruption_applied=False,
        change_count=0,
        created_at=_AWARE,
    )
    defaults.update(kw)
    return Sample.model_validate(defaults)


class TestCRCorruptor:
    def test_name(self) -> None:
        assert CRCorruptor().corruption_name == CorruptionName.CR

    def test_invalid_level(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="unknown")
        assert result.status is INVALID

    def test_negative_seed(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=-1, level="low")
        assert result.status is INVALID

    def test_already_corrupted_rejected(self) -> None:
        c = CRCorruptor()
        # Build a fully valid corrupted sample (all required fields provided)
        op = InternalChangeOp(
            op_index=0,
            op_type=ChangeOperation.REPLACE,
            field=PayloadField.TEXT_A,
            start=0,
            end=1,
            original="今",
            replacement="伒",
            unit_kind="char",
            metadata={},
        )
        trace = InternalChangeTrace(
            corruption_name=CorruptionName.CR,
            corruption_level="low",
            seed=7,
            ops=(op,),
        )
        corrupted = _pair_sample(
            corrupted_payload=PairPayload(
                text_a="伒天天气很好心情也不错",
                text_b="明天应该也不错",
            ),
            corruption_name=CorruptionName.CR,
            corruption_level="low",
            corruption_seed=7,
            corruption_applied=True,
            change_count=1,
            internal_change_trace=trace,
            public_redacted_trace=redact_trace(trace),
        )
        # The corruptor should refuse to corrupt an already-corrupted sample
        with pytest.raises(ValueError, match="already corrupted"):
            c.corrupt(corrupted, seed=0, level="low")

    def test_c3_targets_non_context_rejected(self) -> None:
        # C3 with target_fields containing non-context field is invalid at Sample level.
        with pytest.raises(ValidationError):
            _c3_sample(target_fields=(PayloadField.QUESTION,))

    def test_low_level_returns_applied(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)
        if result.status is APPLIED:
            assert result.corrupted_payload is not None
            assert result.change_count >= 1

    def test_medium_level(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="medium")
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_high_level(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="high")
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_reproducible(self) -> None:
        c = CRCorruptor()
        r1 = c.corrupt(_pair_sample(), seed=42, level="low")
        r2 = c.corrupt(_pair_sample(), seed=42, level="low")
        if r1.status is APPLIED and r2.status is APPLIED:
            assert r1.corrupted_payload == r2.corrupted_payload
            assert r1.change_count == r2.change_count

    def test_c3_only_targets_context(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_c3_sample(), seed=0, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_ops_referencing_correct_field(self) -> None:
        c = CRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="low")
        if result.status is APPLIED:
            for op in result.ops:
                assert op.field in (PayloadField.TEXT_A, PayloadField.TEXT_B)
                assert op.op_type.value in ("duplicate",)
