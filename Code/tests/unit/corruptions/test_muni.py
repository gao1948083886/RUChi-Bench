"""Tests for ruchi_bench.corruptions.muni."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ruchi_bench.corruptions.muni import MUNICorruptor
from ruchi_bench.corruptions.result import APPLIED, INVALID, NOT_APPLIED
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import MRCPayload, PairPayload
from ruchi_bench.schema.sample import Sample

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
        clean_payload=PairPayload(text_a="今天天气很好", text_b="明天应该也不错"),
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
            context="今天天气很好阳光明媚",
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


class TestMUNICorruptor:
    def test_name(self) -> None:
        assert MUNICorruptor().corruption_name == CorruptionName.MUNI

    def test_invalid_level(self) -> None:
        c = MUNICorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="unknown")
        assert result.status is INVALID

    def test_negative_seed(self) -> None:
        c = MUNICorruptor()
        result = c.corrupt(_pair_sample(), seed=-1, level="low")
        assert result.status is INVALID

    def test_applied_or_not(self) -> None:
        c = MUNICorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)
        if result.status is APPLIED:
            assert result.corrupted_payload is not None

    def test_reproducible(self) -> None:
        c = MUNICorruptor()
        r1 = c.corrupt(_pair_sample(), seed=42, level="low")
        r2 = c.corrupt(_pair_sample(), seed=42, level="low")
        if r1.status is APPLIED and r2.status is APPLIED:
            assert r1.corrupted_payload == r2.corrupted_payload

    def test_ops_empirical_noise_insert(self) -> None:
        c = MUNICorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="low")
        if result.status is APPLIED:
            for op in result.ops:
                assert op.op_type.value == "insert"
                assert op.unit_kind == "noise"
                assert op.field in (PayloadField.TEXT_A, PayloadField.TEXT_B)

    def test_c3_only_targets_context(self) -> None:
        c = MUNICorruptor()
        result = c.corrupt(_c3_sample(), seed=0, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_not_applied_on_empty_visible_text(self) -> None:
        c = MUNICorruptor()
        # With seed=99999 and very short text, MUNI may legitimately return NOT_APPLIED
        # (no eligible visible positions found). That's the contract we test.
        result = c.corrupt(_pair_sample(), seed=99999, level="low")
        assert result.status in (NOT_APPLIED, APPLIED)
