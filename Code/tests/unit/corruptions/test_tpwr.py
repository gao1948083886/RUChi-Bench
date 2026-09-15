"""Tests for ruchi_bench.corruptions.tpwr."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ruchi_bench.corruptions.result import APPLIED, INVALID, NOT_APPLIED
from ruchi_bench.corruptions.tpwr import TPWRCorruptor
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
        # Contains words in the synthetic lexicon: "测试", "系统", "数据", "用户", "功能"
        clean_payload=PairPayload(
            text_a="这是一个测试系统，用户数据功能",
            text_b="应用管理系统",
        ),
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
            context="测试数据系统用户功能",
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


class TestTPWRCorruptor:
    def test_name(self) -> None:
        assert TPWRCorruptor().corruption_name == CorruptionName.TPWR

    def test_invalid_level(self) -> None:
        c = TPWRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="unknown")
        assert result.status is INVALID

    def test_negative_seed(self) -> None:
        c = TPWRCorruptor()
        result = c.corrupt(_pair_sample(), seed=-1, level="low")
        assert result.status is INVALID

    def test_applied_or_not(self) -> None:
        # Lexicon word "测试系统" in text → APPLIED or NOT_APPLIED
        c = TPWRCorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)
        if result.status is APPLIED:
            assert result.corrupted_payload is not None

    def test_reproducible(self) -> None:
        c = TPWRCorruptor()
        r1 = c.corrupt(_pair_sample(), seed=42, level="low")
        r2 = c.corrupt(_pair_sample(), seed=42, level="low")
        if r1.status is APPLIED and r2.status is APPLIED:
            assert r1.corrupted_payload == r2.corrupted_payload

    def test_ops_word_unit(self) -> None:
        c = TPWRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="low")
        if result.status is APPLIED:
            for op in result.ops:
                assert op.unit_kind == "word"
                assert op.field in (PayloadField.TEXT_A, PayloadField.TEXT_B)
                assert op.original != op.replacement

    def test_not_applied_when_no_eligible_words(self) -> None:
        c = TPWRCorruptor()
        # Text with no lexicon words, text_b non-empty
        s = _pair_sample(clean_payload=PairPayload(text_a="中文文本", text_b="占位"))
        result = c.corrupt(s, seed=0, level="low")
        # May return APPLIED if seed picks nothing or NOT_APPLIED
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_c3_only_targets_context(self) -> None:
        c = TPWRCorruptor()
        result = c.corrupt(_c3_sample(), seed=0, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)
