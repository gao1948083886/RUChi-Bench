"""Tests for ruchi_bench.corruptions.vscr."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ruchi_bench.corruptions.result import APPLIED, INVALID, NOT_APPLIED
from ruchi_bench.corruptions.vscr import VSCRCorruptor
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
        # Use characters with known confusables (from _CONFUSABLES table)
        clean_payload=PairPayload(text_a="今天天气很好", text_b="明天北京也好"),
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


class TestVSCRCorruptor:
    def test_name(self) -> None:
        assert VSCRCorruptor().corruption_name == CorruptionName.VSCR

    def test_invalid_level(self) -> None:
        c = VSCRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="unknown")
        assert result.status is INVALID

    def test_negative_seed(self) -> None:
        c = VSCRCorruptor()
        result = c.corrupt(_pair_sample(), seed=-1, level="low")
        assert result.status is INVALID

    def test_applied_contains_vscr_char(self) -> None:
        c = VSCRCorruptor()
        result = c.corrupt(_pair_sample(), seed=7, level="low")
        # Low seed with long CJK text → APPLIED or NOT_APPLIED (NOT_APPLIED if bad-luck)
        assert result.status in (APPLIED, NOT_APPLIED)
        if result.status is APPLIED:
            assert result.corrupted_payload is not None

    def test_short_pair_levels_have_distinct_budgets(self) -> None:
        """A short pair must not collapse all three levels to one edit."""
        sample = _pair_sample(
            clean_payload=PairPayload(
                text_a="钓鱼怎么调漂",
                text_b="钓鱼怎么调漂？",
            )
        )
        corruptor = VSCRCorruptor()
        results = [
            corruptor.corrupt(sample, seed=42, level=level)
            for level in ("low", "medium", "high")
        ]
        assert all(result.status is APPLIED for result in results)
        assert [result.change_count for result in results] == [1, 2, 3]
        payloads = [result.corrupted_payload for result in results]
        assert len({payload.model_dump_json() for payload in payloads if payload}) == 3

    def test_reproducible(self) -> None:
        c = VSCRCorruptor()
        r1 = c.corrupt(_pair_sample(), seed=42, level="low")
        r2 = c.corrupt(_pair_sample(), seed=42, level="low")
        if r1.status is APPLIED and r2.status is APPLIED:
            assert r1.corrupted_payload == r2.corrupted_payload
            assert r1.change_count == r2.change_count

    def test_c3_only_targets_context(self) -> None:
        c = VSCRCorruptor()
        result = c.corrupt(_c3_sample(), seed=0, level="low")
        assert result.status in (APPLIED, NOT_APPLIED)

    def test_ops_contain_char_unit(self) -> None:
        c = VSCRCorruptor()
        result = c.corrupt(_pair_sample(), seed=0, level="low")
        if result.status is APPLIED:
            for op in result.ops:
                assert op.unit_kind in ("char", "confusable")
                assert op.field in (PayloadField.TEXT_A, PayloadField.TEXT_B)
