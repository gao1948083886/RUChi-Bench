"""Unit tests for dataset adapters. All text is synthetic (never real restricted data).

Fixtures are self-authored, clearly free-to-use, and marked with a provenance comment.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

import pytest

from ruchi_bench.data.adapters import (
    AdaptationResult,
    AdaptationStatus,
    ASAPAdapter,
    C3Adapter,
    LCQMCAdapter,
    PAWSXAdapter,
    XNLIAdapter,
)
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import MRCPayload, PairPayload


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------
def _aware() -> datetime:
    return datetime(2026, 7, 27, tzinfo=UTC)


def _raw(**fields: Any) -> dict[str, object]:
    """Cast a dict literal to dict[str, object] so it satisfies the adapter signature."""
    return cast(dict[str, object], fields)


# ------------------------------------------------------------------
# AdaptationResult invariants
# ------------------------------------------------------------------
def test_accepted_result_requires_sample() -> None:
    with pytest.raises(ValueError, match="must carry a Sample"):
        AdaptationResult(status=AdaptationStatus.ACCEPTED)


def test_excluded_result_requires_reason() -> None:
    with pytest.raises(ValueError, match="must carry an exclusion_reason"):
        AdaptationResult(status=AdaptationStatus.EXCLUDED)


def test_invalid_result_requires_reason() -> None:
    with pytest.raises(ValueError, match="must carry an invalid_reason"):
        AdaptationResult(status=AdaptationStatus.INVALID)


def test_accepted_result_cannot_also_carry_exclusion_reason() -> None:
    # ACCEPTED without sample → raises "must carry a Sample" first.
    with pytest.raises(ValueError, match="must carry"):
        AdaptationResult(status=AdaptationStatus.ACCEPTED, exclusion_reason="x")

    # ACCEPTED with sample but also exclusion_reason → raises "must not carry".
    dummy_sample = _pawsx_sample()
    with pytest.raises(ValueError, match="must not carry"):
        AdaptationResult(
            status=AdaptationStatus.ACCEPTED,
            sample=dummy_sample,
            exclusion_reason="x",
        )


def test_excluded_result_cannot_also_carry_sample() -> None:
    dummy_sample = _pawsx_sample()
    with pytest.raises(ValueError, match="must not carry"):
        AdaptationResult(
            status=AdaptationStatus.EXCLUDED,
            sample=dummy_sample,
            exclusion_reason="neutral",
        )


def _pawsx_sample() -> Any:
    from ruchi_bench.schema.sample import Sample

    return Sample(
        sample_id="x",
        source_sample_id="x",
        dataset_name=DatasetName.PAWSX_ZH,
        dataset_version="v1",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.PAIR_PARAPHRASE,
        language=Language.ZH,
        clean_payload=PairPayload(text_a="a", text_b="b"),
        gold_label=1,
        gold_label_text="paraphrase",
        target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_B),
        label_projection_status=ProjectionStatus.NOT_APPLICABLE,
        corruption_applied=False,
        change_count=0,
        created_at=_aware(),
    )


# ------------------------------------------------------------------
# PAWS-X zh
# ------------------------------------------------------------------
def test_pawsx_adapter_paraphrase_accepted() -> None:
    """Valid PAWS-X paraphrase pair → ACCEPTED."""
    adapter = PAWSXAdapter()
    raw = _raw(id="1", sentence1="这是一句话。", sentence2="这是相同的意思。", label=1)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.dataset_name is DatasetName.PAWSX_ZH
    assert sample.benchmark_task_type is BenchmarkTaskType.PAIR_PARAPHRASE
    assert sample.gold_label == 1
    assert sample.gold_label_text == "paraphrase"
    assert cast(PairPayload, sample.clean_payload).text_a == "这是一句话。"  # noqa: PLR0915


def test_pawsx_adapter_not_paraphrase_accepted() -> None:
    """Valid PAWS-X non-paraphrase pair → ACCEPTED."""
    adapter = PAWSXAdapter()
    raw = _raw(id="2", sentence1="第一句", sentence2="完全不同的内容。", label=0)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == 0
    assert sample.gold_label_text == "different_meaning"


def test_pawsx_adapter_missing_field() -> None:
    """Missing sentence2 → INVALID."""
    adapter = PAWSXAdapter()
    raw = _raw(id="3", sentence1="只有一句")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "missing" in cast(str, result.invalid_reason)


def test_pawsx_adapter_blank_field() -> None:
    """Blank sentence1 → INVALID."""
    adapter = PAWSXAdapter()
    raw = _raw(id="4", sentence1="   ", sentence2="第二句", label=0)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID


def test_pawsx_adapter_unknown_label() -> None:
    """Unknown label → INVALID."""
    adapter = PAWSXAdapter()
    raw = _raw(id="5", sentence1="句一", sentence2="句二", label=9)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "not in" in cast(str, result.invalid_reason)


# ------------------------------------------------------------------
# XNLI zh
# ------------------------------------------------------------------
def test_xnli_adapter_entailment() -> None:
    """Valid XNLI entailment → ACCEPTED."""
    adapter = XNLIAdapter()
    raw = _raw(
        pairID="xnli-1",
        sentence1="前提文本。",
        sentence2="从前提推导的假设。",
        gold_label="entailment",
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.dataset_name is DatasetName.XNLI_ZH
    assert sample.benchmark_task_type is BenchmarkTaskType.NLI
    assert sample.gold_label == "entailment"
    assert sample.gold_label_text == "entailment"


def test_xnli_adapter_neutral() -> None:
    """Valid XNLI neutral → ACCEPTED."""
    adapter = XNLIAdapter()
    raw = _raw(
        pairID="xnli-2",
        sentence1="这个家庭有三个孩子。",
        sentence2="这家有正好三个孩子。",
        gold_label="neutral",
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == "neutral"


def test_xnli_adapter_contradiction() -> None:
    """Valid XNLI contradiction → ACCEPTED."""
    adapter = XNLIAdapter()
    raw = _raw(
        pairID="xnli-3",
        sentence1="他是成年人。",
        sentence2="他是个婴儿。",
        gold_label="contradiction",
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == "contradiction"


def test_xnli_adapter_missing_field() -> None:
    """Missing gold_label → INVALID."""
    adapter = XNLIAdapter()
    raw = _raw(pairID="xnli-4", sentence1="前提", sentence2="假设")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "missing" in cast(str, result.invalid_reason)


def test_xnli_adapter_unknown_label() -> None:
    """Unknown gold_label → INVALID."""
    adapter = XNLIAdapter()
    raw = _raw(pairID="xnli-5", sentence1="前提", sentence2="假设", gold_label="random")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "not in" in cast(str, result.invalid_reason)


# ------------------------------------------------------------------
# C3
# ------------------------------------------------------------------
def test_c3_adapter_valid() -> None:
    """Valid C3 record → ACCEPTED with correct answer_index."""
    adapter = C3Adapter()
    raw = _raw(
        id="c3-1",
        context="对话发生的场景上下文。",
        question="关于对话内容的问题是什么？",
        options=["选项甲", "选项乙", "选项丙"],
        answer="选项乙",
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.dataset_name is DatasetName.C3
    assert sample.benchmark_task_type is BenchmarkTaskType.MULTIPLE_CHOICE_MRC
    assert sample.gold_label == 1  # "选项乙" is index 1
    assert sample.gold_label_text == "选项乙"
    assert cast(MRCPayload, sample.clean_payload).context == "对话发生的场景上下文。"  # noqa: PLR0915
    assert sample.target_fields == (PayloadField.CONTEXT,)


def test_c3_adapter_answer_not_in_options() -> None:
    """Answer not in options → INVALID."""
    adapter = C3Adapter()
    raw = _raw(
        id="c3-2",
        context="上下文",
        question="问？",
        options=["甲", "乙"],
        answer="丁",  # not in options
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "not found in" in cast(str, result.invalid_reason)


def test_c3_adapter_missing_context() -> None:
    """Missing context → INVALID."""
    adapter = C3Adapter()
    raw = _raw(id="c3-3", question="问？", options=["甲", "乙"], answer="甲")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "missing" in cast(str, result.invalid_reason)


def test_c3_adapter_empty_options() -> None:
    """Empty options list → INVALID."""
    adapter = C3Adapter()
    raw = _raw(id="c3-4", context="上下文", question="问？", options=[], answer="甲")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "empty" in cast(str, result.invalid_reason)


def test_c3_adapter_blank_option_item() -> None:
    """Blank item in options → INVALID."""
    adapter = C3Adapter()
    raw = _raw(
        id="c3-5", context="上下文", question="问？", options=["甲", "  ", "丙"], answer="甲"
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID


# ------------------------------------------------------------------
# LCQMC
# ------------------------------------------------------------------
def test_lcqmc_adapter_valid() -> None:
    """Valid LCQMC matched pair → ACCEPTED."""
    adapter = LCQMCAdapter()
    raw = _raw(
        id="lcqmc-1",
        sentence1="相似的问题一",
        sentence2="相似的问题二",
        label=1,
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.dataset_name is DatasetName.LCQMC
    assert sample.benchmark_task_type is BenchmarkTaskType.QUESTION_MATCHING
    assert sample.gold_label == 1
    assert sample.gold_label_text == "matched_intent"


def test_lcqmc_adapter_not_matched() -> None:
    """Valid LCQMC unmatched pair → ACCEPTED."""
    adapter = LCQMCAdapter()
    raw = _raw(
        id="lcqmc-2",
        question1="第一个问题",  # variant field name
        question2="完全不相关的另一个问题",
        label=0,
    )
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == 0
    assert sample.gold_label_text == "not_matched"


def test_lcqmc_adapter_missing_both_field_variants() -> None:
    """Neither sentence1 nor question1 present → INVALID."""
    adapter = LCQMCAdapter()
    raw = _raw(id="lcqmc-3", sentence2="第二句", label=1)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "neither" in cast(str, result.invalid_reason)


def test_lcqmc_adapter_unknown_label() -> None:
    """Unknown label → INVALID."""
    adapter = LCQMCAdapter()
    raw = _raw(id="lcqmc-4", sentence1="句一", sentence2="句二", label=9)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID


# ------------------------------------------------------------------
# ASAP
# ------------------------------------------------------------------
@pytest.mark.parametrize("star", [1, 2])
def test_asap_adapter_star_1_2_accepted_as_negative(star: int) -> None:
    """Star 1–2 → ACCEPTED as negative polarity."""
    adapter = ASAPAdapter()
    raw = _raw(id=f"asap-s{star}", review=f"非常差的{star}星评论。", star=star)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.dataset_name is DatasetName.ASAP
    assert sample.benchmark_task_type is BenchmarkTaskType.SENTIMENT_POLARITY
    assert sample.source_gold_label == star
    assert sample.gold_label == "negative"
    assert sample.gold_label_text == "negative"
    assert sample.label_projection_name == "asap_polarity"
    assert sample.label_projection_version == "1.0.0"
    assert sample.label_projection_status is ProjectionStatus.ACCEPTED


@pytest.mark.parametrize("star", [4, 5])
def test_asap_adapter_star_4_5_accepted_as_positive(star: int) -> None:
    """Star 4–5 → ACCEPTED as positive polarity."""
    adapter = ASAPAdapter()
    raw = _raw(id=f"asap-s{star}", review=f"很好的{star}星评论。", star=star)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == "positive"


def test_asap_adapter_star_3_excluded() -> None:
    """Star 3 (neutral) → EXCLUDED."""
    adapter = ASAPAdapter()
    raw = _raw(id="asap-s3", review="一般般的评论。", star=3)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.EXCLUDED
    assert result.sample is None
    reason = cast(str, result.exclusion_reason)
    assert "star=3" in reason
    assert "neutral" in reason


@pytest.mark.parametrize("star", [0, 6, 99, -1])
def test_asap_adapter_out_of_range_star_invalid(star: int) -> None:
    """Star outside 1–5 → INVALID."""
    adapter = ASAPAdapter()
    raw = _raw(id=f"asap-s{star}", review="评论文本。", star=star)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "not in the valid range" in cast(str, result.invalid_reason)


def test_asap_adapter_missing_review() -> None:
    """Missing review text → INVALID."""
    adapter = ASAPAdapter()
    raw = _raw(id="asap-missing", star=5)
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.INVALID
    assert "missing" in cast(str, result.invalid_reason)


def test_asap_adapter_string_star_valid() -> None:
    """Star as string "5" → still ACCEPTED (type coercion)."""
    adapter = ASAPAdapter()
    raw = _raw(id="asap-str", review="很好的评论。", star="5")
    result = adapter.adapt_record(raw)
    assert result.status is AdaptationStatus.ACCEPTED
    sample = result.sample
    assert sample is not None
    assert sample.gold_label == "positive"


# ------------------------------------------------------------------
# Synthetic provenance
# ------------------------------------------------------------------
# All fixtures in this file are self-authored Chinese text created solely
# for testing. They are not copied from any restricted source dataset
# and are not suitable for benchmark results or model training.
