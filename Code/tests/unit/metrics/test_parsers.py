from __future__ import annotations

import pytest

from ruchi_bench.metrics.parsers import parse_label
from ruchi_bench.schema.enums import BenchmarkTaskType


@pytest.mark.parametrize(
    ("task", "raw", "expected"),
    [
        (BenchmarkTaskType.PAIR_PARAPHRASE, "0", "0"),
        (BenchmarkTaskType.PAIR_PARAPHRASE, "１", "1"),
        (BenchmarkTaskType.QUESTION_MATCHING, "1", "1"),
        (BenchmarkTaskType.NLI, "ENTAILMENT", "entailment"),
        (BenchmarkTaskType.NLI, "neutral", "neutral"),
        (BenchmarkTaskType.SENTIMENT_POLARITY, " Positive ", "positive"),
        ("sentiment_rating", "5", "5"),
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "C", "C"),
    ],
)
def test_parse_label_accepts_exact_v2_labels(
    task: BenchmarkTaskType, raw: str, expected: str
) -> None:
    result = parse_label(raw, task)
    assert result.is_success
    assert result.label == expected
    assert result.parser_version == "strict-v2"


@pytest.mark.parametrize(
    ("task", "raw"),
    [
        (BenchmarkTaskType.PAIR_PARAPHRASE, "1."),
        (BenchmarkTaskType.PAIR_PARAPHRASE, "The answer is 1"),
        (BenchmarkTaskType.QUESTION_MATCHING, "答案是1"),
        (BenchmarkTaskType.NLI, "Label: entailment"),
        (BenchmarkTaskType.NLI, "entailment because ..."),
        (BenchmarkTaskType.SENTIMENT_POLARITY, "好评"),
        ("sentiment_rating", "5."),
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "A."),
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "A because ..."),
    ],
)
def test_parse_label_rejects_non_exact_labels(
    task: BenchmarkTaskType, raw: str
) -> None:
    result = parse_label(raw, task)
    assert not result.is_success
    assert result.label is None


@pytest.mark.parametrize(
    ("task", "raw", "expected"),
    [
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "B\n\n答案：", "B"),
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "D\n\n<s", "D"),
        (BenchmarkTaskType.PAIR_PARAPHRASE, "1\n</s>", "1"),
    ],
)
def test_parse_label_accepts_unambiguous_safe_suffixes(
    task: BenchmarkTaskType, raw: str, expected: str
) -> None:
    result = parse_label(raw, task)
    assert result.is_success
    assert result.label == expected


@pytest.mark.parametrize(
    ("task", "raw"),
    [
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "B C D"),
        (BenchmarkTaskType.MULTIPLE_CHOICE_MRC, "A because ..."),
    ],
)
def test_parse_label_rejects_ambiguous_or_explanatory_suffixes(
    task: BenchmarkTaskType, raw: str
) -> None:
    result = parse_label(raw, task)
    assert not result.is_success
    assert result.label is None


def test_parse_label_rejects_c3_letter_outside_allowed_set() -> None:
    result = parse_label(
        "D",
        BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
        c3_allowed_letters="ABC",
    )
    assert result.label is None
    assert result.status == "failure_invalid_label"


def test_parse_label_rejects_unknown_task() -> None:
    with pytest.raises(ValueError, match="Unknown benchmark_task_type"):
        parse_label("0", "unknown_task")
