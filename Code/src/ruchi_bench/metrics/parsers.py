"""Strict label parsers for Prompt Suite v2.

The parser contract is intentionally narrow:
- normalize with NFKC, then strip leading/trailing whitespace;
- accept only task-specific exact labels;
- treat prefixes, punctuation, explanations, and multiple tokens as parse failures.

Callers must distinguish parse failure from a parsed but wrong label.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from ruchi_bench.schema.enums import BenchmarkTaskType

__all__ = ["ParsedResult", "parse_label"]


@dataclass(frozen=True, slots=True)
class ParsedResult:
    """One strict parse result."""

    label: str | None
    status: str
    parser_version: str = "strict-v2"

    @property
    def is_success(self) -> bool:
        return self.status == "success"


def _normalize(raw: str) -> str:
    return unicodedata.normalize("NFKC", raw).strip()


def _failure_for_non_exact(text: str) -> str:
    if not text:
        return "failure_empty"
    if any(ch.isspace() for ch in text):
        return "failure_multiple_tokens"
    if text[:1].lower() in {"a", "b", "c", "d"} and len(text) > 1:
        return "failure_prefix"
    if any(not ch.isalnum() and ch != "_" for ch in text):
        return "failure_punctuation"
    return "failure_invalid_label"


def _parse_exact(raw: str, valid: set[str], *, lowercase: bool) -> ParsedResult:
    text = _normalize(raw)
    candidate = text.lower() if lowercase else text
    if candidate in valid:
        return ParsedResult(candidate, "success")
    # Some local instruct models append a harmless answer cue or an incomplete
    # end-of-sequence marker after an otherwise unambiguous label.  Accept only
    # these exact suffixes; explanations and multiple labels remain failures.
    for label in sorted(valid, key=len, reverse=True):
        normalized_label = label.lower() if lowercase else label
        suffix_pattern = (
            rf"{re.escape(label)}[ \t\r\n]+"
            r"(?:答案[:：]?|</?s>?|<\|(?:endoftext|eot_id)\|>)"
        )
        if re.fullmatch(suffix_pattern, text, flags=re.IGNORECASE if lowercase else 0):
            return ParsedResult(normalized_label, "success")
    return ParsedResult(None, _failure_for_non_exact(text))


def _task_value(task_type: BenchmarkTaskType | str) -> str:
    if isinstance(task_type, BenchmarkTaskType):
        return task_type.value
    return task_type


def parse_label(
    raw: str,
    benchmark_task_type: BenchmarkTaskType | str,
    *,
    c3_allowed_letters: str = "ABCD",
) -> ParsedResult:
    """Parse a model response according to the sample's benchmark task type."""
    task_value = _task_value(benchmark_task_type)

    if task_value in {
        BenchmarkTaskType.PAIR_PARAPHRASE.value,
        BenchmarkTaskType.QUESTION_MATCHING.value,
    }:
        return _parse_exact(raw, {"0", "1"}, lowercase=False)
    if task_value == BenchmarkTaskType.NLI.value:
        return _parse_exact(
            raw,
            {"entailment", "neutral", "contradiction"},
            lowercase=True,
        )
    if task_value == BenchmarkTaskType.SENTIMENT_POLARITY.value:
        return _parse_exact(raw, {"positive", "negative"}, lowercase=True)
    if task_value == "sentiment_rating":
        return _parse_exact(raw, {"1", "2", "3", "4", "5"}, lowercase=False)
    if task_value == BenchmarkTaskType.MULTIPLE_CHOICE_MRC.value:
        return _parse_exact(raw, set(c3_allowed_letters), lowercase=False)

    raise ValueError(f"Unknown benchmark_task_type: {benchmark_task_type!r}")
