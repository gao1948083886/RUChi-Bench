"""Transparent word-repetition corruption (the seventh strategy)."""

from __future__ import annotations

import re

from ruchi_bench.corruptions.base import _check_sample_can_be_corrupted
from ruchi_bench.corruptions.result import (
    APPLIED,
    INVALID,
    NOT_APPLIED,
    CorruptionOp,
    CorruptionResult,
)
from ruchi_bench.corruptions.utils import (
    apply_ops,
    build_trace_ops,
    local_rng,
    select_k_unique,
    visible_length,
)
from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, DatasetName, PayloadField
from ruchi_bench.schema.sample import Sample

_RE_ALNUM = re.compile(r"^[A-Za-z0-9]+$")


def _has_cjk(text: str) -> bool:
    return any(0x3400 <= ord(ch) <= 0x9FFF for ch in text)


def _segment_candidates(text: str) -> list[tuple[int, int, str]]:
    import jieba

    candidates: list[tuple[int, int, str]] = []
    for word, start, end in jieba.tokenize(text, mode="default"):
        if word.strip() and not _RE_ALNUM.fullmatch(word) and _has_cjk(word):
            candidates.append((start, end, word))
    return candidates


class REDWordCorruptor:
    """Duplicate complete jieba words while excluding punctuation-only spans."""

    corruption_name = CorruptionName.RED_WORD
    # Low changes one randomly selected target field, medium changes each target
    # field once, and high changes each field plus one additional word when
    # available.  No word is duplicated more than once.
    LEVEL_PARAMS = {"low": 1, "medium": 1, "high": 2}

    def is_valid_target(self, sample: Sample, field: PayloadField) -> bool:
        if sample.dataset_name is DatasetName.C3:
            return field is PayloadField.CONTEXT
        return field in sample.clean_payload.CORRUPTIBLE_FIELDS

    def corrupt(self, sample: Sample, *, seed: int, level: str) -> CorruptionResult:
        _check_sample_can_be_corrupted(sample)
        if level not in self.LEVEL_PARAMS:
            return CorruptionResult(
                status=INVALID,
                corruption_name=self.corruption_name,
                failure_reason=f"unknown level {level!r}",
            )
        if seed < 0:
            return CorruptionResult(
                status=INVALID,
                corruption_name=self.corruption_name,
                failure_reason="seed must be non-negative",
            )
        if any(not self.is_valid_target(sample, field) for field in sample.target_fields):
            return CorruptionResult(
                status=INVALID,
                corruption_name=self.corruption_name,
                failure_reason="one or more target fields are illegal",
            )

        rng = local_rng(seed)
        candidates_by_field: dict[PayloadField, list[tuple[int, int, str]]] = {}
        for field in sample.target_fields:
            text = getattr(sample.clean_payload, field.value)
            segments = _segment_candidates(text)
            if not segments:
                continue
            candidates_by_field[field] = segments

        if not candidates_by_field:
            return CorruptionResult(
                status=NOT_APPLIED,
                corruption_name=self.corruption_name,
                failure_reason="no eligible Chinese words",
            )

        fields = list(candidates_by_field)
        if level == "low":
            fields = [rng.choice(fields)]
        all_ops: list[CorruptionOp] = []
        for field in fields:
            segments = candidates_by_field[field]
            for idx in select_k_unique(
                rng,
                list(range(len(segments))),
                self.LEVEL_PARAMS[level],
            ):
                start, end, word = segments[idx]
                all_ops.append(
                    CorruptionOp(
                        op_index=len(all_ops),
                        op_type=ChangeOperation.DUPLICATE,
                        field=field,
                        start=start,
                        end=end,
                        original=word,
                        replacement=word,
                        unit_kind="word",
                        metadata={"candidate_count": len(segments)},
                    )
                )

        corrupted = apply_ops(sample.clean_payload, all_ops)
        build_trace_ops(self.corruption_name, level, seed, all_ops, sample.clean_payload)
        return CorruptionResult(
            status=APPLIED,
            corruption_name=self.corruption_name,
            corrupted_payload=corrupted,
            ops=tuple(all_ops),
            change_count=len(all_ops),
        )


WRCorruptor = REDWordCorruptor
