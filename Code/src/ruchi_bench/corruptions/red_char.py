"""Transparent character-repetition corruption (the sixth strategy)."""

from __future__ import annotations

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


def _is_cjk(ch: str) -> bool:
    return (0x3400 <= ord(ch) <= 0x4DBF) or (0x4E00 <= ord(ch) <= 0x9FFF)


class REDCharCorruptor:
    """Repeat selected Chinese characters, with genuinely increasing intensity."""

    corruption_name = CorruptionName.RED_CHAR
    # The level controls the number of independent repeated characters.  For a
    # pair, low selects one side at random, medium touches both sides, and high
    # adds a second repeat where the text has enough eligible characters.  This
    # removes the old text_a-only bias while keeping high human-readable.
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
        candidates_by_field: dict[PayloadField, list[int]] = {}
        for field in sample.target_fields:
            text = getattr(sample.clean_payload, field.value)
            eligible = [i for i, ch in enumerate(text) if _is_cjk(ch)]
            if not eligible or visible_length(text) < 2:
                continue
            candidates_by_field[field] = eligible

        if not candidates_by_field:
            return CorruptionResult(
                status=NOT_APPLIED,
                corruption_name=self.corruption_name,
                failure_reason="no eligible Chinese characters",
            )

        fields = list(candidates_by_field)
        if level == "low":
            fields = [rng.choice(fields)]
        all_ops: list[CorruptionOp] = []
        for field in fields:
            text = getattr(sample.clean_payload, field.value)
            positions = select_k_unique(
                rng,
                candidates_by_field[field],
                self.LEVEL_PARAMS[level],
            )
            for pos in positions:
                char = text[pos]
                extra = 1
                all_ops.append(
                    CorruptionOp(
                        op_index=len(all_ops),
                        op_type=ChangeOperation.DUPLICATE,
                        field=field,
                        start=pos,
                        end=pos + 1,
                        original=char,
                        replacement=char * extra,
                        unit_kind="char",
                        metadata={"repeat_count": extra},
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


CRCorruptor = REDCharCorruptor
