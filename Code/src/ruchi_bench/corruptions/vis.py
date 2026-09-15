"""Visual-similarity replacement using the Confused_Chinese resource snapshot."""

from __future__ import annotations

import json
from importlib.resources import files

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
)
from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, DatasetName, PayloadField
from ruchi_bench.schema.sample import Sample


def _load_confusions() -> dict[str, tuple[str, ...]]:
    resource = files("ruchi_bench.corruptions").joinpath("resources/visual_confusions.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    return {source: tuple(candidates) for source, candidates in data.items()}


_VISUAL_CONFUSIONS = _load_confusions()


def is_visual_source_eligible(char: str) -> bool:
    """Return whether a character is safe to consider for visual corruption."""
    return char in _VISUAL_CONFUSIONS and char not in VISCorruptor._EXCLUDED_SOURCES


class VISCorruptor:
    """Replace eligible characters with Confused_Chinese visual confusions."""

    corruption_name = CorruptionName.VIS
    # Visual substitutions are hard to recover from when applied to function
    # words. Keep the rate conservative; eligibility is filtered below.
    LEVEL_RATES = {"low": 0.05, "medium": 0.10, "high": 0.12}

    # A broad visual-confusion table can contain candidates that destroy the
    # sentence rather than simulate a plausible typo. Exclude common function
    # words and pronouns from this strategy.
    _EXCLUDED_SOURCES = frozenset(
        "我你他她它的是不了在也和与或有无这那其为被将把从对及而都很最怎"
    )
    _BASE_BUDGETS = {"low": 1, "medium": 2, "high": 3}
    _LENGTH_SCALES = {"low": 240, "medium": 200, "high": 160}

    @classmethod
    def _budget(cls, level: str, *, eligible_count: int, text_length: int) -> int:
        """Choose a conservative whole-sample edit budget for one level."""
        base = cls._BASE_BUDGETS[level]
        scale = cls._LENGTH_SCALES[level]
        requested = base + text_length // scale
        return min(eligible_count, requested)

    def is_valid_target(self, sample: Sample, field: PayloadField) -> bool:
        if sample.dataset_name is DatasetName.C3:
            return field is PayloadField.CONTEXT
        return field in sample.clean_payload.CORRUPTIBLE_FIELDS

    def corrupt(self, sample: Sample, *, seed: int, level: str) -> CorruptionResult:
        _check_sample_can_be_corrupted(sample)
        if level not in self.LEVEL_RATES:
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
        all_ops: list[CorruptionOp] = []
        eligible_locations: list[tuple[PayloadField, int]] = []
        for field in sample.target_fields:
            text = getattr(sample.clean_payload, field.value)
            eligible = [
                i
                for i, ch in enumerate(text)
                if is_visual_source_eligible(ch)
            ]
            if not eligible:
                continue
            eligible_locations.extend((field, pos) for pos in eligible)

        if not eligible_locations:
            return CorruptionResult(
                status=NOT_APPLIED,
                corruption_name=self.corruption_name,
                failure_reason="no eligible visual confusions found",
            )

        total_length = sum(
            len(getattr(sample.clean_payload, field.value)) for field in sample.target_fields
        )
        budget = self._budget(
            level,
            eligible_count=len(eligible_locations),
            text_length=total_length,
        )
        selected_indices = rng.sample(range(len(eligible_locations)), budget)
        for selected_index in sorted(selected_indices):
            field, pos = eligible_locations[selected_index]
            text = getattr(sample.clean_payload, field.value)
            original = text[pos]
            candidates = _VISUAL_CONFUSIONS[original]
            # The vendored resource preserves upstream similarity ranking;
            # choose its top candidate instead of an arbitrary lower-ranked
            # one that may be much less readable in context.
            replacement = candidates[0]
            all_ops.append(
                CorruptionOp(
                    op_index=len(all_ops),
                    op_type=ChangeOperation.REPLACE,
                    field=field,
                    start=pos,
                    end=pos + 1,
                    original=original,
                    replacement=replacement,
                    unit_kind="char",
                    metadata={"candidate_count": len(candidates), "source": "confused_chinese"},
                )
            )

        if not all_ops:
            return CorruptionResult(
                status=NOT_APPLIED,
                corruption_name=self.corruption_name,
                failure_reason="no eligible visual confusions found",
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


VSCRCorruptor = VISCorruptor
