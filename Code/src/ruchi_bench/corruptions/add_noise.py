"""Empirical character/symbol noise insertion (JioNLP adapter)."""

from __future__ import annotations

from ruchi_bench.corruptions.open_source import apply_text_transform, jionlp_transform
from ruchi_bench.corruptions.result import CorruptionResult
from ruchi_bench.schema.enums import CorruptionName, PayloadField
from ruchi_bench.schema.sample import Sample


class ADDNoiseCorruptor:
    corruption_name = CorruptionName.ADD_NOISE
    # Symbol/character noise was acceptable in the first preview. Keep its
    # relatively broad range; the other strategies use tighter budgets.
    LEVEL_RATES = {"low": 0.05, "medium": 0.08, "high": 0.12}
    TARGET_EDITS = {"low": 1, "medium": 2, "high": 3}
    MAX_EDIT_RATIOS = {"low": 0.08, "medium": 0.12, "high": 0.16}

    def is_valid_target(self, sample: Sample, field: PayloadField) -> bool:
        from ruchi_bench.corruptions.open_source import valid_target

        return valid_target(sample, field)

    def corrupt(self, sample: Sample, *, seed: int, level: str) -> CorruptionResult:
        return apply_text_transform(
            sample,
            name=self.corruption_name,
            seed=seed,
            level=level,
            rate=self.LEVEL_RATES.get(level, 0.0),
            transform=jionlp_transform(
                "random_add_delete",
                mode="add",
                target_edit_units=self.TARGET_EDITS.get(level),
                max_edit_ratio=self.MAX_EDIT_RATIOS.get(level),
            ),
            unit_kind="noise",
        )
