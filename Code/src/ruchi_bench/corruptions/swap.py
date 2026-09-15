"""Open-source neighboring character transposition (JioNLP adapter)."""

from __future__ import annotations

from ruchi_bench.corruptions.open_source import apply_text_transform, jionlp_transform
from ruchi_bench.corruptions.result import CorruptionResult
from ruchi_bench.schema.enums import CorruptionName, PayloadField
from ruchi_bench.schema.sample import Sample


class SWAPCorruptor:
    corruption_name = CorruptionName.SWAP
    # Keep swaps local and recoverable. The former 0.20 High setting could
    # make short sentences nearly unreadable.
    LEVEL_RATES = {"low": 0.05, "medium": 0.10, "high": 0.12}
    # One adjacent transposition normally changes two code-point positions.
    TARGET_EDITS = {"low": 2, "medium": 4, "high": 6}
    MAX_EDIT_RATIOS = {"low": 0.10, "medium": 0.15, "high": 0.20}
    TARGET_LOCAL_SWAPS = {"low": 1, "medium": 2, "high": 3}
    MAX_LOCAL_SWAPS = {"low": 1, "medium": 2, "high": 3}

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
                "swap_char_position",
                target_edit_units=self.TARGET_EDITS.get(level),
                max_edit_ratio=self.MAX_EDIT_RATIOS.get(level),
                target_local_swaps=self.TARGET_LOCAL_SWAPS.get(level),
                max_local_swaps=self.MAX_LOCAL_SWAPS.get(level),
                swap_scale=0.4,
            ),
            unit_kind="char",
        )
