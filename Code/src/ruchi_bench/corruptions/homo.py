"""Open-source homophone replacement strategy (JioNLP adapter)."""

from __future__ import annotations

from ruchi_bench.corruptions.open_source import apply_text_transform, jionlp_transform
from ruchi_bench.corruptions.result import CorruptionResult
from ruchi_bench.schema.enums import CorruptionName, PayloadField
from ruchi_bench.schema.sample import Sample


class HOMOCorruptor:
    corruption_name = CorruptionName.HOMO
    # Prefer genuine homophones to loose/mispronounced matches. A short
    # sentence may legitimately saturate at one replacement.
    LEVEL_RATES = {"low": 0.05, "medium": 0.08, "high": 0.12}
    TARGET_EDITS = {"low": 1, "medium": 2, "high": 3}
    MAX_EDIT_RATIOS = {"low": 0.06, "medium": 0.10, "high": 0.14}

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
                "homophone_substitution",
                target_edit_units=self.TARGET_EDITS.get(level),
                max_edit_ratio=self.MAX_EDIT_RATIOS.get(level),
            ),
            unit_kind="word",
        )


TPWRCorruptor = HOMOCorruptor
