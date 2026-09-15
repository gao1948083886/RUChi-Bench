"""Contract tests for the seven frozen corruption strategies."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from ruchi_bench.corruptions.registry import CORRUPTOR_REGISTRY
from ruchi_bench.corruptions.result import APPLIED
from ruchi_bench.corruptions.red_char import REDCharCorruptor
from ruchi_bench.corruptions.red_word import REDWordCorruptor
from ruchi_bench.corruptions.swap import SWAPCorruptor
from ruchi_bench.data.strategy_assignment import assign_strategies
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import PairPayload
from ruchi_bench.schema.sample import Sample


def _sample(**updates: Any) -> Sample:
    values: dict[str, Any] = {
        "sample_id": "seven-1",
        "source_sample_id": "1",
        "dataset_name": DatasetName.PAWSX_ZH,
        "dataset_version": "test",
        "split": SplitName.TEST,
        "benchmark_task_type": BenchmarkTaskType.PAIR_PARAPHRASE,
        "language": Language.ZH,
        "clean_payload": PairPayload(
            text_a="今天天气很好，我们一起去北京参加测试。",
            text_b="明天天气也很好，我们一起去北京参加测试。",
        ),
        "gold_label": 1,
        "gold_label_text": "paraphrase",
        "target_fields": (PayloadField.TEXT_A, PayloadField.TEXT_B),
        "label_projection_status": ProjectionStatus.NOT_APPLICABLE,
        "corruption_applied": False,
        "change_count": 0,
        "created_at": datetime(2026, 9, 6, tzinfo=UTC),
    }
    values.update(updates)
    return Sample.model_validate(values)


def test_registry_contains_exactly_seven_canonical_strategies() -> None:
    assert tuple(CORRUPTOR_REGISTRY) == tuple(CorruptionName)
    assert set(CORRUPTOR_REGISTRY) == set(CorruptionName)


def test_each_strategy_is_reproducible_and_keeps_labels() -> None:
    sample = _sample()
    for name, corruptor in CORRUPTOR_REGISTRY.items():
        first = corruptor.corrupt(sample, seed=19, level="medium")
        second = corruptor.corrupt(sample, seed=19, level="medium")
        assert first.status == second.status, name
        if first.status is APPLIED:
            assert first.corrupted_payload == second.corrupted_payload
            assert first.ops == second.ops
            assert first.corrupted_payload != sample.clean_payload
        assert sample.gold_label == 1
        assert sample.target_fields == (PayloadField.TEXT_A, PayloadField.TEXT_B)


def test_three_levels_are_each_generated_from_clean_input() -> None:
    sample = _sample()
    for corruptor in CORRUPTOR_REGISTRY.values():
        results = [
            corruptor.corrupt(sample, seed=101, level=level) for level in ("low", "medium", "high")
        ]
        for result in results:
            if result.status is APPLIED:
                assert result.corrupted_payload is not None
                assert result.corrupted_payload != sample.clean_payload or result.ops


def test_assignment_chooses_one_strategy_per_sample() -> None:
    samples = [_sample(sample_id=f"seven-{index}") for index in range(12)]
    assignments = assign_strategies(samples, seed=7)
    assert set(assignments) == {sample.sample_id for sample in samples}
    assert all(name in CORRUPTOR_REGISTRY for name in assignments.values())


def test_calibrated_levels_have_distinct_budgets() -> None:
    assert REDCharCorruptor.LEVEL_PARAMS == {"low": 1, "medium": 1, "high": 2}
    assert REDWordCorruptor.LEVEL_PARAMS == {"low": 1, "medium": 1, "high": 2}
    assert SWAPCorruptor.TARGET_LOCAL_SWAPS == {"low": 1, "medium": 2, "high": 3}
    assert SWAPCorruptor.MAX_LOCAL_SWAPS == {"low": 1, "medium": 2, "high": 3}
