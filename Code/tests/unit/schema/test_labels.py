"""Unit tests for ruchi_bench.schema.labels."""

from __future__ import annotations

import pytest

from ruchi_bench.schema.enums import BenchmarkTaskType, DatasetName, ProjectionStatus
from ruchi_bench.schema.labels import (
    ASAP_EXCLUDED_STARS,
    ASAP_POLARITY_LABELS,
    ASAP_POLARITY_STATUS,
    ASAP_STAR_TO_POLARITY,
    ASAP_VALID_STARS,
    DATASET_BENCHMARK_TASK,
    DATASET_LABEL_SPACE,
    LCQMC_LABEL_DECODE,
    PAWSX_LABEL_DECODE,
    XNLI_LABELS,
    polarity_for_star,
)


def test_dataset_benchmark_task_covers_exactly_all_datasets() -> None:
    """DATASET_BENCHMARK_TASK maps exactly the five pilot datasets — no more, no less."""
    assert set(DATASET_BENCHMARK_TASK) == set(DatasetName)
    assert set(DATASET_BENCHMARK_TASK) == {
        DatasetName.PAWSX_ZH,
        DatasetName.XNLI_ZH,
        DatasetName.C3,
        DatasetName.LCQMC,
        DatasetName.ASAP,
    }


def test_benchmark_task_bindings() -> None:
    assert DATASET_BENCHMARK_TASK[DatasetName.PAWSX_ZH] is BenchmarkTaskType.PAIR_PARAPHRASE
    assert DATASET_BENCHMARK_TASK[DatasetName.XNLI_ZH] is BenchmarkTaskType.NLI
    assert DATASET_BENCHMARK_TASK[DatasetName.C3] is BenchmarkTaskType.MULTIPLE_CHOICE_MRC
    assert DATASET_BENCHMARK_TASK[DatasetName.LCQMC] is BenchmarkTaskType.QUESTION_MATCHING
    assert DATASET_BENCHMARK_TASK[DatasetName.ASAP] is BenchmarkTaskType.SENTIMENT_POLARITY


def test_pawsx_and_lcqmc_split_despite_same_structural_type() -> None:
    """PAWS-X and LCQMC are both structurally 'pair_classification' in the dataset
    configs, yet the semantic benchmark layer separates them into distinct tasks.
    This is the whole point of BenchmarkTaskType existing as its own layer.
    """
    tasks = {name: DATASET_BENCHMARK_TASK[name] for name in DatasetName}
    pawsx_task = tasks[DatasetName.PAWSX_ZH]
    lcqmc_task = tasks[DatasetName.LCQMC]
    # Distinctness first, before any identity narrowing: same structural config type
    # ('pair_classification'), but the semantic layer keeps them distinct.
    assert pawsx_task != lcqmc_task
    assert len({pawsx_task, lcqmc_task}) == 2
    # And each maps to the specific expected semantic task.
    assert pawsx_task is BenchmarkTaskType.PAIR_PARAPHRASE
    assert lcqmc_task is BenchmarkTaskType.QUESTION_MATCHING


def test_every_dataset_has_a_label_space_entry() -> None:
    assert set(DATASET_LABEL_SPACE) == set(DatasetName)


def test_closed_label_spaces_are_nonempty_unique_tuples() -> None:
    for name, space in DATASET_LABEL_SPACE.items():
        if name is DatasetName.C3:
            assert space is None  # open per-sample space
            continue
        assert isinstance(space, tuple)
        assert len(space) >= 2
        assert all(isinstance(v, str) and v for v in space)
        assert len(space) == len(set(space))


def test_pawsx_decode() -> None:
    assert PAWSX_LABEL_DECODE == {0: "different_meaning", 1: "paraphrase"}
    assert tuple(PAWSX_LABEL_DECODE.values()) == DATASET_LABEL_SPACE[DatasetName.PAWSX_ZH]


def test_xnli_labels() -> None:
    assert XNLI_LABELS == ("entailment", "neutral", "contradiction")
    assert DATASET_LABEL_SPACE[DatasetName.XNLI_ZH] == XNLI_LABELS


def test_lcqmc_decode() -> None:
    assert LCQMC_LABEL_DECODE == {0: "not_matched", 1: "matched_intent"}
    assert tuple(LCQMC_LABEL_DECODE.values()) == DATASET_LABEL_SPACE[DatasetName.LCQMC]


def test_asap_polarity_labels_and_status() -> None:
    assert ASAP_POLARITY_LABELS == ("negative", "positive")
    assert DATASET_LABEL_SPACE[DatasetName.ASAP] == ASAP_POLARITY_LABELS
    # ASAP-Polarity ACCEPTED after Phase 04 count check 2026-07-30 (DEC-008).
    assert ASAP_POLARITY_STATUS is ProjectionStatus.ACCEPTED


def test_asap_star_mapping_frozen_shape() -> None:
    assert ASAP_STAR_TO_POLARITY == {1: "negative", 2: "negative", 4: "positive", 5: "positive"}
    assert set(ASAP_EXCLUDED_STARS) == {3}
    assert set(ASAP_VALID_STARS) == {1, 2, 3, 4, 5}
    # 3 is excluded, never mapped to a polarity.
    assert 3 not in ASAP_STAR_TO_POLARITY
    # Mapped values stay within the declared polarity space.
    assert set(ASAP_STAR_TO_POLARITY.values()) == set(ASAP_POLARITY_LABELS)


@pytest.mark.parametrize(
    ("star", "expected"),
    [
        (1, "negative"),
        (2, "negative"),
        (3, None),
        (4, "positive"),
        (5, "positive"),
    ],
)
def test_polarity_for_star_valid(star: int, expected: str | None) -> None:
    assert polarity_for_star(star) == expected


@pytest.mark.parametrize("bad_star", [0, 6, -1, 10])
def test_polarity_for_star_out_of_range_raises(bad_star: int) -> None:
    with pytest.raises(ValueError):
        polarity_for_star(bad_star)


def test_mapping_tables_are_read_only() -> None:
    """MappingProxyType prevents accidental mutation of the frozen label tables.

    (DATASET_BENCHMARK_TASK is a plain dict by design — its type contract is
    ``Final[dict[...]]`` — so it is intentionally not covered by this immutability
    check; its correctness is guarded by the coverage/binding tests above.)
    """
    with pytest.raises(TypeError):
        ASAP_STAR_TO_POLARITY[3] = "negative"  # type: ignore[index]
    with pytest.raises(TypeError):
        DATASET_LABEL_SPACE[DatasetName.C3] = ("x",)  # type: ignore[index]
