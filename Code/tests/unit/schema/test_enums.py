"""Unit tests for ruchi_bench.schema.enums."""

from __future__ import annotations

from enum import Enum

import pytest

from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)

ALL_ENUMS = [
    DatasetName,
    Language,
    BenchmarkTaskType,
    SplitName,
    PayloadField,
    CorruptionName,
    ProjectionStatus,
]


def test_exactly_seven_enums() -> None:
    """The schema exposes exactly seven closed enumerations."""
    assert len(ALL_ENUMS) == 7
    assert len({e.__name__ for e in ALL_ENUMS}) == 7


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_all_members_are_str(enum_cls: type[Enum]) -> None:
    """Every member subclasses str and equals its own value string."""
    for member in enum_cls:
        assert isinstance(member, str)
        assert member == member.value
        assert isinstance(member.value, str)


@pytest.mark.parametrize("enum_cls", ALL_ENUMS)
def test_values_unique(enum_cls: type[Enum]) -> None:
    values = [m.value for m in enum_cls]
    assert len(values) == len(set(values))


def test_dataset_name_exact_membership() -> None:
    assert {d.value for d in DatasetName} == {
        "pawsx_zh",
        "xnli_zh",
        "c3",
        "lcqmc",
        "asap",
    }


def test_benchmark_task_type_exact_membership() -> None:
    assert {t.value for t in BenchmarkTaskType} == {
        "pair_paraphrase",
        "nli",
        "multiple_choice_mrc",
        "question_matching",
        "sentiment_polarity",
    }


def test_language_membership() -> None:
    """Pilot is Chinese-only."""
    assert {lang.value for lang in Language} == {"zh"}
    assert Language.ZH.value == "zh"


def test_split_name_domain_complete() -> None:
    """SplitName is the full split space (test-only restriction lives elsewhere)."""
    assert {s.value for s in SplitName} == {"train", "validation", "test"}


def test_payload_field_membership() -> None:
    assert {f.value for f in PayloadField} == {
        "text_a",
        "text_b",
        "context",
        "question",
        "options",
    }


def test_corruption_membership() -> None:
    assert {c.value for c in CorruptionName} == {
        "homo", "vis", "del", "swap", "add_noise", "red_char", "red_word"
    }
    assert CorruptionName.TPWR is CorruptionName.HOMO
    assert CorruptionName.VSCR is CorruptionName.VIS
    assert CorruptionName.CR is CorruptionName.RED_CHAR
    assert CorruptionName.WR is CorruptionName.RED_WORD
    assert CorruptionName.MUNI is CorruptionName.ADD_NOISE


def test_projection_status_membership() -> None:
    assert {p.value for p in ProjectionStatus} == {
        "not_applicable",
        "provisional",
        "accepted",
    }


def test_lookup_by_value_roundtrip() -> None:
    assert DatasetName("asap") is DatasetName.ASAP
    assert BenchmarkTaskType("nli") is BenchmarkTaskType.NLI
    assert Language("zh") is Language.ZH
    assert ProjectionStatus("provisional") is ProjectionStatus.PROVISIONAL


def test_unknown_value_raises() -> None:
    with pytest.raises(ValueError):
        DatasetName("not_a_dataset")
