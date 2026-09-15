"""Tests for ruchi_bench.corruptions.utils."""

from __future__ import annotations

from typing import cast

from ruchi_bench.corruptions.result import CorruptionOp
from ruchi_bench.corruptions.utils import (
    apply_ops,
    cjk_positions,
    has_cjk,
    local_rng,
    round_rate,
    select_k_unique,
    visible_char_positions,
    visible_length,
)
from ruchi_bench.schema.enums import ChangeOperation, PayloadField
from ruchi_bench.schema.payloads import PairPayload


class TestHasCJK:
    def test_cjk_present(self) -> None:
        assert has_cjk("第一句") is True

    def test_cjk_absent(self) -> None:
        assert has_cjk("ABC123") is False

    def test_mixed(self) -> None:
        assert has_cjk("句ABC") is True


class TestCJKPositions:
    def test_all_cjk(self) -> None:
        assert cjk_positions("测试") == [0, 1]

    def test_mixed(self) -> None:
        # "测" and "试" are CJK; "A" is not.
        pos = cjk_positions("测A试")
        assert set(pos) == {0, 2}

    def test_empty(self) -> None:
        assert cjk_positions("") == []


class TestVisibleLength:
    def test_whitespace_stripped(self) -> None:
        # Unicode category Zs (space separators) are invisible.
        assert visible_length("  第一句  ") == 3  # 第 + 一 + 句

    def test_chinese(self) -> None:
        assert visible_length("测试") == 2

    def test_mixed(self) -> None:
        assert visible_length("测A试2") == 4


class TestVisibleCharPositions:
    def test_strips_whitespace(self) -> None:
        assert 0 not in visible_char_positions(" 测试")
        assert 1 in visible_char_positions(" 测试")

    def test_empty(self) -> None:
        assert visible_char_positions("   ") == []


class TestLocalRNG:
    def test_seed_reproducible(self) -> None:
        rng1 = local_rng(42)
        rng2 = local_rng(42)
        assert [rng1.random() for _ in range(5)] == [rng2.random() for _ in range(5)]

    def test_different_seeds_different(self) -> None:
        rng1 = local_rng(1)
        rng2 = local_rng(2)
        seq1 = [rng1.random() for _ in range(3)]
        seq2 = [rng2.random() for _ in range(3)]
        assert seq1 != seq2


class TestSelectKUnique:
    def test_exact_k(self) -> None:
        import random

        rng = random.Random(0)
        result = select_k_unique(rng, list(range(10)), k=3)
        assert len(result) == 3
        assert len(set(result)) == 3

    def test_k_exceeds_pool(self) -> None:
        import random

        rng = random.Random(0)
        result = select_k_unique(rng, list(range(3)), k=5)
        assert len(result) == 3

    def test_empty_pool(self) -> None:
        import random

        rng = random.Random(0)
        result = select_k_unique(rng, [], k=3)
        assert result == []


class TestRoundRate:
    def test_basic(self) -> None:
        assert round_rate(10, 0.1) == 1  # 10*0.1 = 1
        assert round_rate(100, 0.15) == 15  # exact

    def test_rounds_up_from_half(self) -> None:
        # 10 * 0.05 = 0.5 → round → 1 (banker's rounding: Python rounds half to even)
        # round(0.5) = 0, round(1.5) = 2, round(2.5) = 2
        # But max(1, ...) ensures minimum of 1
        assert round_rate(20, 0.05) == 1  # 20*0.05=1.0

    def test_minimum_one(self) -> None:
        assert round_rate(1, 0.01) == 1  # 0.01 → 0 → max(1,0) = 1


class TestApplyOps:
    def test_replace(self) -> None:
        payload = PairPayload(text_a="第一句", text_b="第二句")
        ops = [
            CorruptionOp(
                op_index=0,
                op_type=ChangeOperation.REPLACE,
                field=PayloadField.TEXT_A,
                start=0,
                end=1,
                original="第",
                replacement="苐",
                unit_kind="char",
            ),
        ]
        result = cast(PairPayload, apply_ops(payload, ops))
        assert result.text_a == "苐一句"
        assert result.text_b == "第二句"  # unchanged

    def test_duplicate(self) -> None:
        payload = PairPayload(text_a="第一句", text_b="第二句")
        ops = [
            CorruptionOp(
                op_index=0,
                op_type=ChangeOperation.DUPLICATE,
                field=PayloadField.TEXT_A,
                start=0,
                end=1,
                original="第",
                replacement="第",
                unit_kind="char",
            ),
        ]
        result = cast(PairPayload, apply_ops(payload, ops))
        # DUPLICATE: insert original after position
        assert "第" in result.text_a
        assert len(result.text_a) > len("第一句")

    def test_insert(self) -> None:
        payload = PairPayload(text_a="测试", text_b="占位")
        ops = [
            CorruptionOp(
                op_index=0,
                op_type=ChangeOperation.INSERT,
                field=PayloadField.TEXT_A,
                start=1,
                end=1,
                original="",
                replacement="α",
                unit_kind="unicode",
            ),
        ]
        result = cast(PairPayload, apply_ops(payload, ops))
        assert "α" in result.text_a

    def test_ops_in_reverse_order(self) -> None:
        # Two replacements at different positions
        payload = PairPayload(text_a="测试", text_b="占位")
        ops = [
            CorruptionOp(
                op_index=0,
                op_type=ChangeOperation.REPLACE,
                field=PayloadField.TEXT_A,
                start=0,
                end=1,
                original="测",
                replacement="X",
                unit_kind="char",
            ),
            CorruptionOp(
                op_index=1,
                op_type=ChangeOperation.REPLACE,
                field=PayloadField.TEXT_A,
                start=1,
                end=2,
                original="试",
                replacement="Y",
                unit_kind="char",
            ),
        ]
        result = cast(PairPayload, apply_ops(payload, ops))
        assert result.text_a == "XY"
