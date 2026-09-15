"""Tests for ruchi_bench.corruptions.result."""

from __future__ import annotations

from typing import Any

import pytest

from ruchi_bench.corruptions.result import (
    APPLIED,
    INVALID,
    NOT_APPLIED,
    CorruptionOp,
    CorruptionResult,
)
from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, PayloadField

_STATUSES = [APPLIED, NOT_APPLIED, INVALID]


class TestCorruptionStatus:
    def test_three_legal_statuses(self) -> None:
        assert len(_STATUSES) == 3

    def test_equality(self) -> None:
        assert APPLIED == APPLIED
        assert APPLIED != NOT_APPLIED
        assert APPLIED == "APPLIED"
        assert APPLIED != "applied"
        assert NOT_APPLIED == NOT_APPLIED
        assert INVALID == INVALID

    def test_repr(self) -> None:
        assert repr(APPLIED) == "CorruptionStatus.APPLIED"
        assert repr(NOT_APPLIED) == "CorruptionStatus.NOT_APPLIED"
        assert repr(INVALID) == "CorruptionStatus.INVALID"

    def test_hash(self) -> None:
        assert len({APPLIED, NOT_APPLIED, INVALID}) == 3


class TestCorruptionOp:
    def test_minimal_op(self) -> None:
        op = CorruptionOp(
            op_index=0,
            op_type=ChangeOperation.REPLACE,
            field=PayloadField.TEXT_A,
            start=0,
            end=1,
            original="原",
            replacement="新",
            unit_kind="char",
        )
        assert op.op_index == 0
        assert op.unit_kind == "char"

    def test_op_with_metadata(self) -> None:
        op = CorruptionOp(
            op_index=0,
            op_type=ChangeOperation.INSERT,
            field=PayloadField.CONTEXT,
            start=5,
            end=5,
            original="",
            replacement="α",
            unit_kind="unicode",
            metadata={"unicode_name": "GREEK SMALL LETTER ALPHA"},
        )
        assert op.metadata["unicode_name"] == "GREEK SMALL LETTER ALPHA"


class TestCorruptionResult_APPLIED:
    def _applied_kwargs(self, **kw: Any) -> dict[str, Any]:
        from ruchi_bench.schema.payloads import PairPayload

        return {
            "status": APPLIED,
            "corruption_name": CorruptionName.VSCR,
            "corrupted_payload": PairPayload(text_a="苐", text_b="二"),
            "ops": (
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
            ),
            "change_count": 1,
            "failure_reason": None,
        }

    def test_valid_applied(self) -> None:
        result = CorruptionResult(**self._applied_kwargs())
        assert result.status is APPLIED
        assert result.corrupted_payload is not None
        assert result.change_count == 1

    def test_applied_missing_payload_raises(self) -> None:
        kw = self._applied_kwargs()
        del kw["corrupted_payload"]
        with pytest.raises(ValueError, match="corrupted_payload"):
            CorruptionResult(**kw)

    def test_applied_empty_ops_raises(self) -> None:
        kw = self._applied_kwargs()
        kw["ops"] = ()
        with pytest.raises(ValueError, match="at least one op"):
            CorruptionResult(**kw)

    def test_applied_change_count_zero_raises(self) -> None:
        kw = self._applied_kwargs()
        kw["change_count"] = 0
        with pytest.raises(ValueError, match="change_count > 0"):
            CorruptionResult(**kw)

    def test_applied_with_failure_reason_raises(self) -> None:
        kw = self._applied_kwargs()
        kw["failure_reason"] = "nope"
        with pytest.raises(ValueError, match="failure_reason"):
            CorruptionResult(**kw)


class TestCorruptionResult_NOT_APPLIED:
    def _not_applied_kwargs(self, **kw: Any) -> dict[str, Any]:
        return {
            "status": NOT_APPLIED,
            "corruption_name": CorruptionName.TPWR,
            "corrupted_payload": None,
            "ops": (),
            "change_count": 0,
            "failure_reason": "no eligible words found",
        }

    def test_valid_not_applied(self) -> None:
        result = CorruptionResult(**self._not_applied_kwargs())
        assert result.status is NOT_APPLIED
        assert result.failure_reason == "no eligible words found"

    def test_not_applied_with_payload_raises(self) -> None:
        from ruchi_bench.schema.payloads import PairPayload

        kw = self._not_applied_kwargs()
        kw["corrupted_payload"] = PairPayload(text_a="x", text_b="y")
        with pytest.raises(ValueError, match="not carry a corrupted_payload"):
            CorruptionResult(**kw)

    def test_not_applied_without_reason_raises(self) -> None:
        kw = self._not_applied_kwargs()
        kw["failure_reason"] = None
        with pytest.raises(ValueError, match="failure_reason"):
            CorruptionResult(**kw)


class TestCorruptionResult_INVALID:
    def _invalid_kwargs(self, **kw: Any) -> dict[str, Any]:
        return {
            "status": INVALID,
            "corruption_name": CorruptionName.CR,
            "corrupted_payload": None,
            "ops": (),
            "change_count": 0,
            "failure_reason": "unknown level 'unknown'",
        }

    def test_valid_invalid(self) -> None:
        result = CorruptionResult(**self._invalid_kwargs())
        assert result.status is INVALID

    def test_invalid_without_reason_raises(self) -> None:
        kw = self._invalid_kwargs()
        kw["failure_reason"] = None
        with pytest.raises(ValueError, match="failure_reason"):
            CorruptionResult(**kw)
