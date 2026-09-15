"""Unit tests for ruchi_bench.schema.trace. All text is synthetic."""

from __future__ import annotations

import pytest
from pydantic import JsonValue, ValidationError

from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, PayloadField
from ruchi_bench.schema.errors import TraceRedactionError
from ruchi_bench.schema.trace import (
    InternalChangeOp,
    InternalChangeTrace,
    PublicChangeOp,
    PublicRedactedTrace,
    redact_trace,
)

# The raw text that MUST NOT appear in any public/redacted output.
_SECRET_ORIGINAL = "机密原文"
_SECRET_REPLACEMENT = "泄露替换"


def _internal_op(
    op_index: int = 0,
    *,
    op_type: ChangeOperation = ChangeOperation.REPLACE,
    field: PayloadField = PayloadField.CONTEXT,
    start: int = 2,
    end: int = 4,
    metadata: dict[str, JsonValue] | None = None,
) -> InternalChangeOp:
    return InternalChangeOp(
        op_index=op_index,
        op_type=op_type,
        field=field,
        start=start,
        end=end,
        original=_SECRET_ORIGINAL,
        replacement=_SECRET_REPLACEMENT,
        unit_kind="char",
        metadata=metadata or {},
    )


def _internal_trace(*ops: InternalChangeOp) -> InternalChangeTrace:
    return InternalChangeTrace(
        corruption_name=CorruptionName.VSCR,
        corruption_level="low",
        seed=42,
        ops=tuple(ops),
    )


def test_half_open_span_ok() -> None:
    op = _internal_op(start=3, end=5)
    assert op.end >= op.start


def test_insertion_zero_length_span_ok() -> None:
    op = _internal_op(op_type=ChangeOperation.INSERT, start=4, end=4)
    assert op.start == op.end


@pytest.mark.parametrize(("start", "end"), [(5, 3), (0, -1), (-1, 2)])
def test_invalid_span_rejected(start: int, end: int) -> None:
    with pytest.raises(ValidationError):
        _internal_op(start=start, end=end)


def test_noncontiguous_op_index_rejected() -> None:
    with pytest.raises(ValidationError):
        _internal_trace(_internal_op(0), _internal_op(2))


def test_contiguous_ops_ok() -> None:
    trace = _internal_trace(_internal_op(0), _internal_op(1))
    assert tuple(op.op_index for op in trace.ops) == (0, 1)


def test_public_op_has_no_text_fields() -> None:
    """PublicChangeOp must not even declare original/replacement fields."""
    assert "original" not in PublicChangeOp.model_fields
    assert "replacement" not in PublicChangeOp.model_fields


def test_redact_drops_all_raw_text() -> None:
    trace = _internal_trace(
        _internal_op(0, metadata={"candidate_count": 3, "unicode_category": "Lo"})
    )
    public = redact_trace(trace)
    dumped = repr(public.model_dump(mode="json"))
    assert _SECRET_ORIGINAL not in dumped
    assert _SECRET_REPLACEMENT not in dumped


def test_redact_metadata_allowlist_only() -> None:
    trace = _internal_trace(
        _internal_op(
            0,
            metadata={
                "candidate_count": 5,
                "unicode_category": "Nd",
                "leaky_context": "秘密窗口",
                "original_hint": _SECRET_ORIGINAL,
            },
        )
    )
    public = redact_trace(trace)
    md = public.ops[0].metadata
    assert md == {"candidate_count": 5, "unicode_category": "Nd"}
    assert "leaky_context" not in md
    assert "original_hint" not in md


def test_redact_is_deterministic() -> None:
    trace = _internal_trace(
        _internal_op(0, metadata={"candidate_count": 2, "unicode_category": "Lo"})
    )
    first = redact_trace(trace).model_dump(mode="json")
    second = redact_trace(trace).model_dump(mode="json")
    assert first == second


def test_redact_does_not_mutate_input() -> None:
    original_md: dict[str, JsonValue] = {"candidate_count": 4, "leaky": "x"}
    trace = _internal_trace(_internal_op(0, metadata=original_md))
    redact_trace(trace)
    assert trace.ops[0].metadata == original_md
    assert trace.ops[0].original == _SECRET_ORIGINAL


def test_redact_fails_closed_on_bad_allowlisted_metadata() -> None:
    """An allowlisted key with an invalid value must raise, not degrade."""
    bad = _internal_trace(_internal_op(0, metadata={"candidate_count": -1}))
    with pytest.raises(TraceRedactionError):
        redact_trace(bad)
    bad2 = _internal_trace(_internal_op(0, metadata={"unicode_category": "长文本"}))
    with pytest.raises(TraceRedactionError):
        redact_trace(bad2)


def test_public_trace_preserves_positions_and_kind() -> None:
    trace = _internal_trace(_internal_op(0, start=1, end=3))
    public = redact_trace(trace)
    assert public.ops[0].start == 1
    assert public.ops[0].end == 3
    assert public.ops[0].op_type is ChangeOperation.REPLACE
    assert public.corruption_name is CorruptionName.VSCR
    assert public.seed == 42


def test_public_trace_round_trip() -> None:
    trace = _internal_trace(_internal_op(0))
    public = redact_trace(trace)
    restored = PublicRedactedTrace.model_validate(public.model_dump(mode="json"))
    assert restored == public
