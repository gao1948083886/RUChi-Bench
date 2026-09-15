"""Unit tests for ruchi_bench.schema.payloads. All text is synthetic."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from ruchi_bench.schema.enums import PayloadField, PayloadKind
from ruchi_bench.schema.payloads import (
    MRCPayload,
    PairPayload,
    Payload,
    SingleTextPayload,
)

_PAYLOAD_ADAPTER: TypeAdapter[object] = TypeAdapter(Payload)


def test_pair_payload_valid() -> None:
    p = PairPayload(text_a="第一句", text_b="第二句")
    assert p.payload_kind is PayloadKind.PAIR
    assert {PayloadField.TEXT_A, PayloadField.TEXT_B} == p.OWNED_FIELDS
    assert {PayloadField.TEXT_A, PayloadField.TEXT_B} == p.CORRUPTIBLE_FIELDS


def test_single_text_payload_valid() -> None:
    p = SingleTextPayload(text_a="一段评论")
    assert p.payload_kind is PayloadKind.SINGLE_TEXT
    assert {PayloadField.TEXT_A} == p.OWNED_FIELDS
    assert {PayloadField.TEXT_A} == p.CORRUPTIBLE_FIELDS


def test_mrc_payload_valid() -> None:
    p = MRCPayload(context="上下文", question="问题?", options=("甲", "乙", "丙"))
    assert p.payload_kind is PayloadKind.MRC
    assert p.options == ("甲", "乙", "丙")
    assert {
        PayloadField.CONTEXT,
        PayloadField.QUESTION,
        PayloadField.OPTIONS,
    } == p.OWNED_FIELDS
    assert {PayloadField.CONTEXT} == p.CORRUPTIBLE_FIELDS


@pytest.mark.parametrize("blank", ["", "   ", "\t\n"])
def test_pair_rejects_blank_text(blank: str) -> None:
    with pytest.raises(ValidationError):
        PairPayload(text_a=blank, text_b="ok")


def test_payload_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        SingleTextPayload(text_a="ok", extra="nope")  # type: ignore[call-arg]


def test_mrc_rejects_fewer_than_two_options() -> None:
    with pytest.raises(ValidationError):
        MRCPayload(context="c", question="q", options=("only",))


def test_mrc_rejects_blank_option() -> None:
    with pytest.raises(ValidationError):
        MRCPayload(context="c", question="q", options=("ok", "  "))


def test_mrc_has_no_combined_text_field() -> None:
    """Structure must not offer a concatenated prompt/combined field."""
    assert "prompt" not in MRCPayload.model_fields
    assert "combined_text" not in MRCPayload.model_fields


def test_discriminated_union_dispatches_on_payload_kind() -> None:
    pair = _PAYLOAD_ADAPTER.validate_python({"payload_kind": "pair", "text_a": "a", "text_b": "b"})
    assert isinstance(pair, PairPayload)
    mrc = _PAYLOAD_ADAPTER.validate_python(
        {"payload_kind": "mrc", "context": "c", "question": "q", "options": ["x", "y"]}
    )
    assert isinstance(mrc, MRCPayload)


def test_discriminated_union_rejects_bad_discriminator() -> None:
    with pytest.raises(ValidationError):
        _PAYLOAD_ADAPTER.validate_python(
            {"payload_kind": "not_a_kind", "text_a": "a", "text_b": "b"}
        )


def test_discriminated_union_rejects_kind_field_mismatch() -> None:
    """A pair discriminator with MRC fields must fail (extra fields forbidden)."""
    with pytest.raises(ValidationError):
        _PAYLOAD_ADAPTER.validate_python(
            {"payload_kind": "pair", "context": "c", "question": "q", "options": ["x", "y"]}
        )


def test_payload_json_round_trip() -> None:
    p = MRCPayload(context="上下文", question="问题?", options=("甲", "乙"))
    restored = _PAYLOAD_ADAPTER.validate_python(p.model_dump(mode="json"))
    assert restored == p
