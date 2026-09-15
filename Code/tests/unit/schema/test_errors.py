"""Unit tests for ruchi_bench.schema.errors."""

from __future__ import annotations

import pytest

from ruchi_bench.schema.errors import (
    AdapterError,
    C3StructureError,
    CStructBenchError,
    FieldMappingError,
    InvalidSplitError,
    LabelSpaceError,
    SerializationError,
    TraceRedactionError,
)

ADAPTER_SUBCLASSES = [
    InvalidSplitError,
    LabelSpaceError,
    FieldMappingError,
    C3StructureError,
]
ALL_DOMAIN_ERRORS = [
    AdapterError,
    SerializationError,
    TraceRedactionError,
    *ADAPTER_SUBCLASSES,
]


@pytest.mark.parametrize("exc_cls", ALL_DOMAIN_ERRORS)
def test_all_inherit_base(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, CStructBenchError)
    assert issubclass(exc_cls, Exception)


@pytest.mark.parametrize("exc_cls", ADAPTER_SUBCLASSES)
def test_adapter_subclasses_inherit_adapter_error(exc_cls: type[Exception]) -> None:
    assert issubclass(exc_cls, AdapterError)


def test_serialization_and_trace_are_not_adapter_errors() -> None:
    """These sit directly under the base, not under AdapterError."""
    assert not issubclass(SerializationError, AdapterError)
    assert not issubclass(TraceRedactionError, AdapterError)


@pytest.mark.parametrize("exc_cls", ALL_DOMAIN_ERRORS)
def test_catchable_as_base(exc_cls: type[Exception]) -> None:
    with pytest.raises(CStructBenchError):
        raise exc_cls("boom")


def test_adapter_family_catchable_as_adapter_error() -> None:
    for exc_cls in ADAPTER_SUBCLASSES:
        with pytest.raises(AdapterError):
            raise exc_cls("boom")


def test_cause_chaining_is_preserved() -> None:
    """Wrapping a lower-level error must preserve __cause__ (never swallow)."""
    original = ValueError("raw parse failed")
    try:
        try:
            raise original
        except ValueError as exc:
            raise FieldMappingError("field 'review' missing") from exc
    except FieldMappingError as wrapped:
        assert wrapped.__cause__ is original
    else:  # pragma: no cover - defensive
        pytest.fail("FieldMappingError was not raised")
