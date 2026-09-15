"""Corruption execution result types.

Defines the three legal execution outcomes without conflating them with exceptions:

- ``APPLIED``: corruption succeeded; ``corrupted_payload`` and both traces are present.
- ``NOT_APPLIED``: input is structurally valid but no eligible target was found
  (e.g. no Chinese characters for VSCR, no eligible words for TPWR).
- ``INVALID``: input structure, config, or target fields are illegal for this corruptor
  (e.g. C3 targeting non-context fields, malformed payload).

``NOT_APPLIED`` is a normal, non-error outcome. ``INVALID`` signals a programming
error or a misuse of the corruptor API.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dc_field
from typing import Any

from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, PayloadField
from ruchi_bench.schema.payloads import Payload

__all__ = ["CorruptionStatus", "CorruptionResult"]


class CorruptionStatus:
    """Corruption execution outcome (closed value space)."""

    __slots__ = ("_value",)
    _value: str

    def __init__(self, value: str) -> None:
        object.__setattr__(self, "_value", value)

    def __repr__(self) -> str:
        return f"CorruptionStatus.{self._value}"

    def __str__(self) -> str:
        return self._value

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CorruptionStatus):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._value)


#: Corruption succeeded; the payload was modified.
APPLIED = CorruptionStatus("APPLIED")
#: No eligible target; input is valid but nothing to corrupt.
NOT_APPLIED = CorruptionStatus("NOT_APPLIED")
#: Input or configuration is illegal for this corruptor.
INVALID = CorruptionStatus("INVALID")


@dataclass(frozen=True)
class CorruptionOp:
    """One atomic edit within a corruption result.

    All positions use **code-point half-open [start, end)** offsets into the ORIGINAL
    clean field string. ``start == end`` represents an insertion point.
    """

    op_index: int
    op_type: ChangeOperation
    field: PayloadField
    start: int
    end: int
    original: str
    replacement: str
    unit_kind: str
    metadata: dict[str, Any] = dc_field(default_factory=dict)


@dataclass(frozen=True)
class CorruptionResult:
    """Result of a single corruptor invocation.

    An ``APPLIED`` result always carries a non-null ``corrupted_payload``, a non-empty
    list of ops, and ``change_count > 0``. ``NOT_APPLIED`` and ``INVALID`` results never
    carry a ``corrupted_payload``.
    """

    #: Execution outcome.
    status: CorruptionStatus
    #: Which corruption was applied (or attempted).
    corruption_name: CorruptionName
    #: The modified payload (present only when ``status is APPLIED``).
    corrupted_payload: Payload | None = None
    #: Atomic edit list (present only when ``status is APPLIED``).
    ops: tuple[CorruptionOp, ...] = dc_field(default_factory=tuple)
    #: How many fields were modified (present only when ``status is APPLIED``).
    change_count: int = 0
    #: Human-readable reason for ``NOT_APPLIED`` or ``INVALID``.
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is APPLIED:
            if self.corrupted_payload is None:
                raise ValueError("APPLIED result must carry a corrupted_payload")
            if not self.ops:
                raise ValueError("APPLIED result must have at least one op")
            if self.change_count <= 0:
                raise ValueError("APPLIED result must have change_count > 0")
            if self.failure_reason is not None:
                raise ValueError("APPLIED result must not have a failure_reason")
        elif self.status is NOT_APPLIED:
            if self.corrupted_payload is not None:
                raise ValueError("NOT_APPLIED result must not carry a corrupted_payload")
            if self.ops:
                raise ValueError("NOT_APPLIED result must not carry ops")
            if self.change_count != 0:
                raise ValueError("NOT_APPLIED result must have change_count == 0")
            if self.failure_reason is None:
                raise ValueError("NOT_APPLIED result must carry a failure_reason")
        elif self.status is INVALID:
            if self.corrupted_payload is not None:
                raise ValueError("INVALID result must not carry a corrupted_payload")
            if self.ops:
                raise ValueError("INVALID result must not carry ops")
            if self.change_count != 0:
                raise ValueError("INVALID result must have change_count == 0")
            if self.failure_reason is None:
                raise ValueError("INVALID result must carry a failure_reason")
