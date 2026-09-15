"""Change-trace models and the fail-closed redaction transform (WP2).

Two layers, per DEC-010:

- :class:`InternalChangeTrace` (local-only) carries the FULL edit record, including
  the ``original`` and ``replacement`` substrings needed to rebuild every edit.
- :class:`PublicRedactedTrace` is the safe subset: it structurally CANNOT hold
  ``original`` / ``replacement`` / any text window — those fields do not exist on
  :class:`PublicChangeOp`.

:func:`redact_trace` is the single convergence point that turns an internal trace
into a public one. It is deterministic, does not mutate its input, and builds public
metadata from an explicit allowlist (never copy-then-delete). It fails closed with
:class:`~ruchi_bench.schema.errors.TraceRedactionError` rather than degrading to an
unsafe copy.

Position semantics are FROZEN: ``start`` / ``end`` are Python Unicode **code-point**
half-open ``[start, end)`` offsets into the ORIGINAL clean field string — never UTF-8
byte offsets, never rendered glyph positions. Structure only in WP2; Phase 03 fills
the ops.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, PayloadField
from ruchi_bench.schema.errors import TraceRedactionError
from ruchi_bench.schema.payloads import NonBlankStr

__all__ = [
    "PUBLIC_METADATA_ALLOWLIST",
    "REDACTION_VERSION",
    "InternalChangeOp",
    "InternalChangeTrace",
    "PublicChangeOp",
    "PublicRedactedTrace",
    "redact_trace",
]

#: Version stamp for the redaction contract; bump if the public shape/rules change.
REDACTION_VERSION: Final[str] = "1.0.0"

#: The ONLY metadata keys allowed to cross into a public trace. Each must be
#: non-text, non-reconstructive. Empty-by-default philosophy: add keys deliberately.
PUBLIC_METADATA_ALLOWLIST: Final[frozenset[str]] = frozenset(
    {"candidate_count", "unicode_category"}
)

NonNegInt = Annotated[int, Field(ge=0)]


class InternalChangeOp(BaseModel):
    """One edit, full record (local-only). Holds original/replacement substrings."""

    model_config = ConfigDict(extra="forbid")

    op_index: NonNegInt
    op_type: ChangeOperation
    field: PayloadField
    start: NonNegInt
    end: NonNegInt
    original: str
    replacement: str
    unit_kind: NonBlankStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_span(self) -> InternalChangeOp:
        # Half-open [start, end); insertions are zero-length (start == end).
        if self.end < self.start:
            raise ValueError(f"op {self.op_index}: end ({self.end}) < start ({self.start})")
        return self


class PublicChangeOp(BaseModel):
    """One edit, redacted. Structurally cannot carry original/replacement/text."""

    model_config = ConfigDict(extra="forbid")

    op_index: NonNegInt
    op_type: ChangeOperation
    field: PayloadField
    start: NonNegInt
    end: NonNegInt
    unit_kind: NonBlankStr
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


def _check_contiguous_indices(op_indices: tuple[int, ...]) -> None:
    """op_index must be 0,1,2,... contiguous (empty tuple is allowed → no ops)."""
    for position, op_index in enumerate(op_indices):
        if op_index != position:
            raise ValueError(f"op_index not contiguous from 0: expected {position}, got {op_index}")


class InternalChangeTrace(BaseModel):
    """Full, local-only change trace for one corrupted field-set."""

    model_config = ConfigDict(extra="forbid")

    corruption_name: CorruptionName
    corruption_level: NonBlankStr
    seed: NonNegInt
    ops: tuple[InternalChangeOp, ...]

    @model_validator(mode="after")
    def _check_ops(self) -> InternalChangeTrace:
        _check_contiguous_indices(tuple(op.op_index for op in self.ops))
        return self


class PublicRedactedTrace(BaseModel):
    """Safe, publishable change trace. Always the redacted shape regardless of Level.

    Whether Level A/B additionally publishes the internal trace is a release-policy
    decision made elsewhere; it must never make this public trace unsafe.
    """

    model_config = ConfigDict(extra="forbid")

    corruption_name: CorruptionName
    corruption_level: NonBlankStr
    seed: NonNegInt
    ops: tuple[PublicChangeOp, ...]
    redaction_version: NonBlankStr = REDACTION_VERSION

    @model_validator(mode="after")
    def _check_ops(self) -> PublicRedactedTrace:
        _check_contiguous_indices(tuple(op.op_index for op in self.ops))
        return self


def _redact_metadata(metadata: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Build a fresh public metadata dict from the allowlist ONLY.

    Never copies the internal dict and deletes keys — it constructs a new dict by
    reading only allowlisted keys, in sorted order for determinism. Keys outside the
    allowlist are silently dropped (not an error — they simply never leak). An
    allowlisted key whose value fails its type check is a hard failure
    (:class:`TraceRedactionError`): we fail closed rather than emit a suspect value.
    """
    safe: dict[str, JsonValue] = {}
    for key in sorted(PUBLIC_METADATA_ALLOWLIST):
        if key not in metadata:
            continue
        value = metadata[key]
        if key == "candidate_count":
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise TraceRedactionError(
                    f"metadata['candidate_count'] must be a non-negative int, got {value!r}"
                )
            safe[key] = value
        elif key == "unicode_category":
            # A Unicode general-category code: two ASCII letters, e.g. 'Lo', 'Nd'.
            if (
                not isinstance(value, str)
                or len(value) != 2
                or not value.isalpha()
                or not value.isascii()
            ):
                raise TraceRedactionError(
                    f"metadata['unicode_category'] must be a 2-letter ASCII category "
                    f"code, got {value!r}"
                )
            safe[key] = value
    return safe


def redact_trace(internal: InternalChangeTrace) -> PublicRedactedTrace:
    """Convert an internal trace to its safe public form. Fail-closed, deterministic.

    Does not mutate ``internal``. Drops ``original`` / ``replacement`` (they have no
    field on :class:`PublicChangeOp`) and rebuilds metadata from
    :data:`PUBLIC_METADATA_ALLOWLIST`. Raises :class:`TraceRedactionError` on any
    metadata value that is present-but-invalid.
    """
    public_ops = tuple(
        PublicChangeOp(
            op_index=op.op_index,
            op_type=op.op_type,
            field=op.field,
            start=op.start,
            end=op.end,
            unit_kind=op.unit_kind,
            metadata=_redact_metadata(op.metadata),
        )
        for op in internal.ops
    )
    return PublicRedactedTrace(
        corruption_name=internal.corruption_name,
        corruption_level=internal.corruption_level,
        seed=internal.seed,
        ops=public_ops,
        redaction_version=REDACTION_VERSION,
    )
