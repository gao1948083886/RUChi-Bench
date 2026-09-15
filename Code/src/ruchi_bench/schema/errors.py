"""Domain exception hierarchy for the RUChi-Bench schema and adapter layer.

Design rules (see ``CLAUDE.md`` — "Do not silently swallow errors"):

- Every RUChi-Bench-specific failure is a subclass of :class:`CStructBenchError`,
  so callers can catch the whole family with one ``except``.
- Adapter-boundary failures are :class:`AdapterError` subclasses. When an adapter
  wraps a lower-level failure (e.g. a Pydantic ``ValidationError``), it MUST chain
  the original via ``raise ... from exc`` so ``__cause__`` is preserved. These
  classes never suppress or hide the underlying error.
- Native Pydantic ``ValidationError`` is allowed to propagate unchanged *inside* the
  schema layer; it is only wrapped when it crosses the adapter boundary and we want a
  domain-typed error with dataset/field context.

No control-flow-by-exception shortcuts, no bare ``except``, no swallowing.
"""

from __future__ import annotations

__all__ = [
    "AdapterError",
    "C3StructureError",
    "CStructBenchError",
    "FieldMappingError",
    "InvalidSplitError",
    "LabelSpaceError",
    "SerializationError",
    "TraceRedactionError",
]


class CStructBenchError(Exception):
    """Base class for all RUChi-Bench domain errors."""


class AdapterError(CStructBenchError):
    """Base class for failures raised while adapting a raw dataset record.

    Subclasses carry enough context (dataset id, offending value) to make the log
    actionable without re-reading the raw file.
    """


class InvalidSplitError(AdapterError):
    """A record was drawn from a split the pilot does not admit (e.g. not ``test``)."""


class LabelSpaceError(AdapterError):
    """A gold label is outside the dataset's declared/allowed label space.

    Also raised when a derived projection (e.g. ASAP-Polarity) cannot map an
    otherwise-valid source label — that case is a legitimate *exclusion* decision and
    must be surfaced explicitly, never silently coerced to a wrong label.
    """


class FieldMappingError(AdapterError):
    """A required source field is missing, empty, or maps to an unexpected type."""


class C3StructureError(AdapterError):
    """C3 record structure violated an invariant.

    E.g. ``context``/``question``/``options``/``answer`` not all present as separate
    fields, or ``answer`` not found among ``options``. C3 fields must stay separate
    and must never be irreversibly concatenated (frozen rule).
    """


class SerializationError(CStructBenchError):
    """Serializing or deserializing a schema object failed or lost information."""


class TraceRedactionError(CStructBenchError):
    """Producing a public redacted change-trace failed its safety invariants.

    Raised when redaction would leak protected raw text (Level B/C datasets) or when
    the redacted trace can no longer be validated as a safe subset of the internal
    trace. Fail closed — never emit an unverified public trace.
    """
