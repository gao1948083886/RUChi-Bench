"""Structural payload models for the RUChi-Bench unified schema (WP2).

Payloads are split by *input structure*, not by semantic task: PAWS-X, XNLI and
LCQMC all normalize to ``text_a`` + ``text_b`` and therefore share
:class:`PairPayload`; the semantic distinction between them lives in
:class:`~ruchi_bench.schema.enums.BenchmarkTaskType`, not here. The three structural
payloads are :class:`PairPayload`, :class:`SingleTextPayload` and :class:`MRCPayload`.

The union :data:`Payload` is discriminated on ``payload_kind`` (a
:class:`~ruchi_bench.schema.enums.PayloadKind`), NOT on the benchmark task type —
the task type is not part of the payload and several tasks map to one payload kind.

All payloads:

- forbid unknown fields (``extra="forbid"``);
- require every text field to be non-empty after ``str.strip()`` (the stored value is
  the ORIGINAL untrimmed text — trimming is only a non-emptiness check, so code-point
  positions used by change-traces stay faithful to the source);
- expose their owned fields (``OWNED_FIELDS``) and the subset that a corruption may
  target (``CORRUPTIBLE_FIELDS``).

MRC keeps ``context`` / ``question`` / ``options`` as separate fields and never
provides a concatenated ``prompt`` / ``combined_text`` field: the structure itself
makes irreversible concatenation impossible.
"""

from __future__ import annotations

from typing import Annotated, ClassVar, Literal

from pydantic import AfterValidator, BaseModel, ConfigDict, Field

from ruchi_bench.schema.enums import PayloadField, PayloadKind

__all__ = [
    "NonBlankStr",
    "MRCPayload",
    "PairPayload",
    "Payload",
    "SingleTextPayload",
]


def _reject_blank(value: str) -> str:
    """Reject strings that are empty or whitespace-only; return the value unchanged.

    The value is NOT trimmed — only checked — so downstream code-point positions in
    change-traces remain faithful to the original source text.
    """
    if not value.strip():
        raise ValueError("text field must be non-empty after trimming")
    return value


#: A non-empty (after strip) string field, stored verbatim. Reused by trace/sample.
NonBlankStr = Annotated[str, AfterValidator(_reject_blank)]


class PairPayload(BaseModel):
    """Two-text input: ``text_a`` + ``text_b`` (PAWS-X, XNLI, LCQMC)."""

    model_config = ConfigDict(extra="forbid")

    payload_kind: Literal[PayloadKind.PAIR] = PayloadKind.PAIR
    text_a: NonBlankStr
    text_b: NonBlankStr

    OWNED_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset(
        {PayloadField.TEXT_A, PayloadField.TEXT_B}
    )
    CORRUPTIBLE_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset(
        {PayloadField.TEXT_A, PayloadField.TEXT_B}
    )


class SingleTextPayload(BaseModel):
    """Single-text input: ``text_a`` (ASAP / ASAP-Polarity)."""

    model_config = ConfigDict(extra="forbid")

    payload_kind: Literal[PayloadKind.SINGLE_TEXT] = PayloadKind.SINGLE_TEXT
    text_a: NonBlankStr

    OWNED_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset({PayloadField.TEXT_A})
    CORRUPTIBLE_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset({PayloadField.TEXT_A})


class MRCPayload(BaseModel):
    """Multiple-choice MRC input (C3): separate context / question / options.

    ``context``, ``question`` and ``options`` are always stored separately; there is
    intentionally no concatenated field. Only ``context`` is corruptible.
    """

    model_config = ConfigDict(extra="forbid")

    payload_kind: Literal[PayloadKind.MRC] = PayloadKind.MRC
    context: NonBlankStr
    question: NonBlankStr
    options: tuple[NonBlankStr, ...] = Field(min_length=2)

    OWNED_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset(
        {PayloadField.CONTEXT, PayloadField.QUESTION, PayloadField.OPTIONS}
    )
    CORRUPTIBLE_FIELDS: ClassVar[frozenset[PayloadField]] = frozenset({PayloadField.CONTEXT})


#: Discriminated union of the structural payloads, keyed on ``payload_kind``.
Payload = Annotated[
    PairPayload | SingleTextPayload | MRCPayload,
    Field(discriminator="payload_kind"),
]
