"""Shared utilities for all seven corruptors.

All utilities are pure Python (no I/O, no external resources) so they can be used freely
across corruptors. Complex resources (lexicons, confusion tables, Unicode tables) live
in their own per-corruptor modules.
"""

from __future__ import annotations

import random
import re
import unicodedata
from collections.abc import Sequence
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, cast

from pydantic import JsonValue

from ruchi_bench.schema.enums import ChangeOperation, CorruptionName, PayloadField
from ruchi_bench.schema.payloads import Payload

if TYPE_CHECKING:
    from ruchi_bench.corruptions.result import CorruptionOp
    from ruchi_bench.schema.trace import InternalChangeTrace, PublicRedactedTrace


#: Characters classified as "Chinese" (CJK Unified Ideographs and extensions).
_RE_CJK = re.compile(r"[一-鿿]")


def has_cjk(text: str) -> bool:
    """Return True if ``text`` contains at least one CJK character."""
    return bool(_RE_CJK.search(text))


def cjk_positions(text: str) -> list[int]:
    """Return the sorted list of code-point indices that are CJK characters."""
    return [i for i, ch in enumerate(text) if _RE_CJK.match(ch)]


def visible_length(text: str) -> int:
    """Count visible (non-whitespace, non-control) code-points."""
    return sum(1 for ch in text if not unicodedata.category(ch).startswith(("C", "Z")))


def visible_char_positions(text: str) -> list[int]:
    """Code-point indices of visible (non-whitespace, non-control) characters."""
    return [i for i, ch in enumerate(text) if not unicodedata.category(ch).startswith(("C", "Z"))]


def build_trace_ops(
    corruption_name: CorruptionName,
    corruption_level: str,
    seed: int,
    ops: Sequence[CorruptionOp],
    clean_payload: Payload,
) -> tuple[InternalChangeTrace, PublicRedactedTrace]:
    """Build internal and public change traces from a list of CorruptionOps.

    Imports are deferred inside to avoid circular imports at module level.
    """
    from ruchi_bench.schema.trace import (
        InternalChangeOp,
        InternalChangeTrace,
        redact_trace,
    )

    internal = InternalChangeTrace(
        corruption_name=corruption_name,
        corruption_level=corruption_level,
        seed=seed,
        ops=tuple(
            InternalChangeOp(
                op_index=op.op_index,
                op_type=op.op_type,
                field=op.field,
                start=op.start,
                end=op.end,
                original=op.original,
                replacement=op.replacement,
                unit_kind=op.unit_kind,
                metadata=cast("dict[str, JsonValue]", op.metadata),
            )
            for op in ops
        ),
    )
    public = redact_trace(internal)
    return internal, public


def apply_ops(
    clean_payload: Payload,
    ops: Sequence[CorruptionOp],
) -> Payload:
    """Apply a sequence of CorruptionOps to the clean payload, returning a new payload.

    Operations are applied in reverse code-point order per field, so that earlier
    insertions/replacements do not shift the positions of later ops within the same
    field. Ops from different fields are independent.
    """
    from collections import defaultdict

    # Group ops by field so we can sort and apply per field.
    by_field: dict[PayloadField, list[CorruptionOp]] = defaultdict(list)
    for op in ops:
        by_field[op.field].append(op)

    # Work on a mutable copy of the payload fields.
    payload_dict = clean_payload.model_dump()

    for field, field_ops in by_field.items():
        field_name = field.value
        text = payload_dict[field_name]
        assert isinstance(text, str)

        # Sort by position descending — work from end to start to avoid index drift.
        sorted_ops = sorted(field_ops, key=lambda op: (op.start, op.end), reverse=True)

        for op in sorted_ops:
            # All positions are in the ORIGINAL clean text; no operation applied yet,
            # so offsets are always valid against the current state of `text`.
            if op.op_type is ChangeOperation.REPLACE:
                text = text[: op.start] + op.replacement + text[op.end :]
            elif op.op_type is ChangeOperation.INSERT:
                text = text[: op.start] + op.replacement + text[op.start :]
            elif op.op_type is ChangeOperation.DELETE:
                text = text[: op.start] + text[op.end :]
            elif op.op_type is ChangeOperation.DUPLICATE:
                text = text[: op.end] + op.replacement + text[op.end :]

        payload_dict[field_name] = text

    return type(clean_payload).model_validate(payload_dict)


def local_rng(seed: int) -> random.Random:
    """Return a fresh local Random seeded with ``seed``; no global state."""
    rng = random.Random(seed)
    return rng


def select_k_unique(rng: random.Random, pool: Sequence[int], k: int) -> list[int]:
    """Select up to ``k`` unique items from ``pool`` using ``rng``.

    Returns all available items if ``k`` >= len(pool).
    Returns an empty list if ``pool`` is empty.
    """
    if not pool:
        return []
    k = min(k, len(pool))
    indices = rng.sample(range(len(pool)), k)
    return [pool[i] for i in sorted(indices)]


def round_rate(total: int, rate: float) -> int:
    """Apply a rate to a total count, returning max(1, round(total * rate)).

    This implements the k=max(1, round(...)) pattern used by rate-based strategies.
    """
    return max(1, round(total * rate))


def diff_to_ops(
    clean: str,
    corrupted: str,
    *,
    field: PayloadField,
    unit_kind: str,
    metadata: dict[str, object] | None = None,
    start_index: int = 0,
) -> list[CorruptionOp]:
    """Convert a clean/corrupted pair into traceable code-point edit operations.

    This is the adapter boundary for third-party augmenters.  The external library
    is free to choose its output string; the benchmark still records exact offsets
    against the clean input and reconstructs the output through :func:`apply_ops`.
    """
    from ruchi_bench.corruptions.result import CorruptionOp

    if clean == corrupted:
        return []
    matcher = SequenceMatcher(a=clean, b=corrupted, autojunk=False)
    common_metadata = dict(metadata or {})
    ops: list[CorruptionOp] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        original = clean[i1:i2]
        replacement = corrupted[j1:j2]
        if tag == "delete":
            op_type = ChangeOperation.DELETE
        elif tag == "insert":
            op_type = ChangeOperation.INSERT
        else:
            op_type = ChangeOperation.REPLACE
        ops.append(
            CorruptionOp(
                op_index=start_index + len(ops),
                op_type=op_type,
                field=field,
                start=i1,
                end=i2,
                original=original,
                replacement=replacement,
                unit_kind=unit_kind,
                metadata=dict(common_metadata),
            )
        )
    return ops


def is_word_boundary_left(text: str, pos: int) -> bool:
    """Return True if ``pos`` is a valid word-boundary insert point on the left."""
    if pos == 0:
        return True
    prev = text[pos - 1]
    if prev.isspace():
        return True
    return unicodedata.category(prev) in ("Ps", "Pe", "Po", "Pd")


def is_word_boundary_right(text: str, pos: int) -> bool:
    """Return True if ``pos`` is a valid word-boundary insert point on the right."""
    if pos >= len(text):
        return True
    curr = text[pos]
    if curr.isspace():
        return True
    return unicodedata.category(curr) in ("Ps", "Pe", "Po", "Pd")


def unicode_category(ch: str) -> str:
    """Return the Unicode general category for character ``ch`` (e.g. 'Lo', 'Nd')."""
    return unicodedata.category(ch)
