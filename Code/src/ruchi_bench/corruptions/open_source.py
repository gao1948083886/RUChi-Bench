"""Adapters for the selected open-source Chinese text augmentation methods.

The third-party packages return strings rather than edit traces.  This module keeps
that boundary explicit: the package produces a candidate string, and RUChi-Bench
converts the clean/candidate pair into code-point operations before accepting it.
"""

from __future__ import annotations

from collections.abc import Callable
from difflib import SequenceMatcher
from functools import cache
from typing import Any

from ruchi_bench.corruptions.base import _check_sample_can_be_corrupted
from ruchi_bench.corruptions.result import (
    APPLIED,
    INVALID,
    NOT_APPLIED,
    CorruptionOp,
    CorruptionResult,
)
from ruchi_bench.corruptions.utils import apply_ops, diff_to_ops
from ruchi_bench.schema.enums import CorruptionName, DatasetName, PayloadField
from ruchi_bench.schema.sample import Sample

TextTransform = Callable[[str, float, int], str | None]
_NORMAL_PUNCTUATION = "，。！？；：、（）【】《》“”‘’…——·「」『』"


def _is_normal_noise_char(char: str) -> bool:
    """Allow ordinary Chinese/ASCII text, punctuation, and whitespace only."""
    # ``str.isspace()`` also accepts U+2028/U+2029.  Those characters are legal
    # inside JSON strings, but many JSONL readers treat them as physical record
    # separators.  Restrict whitespace to the four characters that are safe in
    # our line-oriented format.
    if char in " \t\r\n":
        return True
    codepoint = ord(char)
    if 0x3400 <= codepoint <= 0x4DBF or 0x4E00 <= codepoint <= 0x9FFF:
        return True
    return char in _NORMAL_PUNCTUATION or (
        char.isascii()
        and (char.isalnum() or char in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    )


def _sanitize_added_chars(source: str, candidate: str) -> str | None:
    """Remove disallowed inserted characters while preserving clean text."""
    sanitized: list[str] = []
    matcher = SequenceMatcher(a=source, b=candidate, autojunk=False)
    for tag, _i1, _i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            sanitized.append(candidate[j1:j2])
        elif tag == "insert":
            sanitized.append(
                "".join(char for char in candidate[j1:j2] if _is_normal_noise_char(char))
            )
        elif tag == "replace":
            # ADD_NOISE should not delete clean content. Keep only normal
            # characters from a replacement-like alignment and let the edit
            # budget filter decide whether the candidate is acceptable.
            sanitized.append(
                "".join(char for char in candidate[j1:j2] if _is_normal_noise_char(char))
            )
        else:  # pragma: no cover - defensive for the add-only augmenter
            return None
    return "".join(sanitized)


def _has_disallowed_added_chars(source: str, candidate: str) -> bool:
    """Return whether sanitizing the added characters would change the output."""
    sanitized = _sanitize_added_chars(source, candidate)
    return sanitized is None or sanitized != candidate


@cache
def _jionlp_augmenter(method: str, mode: str, rate: float) -> Any:
    """Create one configured JioNLP instance and reuse its loaded resources.

    JioNLP's public singleton rebuilds the large character distribution whenever its
    seed changes. We instead cache one instance per configuration and control the
    NumPy state locally below; this preserves the upstream algorithm without paying
    the resource-loading cost for every sample.
    """
    if method == "homophone_substitution":
        from jionlp.textaug.homophone_substitution import (  # type: ignore[import-untyped]
            HomophoneSubstitution,
        )

        return HomophoneSubstitution()
    if method == "random_add_delete":
        from jionlp.textaug.random_add_delete import RandomAddDelete  # type: ignore[import-untyped]

        return RandomAddDelete()
    if method == "swap_char_position":
        from jionlp.textaug.swap_char_position import (  # type: ignore[import-untyped]
            SwapCharPosition,
        )

        return SwapCharPosition()
    raise ValueError(f"unsupported JioNLP method {method!r}")


def valid_target(sample: Sample, field: PayloadField) -> bool:
    """Return whether a field is legal for the current sample."""
    if sample.dataset_name is DatasetName.C3:
        return field is PayloadField.CONTEXT
    return field in sample.clean_payload.CORRUPTIBLE_FIELDS


def validate_call(
    sample: Sample, *, name: CorruptionName, seed: int, level: str, levels: tuple[str, ...]
) -> str | None:
    """Run common input checks and return an error message, if any."""
    _check_sample_can_be_corrupted(sample)
    if level not in levels:
        return f"unknown level {level!r}"
    if seed < 0:
        return "seed must be non-negative"
    if any(not valid_target(sample, field) for field in sample.target_fields):
        return "one or more target fields are illegal for this sample"
    return None


def apply_text_transform(
    sample: Sample,
    *,
    name: CorruptionName,
    seed: int,
    level: str,
    rate: float,
    transform: TextTransform,
    unit_kind: str,
    levels: tuple[str, ...] = ("low", "medium", "high"),
) -> CorruptionResult:
    """Apply one open-source string augmenter independently to each target field."""
    error = validate_call(sample, name=name, seed=seed, level=level, levels=levels)
    if error is not None:
        return CorruptionResult(status=INVALID, corruption_name=name, failure_reason=error)

    all_ops: list[CorruptionOp] = []
    for field in sample.target_fields:
        clean = getattr(sample.clean_payload, field.value)
        candidate = transform(clean, rate, seed + len(all_ops))
        if not isinstance(candidate, str) or candidate == clean:
            continue
        all_ops.extend(
            diff_to_ops(
                clean,
                candidate,
                field=field,
                unit_kind=unit_kind,
                metadata={"source": "open_source"},
                start_index=len(all_ops),
            )
        )

    if not all_ops:
        return CorruptionResult(
            status=NOT_APPLIED,
            corruption_name=name,
            failure_reason="open-source augmenter produced no changed target field",
        )

    # Re-index after per-field diffs so the trace is contiguous across fields.
    all_ops = [op.__class__(**{**op.__dict__, "op_index": i}) for i, op in enumerate(all_ops)]
    corrupted = apply_ops(sample.clean_payload, all_ops)
    return CorruptionResult(
        status=APPLIED,
        corruption_name=name,
        corrupted_payload=corrupted,
        ops=tuple(all_ops),
        change_count=len(all_ops),
    )


def _edit_units(source: str, candidate: str) -> int:
    """Count changed code-point units in a candidate string."""
    return sum(
        max(i2 - i1, j2 - j1)
        for tag, i1, i2, j1, j2 in SequenceMatcher(
            a=source, b=candidate, autojunk=False
        ).get_opcodes()
        if tag != "equal"
    )


def _local_adjacent_swap_count(source: str, candidate: str) -> int | None:
    """Return the number of disjoint adjacent swaps, or None if not local."""
    if len(source) != len(candidate):
        return None
    index = 0
    swaps = 0
    while index < len(source):
        if source[index] == candidate[index]:
            index += 1
            continue
        if (
            index + 1 >= len(source)
            or source[index] != candidate[index + 1]
            or source[index + 1] != candidate[index]
        ):
            return None
        swaps += 1
        index += 2
    return swaps


def jionlp_transform(
    method: str,
    *,
    mode: str = "default",
    target_edit_units: int | None = None,
    max_edit_units: int | None = None,
    max_edit_ratio: float | None = None,
    target_local_swaps: int | None = None,
    max_local_swaps: int | None = None,
    swap_scale: float = 1.0,
) -> TextTransform:
    """Build a deterministic adapter around one JioNLP text augmentation method."""

    def transform(text: str, rate: float, seed: int) -> str | None:
        try:
            # Import check is deliberately lazy so schema-only usage does not require
            # the optional data augmentation dependencies.
            import jionlp  # type: ignore[import-untyped]  # noqa: F401
        except ImportError:
            return None

        fn = _jionlp_augmenter(method, mode, rate)
        effective_rate = rate
        if method == "swap_char_position" and target_local_swaps is not None:
            cjk_count = sum(0x3400 <= ord(char) <= 0x9FFF for char in text)
            # JioNLP samples a probability per character. Scale that
            # probability by text length so a long review does not receive
            # dozens of swaps when the benchmark asks for one or two.
            effective_rate = min(
                rate,
                1.5 * target_local_swaps / max(cjk_count, 1),
            )
        # First-call initialization in JioNLP reseeds NumPy internally. Warm each
        # cached instance once with zero requested augmentations, then seed the RNG
        # immediately before the real call so first and subsequent calls agree.
        if method == "homophone_substitution" and fn.word_pinyin_dict is None:
            fn("", augmentation_num=0, homo_ratio=rate, allow_mispronounce=False, seed=1)
        elif method == "random_add_delete" and fn.char_keys is None:
            fn(
                "",
                augmentation_num=0,
                add_ratio=rate if mode == "add" else 0.0,
                delete_ratio=rate if mode == "delete" else 0.0,
                seed=1,
            )
        elif method == "swap_char_position" and fn.random is None:
            fn("", augmentation_num=0, swap_ratio=rate, seed=1, scale=swap_scale)
        # JioNLP 1.5.29 uses the process-wide NumPy RNG internally. Save and restore
        # it so repeated benchmark calls are deterministic and do not perturb other
        # experiments in the same Python process.
        import numpy as np

        state = np.random.get_state()
        np.random.seed(seed)
        try:
            if method == "homophone_substitution":
                candidates = fn(
                    text,
                    augmentation_num=8,
                    homo_ratio=effective_rate,
                    allow_mispronounce=False,
                    seed=1,
                )
            elif method == "random_add_delete":
                candidates = fn(
                    text,
                    augmentation_num=8,
                    add_ratio=effective_rate if mode == "add" else 0.0,
                    delete_ratio=effective_rate if mode == "delete" else 0.0,
                    seed=1,
                )
            elif method == "swap_char_position":
                candidates = fn(
                    text,
                    augmentation_num=8,
                    swap_ratio=effective_rate,
                    seed=1,
                    scale=swap_scale,
                )
            else:  # pragma: no cover - protected by strategy constructors
                raise ValueError(f"unsupported JioNLP method {method!r}")
        finally:
            np.random.set_state(state)
        valid: list[tuple[int, int, str]] = []
        for candidate in candidates:
            if not isinstance(candidate, str) or candidate == text:
                continue
            if method == "random_add_delete" and mode == "add":
                candidate = _sanitize_added_chars(text, candidate)
                if not isinstance(candidate, str) or candidate == text:
                    continue
            units = _edit_units(text, candidate)
            if max_edit_units is not None and units > max_edit_units:
                continue
            # Use a floor of 20 code points so that one ordinary edit in a
            # short sentence is not rejected merely because the sentence is
            # short; longer texts remain bounded proportionally.
            if max_edit_ratio is not None and units / max(len(text), 20) > max_edit_ratio:
                continue
            local_swaps = None
            if method == "swap_char_position":
                local_swaps = _local_adjacent_swap_count(text, candidate)
                if local_swaps is None:
                    continue
                if max_local_swaps is not None and local_swaps > max_local_swaps:
                    continue
            edit_distance = (
                abs(units - target_edit_units) if target_edit_units is not None else 0
            )
            local_distance = (
                abs(local_swaps - target_local_swaps)
                if local_swaps is not None and target_local_swaps is not None
                else 0
            )
            valid.append((edit_distance + local_distance, units, candidate))
        if valid:
            # Prefer the candidate closest to the requested severity; when
            # equally close, prefer fewer edits for a more conservative result.
            valid.sort(key=lambda item: (item[0], item[1]))
            return valid[0][2]
        return None

    return transform
