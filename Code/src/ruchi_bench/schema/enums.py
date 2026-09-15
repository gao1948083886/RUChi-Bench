"""Closed enumerations for the RUChi-Bench unified schema.

These enums define the *domain-complete* value spaces used across the benchmark
(datasets, task types, dataset splits, payload fields, corruptions, and the
ASAP-Polarity projection status). They are intentionally free of any I/O, adapter,
or corruption logic; higher layers (admission validators, adapters) enforce
context-specific rules such as "pilot uses the ``test`` split only".

All enums subclass ``str`` so that members compare equal to their string value and
serialize as plain strings (Pydantic ``use_enum_values`` / JSON both round-trip to
the ``value``). Membership order is stable and used by tests as the canonical order.

Frozen decisions (see ``Docs/DECISIONS.md``):

- Pilot datasets: PAWS-X zh, XNLI zh, C3, LCQMC, ASAP (DEC-007).
- Pilot corruptions: HOMO, VIS, DEL, SWAP, ADD_NOISE, RED_CHAR, RED_WORD.
- ASAP-Polarity is a *derived* binary task, PROVISIONAL until Phase 04 count check
  (DEC-008).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "BenchmarkTaskType",
    "ChangeOperation",
    "CorruptionName",
    "DatasetName",
    "Language",
    "PayloadField",
    "PayloadKind",
    "ProjectionStatus",
    "SplitName",
]


class DatasetName(StrEnum):
    """The five pilot datasets (DEC-007). Values are the canonical dataset ids."""

    PAWSX_ZH = "pawsx_zh"
    XNLI_ZH = "xnli_zh"
    C3 = "c3"
    LCQMC = "lcqmc"
    ASAP = "asap"


class Language(StrEnum):
    """Sample language. The pilot is Chinese-only; extend as scope grows."""

    ZH = "zh"


class BenchmarkTaskType(StrEnum):
    """Semantic benchmark task type of a unified sample.

    This is the *semantic* task layer, distinct from the *structural* ``task_type``
  recorded in the dataset configuration (e.g. two datasets that are both
    structurally ``pair_classification`` map to different semantic benchmark task
    types here — PAWS-X → ``PAIR_PARAPHRASE`` vs LCQMC → ``QUESTION_MATCHING``).
    """

    PAIR_PARAPHRASE = "pair_paraphrase"  # PAWS-X zh
    NLI = "nli"  # XNLI zh
    MULTIPLE_CHOICE_MRC = "multiple_choice_mrc"  # C3
    QUESTION_MATCHING = "question_matching"  # LCQMC
    SENTIMENT_POLARITY = "sentiment_polarity"  # ASAP-Polarity (derived)


class SplitName(StrEnum):
    """Dataset splits. Domain-complete on purpose.

    The pilot only ever admits the official ``test`` split, but that restriction is
    a research rule enforced at the admission-validation layer — not encoded here —
    so the enum stays a faithful, reusable description of the split space.
    """

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class PayloadField(StrEnum):
    """Logical text fields a corruption may target inside a sample payload.

    C3 keeps ``context``/``question``/``options`` as separate fields and only
    ``context`` is ever corrupted (DEC frozen rule); the enum lists all addressable
    fields, and which fields are legal targets per dataset is enforced elsewhere.
    """

    TEXT_A = "text_a"
    TEXT_B = "text_b"
    CONTEXT = "context"
    QUESTION = "question"
    OPTIONS = "options"


class PayloadKind(StrEnum):
    """Structural kind of a sample payload — the discriminator for the payload union.

    This is the *structural* input shape, deliberately coarser than
    :class:`BenchmarkTaskType`: several semantic tasks share one structure (e.g.
    ``PAIR_PARAPHRASE``, ``QUESTION_MATCHING`` and ``NLI`` all use :data:`PAIR`).
    """

    PAIR = "pair"
    SINGLE_TEXT = "single_text"
    MRC = "mrc"


class ChangeOperation(StrEnum):
    """Kind of edit recorded in a change-trace op."""

    REPLACE = "replace"
    INSERT = "insert"
    DELETE = "delete"
    DUPLICATE = "duplicate"


class CorruptionName(StrEnum):
    """The seven frozen perturbation strategies.

    The legacy names remain aliases so old serialized fixtures and imports can be
    migrated without silently changing their meaning. The public interface exposes
    only the seven canonical strategies below.
    """

    HOMO = "homo"  # JioNLP tone-agnostic homophone substitution
    VIS = "vis"  # Confused_Chinese visual-similarity replacement
    DEL = "del"  # JioNLP character deletion
    SWAP = "swap"  # JioNLP neighboring character transposition
    ADD_NOISE = "add_noise"  # JioNLP empirical noise insertion
    RED_CHAR = "red_char"  # transparent character repetition
    RED_WORD = "red_word"  # transparent word repetition

    # Deprecated round-1 aliases.
    TPWR = HOMO
    VSCR = VIS
    CR = RED_CHAR
    WR = RED_WORD
    MUNI = ADD_NOISE


class ProjectionStatus(StrEnum):
    """Status of a derived label projection (currently only ASAP-Polarity).

    - ``NOT_APPLICABLE``: dataset uses its official label space directly.
    - ``PROVISIONAL``: derived mapping designed but not yet count-verified (DEC-008).
    - ``ACCEPTED``: derived mapping verified and frozen (post Phase 04 check).
    """

    NOT_APPLICABLE = "not_applicable"
    PROVISIONAL = "provisional"
    ACCEPTED = "accepted"
