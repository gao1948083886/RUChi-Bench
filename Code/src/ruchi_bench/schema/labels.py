"""Canonical label spaces, task-type binding, and the ASAP-Polarity projection.

This module is the single source of truth for:

- which semantic :class:`BenchmarkTaskType` each pilot dataset carries;
- each dataset's *canonical* label space (the label strings the unified schema uses),
  plus the raw-value -> canonical-label decode maps for datasets whose source labels
  are integers or alternate strings;
- the derived ASAP-Polarity star -> polarity projection (DEC-008, ACCEPTED).

It performs no I/O and reads no dataset files. Adapters (a later work package) use
these tables to validate and normalize records; the exclusion of ASAP 3-star reviews
is an adapter decision and is NOT executed here — this module only *declares* that
3 maps to "excluded".

Label-space representation
--------------------------
- Closed-space datasets (PAWS-X, XNLI, LCQMC, ASAP-Polarity) expose an ordered tuple
  of canonical label strings in :data:`DATASET_LABEL_SPACE`.
- C3 has an *open*, per-sample label space (the gold answer is one of that sample's
  option strings), so it maps to ``None`` here; its validity is checked structurally
  by the C3 adapter, not against a fixed set.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import Final

from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    DatasetName,
    PayloadKind,
    ProjectionStatus,
)

__all__ = [
    "ASAP_EXCLUDED_STARS",
    "ASAP_POLARITY_LABELS",
    "ASAP_POLARITY_STATUS",
    "ASAP_PROJECTION_NAME",
    "ASAP_PROJECTION_VERSION",
    "ASAP_STAR_TO_POLARITY",
    "ASAP_VALID_STARS",
    "DATASET_BENCHMARK_TASK",
    "DATASET_LABEL_SPACE",
    "TASK_TO_PAYLOAD_KIND",
    "LCQMC_LABEL_DECODE",
    "PAWSX_LABEL_DECODE",
    "XNLI_LABELS",
    "polarity_for_star",
]


# --- Semantic benchmark task type per dataset -------------------------------------
# NOTE: this is the *semantic* layer. It intentionally differs from the *structural*
# ``task_type`` in the dataset configuration — e.g. PAWS-X and LCQMC are both
# structurally "pair_classification" but split into distinct semantic task types
# (PAWS-X -> PAIR_PARAPHRASE, LCQMC -> QUESTION_MATCHING).
DATASET_BENCHMARK_TASK: Final[dict[DatasetName, BenchmarkTaskType]] = {
    DatasetName.PAWSX_ZH: BenchmarkTaskType.PAIR_PARAPHRASE,
    DatasetName.XNLI_ZH: BenchmarkTaskType.NLI,
    DatasetName.C3: BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
    DatasetName.LCQMC: BenchmarkTaskType.QUESTION_MATCHING,
    DatasetName.ASAP: BenchmarkTaskType.SENTIMENT_POLARITY,
}


# --- PAWS-X zh: source int label {0,1} -> canonical string ------------------------
PAWSX_LABEL_DECODE: Final[MappingProxyType[int, str]] = MappingProxyType(
    {
        0: "different_meaning",
        1: "paraphrase",
    }
)

# --- XNLI zh: source string labels (already canonical) ----------------------------
XNLI_LABELS: Final[tuple[str, ...]] = ("entailment", "neutral", "contradiction")

# --- LCQMC: source int label {0,1} -> canonical string ----------------------------
LCQMC_LABEL_DECODE: Final[MappingProxyType[int, str]] = MappingProxyType(
    {
        0: "not_matched",
        1: "matched_intent",
    }
)


# --- ASAP-Polarity: derived projection (DEC-008) ----------------------------------
# The star->polarity mapping DESIGN is FROZEN (negative=[1,2], positive=[4,5],
# excluded=[3]); it must NOT be re-derived from model results or class distribution.
# Phase 04 count check PASSED 2026-07-30: neg=338, pos=3885. Projection ACCEPTED.
# 3-star is a deliberate *exclusion*, not a label — the adapter (a LATER work package)
# performs the actual exclusion; this module only declares 3 as excluded.
ASAP_POLARITY_LABELS: Final[tuple[str, ...]] = ("negative", "positive")

ASAP_STAR_TO_POLARITY: Final[MappingProxyType[int, str]] = MappingProxyType(
    {
        1: "negative",
        2: "negative",
        4: "positive",
        5: "positive",
    }
)
ASAP_EXCLUDED_STARS: Final[frozenset[int]] = frozenset({3})
ASAP_VALID_STARS: Final[frozenset[int]] = frozenset({1, 2, 3, 4, 5})

# ACCEPTED after Phase 04 count check 2026-07-30 (>=150 per polarity class, DEC-008).
ASAP_POLARITY_STATUS: Final[ProjectionStatus] = ProjectionStatus.ACCEPTED

#: Stable identity of the ASAP-Polarity projection (the mapping DESIGN is frozen;
#: only its verification status is PROVISIONAL). Used as Sample.label_projection_name.
ASAP_PROJECTION_NAME: Final[str] = "asap_polarity"
ASAP_PROJECTION_VERSION: Final[str] = "1.0.0"


def polarity_for_star(star: int) -> str | None:
    """Map an ASAP ``star`` rating to its ASAP-Polarity label.

    Returns the canonical polarity string for 1/2/4/5, or ``None`` for the excluded
    3-star case. Raises :class:`ValueError` for any star outside 1..5 so that
    unexpected source values fail loudly rather than being silently dropped.

    This function only *classifies*; it does not decide whether an excluded sample is
    removed from the pilot — that is the adapter's responsibility.
    """
    if star not in ASAP_VALID_STARS:
        raise ValueError(f"ASAP star rating out of range 1..5: {star!r}")
    return ASAP_STAR_TO_POLARITY.get(star)


# --- Canonical label space per dataset --------------------------------------------
# C3 is open per-sample (answer is one of the sample's options) -> None.
DATASET_LABEL_SPACE: Final[MappingProxyType[DatasetName, tuple[str, ...] | None]] = (
    MappingProxyType(
        {
            DatasetName.PAWSX_ZH: tuple(PAWSX_LABEL_DECODE.values()),
            DatasetName.XNLI_ZH: XNLI_LABELS,
            DatasetName.C3: None,
            DatasetName.LCQMC: tuple(LCQMC_LABEL_DECODE.values()),
            DatasetName.ASAP: ASAP_POLARITY_LABELS,
        }
    )
)


# --- Semantic task -> structural payload kind -------------------------------------
# Several semantic tasks share one structural payload (all *_pair tasks -> PAIR).
# The Sample model enforces that its payload matches this expected kind.
TASK_TO_PAYLOAD_KIND: Final[MappingProxyType[BenchmarkTaskType, PayloadKind]] = MappingProxyType(
    {
        BenchmarkTaskType.PAIR_PARAPHRASE: PayloadKind.PAIR,
        BenchmarkTaskType.NLI: PayloadKind.PAIR,
        BenchmarkTaskType.QUESTION_MATCHING: PayloadKind.PAIR,
        BenchmarkTaskType.SENTIMENT_POLARITY: PayloadKind.SINGLE_TEXT,
        BenchmarkTaskType.MULTIPLE_CHOICE_MRC: PayloadKind.MRC,
    }
)
