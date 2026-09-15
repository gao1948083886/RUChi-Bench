"""Base adapter types: :class:`AdaptationResult`, :class:`AdaptationStatus`,
and :class:`BaseAdapter`.

:class:`AdaptationResult` carries either a :class:`Sample <ruchi_bench.schema.sample.Sample>`
(``ACCEPTED``) or structured metadata about why a record could not be adapted (``EXCLUDED``
or ``INVALID``). It deliberately does **not** carry a ``failure_reason`` string on the
:class:`Sample <ruchi_bench.schema.sample.Sample>` — dataset exclusion and structural
invalidity are first-class outcomes, not error strings buried in a sample field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ruchi_bench.schema.sample import Sample


@dataclass(frozen=True, slots=True)
class AdaptationResult:
    """Result of adapting one source record to the unified schema.

    Exactly one of ``sample`` / ``exclusion_reason`` / ``invalid_reason`` is non-None,
    corresponding to the three possible statuses.
    """

    status: AdaptationStatus
    sample: Sample | None = None
    exclusion_reason: str | None = None
    invalid_reason: str | None = None

    def __post_init__(self) -> None:
        if self.status is AdaptationStatus.ACCEPTED:
            if self.sample is None:
                raise ValueError("ACCEPTED result must carry a Sample")
            if self.exclusion_reason is not None or self.invalid_reason is not None:
                raise ValueError("ACCEPTED result must not carry exclusion/invalid reason")
        elif self.status is AdaptationStatus.EXCLUDED:
            if self.exclusion_reason is None:
                raise ValueError("EXCLUDED result must carry an exclusion_reason")
            if self.sample is not None or self.invalid_reason is not None:
                raise ValueError("EXCLUDED result must not carry Sample or invalid_reason")
        elif self.status is AdaptationStatus.INVALID:
            if self.invalid_reason is None:
                raise ValueError("INVALID result must carry an invalid_reason")
            if self.sample is not None or self.exclusion_reason is not None:
                raise ValueError("INVALID result must not carry Sample or exclusion_reason")


class AdaptationStatus(StrEnum):
    """Outcome of a single-record adaptation.

    - ``ACCEPTED``: a valid :class:`Sample` was produced.
    - ``EXCLUDED``: the record is structurally well-formed but was filtered out by a
      dataset-specific rule (e.g. ASAP 3-star reviews have no binary label).
    - ``INVALID``: the record is malformed or violates the schema's structural
      constraints.
    """

    ACCEPTED = "accepted"
    EXCLUDED = "excluded"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class BaseAdapter:
    """Abstract base for dataset adapters.

    Subclasses must implement :meth:`adapt_record` to translate one dataset-specific
    raw record (as a string-keyed dict loaded from the source file) into an
    :class:`AdaptationResult`.  The adapter must verify the split and apply any
    dataset-specific field-mapping or label-projection rules.
    """

    dataset_name: str
    dataset_version: str
    split: str

    def adapt_record(self, raw: dict[str, Any]) -> AdaptationResult:
        """Translate one raw source record into an :class:`AdaptationResult`.

        Parameters
        ----------
        raw
            A dictionary parsed from the source file.  Keys are field names as
            they appear in the original dataset (e.g. ``"sentence1"`` for PAWS-X).

        Returns
        -------
        AdaptationResult
            One of ``ACCEPTED`` (with a valid :class:`Sample`), ``EXCLUDED`` (with a
            reason), or ``INVALID`` (with a reason).  Never raises; all errors are
            wrapped in an ``INVALID`` result.
        """
        raise NotImplementedError
