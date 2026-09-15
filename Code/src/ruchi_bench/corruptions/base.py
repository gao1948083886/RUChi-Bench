"""Base protocol and shared logic for all seven corruptors."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Protocol

from ruchi_bench.schema.enums import CorruptionName, PayloadField
from ruchi_bench.schema.sample import Sample

if TYPE_CHECKING:
    from ruchi_bench.corruptions.result import CorruptionResult


class C3ContextOnlyError(Exception):
    """Raised when a corruptor is called on C3 with a non-context target field."""

    pass


class BaseCorruptor(Protocol):
    """Protocol for all seven corruption corruptors.

    Every corruptor must:

    - Be a concrete class (not just a module of functions) so it can be instantiated
      and registered by name.
    - Hold a ``corruption_name`` attribute (the canonical identifier).
    - Implement :meth:`corrupt`, returning a :class:`CorruptionResult`.
    """

    corruption_name: CorruptionName

    def corrupt(
        self,
        sample: Sample,
        *,
        seed: int,
        level: str,
    ) -> CorruptionResult:
        """Apply this corruption to ``sample``.

        Parameters
        ----------
        sample:
            The benchmark sample to corrupt. Must be in a clean state (not already
            corrupted) and have at least one target field.
        seed:
            Non-negative integer seed. The corruptor creates a local RNG on every call,
            so the same ``(sample, seed, level)`` triple always produces the same result
            regardless of global random state.
        level:
            Corruption intensity level, e.g. ``"low"``, ``"medium"``, ``"high"``.
            The corruptor defines its own mapping from level to concrete parameters.

        Returns
        -------
        CorruptionResult
            One of:
            - ``APPLIED``: corruption succeeded; result carries the corrupted payload and traces.
            - ``NOT_APPLIED``: input is valid but no eligible target was found.
            - ``INVALID``: input structure or target fields are illegal for this corruptor.

        Raises
        ------
        C3ContextOnlyError
            Never raised at the API boundary; instead returned as an ``INVALID`` result.
        """
        ...

    @abstractmethod
    def is_valid_target(self, sample: Sample, field: PayloadField) -> bool:
        """Return True if ``field`` is a legal corruption target for ``sample``.

        Used by the base implementation to check inputs before delegation.
        """
        ...


def _check_sample_can_be_corrupted(sample: Sample) -> None:
    """Raise ValueError if sample is not in a clean, non-corrupted state."""
    if sample.corruption_applied:
        raise ValueError(
            f"Sample {sample.sample_id!r} is already corrupted; corruptors require a "
            "clean (uncorrupted) input sample"
        )
    if sample.corruption_name is not None:
        raise ValueError(
            f"Sample {sample.sample_id!r} already corrupted "
            f"(corruption_name={sample.corruption_name.value!r})"
        )
    if not sample.target_fields:
        raise ValueError(f"Sample {sample.sample_id!r} has no target_fields")
