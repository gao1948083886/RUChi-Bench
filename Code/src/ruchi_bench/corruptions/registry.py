"""Canonical registry for the seven corruption strategies."""

from __future__ import annotations

from ruchi_bench.corruptions.add_noise import ADDNoiseCorruptor
from ruchi_bench.corruptions.base import BaseCorruptor
from ruchi_bench.corruptions.deletion import DELCorruptor
from ruchi_bench.corruptions.homo import HOMOCorruptor
from ruchi_bench.corruptions.red_char import REDCharCorruptor
from ruchi_bench.corruptions.red_word import REDWordCorruptor
from ruchi_bench.corruptions.swap import SWAPCorruptor
from ruchi_bench.corruptions.vis import VISCorruptor
from ruchi_bench.schema.enums import CorruptionName

__all__ = ["CORRUPTOR_REGISTRY", "get_corruptor"]


CORRUPTOR_REGISTRY: dict[CorruptionName, BaseCorruptor] = {
    CorruptionName.HOMO: HOMOCorruptor(),
    CorruptionName.VIS: VISCorruptor(),
    CorruptionName.DEL: DELCorruptor(),
    CorruptionName.SWAP: SWAPCorruptor(),
    CorruptionName.ADD_NOISE: ADDNoiseCorruptor(),
    CorruptionName.RED_CHAR: REDCharCorruptor(),
    CorruptionName.RED_WORD: REDWordCorruptor(),
}


def get_corruptor(name: CorruptionName) -> BaseCorruptor:
    """Return the canonical corruptor for ``name``."""
    return CORRUPTOR_REGISTRY[name]
