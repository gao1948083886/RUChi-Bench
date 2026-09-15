"""Backward-compatible import for the renamed RED_CHAR strategy."""

from ruchi_bench.corruptions.red_char import REDCharCorruptor

CRCorruptor = REDCharCorruptor

__all__ = ["CRCorruptor"]
