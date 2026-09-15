"""Backward-compatible import for the renamed RED_WORD strategy."""

from ruchi_bench.corruptions.red_word import REDWordCorruptor

WRCorruptor = REDWordCorruptor

__all__ = ["WRCorruptor"]
