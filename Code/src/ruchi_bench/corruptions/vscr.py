"""Backward-compatible import for the renamed VIS strategy."""

from ruchi_bench.corruptions.vis import VISCorruptor

VSCRCorruptor = VISCorruptor

__all__ = ["VSCRCorruptor"]
