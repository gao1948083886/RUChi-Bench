"""Backward-compatible import for the renamed HOMO strategy."""

from ruchi_bench.corruptions.homo import HOMOCorruptor

TPWRCorruptor = HOMOCorruptor

__all__ = ["TPWRCorruptor"]
