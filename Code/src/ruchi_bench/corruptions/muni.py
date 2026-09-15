"""Backward-compatible import for the renamed ADD_NOISE strategy."""

from ruchi_bench.corruptions.add_noise import ADDNoiseCorruptor

MUNICorruptor = ADDNoiseCorruptor

__all__ = ["MUNICorruptor"]
