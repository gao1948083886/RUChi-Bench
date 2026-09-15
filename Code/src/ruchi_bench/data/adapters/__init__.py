"""Dataset adapters that load raw source records into the unified Sample schema.

Each adapter is responsible for one dataset and handles the structural mapping from
the dataset's native format to :class:`~ruchi_bench.schema.sample.Sample`.

All adapters in this package are **fixture-only** for Phase 02 — they operate on
synthetic data, never on real restricted source data.
"""

from __future__ import annotations

from ruchi_bench.data.adapters.asap import ASAPAdapter
from ruchi_bench.data.adapters.base import (
    AdaptationResult,
    AdaptationStatus,
    BaseAdapter,
)
from ruchi_bench.data.adapters.c3 import C3Adapter
from ruchi_bench.data.adapters.lcqmc import LCQMCAdapter
from ruchi_bench.data.adapters.pawsx_zh import PAWSXAdapter
from ruchi_bench.data.adapters.xnli_zh import XNLIAdapter

__all__ = [
    "AdaptationResult",
    "AdaptationStatus",
    "ASAPAdapter",
    "BaseAdapter",
    "C3Adapter",
    "LCQMCAdapter",
    "PAWSXAdapter",
    "XNLIAdapter",
]
