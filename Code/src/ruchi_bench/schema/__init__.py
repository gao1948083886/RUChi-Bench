"""Unified data schema for RUChi-Bench (Phase 02).

Work package 1 provides the foundational, I/O-free building blocks:

- :mod:`ruchi_bench.schema.enums` — closed value spaces (datasets, task types,
  splits, payload fields, corruptions, projection status).
- :mod:`ruchi_bench.schema.errors` — the domain exception hierarchy.
- :mod:`ruchi_bench.schema.labels` — canonical label spaces, dataset -> benchmark
  task-type binding, and the PROVISIONAL ASAP-Polarity projection.

The Sample model, change trace, serialization, and per-dataset adapters are LATER
work packages and are not exported here yet.
"""

from __future__ import annotations

from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    ChangeOperation,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    PayloadKind,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.errors import (
    AdapterError,
    C3StructureError,
    CStructBenchError,
    FieldMappingError,
    InvalidSplitError,
    LabelSpaceError,
    SerializationError,
    TraceRedactionError,
)
from ruchi_bench.schema.labels import (
    ASAP_EXCLUDED_STARS,
    ASAP_POLARITY_LABELS,
    ASAP_POLARITY_STATUS,
    ASAP_PROJECTION_NAME,
    ASAP_PROJECTION_VERSION,
    ASAP_STAR_TO_POLARITY,
    ASAP_VALID_STARS,
    DATASET_BENCHMARK_TASK,
    DATASET_LABEL_SPACE,
    LCQMC_LABEL_DECODE,
    PAWSX_LABEL_DECODE,
    TASK_TO_PAYLOAD_KIND,
    XNLI_LABELS,
    polarity_for_star,
)
from ruchi_bench.schema.payloads import (
    MRCPayload,
    PairPayload,
    Payload,
    SingleTextPayload,
)
from ruchi_bench.schema.sample import SCHEMA_VERSION, Sample
from ruchi_bench.schema.serialization import (
    WriteResult,
    read_jsonl,
    sample_from_json,
    sample_to_json,
    write_jsonl,
)
from ruchi_bench.schema.trace import (
    PUBLIC_METADATA_ALLOWLIST,
    REDACTION_VERSION,
    InternalChangeOp,
    InternalChangeTrace,
    PublicChangeOp,
    PublicRedactedTrace,
    redact_trace,
)

__all__ = [
    "ASAP_EXCLUDED_STARS",
    "ASAP_POLARITY_LABELS",
    "ASAP_POLARITY_STATUS",
    "ASAP_STAR_TO_POLARITY",
    "ASAP_PROJECTION_NAME",
    "ASAP_PROJECTION_VERSION",
    "ASAP_VALID_STARS",
    "DATASET_BENCHMARK_TASK",
    "DATASET_LABEL_SPACE",
    "PUBLIC_METADATA_ALLOWLIST",
    "REDACTION_VERSION",
    "SCHEMA_VERSION",
    "TASK_TO_PAYLOAD_KIND",
    "AdapterError",
    "BenchmarkTaskType",
    "C3StructureError",
    "CStructBenchError",
    "ChangeOperation",
    "CorruptionName",
    "DatasetName",
    "FieldMappingError",
    "InternalChangeOp",
    "InternalChangeTrace",
    "InvalidSplitError",
    "LCQMC_LABEL_DECODE",
    "LabelSpaceError",
    "Language",
    "MRCPayload",
    "PAWSX_LABEL_DECODE",
    "PairPayload",
    "Payload",
    "PayloadField",
    "PayloadKind",
    "ProjectionStatus",
    "PublicChangeOp",
    "PublicRedactedTrace",
    "Sample",
    "SerializationError",
    "SingleTextPayload",
    "SplitName",
    "TraceRedactionError",
    "WriteResult",
    "XNLI_LABELS",
    "polarity_for_star",
    "read_jsonl",
    "redact_trace",
    "sample_from_json",
    "sample_to_json",
    "write_jsonl",
]
