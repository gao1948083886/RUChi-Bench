"""Remote OpenAI-compatible API client and runner (Phase 05-06).

Modules
-------
client :
    Real API client with httpx, timeout, exponential backoff retry, error classification.
mock_client :
    Mock client for dry-run testing (deterministic, no network).
pipeline :
    Inference runner with JSONL write, resume, and real/mock swap.
"""

from ruchi_bench.inference.client import (
    APIAuthError,
    APIClientError,
    APIParseError,
    APIResponse,
    APIServerError,
    APITimeoutError,
    RealAPIClient,
)
from ruchi_bench.inference.mock_client import InferenceLogger, MockAPIClient
from ruchi_bench.inference.pipeline import (
    InferenceResult,
    PipelineConfig,
    create_api_client,
    run_inference,
)

__all__ = [
    # client
    "APIResponse",
    "APIClientError",
    "APITimeoutError",
    "APIAuthError",
    "APIServerError",
    "APIParseError",
    "RealAPIClient",
    # mock_client
    "MockAPIClient",
    "InferenceLogger",
    # pipeline
    "InferenceResult",
    "PipelineConfig",
    "create_api_client",
    "run_inference",
]
