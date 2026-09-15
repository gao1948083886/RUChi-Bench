"""Phase 06 real OpenAI-compatible API client.

Wraps httpx with:
- Configurable timeout (default 30s)
- Exponential backoff retry (max 3 attempts) on transient failures
- Error classification: timeout / 401 / 500 / parse failure
- Interface-compatible with MockAPIClient so the pipeline is interchangeable.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)

__all__ = ["APIResponse", "RealAPIClient"]


# Fixed decoding defaults for the benchmark inference protocol.
DEFAULT_MAX_TOKENS = 8
DEFAULT_TOP_P = 1.0
DEFAULT_TOP_K = 0


class APIClientError(Exception):
    """Base exception for API client errors."""

    pass


class APITimeoutError(APIClientError):
    """Request timed out."""

    pass


class APIAuthError(APIClientError):
    """Authentication/authorization failure (401/403)."""

    pass


class APIServerError(APIClientError):
    """Server-side error (5xx)."""

    pass


class APIParseError(APIClientError):
    """Failed to parse API response."""

    pass


@dataclass
class APIResponse:
    """API response matching the shape needed by the inference pipeline.

    Attributes
    ----------
    sample_id :
        Echo of the sample_id passed to the call.
    model :
        Model name from the response (or fallback).
    content :
        The assistant's message content.
    finish_reason :
        Stop reason (e.g. "stop", "length").
    prompt_tokens :
        Tokens in the prompt.
    completion_tokens :
        Tokens in the completion.
    total_tokens :
        Sum of prompt and completion tokens.
    latency_ms :
        Actual round-trip latency in milliseconds.
    raw :
        Raw parsed JSON dict from the API (for debugging/audit).
    """

    sample_id: str
    model: str
    content: str
    finish_reason: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    raw: dict[str, Any]

    def to_openai_dict(self) -> dict[str, Any]:
        return {
            "id": f"real-{self.sample_id}",
            "object": "chat.completion",
            "created": 0,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.content},
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
            },
        }


@dataclass
class RealAPIClient:
    """OpenAI-compatible API client with retry and timeout.

    Parameters
    ----------
    api_base :
        Base URL of the API (e.g. "https://api.example.com/v1").
        Must end with "/v1".
    api_key :
        Bearer token. Set to "***REDACTED***" in public manifests.
    model_name :
        Model name to send in requests and record in responses.
    timeout :
        Per-request timeout in seconds (default 30).
    max_retries :
        Max retry attempts on transient errors (default 3).
    timeout_seconds :
        httpx timeout configuration (default 30.0 seconds).

    Example
    -------
    client = RealAPIClient(
        api_base="https://api.example.com/v1",
        api_key="***REDACTED***",
        model_name="qwen3.5",
    )
    response = client.call(
        messages=[{"role": "user", "content": "Hello"}],
        sample_id="test-001",
        temperature=0.0,
        max_tokens=DEFAULT_MAX_TOKENS,
        top_p=DEFAULT_TOP_P,
        top_k=DEFAULT_TOP_K,
    )
    print(response.content)
    """

    api_base: str
    api_key: str
    model_name: str
    timeout: float = 30.0
    max_retries: int = 3
    _client: httpx.Client | None = None

    def __post_init__(self) -> None:
        self._client = httpx.Client(
            base_url=self.api_base,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(self.timeout, connect=10.0),
            # The pilot model is local. Prevent shell proxy settings from
            # turning localhost requests into false gateway errors.
            trust_env=False,
        )

    def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> RealAPIClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def call(
        self,
        messages: list[dict[str, object]],
        sample_id: str,
        temperature: float = 0.0,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        top_p: float = DEFAULT_TOP_P,
        top_k: int = DEFAULT_TOP_K,
    ) -> APIResponse:
        """Call the chat completions endpoint with retry logic.

        Parameters
        ----------
        messages :
            OpenAI-style chat messages.
        sample_id :
            Sample identifier (for logging/audit, not sent to the API).
        temperature :
            Sampling temperature (0 = deterministic).
        max_tokens :
            Maximum tokens to generate.
        top_p :
            Nucleus-sampling threshold. Fixed at 1.0 for this benchmark.
        top_k :
            Number of highest-probability tokens retained for sampling. Fixed
            at 0 (disabled) for this benchmark.

        Returns
        -------
        APIResponse

        Raises
        ------
        APITimeoutError
            Request timed out after all retries.
        APIAuthError
            Authentication failed (401/403).
        APIServerError
            Server returned 5xx after all retries.
        APIParseError
            Response body was not valid JSON.
        """
        payload: dict[str, object] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
        }

        last_error: APIClientError | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return self._do_call(payload, sample_id)
            except APITimeoutError:
                wait = 2**attempt * 0.5
                logger.warning(
                    "Timeout on attempt %d/%d for sample_id=%r; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sample_id,
                    wait,
                )
                last_error = APITimeoutError(f"Timeout after {self.timeout}s")
                if attempt < self.max_retries:
                    time.sleep(wait)
            except APIParseError as exc:
                # Non-JSON response (e.g. 502 Bad Gateway with HTML body) — retry
                wait = 2**attempt * 0.5
                logger.warning(
                    "Parse error on attempt %d/%d for sample_id=%r: %s; retrying in %.1fs",
                    attempt + 1,
                    self.max_retries + 1,
                    sample_id,
                    exc,
                    wait,
                )
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(wait)
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in (401, 403):
                    raise APIAuthError(
                        f"Auth failed ({status}): check API_KEY"
                    ) from exc
                if status >= 500:
                    wait = 2**attempt * 0.5
                    logger.warning(
                        "Server error %d on attempt %d/%d for sample_id=%r; retrying in %.1fs",
                        status,
                        attempt + 1,
                        self.max_retries + 1,
                        sample_id,
                        wait,
                    )
                    last_error = APIServerError(f"Server error {status}")
                    if attempt < self.max_retries:
                        time.sleep(wait)
                else:
                    raise APIServerError(
                        f"Unexpected HTTP {status}: {exc.response.text[:200]}"
                    ) from exc
            except httpx.ConnectError as exc:
                raise APIClientError(
                    f"Connection failed to {self.api_base}: {exc}"
                ) from exc

        # All retries exhausted
        raise last_error or APIClientError("Unknown error in API call")

    def _do_call(
        self, payload: dict[str, Any], sample_id: str
    ) -> APIResponse:
        """Execute one HTTP request and parse the response."""
        assert self._client is not None
        start = time.monotonic()
        try:
            response = self._client.post(
                "/chat/completions",
                json=payload,
            )
            elapsed_ms = (time.monotonic() - start) * 1000.0
        except httpx.TimeoutException as exc:
            raise APITimeoutError(f"Request timed out after {self.timeout}s") from exc

        try:
            raw = response.json()
        except json.JSONDecodeError as exc:
            raise APIParseError(
                f"Non-JSON response (HTTP {response.status_code}): "
                f"{response.text[:200]!r}"
            ) from exc
        try:
            choices = raw["choices"]
        except (KeyError, TypeError) as exc:
            raise APIParseError(f"Invalid response shape: {raw!r}") from exc

        if not choices:
            raise APIParseError("Response has no choices")

        first_choice = choices[0]
        message = first_choice.get("message", {})
        content = message.get("content", "")
        finish_reason = first_choice.get("finish_reason", "unknown")

        usage = raw.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)

        model = raw.get("model", self.model_name)

        logger.debug(
            "API call sample_id=%r -> %r (elapsed=%.1fms)",
            sample_id,
            content[:50] if content else "(empty)",
            elapsed_ms,
        )

        return APIResponse(
            sample_id=sample_id,
            model=model,
            content=content,
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=elapsed_ms,
            raw=raw,
        )
