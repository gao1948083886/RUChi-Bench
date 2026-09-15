"""Tests for ruchi_bench.inference.mock_client."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

from ruchi_bench.inference.mock_client import (
    InferenceLogger,
    MockAPIClient,
    MockAPIResponse,
)


class TestMockAPIResponse:
    def test_to_openai_dict(self) -> None:
        resp = MockAPIResponse(
            sample_id="s1",
            model="mock-model/v1",
            content="paraphrase",
            finish_reason="stop",
            latency_ms=42.0,
        )
        d = resp.to_openai_dict()
        assert d["id"] == "mock-s1"
        assert d["model"] == "mock-model/v1"
        assert d["choices"][0]["message"]["content"] == "paraphrase"
        assert d["choices"][0]["finish_reason"] == "stop"
        assert d["usage"]["total_tokens"] == 13


class TestMockAPIClient:
    def test_known_sample(self) -> None:
        client = MockAPIClient(
            expected_responses={"s1": "paraphrase", "s2": "not paraphrase"},
            latency_ms=0.0,
        )
        resp = client.call([], sample_id="s1")
        assert resp.sample_id == "s1"
        assert resp.content == "paraphrase"
        assert resp.model == "mock-model/dummy"

    def test_unknown_sample(self) -> None:
        client = MockAPIClient(expected_responses={}, latency_ms=0.0)
        resp = client.call([], sample_id="unknown")
        assert resp.content == "__UNKNOWN__"

    def test_latency(self) -> None:
        start = time.monotonic()
        client = MockAPIClient(expected_responses={}, latency_ms=100.0)
        client.call([], sample_id="s1")
        elapsed_ms = (time.monotonic() - start) * 1000
        assert elapsed_ms >= 90


class TestInferenceLogger:
    def test_fresh_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            logger = InferenceLogger(out)
            assert not logger.is_done("rk_s1")
            assert not logger.is_done("rk_s2")

            logger.log(
                request_key="rk_s1",
                sample_id="s1",
                model="mock",
                messages=[{"role": "user", "content": "hello"}],
                raw_response={"content": "paraphrase"},
                latency_ms=10.0,
            )
            assert logger.is_done("rk_s1")
            assert not logger.is_done("rk_s2")

            logger.log(
                request_key="rk_s2",
                sample_id="s2",
                model="mock",
                messages=[{"role": "user", "content": "world"}],
                raw_response={"content": "not paraphrase"},
                latency_ms=12.0,
            )
            assert logger.is_done("rk_s2")

            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2
            rec1 = json.loads(lines[0])
            assert rec1["request_key"] == "rk_s1"
            assert rec1["sample_id"] == "s1"
            assert rec1["raw_response"]["content"] == "paraphrase"

    def test_resume_via_request_key(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            out.write_text(
                json.dumps(
                    {
                        "request_key": "rk_s1",
                        "sample_id": "s1",
                        "model": "mock",
                        "messages": [],
                        "raw_response": {},
                        "latency_ms": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            logger = InferenceLogger(out)
            assert logger.is_done("rk_s1")
            assert not logger.is_done("rk_s2")

    def test_resume_skips_bad_lines(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            out.write_text(
                'not json\n{"request_key": "rk_s1", "sample_id": "s1"}\n',
                encoding="utf-8",
            )
            logger = InferenceLogger(out)
            assert logger.is_done("rk_s1")

    def test_legacy_records_detected(self) -> None:
        """Legacy records (no request_key) are counted but not used for dedup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            out.write_text(
                json.dumps(
                    {
                        "sample_id": "s_legacy",
                        "model": "mock",
                        "messages": [],
                        "raw_response": {},
                        "latency_ms": 1.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            logger = InferenceLogger(out)
            assert logger.legacy_count == 1
            # The legacy sample_id is NOT in the done set
            assert not logger.is_done("s_legacy")
            assert not logger.is_done("rk_new")

    def test_record_has_request_key_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.jsonl"
            logger = InferenceLogger(out)
            logger.log(
                request_key="rk_abc",
                sample_id="s99",
                model="mock",
                messages=[{"role": "user", "content": "hi"}],
                raw_response={"content": "ok"},
                latency_ms=1.0,
            )
            rec = json.loads(out.read_text(encoding="utf-8").strip().split("\n")[0])
            assert "request_key" in rec
            assert rec["request_key"] == "rk_abc"
