"""Tests for ruchi_bench.inference.pipeline."""

from __future__ import annotations

import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from ruchi_bench.inference.mock_client import InferenceLogger, MockAPIClient
from ruchi_bench.inference.pipeline import (
    InferenceResult,
    PipelineConfig,
    _build_label_only_messages,
    _build_request_key,
    run_inference,
)
from ruchi_bench.inference.prompts import prompt_for_sample
from ruchi_bench.inference.request_key import sample_uid
from ruchi_bench.schema.enums import (
    BenchmarkTaskType,
    ChangeOperation,
    CorruptionName,
    DatasetName,
    Language,
    PayloadField,
    ProjectionStatus,
    SplitName,
)
from ruchi_bench.schema.payloads import MRCPayload, PairPayload, SingleTextPayload
from ruchi_bench.schema.sample import Sample
from ruchi_bench.schema.trace import InternalChangeOp, InternalChangeTrace, redact_trace


def _pair_sample(sample_id: str = "s1") -> Sample:
    return Sample(
        sample_id=sample_id,
        source_sample_id="0",
        dataset_name=DatasetName.PAWSX_ZH,
        dataset_version="final",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.PAIR_PARAPHRASE,
        language=Language.ZH,
        clean_payload=PairPayload(text_a="第一句", text_b="第二句"),
        gold_label=1,
        gold_label_text="paraphrase",
        target_fields=(PayloadField.TEXT_A, PayloadField.TEXT_B),
        label_projection_status=ProjectionStatus.NOT_APPLICABLE,
        corruption_applied=False,
        change_count=0,
        created_at=datetime.now(tz=UTC),
    )


def _single_sample(sample_id: str = "s2") -> Sample:
    return Sample(
        sample_id=sample_id,
        source_sample_id="1",
        dataset_name=DatasetName.ASAP,
        dataset_version="master",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.SENTIMENT_POLARITY,
        language=Language.ZH,
        clean_payload=SingleTextPayload(text_a="这家店不错"),
        source_gold_label=5,
        gold_label="positive",
        gold_label_text="positive",
        target_fields=(PayloadField.TEXT_A,),
        label_projection_name="asap_polarity",
        label_projection_version="1.0.0",
        label_projection_status=ProjectionStatus.ACCEPTED,
        corruption_applied=False,
        change_count=0,
        created_at=datetime.now(tz=UTC),
    )


def _lcqmc_sample(sample_id: str = "s-lcqmc") -> Sample:
    sample = _pair_sample(sample_id)
    return sample.model_copy(
        update={
            "dataset_name": DatasetName.LCQMC,
            "benchmark_task_type": BenchmarkTaskType.QUESTION_MATCHING,
        }
    )


def _mrc_sample(sample_id: str = "s3") -> Sample:
    return Sample(
        sample_id=sample_id,
        source_sample_id="2",
        dataset_name=DatasetName.C3,
        dataset_version="master",
        split=SplitName.TEST,
        benchmark_task_type=BenchmarkTaskType.MULTIPLE_CHOICE_MRC,
        language=Language.ZH,
        clean_payload=MRCPayload(context="上下文", question="问什么？", options=("甲", "乙")),
        gold_label=0,
        gold_label_text="甲",
        target_fields=(PayloadField.CONTEXT,),
        label_projection_status=ProjectionStatus.NOT_APPLICABLE,
        corruption_applied=False,
        change_count=0,
        created_at=datetime.now(tz=UTC),
    )


def _corrupted_pair_sample(sample_id: str = "s4") -> Sample:
    sample = _pair_sample(sample_id)
    trace = InternalChangeTrace(
        corruption_name=CorruptionName.VSCR,
        corruption_level="low",
        seed=7,
        ops=(
            InternalChangeOp(
                op_index=0,
                op_type=ChangeOperation.REPLACE,
                field=PayloadField.TEXT_A,
                start=0,
                end=1,
                original="第",
                replacement="苐",
                unit_kind="char",
                metadata={"candidate_count": 1},
            ),
        ),
    )
    return sample.model_copy(
        update={
            "corrupted_payload": PairPayload(text_a="苐一句", text_b="第二句"),
            "corruption_name": CorruptionName.VSCR,
            "corruption_level": "low",
            "corruption_seed": 7,
            "corruption_applied": True,
            "change_count": 1,
            "internal_change_trace": trace,
            "public_redacted_trace": redact_trace(trace),
        }
    )


def _corrupted_xnli_sample(sample_id: str = "s5") -> Sample:
    sample = _pair_sample(sample_id).model_copy(
        update={
            "dataset_name": DatasetName.XNLI_ZH,
            "benchmark_task_type": BenchmarkTaskType.NLI,
            "gold_label": "neutral",
            "gold_label_text": "neutral",
        }
    )
    trace = InternalChangeTrace(
        corruption_name=CorruptionName.CR,
        corruption_level="medium",
        seed=8,
        ops=(
            InternalChangeOp(
                op_index=0,
                op_type=ChangeOperation.DUPLICATE,
                field=PayloadField.TEXT_B,
                start=0,
                end=1,
                original="第",
                replacement="第第",
                unit_kind="char",
                metadata={"candidate_count": 1},
            ),
        ),
    )
    return sample.model_copy(
        update={
            "corrupted_payload": PairPayload(text_a="第一句", text_b="第第二句"),
            "corruption_name": CorruptionName.CR,
            "corruption_level": "medium",
            "corruption_seed": 8,
            "corruption_applied": True,
            "change_count": 1,
            "internal_change_trace": trace,
            "public_redacted_trace": redact_trace(trace),
        }
    )


def _corrupted_single_sample(sample_id: str = "s6") -> Sample:
    sample = _single_sample(sample_id)
    trace = InternalChangeTrace(
        corruption_name=CorruptionName.WR,
        corruption_level="high",
        seed=9,
        ops=(
            InternalChangeOp(
                op_index=0,
                op_type=ChangeOperation.DUPLICATE,
                field=PayloadField.TEXT_A,
                start=0,
                end=1,
                original="这",
                replacement="这这",
                unit_kind="word",
                metadata={"candidate_count": 1},
            ),
        ),
    )
    return sample.model_copy(
        update={
            "corrupted_payload": SingleTextPayload(text_a="这这家店不错"),
            "corruption_name": CorruptionName.WR,
            "corruption_level": "high",
            "corruption_seed": 9,
            "corruption_applied": True,
            "change_count": 1,
            "internal_change_trace": trace,
            "public_redacted_trace": redact_trace(trace),
        }
    )


def _corrupted_mrc_sample(sample_id: str = "s7") -> Sample:
    sample = _mrc_sample(sample_id)
    trace = InternalChangeTrace(
        corruption_name=CorruptionName.TPWR,
        corruption_level="low",
        seed=10,
        ops=(
            InternalChangeOp(
                op_index=0,
                op_type=ChangeOperation.REPLACE,
                field=PayloadField.CONTEXT,
                start=0,
                end=1,
                original="上",
                replacement="尚",
                unit_kind="char",
                metadata={"candidate_count": 1},
            ),
        ),
    )
    return sample.model_copy(
        update={
            "corrupted_payload": MRCPayload(
                context="尚下文",
                question="问什么？",
                options=("甲", "乙"),
            ),
            "corruption_name": CorruptionName.TPWR,
            "corruption_level": "low",
            "corruption_seed": 10,
            "corruption_applied": True,
            "change_count": 1,
            "internal_change_trace": trace,
            "public_redacted_trace": redact_trace(trace),
        }
    )


class TestBuildLabelOnlyMessages:
    def test_pair_payload_uses_v2_prompt(self) -> None:
        sample = _pair_sample()
        msgs = _build_label_only_messages(sample)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "句子A：" in msgs[1]["content"]
        assert "句子B：" in msgs[1]["content"]
        assert "允许输出：0 或 1" in msgs[1]["content"]
        assert msgs[1]["content"].endswith("答案：")

    def test_single_text_payload(self) -> None:
        sample = _single_sample()
        msgs = _build_label_only_messages(sample)
        assert len(msgs) == 2
        assert "评论：" in msgs[1]["content"]
        assert "positive 或 negative" in msgs[1]["content"]

    def test_lcqmc_uses_question_matching_prompt(self) -> None:
        messages, version, max_tokens = prompt_for_sample(_lcqmc_sample())
        assert version == "lcqmc-binary-v2"
        assert max_tokens == 4
        assert "问题A：" in messages[1]["content"]
        assert "问题B：" in messages[1]["content"]
        assert "允许输出：0 或 1" in messages[1]["content"]

    def test_mrc_payload(self) -> None:
        sample = _mrc_sample()
        msgs = _build_label_only_messages(sample)
        assert len(msgs) == 2
        assert "材料：" in msgs[1]["content"]
        assert "问题：" in msgs[1]["content"]
        assert "A. 甲" in msgs[1]["content"]
        assert "B. 乙" in msgs[1]["content"]
        assert "允许输出：A B" in msgs[1]["content"]

    def test_corrupted_prompt_uses_corrupted_payload(self) -> None:
        sample = _corrupted_pair_sample()
        clean = _build_label_only_messages(sample, is_clean=True)
        corrupted = _build_label_only_messages(sample, is_clean=False)
        assert "第一句" in clean[1]["content"]
        assert "苐一句" in corrupted[1]["content"]
        assert clean[1]["content"] != corrupted[1]["content"]

    def test_nli_routes_by_task_type(self) -> None:
        sample = _pair_sample().model_copy(
            update={
                "dataset_name": DatasetName.XNLI_ZH,
                "benchmark_task_type": BenchmarkTaskType.NLI,
                "gold_label": "entailment",
                "gold_label_text": "entailment",
            }
        )
        messages, version, max_tokens = prompt_for_sample(sample)
        assert version == "xnli-nli-v2"
        assert max_tokens == 8
        assert "自然语言推断" in messages[1]["content"]
        assert "entailment" in messages[1]["content"]
        assert "paraphrase" not in messages[1]["content"]

    def test_corrupted_prompt_suite_v2_covers_all_tasks(self) -> None:
        samples = [
            (_corrupted_pair_sample(), "pawsx-binary-v2", "苐一句"),
            (_corrupted_xnli_sample(), "xnli-nli-v2", "第第二句"),
            (_corrupted_single_sample(), "asap-polarity-v2", "这这家店不错"),
            (_corrupted_mrc_sample(), "c3-letter-v2", "尚下文"),
        ]

        for sample, expected_version, corrupted_text in samples:
            clean_messages, clean_version, _ = prompt_for_sample(sample, is_clean=True)
            corrupted_messages, corrupted_version, _ = prompt_for_sample(
                sample, is_clean=False
            )
            assert clean_version == corrupted_version == expected_version
            assert clean_messages != corrupted_messages
            assert corrupted_text in corrupted_messages[1]["content"]

        c3_clean, _, _ = prompt_for_sample(_corrupted_mrc_sample(), is_clean=True)
        c3_corrupted, _, _ = prompt_for_sample(_corrupted_mrc_sample(), is_clean=False)
        assert "问题：\n问什么？" in c3_clean[1]["content"]
        assert "问题：\n问什么？" in c3_corrupted[1]["content"]
        assert "A. 甲" in c3_corrupted[1]["content"]
        assert "B. 乙" in c3_corrupted[1]["content"]


class TestRunInference:
    def test_dry_run_writes_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "dryrun.jsonl"
            config = PipelineConfig(output_path=out, model_name="mock-model/v1")
            client = MockAPIClient(expected_responses={}, latency_ms=0.0)
            ilogger = InferenceLogger(out)
            samples = [_pair_sample("s1"), _single_sample("s2")]

            results = run_inference(samples, config, client, ilogger, dry_run=True)

            assert len(results) == 2
            for r in results:
                assert r.raw_content.startswith("[dry-run]")
                assert r.finish_reason == "dry_run"
                assert r.sample_uid is not None
                assert r.request_key is not None

            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2
            for line in lines:
                rec = json.loads(line)
                assert "request_key" in rec
                assert "sample_id" in rec

    def test_resume_skips_via_request_key(self) -> None:
        """Pre-written request_key is skipped; new request_key is executed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "resume.jsonl"
            config = PipelineConfig(output_path=out, model_name="mock-model/v1")

            # First run: generate a request_key for s1
            from ruchi_bench.inference.pipeline import (
                _build_label_only_messages,
            )

            s1 = _pair_sample("s1")
            msgs_s1 = _build_label_only_messages(s1)
            rk_s1 = _build_request_key(s1, config, is_clean=True, messages=msgs_s1)

            # Pre-write s1's record
            out.write_text(
                json.dumps(
                    {
                        "request_key": rk_s1,
                        "sample_id": "s1",
                        "model": "mock",
                        "messages": [],
                        "raw_response": {},
                        "latency_ms": 0.0,
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            client = MockAPIClient(expected_responses={}, latency_ms=0.0)
            ilogger = InferenceLogger(out)

            # s1 should be skipped; s2 is new
            samples = [s1, _single_sample("s2")]
            results = run_inference(samples, config, client, ilogger, dry_run=True)

            # s1 skipped, s2 executed
            assert len(results) == 1
            assert results[0].sample_id == "s2"
            lines = out.read_text(encoding="utf-8").strip().split("\n")
            assert len(lines) == 2  # original s1 + new s2

    def test_result_has_all_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "fields.jsonl"
            config = PipelineConfig(output_path=out, model_name="test-model")
            client = MockAPIClient(expected_responses={}, latency_ms=0.0)
            ilogger = InferenceLogger(out)
            s1 = _pair_sample("s1")
            samples = [s1]

            results = run_inference(samples, config, client, ilogger, dry_run=True)
            assert len(results) == 1
            r = results[0]
            assert r.sample_id == "s1"
            assert r.sample_uid == sample_uid("s1", "pawsx_zh", "test")
            assert r.request_key is not None
            assert len(r.request_key) == 64  # SHA-256 hex
            assert r.model == "test-model"
            assert r.raw_content.startswith("[dry-run]")
            assert r.finish_reason == "dry_run"
            assert r.latency_ms == 0.0
            assert r.error is None


class TestInferenceResult:
    def test_parsed_label(self) -> None:
        r = InferenceResult(
            sample_id="s1",
            sample_uid="x___t___s1",
            request_key="abc123",
            model="mock",
            raw_content="  paraphrase  ",
            finish_reason="stop",
            latency_ms=10.0,
        )
        assert r.parsed_label() == "paraphrase"

    def test_parsed_label_empty(self) -> None:
        r = InferenceResult(
            sample_id="s1",
            sample_uid="x___t___s1",
            request_key="abc123",
            model="mock",
            raw_content="   ",
            finish_reason="stop",
            latency_ms=10.0,
        )
        assert r.parsed_label() is None
