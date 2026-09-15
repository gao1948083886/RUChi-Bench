"""Unit tests for ruchi_bench.schema.serialization. All data is synthetic."""

from __future__ import annotations

import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pytest

from ruchi_bench.schema import (
    BenchmarkTaskType,
    DatasetName,
    Language,
    PairPayload,
    PayloadField,
    ProjectionStatus,
    Sample,
    SingleTextPayload,
    SplitName,
)
from ruchi_bench.schema.errors import SerializationError
from ruchi_bench.schema.serialization import (
    read_jsonl,
    sample_from_json,
    sample_to_json,
    write_jsonl,
)

_AWARE = datetime(2026, 7, 27, tzinfo=UTC)


def _clean_pair() -> Sample:
    return Sample(
        sample_id="p1",
        source_sample_id="0",
        dataset_name=DatasetName.PAWSX_ZH,
        dataset_version="v1",
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
        created_at=_AWARE,
    )


def _clean_asap() -> Sample:
    return Sample(
        sample_id="a1",
        source_sample_id="5",
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
        created_at=_AWARE,
    )


# === object round-trip ===


def test_sample_to_json_round_trip_pair() -> None:
    s = _clean_pair()
    json_str = sample_to_json(s)
    restored = sample_from_json(json_str)
    assert restored.sample_id == s.sample_id
    assert restored.gold_label == s.gold_label
    payload: PairPayload = restored.clean_payload  # type: ignore[assignment]
    assert payload.text_a == "第一句"


def test_sample_to_json_round_trip_asap() -> None:
    s = _clean_asap()
    json_str = sample_to_json(s)
    restored = sample_from_json(json_str)
    assert restored.sample_id == "a1"
    assert restored.label_projection_name == "asap_polarity"


def test_json_contains_utc_datetime() -> None:
    s = _clean_pair()
    d = json.loads(sample_to_json(s))
    assert d["created_at"].endswith("+00:00") or d["created_at"].endswith("Z")


def test_json_contains_str_enums() -> None:
    s = _clean_pair()
    d = json.loads(sample_to_json(s))
    assert d["benchmark_task_type"] == "pair_paraphrase"
    assert d["language"] == "zh"
    assert isinstance(d["target_fields"], list)


def test_json_contains_chinese_verbatim_by_default() -> None:
    s = _clean_pair()
    j = sample_to_json(s)
    assert "第一句" in j
    assert "\\u" not in j


def test_json_escapes_chinese_when_ensure_ascii() -> None:
    s = _clean_pair()
    j = sample_to_json(s, ensure_ascii=True)
    assert "第一句" not in j
    assert "\\u" in j


def test_from_json_unknown_field_rejected() -> None:
    bad = json.dumps(
        {
            "sample_id": "x",
            "unknown_field": 1,
            **{k: v for k, v in _clean_pair().model_dump(mode="json").items()},
        }
    )
    with pytest.raises(SerializationError) as exc_info:
        sample_from_json(bad)
    assert "validation" in str(exc_info.value).lower()


def test_from_json_malformed_json_raises_serialization_error() -> None:
    with pytest.raises(SerializationError) as exc_info:
        sample_from_json("{ bad json }")
    assert "JSON decode error" in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, json.JSONDecodeError)


# === file I/O ===


def _tmp_path() -> Path:
    fd, path = tempfile.mkstemp(suffix=".jsonl")
    os.close(fd)
    return Path(path)


def _read_text(path: Path) -> str:
    with path.open(encoding="utf-8") as fh:
        return fh.read()


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    samples = [_clean_pair(), _clean_asap()]
    result = write_jsonl(iter(samples), tmp_path / "data.jsonl")
    assert result.count == 2
    assert len(result.written) == 2
    lines = _read_text(tmp_path / "data.jsonl").splitlines()
    assert len(lines) == 2
    # Each line is a valid complete JSON object (no trailing comma, no bracket)
    assert not lines[0].startswith("[")
    assert lines[0].strip()


def test_write_jsonl_count_matches_written(tmp_path: Path) -> None:
    samples = [_clean_pair(), _clean_pair(), _clean_asap()]
    result = write_jsonl(samples, tmp_path / "data.jsonl")
    assert result.count == 3
    assert len(result.written) == 3
    assert result.written[0].sample_id == "p1"
    assert result.written[2].sample_id == "a1"


def test_read_jsonl_yields_in_order(tmp_path: Path) -> None:
    samples = [_clean_pair(), _clean_asap()]
    write_jsonl(samples, tmp_path / "data.jsonl")
    read_back = list(read_jsonl(tmp_path / "data.jsonl"))
    assert len(read_back) == 2
    assert read_back[0].sample_id == "p1"
    assert read_back[1].sample_id == "a1"


def test_read_jsonl_rejects_empty_line(tmp_path: Path) -> None:
    valid = sample_to_json(_clean_pair())
    with (tmp_path / "data.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(valid + "\n")
        fh.write("   \n")  # whitespace-only → empty after strip
    with pytest.raises(SerializationError) as exc_info:
        list(read_jsonl(tmp_path / "data.jsonl"))
    assert "empty line" in str(exc_info.value)
    assert "Line 2" in str(exc_info.value)


def test_read_jsonl_rejects_malformed_json(tmp_path: Path) -> None:
    valid = sample_to_json(_clean_pair())
    with (tmp_path / "data.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(valid + "\n")
        fh.write("{ bad }\n")  # line 2: not valid JSON
    with pytest.raises(SerializationError) as exc_info:
        list(read_jsonl(tmp_path / "data.jsonl"))
    assert "Line 2" in str(exc_info.value)
    assert "JSON decode error" in str(exc_info.value)


def test_read_jsonl_rejects_pydantic_error(tmp_path: Path) -> None:
    bad_line = json.dumps(
        {
            "sample_id": "x",
            "source_sample_id": "0",
            "dataset_name": "pawsx_zh",
            "dataset_version": "v",
            "split": "train",  # pilot rejects non-test
            "benchmark_task_type": "pair_paraphrase",
            "language": "zh",
            "clean_payload": {"payload_kind": "pair", "text_a": "a", "text_b": "b"},
            "gold_label": 1,
            "gold_label_text": "p",
            "target_fields": ["text_a"],
            "label_projection_status": "not_applicable",
            "corruption_applied": False,
            "change_count": 0,
            "created_at": "2026-07-27T00:00:00Z",
        }
    )
    with (tmp_path / "data.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(bad_line + "\n")
    with pytest.raises(SerializationError) as exc_info:
        list(read_jsonl(tmp_path / "data.jsonl"))
    assert "Line 1" in str(exc_info.value)
    assert "validation" in str(exc_info.value).lower()


def test_read_jsonl_file_not_found_raises_serialization_error() -> None:
    nonexistent = Path("C:/no/such/path/nowhere.jsonl")
    with pytest.raises(SerializationError) as exc_info:
        list(read_jsonl(nonexistent))
    assert str(nonexistent) in str(exc_info.value)
    assert isinstance(exc_info.value.__cause__, FileNotFoundError)


def test_write_jsonl_file_not_found_raises_serialization_error() -> None:
    bad_path = Path("C:/no/such/dir/file.jsonl")
    with pytest.raises(SerializationError) as exc_info:
        write_jsonl([_clean_pair()], bad_path)
    assert str(bad_path) in str(exc_info.value)


def test_read_jsonl_empty_file_yields_nothing(tmp_path: Path) -> None:
    (tmp_path / "data.jsonl").write_text("", encoding="utf-8")
    assert list(read_jsonl(tmp_path / "data.jsonl")) == []


def test_read_jsonl_single_record(tmp_path: Path) -> None:
    write_jsonl([_clean_pair()], tmp_path / "data.jsonl")
    [s] = list(read_jsonl(tmp_path / "data.jsonl"))
    assert s.sample_id == "p1"


def test_write_and_read_preserves_all_fields(tmp_path: Path) -> None:
    s = _clean_asap()
    write_jsonl([s], tmp_path / "data.jsonl")
    [restored] = list(read_jsonl(tmp_path / "data.jsonl"))
    assert restored.sample_id == s.sample_id
    assert restored.label_projection_name == "asap_polarity"
    assert restored.label_projection_status is ProjectionStatus.ACCEPTED
    assert restored.source_gold_label == 5
    assert restored.created_at == _AWARE
