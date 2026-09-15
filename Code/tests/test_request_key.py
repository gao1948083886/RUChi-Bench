"""Unit tests for the request_key module.

Tests the three-layer identity system (sample_id / sample_uid / request_key).
"""

from __future__ import annotations

import json
from pathlib import Path

from ruchi_bench.inference.request_key import (
    REQUEST_KEY_SCHEMA_VERSION,
    RequestKeyComponents,
    make_request_key,
    read_request_keys_from_jsonl,
    sample_uid,
)


class TestSampleUid:
    def test_different_datasets_same_id(self) -> None:
        """Identical sample_id across datasets must NOT collide."""
        uid1 = sample_uid("0", "pawsx_zh", "test")
        uid2 = sample_uid("0", "xnli_zh", "test")
        uid3 = sample_uid("0", "c3", "test")
        assert uid1 != uid2 != uid3

    def test_different_splits_same_dataset(self) -> None:
        """Identical sample_id across splits must NOT collide."""
        uid1 = sample_uid("42", "pawsx_zh", "test")
        uid2 = sample_uid("42", "pawsx_zh", "dev")
        uid3 = sample_uid("42", "pawsx_zh", "train")
        assert uid1 != uid2 != uid3

    def test_consistency(self) -> None:
        """Calling sample_uid twice with the same inputs returns identical string."""
        uid_a = sample_uid("abc123", "xnli_zh", "test")
        uid_b = sample_uid("abc123", "xnli_zh", "test")
        assert uid_a == uid_b

    def test_format(self) -> None:
        """sample_uid follows dataset___split___sample_id format."""
        uid = sample_uid("99", "asap", "test")
        assert uid == "asap___test___99"


class TestMakeRequestKey:
    """12 test cases for request_key collisions and uniqueness."""

    def test_tc01_different_datasets_same_id(self) -> None:
        """[TC-01] Identical sample_id across different datasets → unique keys."""
        msgs = [{"role": "user", "content": "hello"}]
        msgs_hash = _stub_messages_hash(msgs)

        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(sample_uid="pawsx_zh___test___0", **base))
        rk2 = make_request_key(RequestKeyComponents(sample_uid="xnli_zh___test___0", **base))
        rk3 = make_request_key(RequestKeyComponents(sample_uid="c3___test___0", **base))
        assert len({rk1, rk2, rk3}) == 3

    def test_tc02_different_splits_same_dataset(self) -> None:
        """[TC-02] Same dataset, different splits → unique keys."""
        msgs = [{"role": "user", "content": "hello"}]
        msgs_hash = _stub_messages_hash(msgs)
        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(sample_uid="pawsx_zh___test___42", **base))
        rk2 = make_request_key(RequestKeyComponents(sample_uid="pawsx_zh___dev___42", **base))
        assert rk1 != rk2

    def test_tc03_clean_vs_corrupted_same_sample(self) -> None:
        """[TC-03] Same sample, clean vs corrupted → unique keys."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk_clean = make_request_key(RequestKeyComponents(
            sample_uid="pawsx_zh___test___0", is_clean=True, **base))
        rk_corrupt = make_request_key(RequestKeyComponents(
            sample_uid="pawsx_zh___test___0", is_clean=False, **base))
        assert rk_clean != rk_corrupt

    def test_tc04_identical_run_same_key(self) -> None:
        """[TC-04] Identical run parameters → identical keys (stable)."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk_a = make_request_key(RequestKeyComponents(sample_uid="asap___test___7", **base))
        rk_b = make_request_key(RequestKeyComponents(sample_uid="asap___test___7", **base))
        assert rk_a == rk_b

    def test_tc05_different_model_same_sample(self) -> None:
        """[TC-05] Same sample, different model → unique keys."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=True, messages_hash=msgs_hash,
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___5", model_uid="qwen3.5", **base))
        rk2 = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___5", model_uid="glm-4-9b", **base))
        assert rk1 != rk2

    def test_tc06_different_temperature(self) -> None:
        """[TC-06] Same sample, different temperature → unique keys."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(
            sample_uid="c3___test___1", temperature=0.0, **base))
        rk2 = make_request_key(RequestKeyComponents(
            sample_uid="c3___test___1", temperature=0.7, **base))
        assert rk1 != rk2

    def test_tc07_different_max_new_tokens(self) -> None:
        """[TC-07] Same sample, different max_new_tokens → unique keys."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0)
        rk1 = make_request_key(RequestKeyComponents(
            sample_uid="pawsx_zh___test___3", max_new_tokens=8, **base))
        rk2 = make_request_key(RequestKeyComponents(
            sample_uid="pawsx_zh___test___3", max_new_tokens=32, **base))
        assert rk1 != rk2

    def test_tc08_different_messages(self) -> None:
        """[TC-08] Same sample, different message content → unique keys."""
        msgs_hash1 = _stub_messages_hash([{"role": "user", "content": "Text A: 你好"}])
        msgs_hash2 = _stub_messages_hash([{"role": "user", "content": "Text A: 您好"}])
        base = dict(is_clean=True, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___9", messages_hash=msgs_hash1, **base))
        rk2 = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___9", messages_hash=msgs_hash2, **base))
        assert rk1 != rk2

    def test_tc09_different_prompt_template_version(self) -> None:
        """[TC-09] Same sample, different prompt_template_version → unique keys."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=True, messages_hash=msgs_hash, model_uid="qwen3.5",
                    temperature=0.0, max_new_tokens=8)
        rk1 = make_request_key(RequestKeyComponents(
            sample_uid="asap___test___2", prompt_template_version=1, **base))
        rk2 = make_request_key(RequestKeyComponents(
            sample_uid="asap___test___2", prompt_template_version=2, **base))
        assert rk1 != rk2

    def test_tc10_sha256_format(self) -> None:
        """[TC-10] request_key is a 64-char hex SHA-256 digest."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "test"}])
        rk = make_request_key(RequestKeyComponents(
            sample_uid="pawsx_zh___test___0", is_clean=True, messages_hash=msgs_hash,
            model_uid="q", prompt_template_version=1, temperature=0.0, max_new_tokens=8))
        assert len(rk) == 64
        assert all(c in "0123456789abcdef" for c in rk)

    def test_tc11_corruption_params_in_key(self) -> None:
        """[TC-11] Corruption name/level/seed contribute to key uniqueness."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "hello"}])
        base = dict(is_clean=False, messages_hash=msgs_hash, model_uid="qwen3.5",
                    prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        rk_tpwr_low = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___0", corruption_name="tpwr",
            corruption_level="low", corruption_seed=42, **base))
        rk_tpwr_high = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___0", corruption_name="tpwr",
            corruption_level="high", corruption_seed=42, **base))
        rk_vscr = make_request_key(RequestKeyComponents(
            sample_uid="xnli_zh___test___0", corruption_name="vscr",
            corruption_level="low", corruption_seed=42, **base))
        assert rk_tpwr_low != rk_tpwr_high != rk_vscr


class TestReadRequestKeysFromJsonl:
    """JSONL reader tests."""

    def test_new_format_with_request_key(self, tmp_path: Path) -> None:
        """[TC-12] Records with request_key are read correctly."""
        p = tmp_path / "new.jsonl"
        records = [
            {"request_key": "aaa111", "sample_id": "0"},
            {"request_key": "bbb222", "sample_id": "0"},
            {"request_key": "ccc333", "sample_id": "42"},
        ]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == {"aaa111", "bbb222", "ccc333"}
        assert legacy == []

    def test_legacy_format_no_request_key(self, tmp_path: Path) -> None:
        """[TC-13] Legacy records without request_key go to legacy list."""
        p = tmp_path / "legacy.jsonl"
        records = [
            {"sample_id": "0"},
            {"sample_id": "1"},
            {"sample_id": "0"},  # duplicate sample_id
        ]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == set()
        assert legacy == ["0", "1", "0"]

    def test_mixed_format(self, tmp_path: Path) -> None:
        """[TC-14] Mixed new and legacy records are correctly split."""
        p = tmp_path / "mixed.jsonl"
        records = [
            {"request_key": "abc", "sample_id": "5"},
            {"sample_id": "3"},  # legacy
            {"request_key": "def", "sample_id": "5"},
            {"sample_id": "2"},  # legacy
        ]
        p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")

        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == {"abc", "def"}
        assert legacy == ["3", "2"]

    def test_empty_file(self, tmp_path: Path) -> None:
        """[TC-15] Empty JSONL returns empty sets."""
        p = tmp_path / "empty.jsonl"
        p.write_text("", encoding="utf-8")
        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == set()
        assert legacy == []

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """[TC-16] Non-existent file returns empty sets."""
        p = tmp_path / "no_such_file.jsonl"
        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == set()
        assert legacy == []

    def test_invalid_json_lines_skipped(self, tmp_path: Path) -> None:
        """[TC-17] Non-JSON lines are silently skipped."""
        p = tmp_path / "bad.jsonl"
        p.write_text(
            '{"request_key": "valid1", "sample_id": "1"}\n'
            'not valid json\n'
            '{"request_key": "valid2", "sample_id": "2"}\n',
            encoding="utf-8",
        )
        request_keys, legacy = read_request_keys_from_jsonl(p)
        assert request_keys == {"valid1", "valid2"}
        assert legacy == []

    def test_schema_version_in_dict(self) -> None:
        """[TC-18] RequestKeyComponents includes schema_version in to_canonical_dict."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "t"}])
        comp = RequestKeyComponents(
            sample_uid="x___t___0", is_clean=True, messages_hash=msgs_hash,
            model_uid="q", prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        d = comp.to_canonical_dict()
        assert d["schema_version"] == REQUEST_KEY_SCHEMA_VERSION
        assert d["schema_version"] == 1

    def test_none_fields_omitted(self) -> None:
        """[TC-19] None optional fields are omitted in canonical dict."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "t"}])
        comp = RequestKeyComponents(
            sample_uid="x___t___0", is_clean=True, messages_hash=msgs_hash,
            model_uid="q", prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        d = comp.to_canonical_dict()
        assert "corruption_name" not in d
        assert "corruption_level" not in d
        assert "corruption_seed" not in d
        assert "target_field" not in d

    def test_non_none_fields_present(self) -> None:
        """[TC-20] Non-None optional fields are included in canonical dict."""
        msgs_hash = _stub_messages_hash([{"role": "user", "content": "t"}])
        comp = RequestKeyComponents(
            sample_uid="x___t___0", is_clean=False,
            corruption_name="tpwr", corruption_level="high", corruption_seed=123,
            target_field="context",
            messages_hash=msgs_hash, model_uid="q",
            prompt_template_version=1, temperature=0.0, max_new_tokens=8)
        d = comp.to_canonical_dict()
        assert d["corruption_name"] == "tpwr"
        assert d["corruption_level"] == "high"
        assert d["corruption_seed"] == 123
        assert d["target_field"] == "context"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _stub_messages_hash(messages: list[dict[str, object]]) -> str:
    """Compute a messages hash using the same logic as pipeline._messages_hash."""
    import hashlib
    canonical = json.dumps(messages, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
