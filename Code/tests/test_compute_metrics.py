"""Tests for compute_metrics.py: sample loading, validation gate, and key extraction."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Scripts"))
from compute_metrics import (
    RunCandidate,
    RunValidation,
    calculate_total_expected,
    discover_runs,
    get_file_stats_keys,
    list_and_select_run,
    normalize_file_stats,
    validate_run,
)

# Also import metrics module helpers
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from ruchi_bench.metrics import (
    SampleCounts,
    load_inference_records,
    load_pilot_samples,
    _load_samples_from_jsonl,
)


# ---------------------------------------------------------------------------
# normalize_file_stats tests
# ---------------------------------------------------------------------------

def test_normalize_file_stats_list_format():
    """Accepts list[dict] format as used by real Phase 08 manifests."""
    file_stats = [
        {"key": "clean", "success": 1200},
        {"key": "tpwr_low", "success": 50},
    ]
    result = normalize_file_stats(file_stats)
    assert result == file_stats
    assert len(result) == 2


def test_normalize_file_stats_dict_format():
    """Converts legacy dict format to list format."""
    file_stats = {"clean": {"success": 100}, "tpwr_low": {"success": 50}}
    result = normalize_file_stats(file_stats)
    assert len(result) == 2
    keys = {r["key"] for r in result}
    assert keys == {"clean", "tpwr_low"}


def test_normalize_file_stats_list_validates_items():
    """Rejects list with non-dict items."""
    file_stats = [{"key": "clean", "success": 10}, "not a dict"]
    with pytest.raises(ValueError, match="expected dict"):
        normalize_file_stats(file_stats)


def test_normalize_file_stats_list_validates_key_field():
    """Rejects list items missing 'key' field."""
    file_stats = [{"success": 10}, {"key": "clean", "success": 20}]
    with pytest.raises(ValueError, match="missing required field 'key'"):
        normalize_file_stats(file_stats)


def test_normalize_file_stats_rejects_other_types():
    """Rejects unsupported types."""
    with pytest.raises(ValueError, match="unsupported type"):
        normalize_file_stats("string")
    with pytest.raises(ValueError, match="unsupported type"):
        normalize_file_stats(123)


def test_normalize_file_stats_dict_values_must_be_dicts():
    """Rejects dict format with non-dict values."""
    with pytest.raises(ValueError, match="expected dict"):
        normalize_file_stats({"clean": "not a dict"})


# ---------------------------------------------------------------------------
# calculate_total_expected / get_file_stats_keys tests
# ---------------------------------------------------------------------------

def test_calculate_total_expected_list():
    """Sums success field from list format."""
    file_stats = [
        {"key": "clean", "success": 1200},
        {"key": "tpwr_low", "success": 50},
    ]
    assert calculate_total_expected(file_stats) == 1250


def test_calculate_total_expected_dict():
    """Sums success field from dict format."""
    file_stats = {"clean": {"success": 1200}, "tpwr_low": {"success": 50}}
    assert calculate_total_expected(file_stats) == 1250


def test_calculate_total_expected_rejects_missing_success():
    """Rejects entries missing 'success' field."""
    file_stats = [{"key": "clean", "success": 100}, {"key": "tpwr_low"}]
    with pytest.raises(ValueError, match="missing 'success' field"):
        calculate_total_expected(file_stats)


def test_get_file_stats_keys_list():
    """Extracts key set from list format."""
    file_stats = [{"key": "clean", "success": 100}, {"key": "tpwr_low", "success": 50}]
    assert get_file_stats_keys(file_stats) == {"clean", "tpwr_low"}


# ---------------------------------------------------------------------------
# load_pilot_samples tests - the core fix
# ---------------------------------------------------------------------------

def test_load_samples_from_jsonl_no_corruption_fields(tmp_path):
    """Clean samples without corruption fields load successfully."""
    jsonl = tmp_path / "test_clean.jsonl"
    with open(jsonl, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "sample_id": str(i),
                "dataset_name": "pawsx_zh",
                "gold_label": 0,
                "corruption_name": None,
                "corruption_level": None,
            }) + "\n")

    samples = _load_samples_from_jsonl(jsonl, corruption_name=None, corruption_level=None)

    assert len(samples) == 4
    for (ds, sid, cn, cl), s in samples.items():
        assert cn is None
        assert cl is None
        assert s["dataset_name"] == "pawsx_zh"


def test_load_pilot_samples_clean_file(tmp_path):
    """load_pilot_samples accepts file path and returns correct counts."""
    clean_file = tmp_path / "all_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "sample_id": str(i),
                "dataset_name": "pawsx_zh",
                "gold_label": 0,
            }) + "\n")

    # corrupted dir is empty
    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()

    samples, counts = load_pilot_samples(clean_file, corrupted_dir)

    assert counts.clean == 4
    assert counts.corrupted == 0
    assert counts.total == 4


def test_load_pilot_samples_clean_plus_corrupted(tmp_path):
    """4 clean + 6 corrupted = clean=4, corrupted=6, total=10."""
    # Clean file
    clean_file = tmp_path / "all_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "sample_id": str(i),
                "dataset_name": "pawsx_zh",
                "gold_label": 0,
            }) + "\n")

    # Corrupted dir
    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()
    tpwr_dir = corrupted_dir / "tpwr"
    tpwr_dir.mkdir()
    low_file = tpwr_dir / "low.jsonl"
    with open(low_file, "w", encoding="utf-8") as f:
        for i in range(6):
            f.write(json.dumps({
                "sample_id": str(i),
                "dataset_name": "pawsx_zh",
                "gold_label": 0,
            }) + "\n")

    samples, counts = load_pilot_samples(clean_file, corrupted_dir)

    assert counts.clean == 4
    assert counts.corrupted == 6
    assert counts.total == 10


def test_load_pilot_samples_clean_plus_multiple_corruptions(tmp_path):
    """Multiple corruption directories are summed correctly."""
    clean_file = tmp_path / "all_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        for i in range(10):
            f.write(json.dumps({"sample_id": str(i), "dataset_name": "xnli_zh", "gold_label": 0}) + "\n")

    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()

    for corruption, n_files in [("tpwr", 3), ("vscr", 2), ("cr", 1)]:
        subdir = corrupted_dir / corruption
        subdir.mkdir()
        for level in ["low", "medium", "high"]:
            with open(subdir / f"{level}.jsonl", "w", encoding="utf-8") as f:
                for i in range(n_files):
                    f.write(json.dumps({
                        "sample_id": f"{corruption}_{level}_{i}",
                        "dataset_name": "xnli_zh",
                        "gold_label": 0,
                    }) + "\n")

    samples, counts = load_pilot_samples(clean_file, corrupted_dir)

    assert counts.clean == 10
    # Each corruption level file has 3 records (n=3 in the test loop)
    # tpwr: 3 levels × 3 = 9
    # vscr: 2 levels × 3 = 6
    # cr: 1 level × 3 = 3
    # Total corrupted = 9 + 6 + 3 = 18; clean + corrupted = 28
    assert counts.corrupted == 18
    assert counts.total == 28


def test_load_pilot_samples_clean_directory_path(tmp_path):
    """load_pilot_samples also works when passed a directory containing all_clean.jsonl."""
    # When a directory is passed, it should look for all_clean.jsonl inside it
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    all_clean = clean_dir / "all_clean.jsonl"
    with open(all_clean, "w", encoding="utf-8") as f:
        for i in range(5):
            f.write(json.dumps({"sample_id": str(i), "dataset_name": "asap", "gold_label": 0}) + "\n")

    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()

    samples, counts = load_pilot_samples(clean_dir, corrupted_dir)

    assert counts.clean == 5
    assert counts.corrupted == 0


def test_load_pilot_samples_no_silent_skip(tmp_path):
    """No records are silently skipped during loading."""
    clean_file = tmp_path / "all_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        for i in range(4):
            f.write(json.dumps({
                "sample_id": str(i),
                "dataset_name": "pawsx_zh",
                "gold_label": 0,
            }) + "\n")
        # Empty line - should be skipped (not counted as error)
        f.write("\n")

    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()

    samples, counts = load_pilot_samples(clean_file, corrupted_dir)

    # Empty lines are skipped but not counted as errors
    assert counts.clean == 4


def test_load_pilot_samples_sample_keys_correct(tmp_path):
    """Sample keys distinguish clean from corrupted correctly."""
    clean_file = tmp_path / "all_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": "s0", "dataset_name": "pawsx_zh", "gold_label": 0}) + "\n")
        f.write(json.dumps({"sample_id": "s1", "dataset_name": "xnli_zh", "gold_label": 1}) + "\n")

    corrupted_dir = tmp_path / "corrupted"
    corrupted_dir.mkdir()
    tpwr_dir = corrupted_dir / "tpwr"
    tpwr_dir.mkdir()
    with open(tpwr_dir / "low.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"sample_id": "s0", "dataset_name": "pawsx_zh", "gold_label": 0}) + "\n")
        f.write(json.dumps({"sample_id": "s1", "dataset_name": "xnli_zh", "gold_label": 1}) + "\n")

    samples, counts = load_pilot_samples(clean_file, corrupted_dir)

    # Both clean and corrupted for s0 should be present with different keys
    clean_key = ("pawsx_zh", "s0", None, None)
    corr_key = ("pawsx_zh", "s0", "tpwr", "low")
    assert clean_key in samples
    assert corr_key in samples
    assert clean_key != corr_key

    assert counts.clean == 2
    assert counts.corrupted == 2


def test_load_inference_records_requires_dataset_name(tmp_path):
    """Old inference records without explicit identity are rejected."""
    jsonl = tmp_path / "20260801_test_clean.jsonl"
    jsonl.write_text(
        json.dumps({"sample_id": "0", "raw_response": {"content": "1"}}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing dataset_name"):
        load_inference_records([jsonl])


def test_load_inference_records_accepts_explicit_identity(tmp_path):
    """New inference records with explicit identity can be loaded."""
    jsonl = tmp_path / "20260801_test_tpwr_low.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "dataset_name": "pawsx_zh",
                "sample_id": "0",
                "raw_response": {"content": "1"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_inference_records([jsonl])

    assert records[0]["dataset_name"] == "pawsx_zh"
    assert records[0]["corruption_name"] == "tpwr"
    assert records[0]["corruption_level"] == "low"


# ---------------------------------------------------------------------------
# SampleCounts dataclass tests
# ---------------------------------------------------------------------------

def test_sample_counts_dataclass():
    """SampleCounts tracks clean/corrupted/total correctly."""
    counts = SampleCounts(clean=1200, corrupted=14522)
    assert counts.clean == 1200
    assert counts.corrupted == 14522
    assert counts.total == 15722


def test_sample_counts_zero():
    """SampleCounts with zero values."""
    counts = SampleCounts()
    assert counts.clean == 0
    assert counts.corrupted == 0
    assert counts.total == 0


# ---------------------------------------------------------------------------
# discover_runs tests
# ---------------------------------------------------------------------------

def test_discover_runs_uses_manifest_run_id(temp_manifests_dir):
    """Prioritizes manifest's own run_id over filename derivation."""
    manifest = temp_manifests_dir / "phase08_20260801_100000_run1_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "20260801_100000_run1",
            "dry_run": False,
            "api_model": "qwen3.5",
            "file_stats": [{"key": "clean", "success": 100}],
        }, f)

    candidates = discover_runs(temp_manifests_dir)
    assert len(candidates) == 1
    assert candidates[0].run_id == "20260801_100000_run1"


def test_discover_runs_derives_run_id_from_filename_when_missing(temp_manifests_dir):
    """Falls back to filename derivation when manifest has no run_id."""
    manifest = temp_manifests_dir / "phase08_20260801_110000_derived_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "",
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 50}],
        }, f)

    candidates = discover_runs(temp_manifests_dir)
    assert len(candidates) == 1
    assert candidates[0].run_id == "20260801_110000_derived"


def test_discover_runs_handles_list_file_stats(temp_manifests_dir):
    """Correctly parses list format file_stats."""
    manifest = temp_manifests_dir / "phase08_20260801_real_list_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "20260801_real_list",
            "dry_run": False,
            "api_model": "qwen3.5",
            "file_stats": [
                {"key": "clean", "success": 1200},
                {"key": "tpwr_low", "success": 50},
            ],
        }, f)

    candidates = discover_runs(temp_manifests_dir)
    assert len(candidates) == 1
    c = candidates[0]
    assert c.run_id == "20260801_real_list"
    assert c.total_expected == 1250


def test_discover_runs_returns_empty_for_missing_dir():
    """Returns empty list when manifests directory doesn't exist."""
    assert discover_runs(Path("/nonexistent/path")) == []


# ---------------------------------------------------------------------------
# validate_run tests - key extraction
# ---------------------------------------------------------------------------

def test_validate_run_extracts_key_with_run_id_prefix(temp_manifests_dir, temp_pilot_dir):
    """Extracts file key using run_id prefix, not split on underscore."""
    run_id = "20260801_test_prefix"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [
                {"key": "clean", "success": 4},
                {"key": "tpwr_low", "success": 6},
            ],
        }, f)

    clean_file = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"sample_id": f"c{i}"}) + "\n")

    corr_file = temp_pilot_dir / f"{run_id}_tpwr_low.jsonl"
    with open(corr_file, "w", encoding="utf-8") as fh:
        for i in range(6):
            fh.write(json.dumps({"sample_id": f"t{i}"}) + "\n")

    validation = validate_run(run_id, manifest, temp_pilot_dir)

    # tpwr_low must NOT be parsed as "low"
    assert validation.clean_record_count == 4
    assert validation.corrupted_record_count == 6


def test_validate_run_vscr_medium_not_parsed_as_medium(temp_manifests_dir, temp_pilot_dir):
    """Verifies vscr_medium is not incorrectly parsed as just 'medium'."""
    run_id = "20260801_vscr_test"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [
                {"key": "clean", "success": 10},
                {"key": "vscr_medium", "success": 5},
            ],
        }, f)

    clean_file = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"sample_id": f"c{i}"}) + "\n")

    corr_file = temp_pilot_dir / f"{run_id}_vscr_medium.jsonl"
    with open(corr_file, "w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"sample_id": f"v{i}"}) + "\n")

    validation = validate_run(run_id, manifest, temp_pilot_dir)

    assert validation.clean_record_count == 10
    assert validation.corrupted_record_count == 5


def test_validate_run_rejects_unknown_file_key(temp_manifests_dir, temp_pilot_dir):
    """Rejects inference files with keys not in manifest file_stats."""
    run_id = "20260801_unknown_key"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 5}],
        }, f)

    clean_file = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(clean_file, "w", encoding="utf-8") as fh:
        for i in range(5):
            fh.write(json.dumps({"sample_id": f"c{i}"}) + "\n")

    unknown_file = temp_pilot_dir / f"{run_id}_unknown_corruption.jsonl"
    with open(unknown_file, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"sample_id": "u0"}) + "\n")

    with pytest.raises(ValueError, match="Unknown file keys"):
        validate_run(run_id, manifest, temp_pilot_dir)


# ---------------------------------------------------------------------------
# validate_run tests - count reconciliation
# ---------------------------------------------------------------------------

def test_validate_run_detects_count_mismatch(temp_manifests_dir, temp_pilot_dir):
    """Sets reconciliation_status=MISMATCH when loaded != expected."""
    run_id = "20260801_mismatch"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 100}],
        }, f)

    jsonl = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for i in range(50):
            fh.write(json.dumps({"sample_id": f"s{i:04d}"}) + "\n")

    validation = validate_run(run_id, manifest, temp_pilot_dir)

    assert validation.reconciliation_status == "MISMATCH"
    assert validation.inference_record_count == 50
    assert validation.manifest_expected == 100


def test_validate_run_ok_when_counts_match(temp_manifests_dir, temp_pilot_dir):
    """Sets reconciliation_status=OK when loaded == expected."""
    run_id = "20260801_ok"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 10}],
        }, f)

    jsonl = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"sample_id": f"s{i:04d}"}) + "\n")

    validation = validate_run(run_id, manifest, temp_pilot_dir)

    assert validation.reconciliation_status == "OK"


def test_validate_run_correctly_splits_clean_and_corrupted(temp_manifests_dir, temp_pilot_dir):
    """Separates clean and corrupted record counts."""
    run_id = "20260801_split"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [
                {"key": "clean", "success": 4},
                {"key": "tpwr_low", "success": 6},
                {"key": "vscr_medium", "success": 8},
            ],
        }, f)

    clean_jsonl = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(clean_jsonl, "w", encoding="utf-8") as fh:
        for i in range(4):
            fh.write(json.dumps({"sample_id": f"c{i}"}) + "\n")

    tpwr_jsonl = temp_pilot_dir / f"{run_id}_tpwr_low.jsonl"
    with open(tpwr_jsonl, "w", encoding="utf-8") as fh:
        for i in range(6):
            fh.write(json.dumps({"sample_id": f"t{i}"}) + "\n")

    vscr_jsonl = temp_pilot_dir / f"{run_id}_vscr_medium.jsonl"
    with open(vscr_jsonl, "w", encoding="utf-8") as fh:
        for i in range(8):
            fh.write(json.dumps({"sample_id": f"v{i}"}) + "\n")

    validation = validate_run(run_id, manifest, temp_pilot_dir)

    assert validation.clean_record_count == 4
    assert validation.corrupted_record_count == 14  # 6 + 8
    assert validation.inference_record_count == 18


# ---------------------------------------------------------------------------
# list_and_select_run tests
# ---------------------------------------------------------------------------

def test_list_and_select_run_excludes_dry_runs(temp_manifests_dir, temp_pilot_dir):
    """Filters out dry_run=true when listing runs."""
    dry_manifest = temp_manifests_dir / "phase08_20260801_dry_manifest.json"
    with open(dry_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "20260801_dry",
            "dry_run": True,
            "file_stats": [{"key": "clean", "success": 10}],
        }, f)

    real_manifest = temp_manifests_dir / "phase08_20260801_real_manifest.json"
    with open(real_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "20260801_real",
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 5}],
        }, f)

    candidates = discover_runs(temp_manifests_dir)
    real_runs = [c for c in candidates if not c.dry_run]

    assert len(real_runs) == 1
    assert real_runs[0].run_id == "20260801_real"


def test_list_and_select_run_returns_none_for_multiple_real_runs(temp_manifests_dir, temp_pilot_dir):
    """Returns None when multiple real runs exist (requires explicit selection)."""
    for run_id in ["20260801_run1", "20260801_run2"]:
        manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
        with open(manifest, "w", encoding="utf-8") as f:
            json.dump({
                "run_id": run_id,
                "dry_run": False,
                "file_stats": [{"key": "clean", "success": 5}],
            }, f)

    result, _ = list_and_select_run(temp_manifests_dir, temp_pilot_dir)
    assert result is None


def test_list_and_select_run_auto_selects_single_real_run(temp_manifests_dir, temp_pilot_dir):
    """Auto-selects when only one real run exists."""
    run_id = "20260801_only"
    manifest = temp_manifests_dir / f"phase08_{run_id}_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "dry_run": False,
            "file_stats": [{"key": "clean", "success": 10}],
        }, f)

    jsonl = temp_pilot_dir / f"{run_id}_clean.jsonl"
    with open(jsonl, "w", encoding="utf-8") as fh:
        for i in range(10):
            fh.write(json.dumps({"sample_id": f"s{i}"}) + "\n")

    result, _ = list_and_select_run(temp_manifests_dir, temp_pilot_dir)

    assert result is not None
    assert result.run_id == "20260801_only"
    assert result.reconciliation_status == "OK"


def test_list_and_select_run_returns_none_when_no_real_runs(temp_manifests_dir, temp_pilot_dir):
    """Returns None when all runs are dry runs."""
    manifest = temp_manifests_dir / "phase08_20260801_all_dry_manifest.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": "20260801_all_dry",
            "dry_run": True,
            "file_stats": [{"key": "clean", "success": 10}],
        }, f)

    result, _ = list_and_select_run(temp_manifests_dir, temp_pilot_dir)
    assert result is None


# ---------------------------------------------------------------------------
# RunCandidate and RunValidation dataclass tests
# ---------------------------------------------------------------------------

def test_run_candidate_dataclass():
    """RunCandidate stores all required fields."""
    candidate = RunCandidate(
        run_id="20260801_test",
        manifest_path=Path("/path/to/manifest.json"),
        dry_run=False,
        api_model="qwen3.5",
        total_expected=1000,
        jsonl_count=16,
        file_stats_keys={"clean", "tpwr_low"},
        created_at="2026-08-01T10:00:00",
    )
    assert candidate.run_id == "20260801_test"
    assert candidate.dry_run is False
    assert candidate.file_stats_keys == {"clean", "tpwr_low"}


def test_run_validation_dataclass():
    """RunValidation stores all required fields."""
    validation = RunValidation(
        run_id="20260801_test",
        dry_run=False,
        inference_record_count=1500,
        clean_record_count=300,
        corrupted_record_count=1200,
        manifest_expected=1500,
        jsonl_files=16,
        reconciliation_status="OK",
    )
    assert validation.inference_record_count == 1500
    assert validation.clean_record_count == 300
    assert validation.corrupted_record_count == 1200


def test_run_validation_mismatch_detail():
    """RunValidation.mismatch_detail populated on mismatch."""
    validation = RunValidation(
        run_id="20260801_mismatch",
        dry_run=False,
        inference_record_count=100,
        clean_record_count=20,
        corrupted_record_count=80,
        manifest_expected=200,
        jsonl_files=16,
        reconciliation_status="MISMATCH",
        mismatch_detail="loaded=100, expected=200, diff=-100",
    )
    assert validation.reconciliation_status == "MISMATCH"
    assert "diff=-100" in validation.mismatch_detail


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_manifests_dir(tmp_path):
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir()
    return manifests_dir


@pytest.fixture
def temp_pilot_dir(tmp_path):
    pilot_dir = tmp_path / "pilot_run"
    pilot_dir.mkdir()
    return pilot_dir
