"""Compute metrics from RUChi-Bench inference results.

Usage:
    python Scripts/compute_metrics.py --run-id <RUN_ID>
    python Scripts/compute_metrics.py --run-id 20260801_120000_123456

Without --run-id, lists available real runs and requires explicit selection.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Fix Windows GBK stdout encoding for Chinese characters
with suppress(Exception):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure package is importable when this script is run from any working directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "src"))

from ruchi_bench.metrics import (  # noqa: E402
    compute_metrics,
    format_metrics_table,
    load_inference_records,
    load_pilot_samples,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Manifest schema normalization
# ---------------------------------------------------------------------------


def normalize_file_stats(file_stats: Any) -> list[dict[str, Any]]:
    """Normalize file_stats to a consistent list format.

    Phase 08 manifests have file_stats as a list:
        [{"key": "clean", "success": 1200, ...}, ...]

    Legacy/dict format:
        {"clean": {"success": 1200}, ...}

    Both are normalized to:
        [{"key": "clean", "success": 1200}, ...]
    """
    if isinstance(file_stats, list):
        # Validate list items
        for i, item in enumerate(file_stats):
            if not isinstance(item, dict):
                raise ValueError(
                    f"file_stats[{i}] is {type(item).__name__}, expected dict"
                )
            if "key" not in item:
                raise ValueError(
                    f"file_stats[{i}] missing required field 'key': {item!r}"
                )
        return list(file_stats)

    if isinstance(file_stats, dict):
        # Convert dict format to list format
        result = []
        for key, value in file_stats.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"file_stats['{key}'] is {type(value).__name__}, expected dict"
                )
            stat = dict(value)
            stat["key"] = key
            result.append(stat)
        return result

    raise ValueError(
        f"file_stats has unsupported type {type(file_stats).__name__}: "
        f"expected list[dict] or dict"
    )


def calculate_total_expected(file_stats: Any) -> int:
    """Calculate total expected records from file_stats."""
    normalized = normalize_file_stats(file_stats)
    total = 0
    for stat in normalized:
        if "success" not in stat:
            raise ValueError(
                f"file_stats entry missing 'success' field: {stat!r}"
            )
        total += stat["success"]
    return total


def get_file_stats_keys(file_stats: Any) -> set[str]:
    """Get the set of valid file keys from file_stats."""
    normalized = normalize_file_stats(file_stats)
    return {stat["key"] for stat in normalized}


# ---------------------------------------------------------------------------
# Run discovery and validation
# ---------------------------------------------------------------------------


@dataclass
class RunCandidate:
    """A candidate pilot run for metrics computation."""
    run_id: str
    manifest_path: Path
    dry_run: bool
    api_model: str
    total_expected: int  # sum of success across all file_stats
    jsonl_count: int
    file_stats_keys: set[str]
    created_at: str | None = None


@dataclass
class RunValidation:
    """Result of validating a run against inference files."""
    run_id: str
    dry_run: bool
    inference_record_count: int
    clean_record_count: int
    corrupted_record_count: int
    manifest_expected: int
    jsonl_files: int
    reconciliation_status: str  # "OK" or "MISMATCH"
    mismatch_detail: str = ""


def discover_runs(manifests_dir: Path) -> list[RunCandidate]:
    """Discover all pilot run manifests in the manifests directory."""
    if not manifests_dir.exists():
        return []

    candidates: list[RunCandidate] = []
    for mf_path in sorted(manifests_dir.glob("phase08_*_manifest.json")):
        try:
            with mf_path.open(encoding="utf-8") as f:
                mdata = json.load(f)

            # Priority 1: use manifest's own run_id if non-empty
            run_id = mdata.get("run_id", "")
            if not run_id:
                # Priority 2: derive from filename (phase08_<RUN_ID>_manifest.json)
                filename = mf_path.stem  # e.g. "phase08_20260801_100000_run1_manifest"
                if filename.startswith("phase08_"):
                    run_id = filename[len("phase08_"):]
                    run_id = run_id.replace("_manifest", "")
                else:
                    run_id = filename.replace("_manifest", "")

            dry_run = mdata.get("dry_run", False)
            api_model = mdata.get("api_model", "unknown")
            created_at = mdata.get("created_at")
            file_stats = mdata.get("file_stats", [])

            # Calculate expected total and get valid keys
            total_expected = calculate_total_expected(file_stats)
            file_stats_keys = get_file_stats_keys(file_stats)

            candidates.append(RunCandidate(
                run_id=run_id,
                manifest_path=mf_path,
                dry_run=dry_run,
                api_model=api_model,
                total_expected=total_expected,
                jsonl_count=len(file_stats),
                file_stats_keys=file_stats_keys,
                created_at=created_at,
            ))
        except Exception as e:
            logger.warning("Failed to parse manifest %s: %s", mf_path, e)
            continue

    return candidates


def validate_run(
    run_id: str,
    manifest_path: Path,
    pilot_dir: Path,
) -> RunValidation:
    """Validate a run against its inference JSONL files.

    Returns RunValidation with count reconciliation status.
    """
    with manifest_path.open(encoding="utf-8") as f:
        mdata = json.load(f)

    dry_run = mdata.get("dry_run", False)
    file_stats = mdata.get("file_stats", [])
    manifest_expected = calculate_total_expected(file_stats)
    valid_keys = get_file_stats_keys(file_stats)

    # Find inference JSONL files for this run_id
    prefix = f"{run_id}_"
    jsonl_paths = sorted(pilot_dir.glob(f"{run_id}_*.jsonl"))

    # Validate file keys
    unknown_keys: list[str] = []
    for p in jsonl_paths:
        stem = p.stem
        if not stem.startswith(prefix):
            logger.warning("Skipping file not matching run_id prefix: %s", p.name)
            continue
        file_key = stem[len(prefix):]  # e.g. "clean", "tpwr_low", "vscr_medium"
        if file_key and file_key not in valid_keys:
            unknown_keys.append(file_key)

    if unknown_keys:
        raise ValueError(
            f"Unknown file keys in inference output: {unknown_keys}. "
            f"Valid keys from manifest: {valid_keys}"
        )

    # Load all records
    jsonl_count = len(jsonl_paths)
    clean_count = 0
    corrupted_count = 0

    for p in jsonl_paths:
        stem = p.stem
        if not stem.startswith(prefix):
            continue
        file_key = stem[len(prefix):]

        rows = 0
        with p.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows += 1

        if file_key == "clean":
            clean_count += rows
        else:
            corrupted_count += rows

    total_records = clean_count + corrupted_count

    # Check reconciliation
    if total_records == manifest_expected:
        status = "OK"
        detail = ""
    else:
        status = "MISMATCH"
        detail = (
            f"loaded={total_records}, expected={manifest_expected}, "
            f"diff={total_records - manifest_expected}"
        )

    return RunValidation(
        run_id=run_id,
        dry_run=dry_run,
        inference_record_count=total_records,
        clean_record_count=clean_count,
        corrupted_record_count=corrupted_count,
        manifest_expected=manifest_expected,
        jsonl_files=jsonl_count,
        reconciliation_status=status,
        mismatch_detail=detail,
    )


def list_and_select_run(
    manifests_dir: Path, pilot_dir: Path
) -> tuple[RunValidation | None, Path | None]:
    """List available runs, filter out dry-runs, require explicit selection.

    Returns (validation, manifest_path) if a run can be selected, else (None, None).
    """
    candidates = discover_runs(manifests_dir)

    if not candidates:
        logger.error("No pilot run manifests found in %s", manifests_dir)
        return None, None

    # Separate dry-run and real runs
    real_runs = [c for c in candidates if not c.dry_run]
    dry_runs = [c for c in candidates if c.dry_run]

    # Print available runs
    print("\n" + "=" * 60)
    print("Available Pilot Runs")
    print("=" * 60)

    if real_runs:
        print("\n[Real Runs] - dry_run=false:")
        for i, c in enumerate(real_runs, 1):
            marker = " *" if c.total_expected > 0 else ""
            created = f" ({c.created_at})" if c.created_at else ""
            print(
                f"  {i}. {c.run_id}{marker}\n"
                f"     Model: {c.api_model} | Expected records: {c.total_expected} | "
                f"Files: {c.jsonl_count} | Keys: {sorted(c.file_stats_keys)}{created}"
            )
    else:
        print("\n[Real Runs] - None found")

    if dry_runs:
        print("\n[Dry Runs] - EXCLUDED (dry_run=true):")
        for c in dry_runs:
            print(f"  - {c.run_id} (expected={c.total_expected})")

    print("\n" + "-" * 60)

    if not real_runs:
        print("\nERROR: No valid (dry_run=false) runs found.")
        return None, None

    if len(real_runs) == 1:
        # Auto-select the only real run
        c = real_runs[0]
        print(f"\nAuto-selected only real run: {c.run_id}")
        try:
            validation = validate_run(c.run_id, c.manifest_path, pilot_dir)
            return validation, c.manifest_path
        except ValueError as e:
            print(f"\nERROR: Validation failed for {c.run_id}: {e}")
            return None, None

    # Multiple real runs - require explicit selection
    print("\nMultiple real runs found. Please specify --run-id explicitly:")
    print("  python Scripts/compute_metrics.py --run-id <RUN_ID>")
    print("\nValid run IDs:")
    for c in real_runs:
        print(f"  {c.run_id}")
    return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute RUChi-Bench metrics from inference results"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help=(
            "Specific run_id to process. Required when multiple real runs exist. "
            "Example: --run-id 20260801_120000_123456"
        ),
    )
    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=ROOT / "Results" / "manifests",
        help="Directory containing inference run manifest JSON files",
    )
    parser.add_argument(
        "--clean",
        "--pilot-clean",
        dest="pilot_clean",
        type=Path,
        default=ROOT / "Data" / "full_test" / "standardized",
        help="Clean JSONL file or directory (the released default is full_test/standardized)",
    )
    parser.add_argument(
        "--perturbed",
        "--pilot-corrupted",
        dest="pilot_corrupted",
        type=Path,
        default=ROOT / "Data" / "full_test" / "perturbed",
        help="Perturbed-data directory (the released default is full_test/perturbed)",
    )
    parser.add_argument(
        "--inference-dir",
        "--pilot-dir",
        dest="pilot_dir",
        type=Path,
        default=ROOT / "Results" / "inference",
        help="Directory containing inference JSONL files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "Results" / "metrics",
        help="Directory for output metrics files",
    )
    args = parser.parse_args()

    # Resolve run_id
    manifest_path: Path | None = None
    if args.run_id:
        run_id = args.run_id
        # Find manifest for this run_id - search by both filename and content
        manifest_paths = list(args.manifests_dir.glob(f"phase08_{run_id}_manifest.json"))
        if not manifest_paths:
            manifest_paths = list(args.manifests_dir.glob("phase08_*_manifest.json"))
            # Check if any manifest has this run_id in content
            for mp in manifest_paths:
                with mp.open(encoding="utf-8") as f:
                    mdata = json.load(f)
                if mdata.get("run_id") == run_id:
                    manifest_paths = [mp]
                    break

        if not manifest_paths:
            logger.error("No manifest found for run_id=%s in %s", run_id, args.manifests_dir)
            sys.exit(1)
        manifest_path = manifest_paths[0]

        # Validate run
        try:
            validation = validate_run(run_id, manifest_path, args.pilot_dir)
        except ValueError as e:
            logger.error("Validation failed: %s", e)
            sys.exit(1)

        # Check dry_run
        if validation.dry_run:
            logger.error(
                "ERROR: run_id=%s has dry_run=true. Cannot compute metrics for dry runs.\n"
                "Please use a real run (dry_run=false) or omit --run-id to see available runs.",
                run_id
            )
            sys.exit(1)

        logger.info("Selected run_id=%s (real run, dry_run=false)", run_id)
    else:
        # No run_id provided - list and select
        validation, manifest_path = list_and_select_run(args.manifests_dir, args.pilot_dir)
        if validation is None:
            sys.exit(1)
        run_id = validation.run_id

    # Print validation summary
    print("\n" + "=" * 60)
    print("Run Validation Summary")
    print("=" * 60)
    print(f"  Run ID:              {validation.run_id}")
    print(f"  Dry Run:             {validation.dry_run}")
    print(f"  JSONL Files:         {validation.jsonl_files}")
    print(f"  Inference Records:   {validation.inference_record_count}")
    print(f"  Clean Records:       {validation.clean_record_count}")
    print(f"  Corrupted Records:   {validation.corrupted_record_count}")
    print(f"  Manifest Expected:   {validation.manifest_expected}")
    print(f"  Reconciliation:      {validation.reconciliation_status}")
    if validation.mismatch_detail:
        print(f"  Detail:              {validation.mismatch_detail}")

    # Check reconciliation
    if validation.reconciliation_status == "MISMATCH":
        print("\n" + "!" * 60)
        print("ERROR: Inference record count mismatch!")
        print("!" * 60)
        print(f"  Loaded: {validation.inference_record_count}")
        print(f"  Expected (from manifest): {validation.manifest_expected}")
        print(f"  Difference: {validation.inference_record_count - validation.manifest_expected}")
        print("\nCannot generate metrics with inconsistent counts.")
        print("Please check the inference output and re-run if needed.")
        sys.exit(1)

    print("=" * 60 + "\n")

    # Load manifest for model name
    with manifest_path.open(encoding="utf-8") as f:
        mdata = json.load(f)
    model_name = mdata.get("api_model", "unknown")
    logger.info("Processing run_id=%s, model=%s", run_id, model_name)

    # Load benchmark samples
    logger.info("Loading benchmark samples...")
    samples, sample_counts = load_pilot_samples(args.pilot_clean, args.pilot_corrupted)
    logger.info(
        "Sample breakdown: clean=%d, corrupted=%d, total=%d",
        sample_counts.clean, sample_counts.corrupted, sample_counts.total
    )

    # Load inference records
    jsonl_paths = sorted(args.pilot_dir.glob(f"{run_id}_*.jsonl"))
    try:
        records = load_inference_records(jsonl_paths, manifest_path=manifest_path)
    except ValueError as exc:
        print("\n" + "!" * 60)
        print("METRICS IDENTITY GATE FAILED")
        print("!" * 60)
        print(str(exc))
        print("\nNo metrics were computed or written.")
        sys.exit(1)
    logger.info("Loaded %d inference records", len(records))

    # -----------------------------------------------------------------------
    # Validation Gate - verify all counts before generating metrics
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Metrics Input Validation Gate")
    print("=" * 60)

    gate_passed = True

    # Check that inference files reconcile with the selected benchmark data.
    if validation.clean_record_count != sample_counts.clean:
        print(
            f"  MISMATCH: clean_inference={validation.clean_record_count}, "
            f"expected={sample_counts.clean}"
        )
        gate_passed = False
    else:
        print(f"  OK: clean_samples={sample_counts.clean}")

    if validation.corrupted_record_count != sample_counts.corrupted:
        print(
            f"  MISMATCH: corrupted_inference={validation.corrupted_record_count}, "
            f"expected={sample_counts.corrupted}"
        )
        gate_passed = False
    else:
        print(f"  OK: corrupted_samples={sample_counts.corrupted}")

    if sample_counts.total <= 0:
        print("  MISMATCH: no benchmark samples were loaded")
        gate_passed = False
    else:
        print(f"  OK: total_samples={sample_counts.total}")

    if validation.inference_record_count != sample_counts.total:
        print(
            f"  MISMATCH: total_inference={validation.inference_record_count}, "
            f"expected={sample_counts.total}"
        )
        gate_passed = False
    else:
        print(f"  OK: total_inference={validation.inference_record_count}")

    if not gate_passed:
        print("\n" + "!" * 60)
        print("VALIDATION GATE FAILED")
        print("!" * 60)
        print("Cannot generate metrics. Please check benchmark data and re-run.")
        sys.exit(1)

    print("\n  VALIDATION GATE: PASS")
    print("=" * 60 + "\n")

    # Compute metrics
    result = compute_metrics(
        samples=samples,
        sample_counts=sample_counts,
        records=records,
        model_name=model_name,
        run_id=run_id,
        inference_clean=validation.clean_record_count,
        inference_corrupted=validation.corrupted_record_count,
    )

    # Write JSON with extended metadata
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{run_id}_metrics.json"

    result_dict = {
        # Identity
        "run_id": result.run_id,
        "model": result.model,
        "dry_run": validation.dry_run,
        # Count reconciliation
        "inference_record_count": validation.inference_record_count,
        "clean_record_count": validation.clean_record_count,
        "corrupted_record_count": validation.corrupted_record_count,
        "manifest_expected_count": validation.manifest_expected,
        "reconciliation_status": validation.reconciliation_status,
        # Sample counts
        "total_samples": result.total_samples,
        "clean_samples": result.clean_samples,
        "corrupted_samples": result.corrupted_samples,
        # Parse failure rates
        "overall_parse_failure_rate": round(result.overall_parse_failure_rate, 2),
        "clean_parse_failure_rate": round(result.clean_parse_failure_rate, 2),
        "corrupted_parse_failure_rate": round(result.corrupted_parse_failure_rate, 2),
        # Core metrics
        "overall_clean_accuracy": round(result.overall_clean_accuracy, 2),
        "corruption_crr": {k: round(v, 2) for k, v in result.corruption_crr.items()},
        "corruption_drop": {k: round(v, 2) for k, v in result.corruption_drop.items()},
        "corruption_success_rate": {
            k: round(v, 2) for k, v in result.corruption_success_rate.items()
        },
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    logger.info("Metrics JSON written to %s", output_path)

    # Print table
    table = format_metrics_table(result)
    print(table, flush=True)

    # Save markdown
    md_path = output_dir / f"{run_id}_metrics.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write(table)
    logger.info("Metrics table written to %s", md_path)

    print(f"\n{'=' * 60}")
    print(f"Metrics computed successfully for run_id={run_id}")
    print(f"  JSON:  {output_path}")
    print(f"  Markdown: {md_path}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
