"""Generate full-test corruption files in parallel by dataset."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Code" / "src"))

from ruchi_bench.data.run_standardized_corruptions import (
    run_standardized_corruptions,
)

STANDARDIZED = ROOT / "Data" / "full_test" / "standardized"
OUTPUT = ROOT / "Data" / "full_test" / "perturbed"
COUNTS = {"pawsx_zh": 1975, "xnli_zh": 5010, "lcqmc": 12500, "c3": 3892, "asap": 4940}
DATASETS = tuple(COUNTS)


def _one(dataset: str) -> dict:
    return run_standardized_corruptions(
        STANDARDIZED,
        OUTPUT,
        seed=42,
        schema_version="full_test_perturbed",
        expected_counts=COUNTS,
        strict_preflight=False,
        dataset_names=(dataset,),
        write_manifest=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate the released full-test perturbation files."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=len(DATASETS),
        help="Number of dataset workers (default: one worker per dataset)",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, dict] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(DATASETS))) as pool:
        futures = {pool.submit(_one, dataset): dataset for dataset in DATASETS}
        for future in as_completed(futures):
            dataset = futures[future]
            manifests.update(future.result()["datasets"])
            print("DATASET_DONE", dataset, manifests[dataset]["clean_count"], flush=True)
    manifest = {
        "schema_version": "full_test_perturbed",
        "started_at": datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "seed": 42,
        "levels": ["low", "medium", "high"],
        "corruption_policy": {
            "all_levels_from_clean": True,
            "labels_unchanged": True,
            "one_strategy_per_sample": True,
            "same_strategy_across_levels": True,
        },
        "datasets": dict(sorted(manifests.items())),
    }
    (OUTPUT / "corruption_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=dict), encoding="utf-8"
    )
    print("FULL_CORRUPTION_DONE", json.dumps({k: v["assignment_counts"] for k, v in manifests.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
