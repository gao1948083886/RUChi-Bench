"""Audit the frozen full-test clean and corruption datasets."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
STANDARDIZED = ROOT / "Data" / "full_test" / "standardized"
CORRUPTED = ROOT / "Data" / "full_test" / "perturbed"
REPORT = ROOT / "Data" / "full_test" / "corruption_audit.json"

COUNTS = {"pawsx_zh": 1975, "xnli_zh": 5010, "lcqmc": 12500, "c3": 3892, "asap": 4940}
FILES = {
    "pawsx_zh": "pawsx_zh_clean.jsonl",
    "xnli_zh": "xnli_zh_clean.jsonl",
    "lcqmc": "lcqmc_clean.jsonl",
    "c3": "c3_clean.jsonl",
    "asap": "asap_clean.jsonl",
}
STRATEGIES = ("add_noise", "del", "homo", "red_char", "red_word", "swap", "vis")
LEVELS = ("low", "medium", "high")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def contains_unsafe_line_separator(value: Any) -> bool:
    if isinstance(value, str):
        return "\u2028" in value or "\u2029" in value
    if isinstance(value, dict):
        return any(contains_unsafe_line_separator(item) for item in value.values())
    if isinstance(value, list):
        return any(contains_unsafe_line_separator(item) for item in value)
    return False


def main() -> None:
    argparse.ArgumentParser(
        description="Audit the included full-test standardized and perturbed data."
    ).parse_args()
    errors: list[str] = []
    report: dict[str, Any] = {"datasets": {}, "total_clean": 0, "total_corrupted": 0}
    manifest = json.loads((CORRUPTED / "corruption_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "full_test_perturbed":
        errors.append("corruption manifest schema_version is not full_test_perturbed")

    for dataset, expected_count in COUNTS.items():
        clean_path = STANDARDIZED / FILES[dataset]
        clean_rows = read_jsonl(clean_path)
        clean_by_id = {str(row["sample_id"]): row for row in clean_rows}
        if len(clean_rows) != expected_count:
            errors.append(f"{dataset}: expected {expected_count} clean rows, found {len(clean_rows)}")
        if len(clean_by_id) != len(clean_rows):
            errors.append(f"{dataset}: duplicate clean sample_id values")
        if any(row.get("split") != "test" or row.get("language") != "zh" for row in clean_rows):
            errors.append(f"{dataset}: clean split/language is not uniformly test/zh")

        records_by_id: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        strategy_counts: Counter[str] = Counter()
        change_counts: dict[str, list[int]] = defaultdict(list)
        unsafe_records = 0
        for strategy in STRATEGIES:
            for level in LEVELS:
                path = CORRUPTED / dataset / strategy / f"{level}.jsonl"
                if not path.exists():
                    errors.append(f"{dataset}: missing {strategy}/{level}.jsonl")
                    continue
                rows = read_jsonl(path)
                if not rows:
                    errors.append(f"{dataset}: empty {strategy}/{level}.jsonl")
                for row in rows:
                    sample_id = str(row.get("sample_id"))
                    if sample_id not in clean_by_id:
                        errors.append(f"{dataset}: unknown sample_id {sample_id}")
                        continue
                    if level in records_by_id[sample_id]:
                        errors.append(f"{dataset}: duplicate {sample_id} at level {level}")
                    records_by_id[sample_id][level] = row
                    if row.get("corruption_name") != strategy or row.get("corruption_level") != level:
                        errors.append(f"{dataset}/{sample_id}: path metadata mismatch")
                    clean = clean_by_id[sample_id]
                    for key, value in clean.items():
                        if row.get(key) != value:
                            errors.append(f"{dataset}/{sample_id}: clean field changed: {key}")
                            break
                    if row.get("corrupted_payload") == clean.get("clean_payload"):
                        errors.append(f"{dataset}/{sample_id}/{level}: payload did not change")
                    if row.get("corruption_applied") is not True or int(row.get("change_count", 0)) <= 0:
                        errors.append(f"{dataset}/{sample_id}/{level}: invalid applied/change_count flag")
                    if contains_unsafe_line_separator(row):
                        unsafe_records += 1
                    change_counts[f"{strategy}/{level}"].append(int(row["change_count"]))

        for sample_id, levels in records_by_id.items():
            if set(levels) != set(LEVELS):
                errors.append(f"{dataset}/{sample_id}: not exactly three corruption levels")
            names = {row.get("corruption_name") for row in levels.values()}
            if len(names) != 1:
                errors.append(f"{dataset}/{sample_id}: strategy differs across levels")
            else:
                strategy_counts.update(names)
        if len(records_by_id) != expected_count:
            errors.append(f"{dataset}: expected {expected_count} corrupted sample IDs, found {len(records_by_id)}")
        if sum(strategy_counts.values()) != expected_count:
            errors.append(f"{dataset}: strategy assignment count is inconsistent")
        if set(strategy_counts) != set(STRATEGIES):
            errors.append(f"{dataset}: not all seven strategies are assigned")
        if unsafe_records:
            errors.append(f"{dataset}: {unsafe_records} records contain U+2028/U+2029")

        summaries = {}
        for key, values in sorted(change_counts.items()):
            summaries[key] = {
                "n": len(values),
                "mean": round(statistics.mean(values), 4) if values else None,
                "median": statistics.median(values) if values else None,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
            }
        monotonic = {}
        for strategy in STRATEGIES:
            means = [summaries[f"{strategy}/{level}"]["mean"] for level in LEVELS]
            monotonic[strategy] = {
                "means": means,
                "low_le_medium": means[0] <= means[1],
                "medium_le_high": means[1] <= means[2],
            }
        report["datasets"][dataset] = {
            "clean_count": len(clean_rows),
            "corrupted_count": sum(len(levels) for levels in records_by_id.values()),
            "assignment_counts": dict(sorted(strategy_counts.items())),
            "change_count_summary": summaries,
            "monotonic_level_summary": monotonic,
            "unsafe_line_separator_records": unsafe_records,
        }
        report["total_clean"] += len(clean_rows)
        report["total_corrupted"] += sum(len(levels) for levels in records_by_id.values())

    report["expected_total_clean"] = sum(COUNTS.values())
    report["expected_total_corrupted"] = 3 * report["expected_total_clean"]
    report["errors"] = errors
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(f"AUDIT_FAILED: {len(errors)} error(s)")
    print("AUDIT_OK")


if __name__ == "__main__":
    main()
