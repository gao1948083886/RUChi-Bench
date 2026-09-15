"""Create the frozen common schema for all five official Chinese test sets."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Data" / "raw"
OUT = ROOT / "Data" / "full_test" / "standardized"
SCHEMA_VERSION = "full_test_standardized_v1"


def _write(path: Path, rows: list[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return len(rows)


def _base(
    *, sample_id: Any, dataset_name: str, dataset_version: str, task_type: str,
    payload_kind: str, clean_payload: dict[str, Any], source_gold_label: Any,
    gold_label: Any, gold_label_text: str, target_fields: list[str],
    source_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample = str(sample_id).strip()
    if not sample:
        raise ValueError(f"blank sample_id in {dataset_name}")
    row: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample,
        "source_sample_id": sample,
        "dataset_name": dataset_name,
        "dataset_version": dataset_version,
        "split": "test",
        "language": "zh",
        "task_type": task_type,
        "payload_kind": payload_kind,
        "clean_payload": clean_payload,
        "source_gold_label": source_gold_label,
        "gold_label": gold_label,
        "gold_label_text": gold_label_text,
        "target_fields": target_fields,
    }
    if source_metadata:
        row["source_metadata"] = source_metadata
    return row


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is blank")
    return value.strip()


def _pair_parquet(path: Path, dataset: str, version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(pq.read_table(path).to_pylist()):
        label = int(source["label"] if "label" in source else source["score"])
        text_a, text_b = _text(source["sentence1"], "sentence1"), _text(source["sentence2"], "sentence2")
        labels = {
            "pawsx_zh": {0: "different_meaning", 1: "paraphrase"},
            "lcqmc": {0: "not_matched", 1: "matched_intent"},
        }[dataset]
        sample_id = source.get("id", f"lcqmc-test-{index:05d}")
        rows.append(_base(
            sample_id=sample_id, dataset_name=dataset, dataset_version=version,
            task_type="pair_paraphrase" if dataset == "pawsx_zh" else "question_matching",
            payload_kind="pair", clean_payload={"text_a": text_a, "text_b": text_b},
            source_gold_label=label, gold_label=label, gold_label_text=labels[label],
            target_fields=["text_a", "text_b"],
        ))
    return rows


def _pawsx() -> list[dict[str, Any]]:
    raw = pq.read_table(RAW / "pawsx_zh" / "hf_mirror_zh" / "test.parquet").to_pylist()
    # The downloaded mirror contains 25 literal NS placeholders rather than
    # Chinese test sentences; they are excluded from the valid benchmark set.
    raw = [r for r in raw if "NS" not in str(r["sentence1"]) and "NS" not in str(r["sentence2"])]
    return _pair_parquet_from_rows(raw, "pawsx_zh", "hf_mirror_zh_test")


def _pair_parquet_from_rows(raw: list[dict[str, Any]], dataset: str, version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, source in enumerate(raw):
        label = int(source["label"] if "label" in source else source["score"])
        labels = {"pawsx_zh": {0: "different_meaning", 1: "paraphrase"}, "lcqmc": {0: "not_matched", 1: "matched_intent"}}[dataset]
        sample_id = source.get("id", f"lcqmc-test-{index:05d}")
        rows.append(_base(
            sample_id=sample_id, dataset_name=dataset, dataset_version=version,
            task_type="pair_paraphrase" if dataset == "pawsx_zh" else "question_matching",
            payload_kind="pair", clean_payload={"text_a": _text(source["sentence1"], "sentence1"), "text_b": _text(source["sentence2"], "sentence2")},
            source_gold_label=label, gold_label=label, gold_label_text=labels[label], target_fields=["text_a", "text_b"],
        ))
    return rows


def _xnli() -> list[dict[str, Any]]:
    path = RAW / "xnli_zh" / "XNLI-1.0" / "xnli.test.tsv"
    with path.open(encoding="utf-8", newline="") as handle:
        raw = [r for r in csv.DictReader(handle, delimiter="\t") if r.get("language") == "zh"]
    allowed = {"entailment", "neutral", "contradiction"}
    rows = []
    for source in raw:
        label = _text(source.get("gold_label"), "gold_label")
        if label not in allowed:
            raise ValueError(f"invalid XNLI label {label!r}")
        rows.append(_base(
            sample_id=source.get("pairID"), dataset_name="xnli_zh", dataset_version="XNLI-1.0",
            task_type="nli", payload_kind="pair",
            clean_payload={"text_a": _text(source.get("sentence1"), "sentence1"), "text_b": _text(source.get("sentence2"), "sentence2")},
            source_gold_label=label, gold_label=label, gold_label_text=label,
            target_fields=["text_a", "text_b"], source_metadata={"genre": source.get("genre")},
        ))
    return rows


def _c3() -> list[dict[str, Any]]:
    rows = []
    for subset in ("d", "m"):
        documents = json.loads((RAW / "c3" / "official_data" / f"c3-{subset}-test.json").read_text(encoding="utf-8"))
        for context, questions, source_document_id in documents:
            for question_index, question in enumerate(questions):
                choices = question.get("choice")
                answer = question.get("answer")
                if not isinstance(context, list) or not context or not all(isinstance(x, str) and x.strip() for x in context):
                    raise ValueError("invalid C3 context")
                if not isinstance(choices, list) or len(choices) < 2 or not all(isinstance(x, str) and x.strip() for x in choices):
                    raise ValueError("invalid C3 choices")
                choices = [x.strip() for x in choices]
                answer = _text(answer, "answer")
                if answer not in choices:
                    raise ValueError("C3 answer missing from choices")
                rows.append(_base(
                    sample_id=f"c3-{subset}-test-{source_document_id}-{question_index}", dataset_name="c3", dataset_version="official",
                    task_type="multiple_choice_mrc", payload_kind="mrc",
                    clean_payload={"context": "\n".join(x.strip() for x in context), "question": _text(question.get("question"), "question"), "options": choices},
                    source_gold_label=answer, gold_label=choices.index(answer), gold_label_text=answer,
                    target_fields=["context"], source_metadata={"subset": subset, "source_document_id": source_document_id},
                ))
    return rows


def _lcqmc() -> list[dict[str, Any]]:
    return _pair_parquet(RAW / "lcqmc" / "mirror_c-mteb" / "test.parquet", "lcqmc", "c-mteb_test")


def _asap() -> list[dict[str, Any]]:
    path = RAW / "asap" / "official_data" / "test.csv"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows = []
    for source in raw:
        try: star = int(float(source["star"]))
        except (KeyError, TypeError, ValueError) as exc: raise ValueError("invalid ASAP star") from exc
        if star not in range(1, 6): raise ValueError(f"invalid ASAP star {star}")
        rows.append(_base(
            sample_id=source.get("id"), dataset_name="asap", dataset_version="official",
            task_type="sentiment_rating", payload_kind="single_text", clean_payload={"text_a": _text(source.get("review"), "review")},
            source_gold_label=star, gold_label=star, gold_label_text=str(star), target_fields=["text_a"],
            source_metadata={"label_field": "star"},
        ))
    return rows


def main() -> None:
    datasets = {"pawsx_zh": _pawsx(), "xnli_zh": _xnli(), "lcqmc": _lcqmc(), "c3": _c3(), "asap": _asap()}
    manifest = {"schema_version": SCHEMA_VERSION, "split": "test", "language": "zh", "datasets": {}, "total": 0}
    for dataset, rows in datasets.items():
        ids = [str(row["sample_id"]) for row in rows]
        if len(ids) != len(set(ids)): raise ValueError(f"duplicate sample_id in {dataset}")
        count = _write(OUT / f"{dataset}_clean.jsonl", rows)
        manifest["datasets"][dataset] = {"count": count, "label_counts": dict(Counter(str(row["gold_label"]) for row in rows))}
        manifest["total"] += count
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    argparse.ArgumentParser(
        description="Standardize locally acquired official test files."
    ).parse_args()
    main()
