"""Deterministic identity recovery for Phase 08 inference records.

Uses request_key exact reconstruction to recover dataset_name and other metadata
from inference records that lack explicit dataset_name fields.

Design (frozen):
- Each inference record's request_key encodes sample_uid + condition + config.
- By reconstructing request_keys from known pilot samples and matching against
  observed inference records, we can deterministically recover identity without
  guessing dataset from sample_id or file position.
- CandidateIdentity objects are stored as a list (not dict) to preserve all variants.
- Composite key (sample_uid, corruption_name, corruption_level, corruption_seed)
  is used for matching, not just sample_uid.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RecoveryConfig",
    "CandidateIdentity",
    "RecoveredIdentity",
    "RecoveryAudit",
    "load_run_config",
    "compute_observed_messages_hash",
    "build_clean_candidates",
    "build_corrupted_candidates",
    "reconstruct_candidate_request_key",
    "recover_run_identities",
    "validate_recovery_gate",
]

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True)
class RecoveryConfig:
    """Frozen configuration for identity recovery of one run."""

    run_id: str
    pilot_clean_path: Path
    pilot_corrupted_dir: Path
    inference_dir: Path
    model_uid: str
    temperature: float
    max_new_tokens: int
    prompt_template_version: int


def load_run_config(
    run_id: str,
    pilot_clean_path: Path,
    pilot_corrupted_dir: Path,
    inference_dir: Path,
    manifest_path: Path | None = None,
) -> RecoveryConfig:
    """Load run configuration from manifest or use defaults.

    Parameters
    ----------
    run_id :
        The run identifier (e.g., "20260730_105603_473727").
    pilot_clean_path :
        Path to clean pilot JSONL.
    pilot_corrupted_dir :
        Directory containing corrupted pilot JSONL subdirectories.
    inference_dir :
        Directory containing inference JSONL files.
    manifest_path :
        Optional path to run manifest. If provided, extracts config from it.
    """
    model_uid = "qwen3.5"
    temperature = 0.0
    max_new_tokens = 8
    prompt_template_version = 1

    if manifest_path and manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as f:
            mdata = json.load(f)

        inference_params = mdata.get("inference_params", {})
        model_uid = mdata.get("api_model", model_uid)
        temperature = inference_params.get("temperature", temperature)
        max_new_tokens = inference_params.get("max_new_tokens", max_new_tokens)

    return RecoveryConfig(
        run_id=run_id,
        pilot_clean_path=pilot_clean_path,
        pilot_corrupted_dir=pilot_corrupted_dir,
        inference_dir=inference_dir,
        model_uid=model_uid,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        prompt_template_version=prompt_template_version,
    )


# =============================================================================
# Identity structures
# =============================================================================


@dataclass(frozen=True)
class CandidateIdentity:
    """A possible identity for an inference record, derived from pilot data.

    This represents one unique variant. A single sample may have multiple variants
    due to different corruption settings. All variants are preserved as separate
    CandidateIdentity objects.
    """

    dataset_name: str
    split: str
    sample_id: str
    sample_uid: str
    is_clean: bool
    corruption_name: str | None  # None for clean
    corruption_level: str | None  # None for clean
    corruption_seed: int | None  # None for clean
    source_file: str
    pilot_line_number: int

    @property
    def composite_key(self) -> tuple[str, str | None, str | None, int | None]:
        """Unique composite key for this variant.

        Used to distinguish between different variants of the same sample.
        For clean: (sample_uid, None, None, None)
        For corrupted: (sample_uid, corruption_name, corruption_level, corruption_seed)
        """
        return (
            self.sample_uid,
            self.corruption_name,
            self.corruption_level,
            self.corruption_seed,
        )


@dataclass
class RecoveredIdentity:
    """The result of recovering an inference record's identity."""

    inference_file: str
    inference_line_number: int
    request_key: str
    sample_id: str
    candidate: CandidateIdentity
    recovery_status: str  # "exact_match", "zero_matches", "multiple_matches"
    matched_request_keys: list[str] = field(default_factory=list)


@dataclass
class RecoveryAudit:
    """Audit results for identity recovery of a run.

    This dataclass uses precise field names to distinguish between different
    concepts that were previously conflated:
    - observed duplicates: records in JSONL sharing the same request_key
    - selected duplicates: selected candidates sharing the same request_key
    - candidate attempts: how many times we tried to match each candidate
    """

    run_id: str

    # Record counts
    observed_records: int = 0

    # Candidate counts
    candidate_clean: int = 0
    candidate_corrupted: int = 0
    candidate_total: int = 0
    candidate_attempt_count: int = 0  # Total match attempts across all candidates

    # Match results
    exact_matches: int = 0
    zero_matches: int = 0
    multiple_matches: int = 0

    # Observed request_key statistics (directly from JSONL)
    observed_unique_request_keys: int = 0
    duplicate_observed_key_groups: int = 0  # Number of request_keys appearing >1 time
    duplicate_observed_extra_records: int = 0  # Extra records beyond first per key

    # Selected candidate statistics (after matching)
    selected_unique_request_keys: int = 0
    duplicate_selected_key_groups: int = 0  # Number of selected keys appearing >1 time
    duplicate_selected_extra_records: int = 0  # Extra records beyond first per selected key

    # Missing/corrupted data
    missing_messages: int = 0
    messages_hash_failure: int = 0

    # Unmatched counts
    unmatched_observed: int = 0
    unmatched_candidates: int = 0

    # Recovery counts
    clean_recovered: int = 0
    corrupted_recovered: int = 0

    # Validation
    sample_uid_mismatch: int = 0
    file_bucket_mismatch: int = 0

    # Per-dataset/severity breakdowns
    per_dataset_clean: dict[str, int] = field(default_factory=dict)
    per_corruption_severity: dict[str, int] = field(default_factory=dict)

    # Gate status
    gate_passed: bool = False
    gate_failures: list[str] = field(default_factory=list)


# =============================================================================
# Hash computation
# =============================================================================


def _messages_hash(messages: list[dict[str, object]]) -> str:
    """Stable SHA-256 of the canonical JSON of a messages list.

    This matches the _messages_hash() function in pipeline.py exactly.
    """
    canonical = json.dumps(
        messages, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_observed_messages_hash(record: dict[str, Any]) -> str | None:
    """Compute messages_hash from an inference record.

    Returns None if the record lacks messages.
    """
    messages = record.get("messages")
    if messages is None:
        return None
    return _messages_hash(messages)


# =============================================================================
# sample_uid generation
# =============================================================================


def _make_sample_uid(dataset_name: str, split: str, sample_id: str) -> str:
    """Generate sample_uid matching the project's sample_uid() function.

    Format: {dataset_name}___{split}___{sample_id}
    """
    return f"{dataset_name}___{split}___{sample_id}"


# =============================================================================
# RequestKey reconstruction
# =============================================================================


def reconstruct_candidate_request_key(
    candidate: CandidateIdentity,
    config: RecoveryConfig,
    messages_hash: str,
) -> str:
    """Reconstruct the request_key that would be generated for a candidate.

    This matches RequestKeyComponents.to_canonical_dict() and make_request_key()
    from request_key.py exactly.
    """
    # Build canonical dict matching RequestKeyComponents
    d = {
        "schema_version": 1,
        "sample_uid": candidate.sample_uid,
        "is_clean": candidate.is_clean,
        "model_uid": config.model_uid,
        "prompt_template_version": config.prompt_template_version,
        "temperature": config.temperature,
        "max_new_tokens": config.max_new_tokens,
        "messages_hash": messages_hash,
    }

    # Add optional fields only if not None
    if candidate.corruption_name is not None:
        d["corruption_name"] = candidate.corruption_name
    if candidate.corruption_level is not None:
        d["corruption_level"] = candidate.corruption_level
    if candidate.corruption_seed is not None:
        d["corruption_seed"] = candidate.corruption_seed

    # Canonical JSON (sorted keys) -> SHA-256
    canonical = json.dumps(d, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# =============================================================================
# Candidate generation
# =============================================================================


def build_clean_candidates(pilot_clean_path: Path) -> list[CandidateIdentity]:
    """Build list of all clean pilot samples.

    Returns list of CandidateIdentity objects (one per sample).
    Each sample appears exactly once.
    """
    candidates: list[CandidateIdentity] = []

    if not pilot_clean_path.exists():
        logger.warning("Clean pilot file not found: %s", pilot_clean_path)
        return candidates

    with pilot_clean_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            rec = json.loads(line)

            dataset_name = rec.get("dataset_name")
            sample_id = str(rec.get("sample_id", ""))
            split = rec.get("split", "test")

            if not dataset_name or not sample_id:
                continue

            sample_uid = _make_sample_uid(dataset_name, split, sample_id)

            candidate = CandidateIdentity(
                dataset_name=dataset_name,
                split=split,
                sample_id=sample_id,
                sample_uid=sample_uid,
                is_clean=True,
                corruption_name=None,
                corruption_level=None,
                corruption_seed=None,
                source_file=str(pilot_clean_path),
                pilot_line_number=line_no,
            )
            candidates.append(candidate)

    logger.info("Built %d clean candidates from %s", len(candidates), pilot_clean_path)
    return candidates


def build_corrupted_candidates(
    pilot_corrupted_dir: Path,
) -> list[CandidateIdentity]:
    """Build list of all APPLIED corrupted pilot samples.

    Returns list of CandidateIdentity objects (one per variant).
    Multiple variants of the same sample are ALL preserved - no overwriting.

    Example: If sample_uid=X has 15 corruption variants (5 corruptions × 3 levels),
    all 15 appear as separate CandidateIdentity objects.
    """
    candidates: list[CandidateIdentity] = []

    if not pilot_corrupted_dir.exists():
        logger.warning("Corrupted pilot dir not found: %s", pilot_corrupted_dir)
        return candidates

    for subdir in pilot_corrupted_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name == "corrupted":
            continue  # skip nested duplicated dir

        corruption_name = subdir.name

        for level_file in sorted(subdir.glob("*.jsonl")):
            level = level_file.stem  # "low", "medium", "high"

            with level_file.open(encoding="utf-8") as f:
                for line_no, line in enumerate(f, start=1):
                    if not line.strip():
                        continue
                    rec = json.loads(line)

                    # Only include APPLIED variants
                    corruption_applied = rec.get("corruption_applied", False)
                    if not corruption_applied:
                        continue

                    dataset_name = rec.get("dataset_name")
                    sample_id = str(rec.get("sample_id", ""))
                    split = rec.get("split", "test")
                    corruption_seed = rec.get("corruption_seed")

                    if not dataset_name or not sample_id:
                        continue

                    sample_uid = _make_sample_uid(dataset_name, split, sample_id)

                    candidate = CandidateIdentity(
                        dataset_name=dataset_name,
                        split=split,
                        sample_id=sample_id,
                        sample_uid=sample_uid,
                        is_clean=False,
                        corruption_name=corruption_name,
                        corruption_level=level,
                        corruption_seed=corruption_seed,
                        source_file=str(level_file),
                        pilot_line_number=line_no,
                    )
                    # APPEND to list - do NOT use sample_uid as key
                    candidates.append(candidate)

    logger.info(
        "Built %d corrupted APPLIED candidates from %s",
        len(candidates),
        pilot_corrupted_dir,
    )
    return candidates


# =============================================================================
# Inference record loading
# =============================================================================


def _parse_file_bucket(
    file_path: Path,
    run_id: str | None = None,
) -> tuple[str | None, str | None, bool]:
    """Parse corruption info from filename.

    Examples:
        "20260730_105603_473727_clean.jsonl" -> (None, None, True)
        "20260730_105603_473727_tpwr_low.jsonl" -> ("tpwr", "low", False)

    Returns (corruption_name, severity, is_clean).
    """
    stem = file_path.stem

    # Remove run_id prefix if provided
    key_part = stem
    if run_id:
        prefix = f"{run_id}_"
        if stem.startswith(prefix):
            key_part = stem[len(prefix):]

    if key_part == "clean":
        return None, None, True

    # Parse corruption_level from key
    # Supported: tpwr_low, vscr_high, cr_medium, wr_low, muni_high
    known_corruptions = ["tpwr", "vscr", "cr", "wr", "muni"]
    for corr in known_corruptions:
        if key_part.startswith(f"{corr}_"):
            level = key_part[len(f"{corr}_"):]
            return corr, level, False

    return None, None, False


# =============================================================================
# Recovery
# =============================================================================


def recover_run_identities(
    config: RecoveryConfig,
    clean_candidates: list[CandidateIdentity],
    corrupted_candidates: list[CandidateIdentity],
) -> tuple[list[RecoveredIdentity], RecoveryAudit]:
    """Recover identities for all inference records in a run.

    Parameters
    ----------
    config :
        Run configuration.
    clean_candidates :
        All clean pilot samples as a list.
    corrupted_candidates :
        All APPLIED corrupted pilot samples as a list.

    Returns
    -------
    (recovered_identities, audit)
    """
    audit = RecoveryAudit(run_id=config.run_id)

    # Candidate counts
    audit.candidate_clean = len(clean_candidates)
    audit.candidate_corrupted = len(corrupted_candidates)
    audit.candidate_total = audit.candidate_clean + audit.candidate_corrupted

    # Build composite index for fast lookup by (is_clean, corruption_name, corruption_level)
    # Index: (is_clean, corruption_name, corruption_level) -> list[CandidateIdentity]
    CompositeKey = tuple[bool, str | None, str | None]
    composite_index: dict[CompositeKey, list[CandidateIdentity]] = {}

    for cand in clean_candidates:
        key: CompositeKey = (True, None, None)
        if key not in composite_index:
            composite_index[key] = []
        composite_index[key].append(cand)

    for cand in corrupted_candidates:
        key = (False, cand.corruption_name, cand.corruption_level)
        if key not in composite_index:
            composite_index[key] = []
        composite_index[key].append(cand)

    # Find all inference files for this run
    inference_files = sorted(config.inference_dir.glob(f"{config.run_id}_*.jsonl"))
    logger.info("Found %d inference files for run %s", len(inference_files), config.run_id)

    recovered: list[RecoveredIdentity] = []

    # Track observed request_keys directly from JSONL
    observed_request_keys: list[str] = []

    # Track selected candidate request_keys (after matching)
    selected_request_keys: list[str] = []

    # Track matched candidates by composite key to detect multiple assignment
    matched_composite_keys: dict[
        tuple[str, str | None, str | None, int | None],
        tuple[Path, int],
    ] = {}  # composite_key -> (inference_file, line_no)

    for inference_file in inference_files:
        corr_name, corr_level, is_clean = _parse_file_bucket(inference_file, config.run_id)

        with inference_file.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue

                rec = json.loads(line)
                audit.observed_records += 1

                # Get observed data
                request_key = rec.get("request_key", "")
                sample_id = str(rec.get("sample_id", ""))

                if request_key:
                    observed_request_keys.append(request_key)

                messages = rec.get("messages")
                if messages is None:
                    # Missing messages - cannot recover
                    audit.zero_matches += 1
                    audit.missing_messages += 1
                    recovered.append(
                        RecoveredIdentity(
                            inference_file=str(inference_file),
                            inference_line_number=line_no,
                            request_key=request_key,
                            sample_id=sample_id,
                            candidate=CandidateIdentity(
                                dataset_name="",
                                split="",
                                sample_id=sample_id,
                                sample_uid="",
                                is_clean=False,
                                corruption_name=None,
                                corruption_level=None,
                                corruption_seed=None,
                                source_file="",
                                pilot_line_number=0,
                            ),
                            recovery_status="missing_messages",
                        )
                    )
                    continue

                # Compute messages_hash
                messages_hash = _messages_hash(messages)

                # Select candidate pool based on file bucket
                pool_key = (is_clean, corr_name, corr_level)
                candidate_pool = composite_index.get(pool_key, [])

                # Try to match each candidate's request_key against observed
                matched_candidates: list[CandidateIdentity] = []
                audit.candidate_attempt_count += len(candidate_pool)

                for candidate in candidate_pool:
                    candidate_rk = reconstruct_candidate_request_key(
                        candidate, config, messages_hash
                    )

                    if candidate_rk == request_key:
                        matched_candidates.append(candidate)

                # Classify result
                if len(matched_candidates) == 1:
                    audit.exact_matches += 1
                    candidate = matched_candidates[0]

                    # Track selected request_key
                    selected_request_keys.append(request_key)

                    # Check for multiple assignment of same candidate
                    composite_key = candidate.composite_key
                    if composite_key in matched_composite_keys:
                        # This candidate was already assigned to another record
                        prev_file, prev_line = matched_composite_keys[composite_key]
                        audit.gate_failures.append(
                            f"candidate multiple assignment: {composite_key} "
                            f"assigned to both {prev_file}:line {prev_line} "
                            f"and {inference_file}:line {line_no}"
                        )
                    matched_composite_keys[composite_key] = (inference_file, line_no)

                    if is_clean:
                        audit.clean_recovered += 1
                        audit.per_dataset_clean[candidate.dataset_name] = (
                            audit.per_dataset_clean.get(candidate.dataset_name, 0) + 1
                        )
                    else:
                        audit.corrupted_recovered += 1
                        sev_key = f"{corr_name}_{corr_level}"
                        audit.per_corruption_severity[sev_key] = (
                            audit.per_corruption_severity.get(sev_key, 0) + 1
                        )

                    recovered.append(
                        RecoveredIdentity(
                            inference_file=str(inference_file),
                            inference_line_number=line_no,
                            request_key=request_key,
                            sample_id=sample_id,
                            candidate=candidate,
                            recovery_status="exact_match",
                            matched_request_keys=[request_key],
                        )
                    )
                elif len(matched_candidates) == 0:
                    audit.zero_matches += 1
                    recovered.append(
                        RecoveredIdentity(
                            inference_file=str(inference_file),
                            inference_line_number=line_no,
                            request_key=request_key,
                            sample_id=sample_id,
                            candidate=CandidateIdentity(
                                dataset_name="",
                                split="",
                                sample_id=sample_id,
                                sample_uid="",
                                is_clean=False,
                                corruption_name=None,
                                corruption_level=None,
                                corruption_seed=None,
                                source_file="",
                                pilot_line_number=0,
                            ),
                            recovery_status="zero_matches",
                        )
                    )
                else:
                    audit.multiple_matches += 1
                    recovered.append(
                        RecoveredIdentity(
                            inference_file=str(inference_file),
                            inference_line_number=line_no,
                            request_key=request_key,
                            sample_id=sample_id,
                            candidate=CandidateIdentity(
                                dataset_name="AMBIGUOUS",
                                split="",
                                sample_id=sample_id,
                                sample_uid="AMBIGUOUS",
                                is_clean=False,
                                corruption_name=None,
                                corruption_level=None,
                                corruption_seed=None,
                                source_file="",
                                pilot_line_number=0,
                            ),
                            recovery_status="multiple_matches",
                            matched_request_keys=[
                                reconstruct_candidate_request_key(c, config, messages_hash)
                                for c in matched_candidates
                            ],
                        )
                    )

    # Compute observed request_key statistics
    observed_key_counter = Counter(observed_request_keys)
    audit.observed_unique_request_keys = len(observed_key_counter)
    audit.duplicate_observed_key_groups = sum(1 for v in observed_key_counter.values() if v > 1)
    audit.duplicate_observed_extra_records = sum(
        v - 1 for v in observed_key_counter.values() if v > 1
    )

    # Compute selected request_key statistics
    selected_key_counter = Counter(selected_request_keys)
    audit.selected_unique_request_keys = len(selected_key_counter)
    audit.duplicate_selected_key_groups = sum(1 for v in selected_key_counter.values() if v > 1)
    audit.duplicate_selected_extra_records = sum(
        v - 1 for v in selected_key_counter.values() if v > 1
    )

    # Compute unmatched
    audit.unmatched_observed = audit.observed_records - audit.exact_matches
    audit.unmatched_candidates = audit.candidate_total - audit.exact_matches

    # Run gate validation
    audit.gate_passed, gate_failures_from_validation = validate_recovery_gate(audit)
    audit.gate_failures.extend(gate_failures_from_validation)

    return recovered, audit


def validate_recovery_gate(audit: RecoveryAudit) -> tuple[bool, list[str]]:
    """Validate that recovery meets all requirements.

    Returns (gate_passed, list_of_failures).
    """
    failures: list[str] = []

    # Check observed records
    expected_total = 15722
    if audit.observed_records != expected_total:
        failures.append(
            f"observed_records={audit.observed_records}, expected={expected_total}"
        )

    # Check candidate counts
    if audit.candidate_clean != 1200:
        failures.append(f"candidate_clean={audit.candidate_clean}, expected=1200")

    if audit.candidate_corrupted != 14522:
        failures.append(f"candidate_corrupted={audit.candidate_corrupted}, expected=14522")

    # Check recovery quality
    if audit.exact_matches != expected_total:
        failures.append(
            f"exact_matches={audit.exact_matches}, expected={expected_total}"
        )

    if audit.zero_matches != 0:
        failures.append(f"zero_matches={audit.zero_matches}, expected=0")

    if audit.multiple_matches != 0:
        failures.append(f"multiple_matches={audit.multiple_matches}, expected=0")

    # Check observed duplicates
    if audit.duplicate_observed_key_groups != 0:
        failures.append(
            f"duplicate_observed_key_groups={audit.duplicate_observed_key_groups}, expected=0"
        )

    # Check selected duplicates
    if audit.duplicate_selected_key_groups != 0:
        failures.append(
            f"duplicate_selected_key_groups={audit.duplicate_selected_key_groups}, expected=0"
        )

    # Check per-dataset counts
    expected_per_dataset = {"pawsx_zh": 300, "xnli_zh": 300, "c3": 300, "asap": 300}
    for ds, expected_count in expected_per_dataset.items():
        actual = audit.per_dataset_clean.get(ds, 0)
        if actual != expected_count:
            failures.append(f"dataset {ds}: clean_count={actual}, expected={expected_count}")

    # Check per-corruption severity counts
    expected_per_corruption = {
        "tpwr_low": 30,
        "tpwr_medium": 30,
        "tpwr_high": 212,
        "vscr_low": 1150,
        "vscr_medium": 1150,
        "vscr_high": 1150,
        "cr_low": 1200,
        "cr_medium": 1200,
        "cr_high": 1200,
        "wr_low": 1200,
        "wr_medium": 1200,
        "wr_high": 1200,
        "muni_low": 1200,
        "muni_medium": 1200,
        "muni_high": 1200,
    }
    for key, expected_count in expected_per_corruption.items():
        actual = audit.per_corruption_severity.get(key, 0)
        if actual != expected_count:
            failures.append(
                f"corruption {key}: count={actual}, expected={expected_count}"
            )

    gate_passed = len(failures) == 0
    return gate_passed, failures


# =============================================================================
# Sidecar generation
# =============================================================================


def write_identity_sidecar(
    recovered: list[RecoveredIdentity],
    output_path: Path,
) -> None:
    """Write identity sidecar JSONL without modifying original records.

    Only writes non-sensitive metadata fields.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for rec in recovered:
            run_prefix = ""
            if rec.candidate.dataset_name and rec.candidate.sample_uid:
                parts = rec.candidate.sample_uid.split("___")
                if parts:
                    run_prefix = parts[0]
            sidecar = {
                "run_id": run_prefix,
                "inference_file": rec.inference_file,
                "inference_line_number": rec.inference_line_number,
                "request_key": rec.request_key,
                "sample_id": rec.sample_id,
                "sample_uid": rec.candidate.sample_uid,
                "dataset_name": rec.candidate.dataset_name,
                "split": rec.candidate.split,
                "condition": "clean" if rec.candidate.is_clean else "corrupted",
                "corruption": rec.candidate.corruption_name,
                "severity": rec.candidate.corruption_level,
                "corruption_seed": rec.candidate.corruption_seed,
                "source_pilot_file": rec.candidate.source_file,
                "recovery_method": "request_key_exact_reconstruction_v1",
                "recovery_status": rec.recovery_status,
            }
            f.write(json.dumps(sidecar, ensure_ascii=False) + "\n")
