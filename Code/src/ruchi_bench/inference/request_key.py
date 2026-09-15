"""Three-layer identity system for inference requests.

Design (frozen):
1. ``sample_id`` — preserved as-is from the original dataset publisher.
   Must NOT be modified to work around cross-dataset collisions.
2. ``sample_uid`` — dataset + split + sample_id, used for intra-project uniqueness.
   Derived deterministically from Sample fields.
3. ``request_key`` — canonical JSON + SHA-256 of the full request fingerprint.
   Includes sample_uid + condition + corruption + model + params.
   Used for resume, deduplication, and JSONL uniqueness.

Key invariants:
- request_key is the ONLY key used for resume/dedup/JSONL uniqueness.
- sample_id is NEVER used alone for any of the above purposes.
- A request_key change means a genuinely different model call.
- An identical request_key on a different run = skip (not re-execute).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "RequestKeyComponents",
    "make_request_key",
    "sample_uid",
    "read_request_keys_from_jsonl",
    "REQUEST_KEY_SCHEMA_VERSION",
]

# Bump when the key structure changes in a non-backward-compatible way.
REQUEST_KEY_SCHEMA_VERSION = 1


def sample_uid(sample_id: str, dataset_name: str, split: str) -> str:
    """Dataset-scoped sample identifier.

    Combines the original publisher sample_id with dataset + split so that
    ``pawsx_zh:test`` and ``xnli_zh:test`` samples with the same bare sample_id
    value (e.g. "0") do not collide.
    """
    return f"{dataset_name}___{split}___{sample_id}"


@dataclass(frozen=True)
class RequestKeyComponents:
    """Canonical components of an inference request key.

    All fields are required. ``None`` fields are treated as absent/missing,
    not as the string "None".
    """

    sample_uid: str
    is_clean: bool
    corruption_name: str | None = None
    corruption_level: str | None = None
    corruption_seed: int | None = None
    target_field: str | None = None
    model_uid: str = ""
    prompt_template_version: int | str = 1
    temperature: float = 0.0
    max_new_tokens: int = 8
    # Hash of the actual model input text (canonical JSON of messages).
    # Changing this always produces a new request_key.
    messages_hash: str = ""

    # Version field — MUST be last, simplifies forward-compat parsing
    schema_version: int = field(default=REQUEST_KEY_SCHEMA_VERSION, repr=False)

    def to_canonical_dict(self) -> dict[str, Any]:
        """Serialize to a deterministic dict for hashing.

        ``None`` values are omitted so that optional fields being absent
        produces the same dict regardless of whether they were set to None
        or simply absent.
        """
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "sample_uid": self.sample_uid,
            "is_clean": self.is_clean,
            "model_uid": self.model_uid,
            "prompt_template_version": self.prompt_template_version,
            "temperature": self.temperature,
            "max_new_tokens": self.max_new_tokens,
            "messages_hash": self.messages_hash,
        }
        if self.corruption_name is not None:
            d["corruption_name"] = self.corruption_name
        if self.corruption_level is not None:
            d["corruption_level"] = self.corruption_level
        if self.corruption_seed is not None:
            d["corruption_seed"] = self.corruption_seed
        if self.target_field is not None:
            d["target_field"] = self.target_field
        return d

    def to_json_bytes(self) -> bytes:
        """Stable canonical JSON bytes (sorted keys, no extra whitespace)."""
        return json.dumps(
            self.to_canonical_dict(),
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")


def make_request_key(components: RequestKeyComponents) -> str:
    """Generate a SHA-256 request key from components.

    The key is the full 64-char hex digest.  It is safe to store in file names
    and as a JSON dict key.
    """
    return hashlib.sha256(components.to_json_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# JSONL reader helpers
# ---------------------------------------------------------------------------

_LEGACY_SCHEMA_VERSION = 0  # Old records written without schema_version field


def read_request_keys_from_jsonl(path: Path) -> tuple[set[str], list[str]]:
    """Read an inference JSONL and extract all request_keys.

    Parameters
    ----------
    path :
        Path to a JSONL file written by InferenceLogger.

    Returns
    -------
    (request_keys, legacy_sample_id_only_keys)
        ``request_keys`` — set of request_key strings found.
        ``legacy_sample_id_only_keys`` — sample_id values that appeared in
        records without a ``request_key`` field. These indicate a pre-fix
        run and are returned separately so callers can decide whether to
        treat them as incompatible or attempt partial resume.
    """
    if not path.exists():
        return set(), []

    request_keys: set[str] = set()
    legacy_sample_ids: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for _line_no, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            # New format: has request_key field
            rk = record.get("request_key")
            if rk is not None:
                request_keys.add(str(rk))
                continue

            # Legacy format: no request_key, fall back to sample_id only
            # These records should NOT be used for resume in new runs
            sid = record.get("sample_id")
            if sid is not None:
                legacy_sample_ids.append(str(sid))

    return request_keys, legacy_sample_ids
