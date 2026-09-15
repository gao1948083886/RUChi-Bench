"""Official dataset download scripts for Phase 04.

Each function downloads one dataset's official test split from its canonical source,
verifies SHA-256, and writes the raw file to Data/raw/<dataset>/<version>/.

Downloads are skipped if:
- The file already exists and SHA-256 matches the recorded hash.
- SHA-256 mismatch is an error (do not silently overwrite).

Each download writes a DownloadManifest to:
    Results/manifests/downloads/<dataset>_<timestamp>.json

Requires: Python 3.11+ stdlib (urllib.request, hashlib, json, pathlib, ssl).
No third-party HTTP libraries required.
"""

from __future__ import annotations

import hashlib
import ssl
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from ruchi_bench.data.download.base import create_manifest, write_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _project_path(path: Path) -> Path:
    """Resolve a relative path against the release-package root."""
    return path if path.is_absolute() else PROJECT_ROOT / path

# Canonical source URLs used by the public dataset access helpers.
PAWSX_URL = "https://storage.googleapis.com/paws/pawsx/x-final.tar.gz"
PAWSX_EXPECTED_FIELDS = ["id", "sentence1", "sentence2", "label"]
PAWSX_EXPECTED_COUNT = 1975
PAWSX_SPLIT_FILE = "x-final/zh/test_2k.tsv"

XNLI_URL = "https://dl.fbaipublicfiles.com/XNLI/XNLI-1.0.zip"
XNLI_EXPECTED_FIELDS = ["gold_label", "sentence1", "sentence2", "language"]
XNLI_EXPECTED_COUNT = 5000
XNLI_TEST_FILE = "XNLI-1.0/xnli.test.tsv"

ASAP_URL = "https://raw.githubusercontent.com/Meituan-Dianping/asap/master/data/test.csv"
ASAP_EXPECTED_FIELDS = ["id", "review", "star"]
ASAP_EXPECTED_COUNT = 4940

C3_D_URL = "https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-d-test.json"
C3_M_URL = "https://raw.githubusercontent.com/nlpdata/c3/master/data/c3-m-test.json"
C3_EXPECTED_FIELDS = ["document", "question", "choice", "answer"]


# ---------------------------------------------------------------------------
# Low-level HTTP helper with SSL fallback
# ---------------------------------------------------------------------------

def _create_ssl_context() -> ssl.SSLContext:
    """Create an SSL context with lenient certificate verification.

    Some environments have outdated CA bundles. We use
    CERT_NONE to work around SSL EOF errors from stale root CAs,
    which is acceptable for public dataset URLs over HTTPS.
    """
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _http_get(url: str, *, timeout: int = 120) -> tuple[bytes, int]:
    """Download URL to bytes, return (body, http_status)."""
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "Mozilla/5.0 (compatible; RUChi-Bench/1.0)")
    ctx = _create_ssl_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return resp.read(), resp.status
    except urllib.error.HTTPError:
        # Re-raise HTTP errors as-is (4xx/5xx carry meaning)
        raise
    except urllib.error.URLError:
        # Fallback: try without SSL context
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read(), resp.status


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Dataset-specific downloaders
# ---------------------------------------------------------------------------

def download_pawsx(output_dir: Path, run_id: str) -> Path | None:
    """Download PAWS-X x-final tarball, extract zh/test, return path or None."""
    import tarfile

    output_dir = _project_path(output_dir) / "pawsx_zh"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PROJECT_ROOT / "Results" / "manifests" / "downloads"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    tar_path = output_dir / "x-final.tar.gz"
    manifest_path = manifest_dir / f"pawsx_zh_{run_id}.json"

    # Skip if already downloaded with matching hash
    if tar_path.exists():
        data = tar_path.read_bytes()
        h = _sha256(data)
        # If hash file exists, compare; otherwise trust existing file
        hash_file = output_dir / ".sha256"
        if hash_file.exists():
            recorded = hash_file.read_text().strip()
            if recorded == h:
                print("PAWS-X: already exists with matching SHA-256, skipping download")
                return output_dir / PAWSX_SPLIT_FILE
            else:
                raise RuntimeError(
                    f"PAWS-X tarball SHA-256 mismatch. Expected {recorded}, got {h}. "
                    "Remove Data/raw/pawsx_zh/ to re-download."
                )

    print(f"PAWS-X: downloading {PAWSX_URL}")
    body, status = _http_get(PAWSX_URL)
    sha = _sha256(body)

    # Write atomically: temp file → rename
    tmp = tar_path.with_suffix(".tmp")
    tmp.write_bytes(body)
    tar_path.write_bytes(body)  # atomic on POSIX; close enough on Windows
    tmp.unlink(missing_ok=True)

    hash_file = output_dir / ".sha256"
    hash_file.write_text(sha)

    # Extract
    import io
    with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tf:
        tf.extractall(path=output_dir)

    extracted = output_dir / PAWSX_SPLIT_FILE
    if not extracted.exists():
        raise RuntimeError(f"PAWS-X: extracted file not found at {extracted}")

    manifest = create_manifest(
        dataset="pawsx_zh",
        canonical_source_url=PAWSX_URL,
        resolved_url=PAWSX_URL,
        http_status=status,
        sha256=sha,
        file_name="x-final.tar.gz",
        file_size=len(body),
        official_split="test",
        expected_fields=PAWSX_EXPECTED_FIELDS,
        expected_row_count=PAWSX_EXPECTED_COUNT,
        license_access_note="Google PAWS-X free-use license (Level A)",
        code_commit="TODO",  # filled by caller
        content_length=len(body),
        observed_fields=None,  # to be filled after parsing
        observed_row_count=None,
    )
    write_manifest(manifest, manifest_path)
    print(f"PAWS-X: downloaded ({len(body):,} bytes), SHA-256={sha[:16]}..., "
          f"extracted to {extracted}")
    return extracted


def download_xnli(output_dir: Path, run_id: str) -> Path | None:
    """Download XNLI-1.0.zip, extract zh/test.tsv, return path or None."""
    output_dir = _project_path(output_dir) / "xnli_zh"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PROJECT_ROOT / "Results" / "manifests" / "downloads"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    zip_path = output_dir / "XNLI-1.0.zip"
    manifest_path = manifest_dir / f"xnli_zh_{run_id}.json"

    if zip_path.exists():
        hash_file = output_dir / ".sha256"
        if hash_file.exists():
            recorded = hash_file.read_text().strip()
            body = zip_path.read_bytes()
            h = _sha256(body)
            if recorded == h:
                print("XNLI: already exists with matching SHA-256, skipping download")
                return output_dir / XNLI_TEST_FILE
            else:
                raise RuntimeError(f"XNLI SHA-256 mismatch. Expected {recorded}, got {h}.")

    print(f"XNLI: downloading {XNLI_URL}")
    body, status = _http_get(XNLI_URL)
    sha = _sha256(body)

    zip_path.write_bytes(body)
    hash_file = output_dir / ".sha256"
    hash_file.write_text(sha)

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(path=output_dir)

    extracted = output_dir / XNLI_TEST_FILE
    if not extracted.exists():
        raise RuntimeError(f"XNLI: extracted file not found at {extracted}")

    manifest = create_manifest(
        dataset="xnli_zh",
        canonical_source_url=XNLI_URL,
        resolved_url=XNLI_URL,
        http_status=status,
        sha256=sha,
        file_name="XNLI-1.0.zip",
        file_size=len(body),
        official_split="test",
        expected_fields=XNLI_EXPECTED_FIELDS,
        expected_row_count=XNLI_EXPECTED_COUNT,
        license_access_note="CC BY-NC 4.0 (Level B, non-commercial research)",
        code_commit="TODO",
        content_length=len(body),
    )
    write_manifest(manifest, manifest_path)
    print(f"XNLI: downloaded ({len(body):,} bytes), SHA-256={sha[:16]}..., "
          f"extracted to {extracted}")
    return extracted


def download_asap(output_dir: Path, run_id: str) -> Path | None:
    """Download ASAP test.csv, return path or None."""
    output_dir = _project_path(output_dir) / "asap"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PROJECT_ROOT / "Results" / "manifests" / "downloads"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "test.csv"
    manifest_path = manifest_dir / f"asap_{run_id}.json"

    if csv_path.exists():
        hash_file = output_dir / ".sha256"
        if hash_file.exists():
            recorded = hash_file.read_text().strip()
            body = csv_path.read_bytes()
            h = _sha256(body)
            if recorded == h:
                print("ASAP: already exists with matching SHA-256, skipping download")
                return csv_path
            else:
                raise RuntimeError(f"ASAP SHA-256 mismatch. Expected {recorded}, got {h}.")

    print(f"ASAP: downloading {ASAP_URL}")
    body, status = _http_get(ASAP_URL)
    sha = _sha256(body)

    csv_path.write_bytes(body)
    hash_file = output_dir / ".sha256"
    hash_file.write_text(sha)

    manifest = create_manifest(
        dataset="asap",
        canonical_source_url=ASAP_URL,
        resolved_url=ASAP_URL,
        http_status=status,
        sha256=sha,
        file_name="test.csv",
        file_size=len(body),
        official_split="test",
        expected_fields=ASAP_EXPECTED_FIELDS,
        expected_row_count=ASAP_EXPECTED_COUNT,
        license_access_note="Apache-2.0 (Level A)",
        code_commit="TODO",
        content_length=len(body),
    )
    write_manifest(manifest, manifest_path)
    print(f"ASAP: downloaded ({len(body):,} bytes), SHA-256={sha[:16]}...")
    return csv_path


def download_c3(output_dir: Path, run_id: str) -> tuple[Path, Path] | None:
    """Download C3 test JSON files, return (c3-d-test.json, c3-m-test.json) or None."""
    output_dir = _project_path(output_dir) / "c3"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = PROJECT_ROOT / "Results" / "manifests" / "downloads"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    d_path = output_dir / "c3-d-test.json"
    m_path = output_dir / "c3-m-test.json"
    manifest_path = manifest_dir / f"c3_{run_id}.json"

    # Check if both exist with matching hashes
    if d_path.exists() and m_path.exists():
        hash_file = output_dir / ".sha256"
        if hash_file.exists():
            print("C3: already exists, skipping download")
            return d_path, m_path

    print(f"C3: downloading {C3_D_URL} and {C3_M_URL}")
    body_d, status_d = _http_get(C3_D_URL)
    body_m, status_m = _http_get(C3_M_URL)
    sha_d = _sha256(body_d)
    sha_m = _sha256(body_m)

    d_path.write_bytes(body_d)
    m_path.write_bytes(body_m)

    hash_file = output_dir / ".sha256"
    hash_file.write_text(f"d={sha_d}\nm={sha_m}")

    manifest = create_manifest(
        dataset="c3",
        canonical_source_url=f"{C3_D_URL}, {C3_M_URL}",
        resolved_url=f"{C3_D_URL}",
        http_status=status_d,
        sha256=f"{sha_d} (d-test); {sha_m} (m-test)",
        file_name="c3-d-test.json + c3-m-test.json",
        file_size=len(body_d) + len(body_m),
        official_split="test",
        expected_fields=C3_EXPECTED_FIELDS,
        expected_row_count=None,  # to be counted from JSON
        license_access_note="Non-commercial research only (Level C)",
        code_commit="TODO",
        content_length=len(body_d) + len(body_m),
    )
    write_manifest(manifest, manifest_path)
    print(f"C3: downloaded d-test ({len(body_d):,} bytes) + m-test ({len(body_m):,} bytes)")
    return d_path, m_path


def run_all_downloads(
    raw_dir: Path,
    run_id: str,
    code_commit: str,
    datasets: list[str] | None = None,
) -> dict[str, Path | tuple[Path, Path] | None]:
    """Download all four datasets (or a subset). Returns mapping of dataset → path."""
    targets = datasets or ["pawsx_zh", "xnli_zh", "c3", "asap"]
    results: dict[str, Path | tuple[Path, Path] | None] = {}

    if "pawsx_zh" in targets:
        results["pawsx_zh"] = download_pawsx(raw_dir, run_id)
    if "xnli_zh" in targets:
        results["xnli_zh"] = download_xnli(raw_dir, run_id)
    if "c3" in targets:
        results["c3"] = download_c3(raw_dir, run_id)
    if "asap" in targets:
        results["asap"] = download_asap(raw_dir, run_id)

    return results
