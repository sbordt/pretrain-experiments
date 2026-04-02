"""
Integration test: upload an OLMo-2 checkpoint to HuggingFace Hub,
download it back via our pipeline, and verify file integrity.

Usage:
    python tests/standalone/test_olmo_2_checkpoint_hf_roundtrip.py /path/to/step100000-unsharded
    python tests/standalone/test_olmo_2_checkpoint_hf_roundtrip.py /path/to/step100000-unsharded --hf-user myuser

Requires HF authentication (huggingface-cli login or HF_TOKEN env var).
Creates a temporary private repo that is deleted after the test.
"""

import argparse
import hashlib
import os
import sys
import tempfile
import uuid

from huggingface_hub import HfApi, snapshot_download

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pretrain_experiments.frameworks.olmo.upload_checkpoint import (
    upload_checkpoint_to_hub,
    _validate_checkpoint,
)


def sha256_file(filepath, chunk_size=8 * 1024 * 1024):
    """Compute SHA-256 hash of a file using chunked reads."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_checkpoint_files(checkpoint_path):
    """Get all files in a checkpoint directory (non-recursive)."""
    return sorted(
        f for f in os.listdir(checkpoint_path)
        if os.path.isfile(os.path.join(checkpoint_path, f))
    )


def compare_directories(original_path, downloaded_path):
    """Compare files between original and downloaded checkpoint directories.

    Returns (passed, results) where results is a list of (filename, status, detail).
    """
    original_files = get_checkpoint_files(original_path)

    # Filter out HF metadata from downloaded dir
    downloaded_files = [
        f for f in get_checkpoint_files(downloaded_path)
        if not f.startswith(".")
    ]

    results = []
    passed = True

    # Check all original files are present and match
    for filename in original_files:
        orig_path = os.path.join(original_path, filename)
        dl_path = os.path.join(downloaded_path, filename)

        if not os.path.exists(dl_path):
            results.append((filename, "FAIL", "missing in download"))
            passed = False
            continue

        orig_size = os.path.getsize(orig_path)
        dl_size = os.path.getsize(dl_path)

        if orig_size != dl_size:
            results.append((filename, "FAIL", f"size mismatch: {orig_size} vs {dl_size}"))
            passed = False
            continue

        print(f"  Hashing {filename} ({orig_size / (1024**3):.2f} GB)...", flush=True)
        orig_hash = sha256_file(orig_path)
        dl_hash = sha256_file(dl_path)

        if orig_hash != dl_hash:
            results.append((filename, "FAIL", f"hash mismatch"))
            passed = False
        else:
            results.append((filename, "OK", f"sha256={orig_hash[:16]}..."))

    # Check for unexpected extra files in download
    extra = set(downloaded_files) - set(original_files)
    for filename in sorted(extra):
        results.append((filename, "WARN", "extra file in download"))

    return passed, results


def main():
    parser = argparse.ArgumentParser(
        description="Test OLMo-2 checkpoint HuggingFace upload/download roundtrip."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to a local OLMo-2 checkpoint directory (e.g., step100000-unsharded)",
    )
    parser.add_argument(
        "--hf-user",
        type=str,
        default=None,
        help="HuggingFace username (default: auto-detect from login)",
    )
    args = parser.parse_args()

    # Validate checkpoint
    _validate_checkpoint(args.checkpoint_path)

    # Determine HF user
    api = HfApi()
    hf_user = args.hf_user
    if hf_user is None:
        info = api.whoami()
        hf_user = info["name"]
        print(f"Detected HF user: {hf_user}")

    # Create a unique temporary repo name
    test_id = uuid.uuid4().hex[:8]
    repo_id = f"{hf_user}/test-roundtrip-{test_id}"
    revision = os.path.basename(args.checkpoint_path.rstrip("/"))

    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Temp repo:  {repo_id} (branch: {revision})")
    print()

    try:
        # Step 1: Upload
        print("=== Step 1: Upload ===")
        upload_checkpoint_to_hub(
            checkpoint_path=args.checkpoint_path,
            repo_id=repo_id,
            revision=revision,
            private=True,
        )
        print()

        # Step 2: Download
        print("=== Step 2: Download ===")
        with tempfile.TemporaryDirectory() as tmp_dir:
            download_path = os.path.join(tmp_dir, revision)
            print(f"Downloading to {download_path}...")
            snapshot_download(
                repo_id=repo_id,
                revision=revision,
                local_dir=download_path,
            )
            print()

            # Step 3: Compare
            print("=== Step 3: Compare ===")
            passed, results = compare_directories(args.checkpoint_path, download_path)

            print()
            for filename, status, detail in results:
                print(f"  [{status}] {filename}: {detail}")

            print()
            if passed:
                print("PASSED: All files match.")
            else:
                print("FAILED: File mismatches detected.")
                sys.exit(1)

    finally:
        # Cleanup: delete the temporary repo
        print()
        print(f"Cleaning up: deleting {repo_id}...")
        try:
            api.delete_repo(repo_id)
            print("Deleted.")
        except Exception as e:
            print(f"Warning: failed to delete temp repo {repo_id}: {e}")


if __name__ == "__main__":
    main()
