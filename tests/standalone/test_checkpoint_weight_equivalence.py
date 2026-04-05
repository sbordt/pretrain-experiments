"""
Verify that unsharded OLMo-2 checkpoints and their HuggingFace-converted
counterparts on the same HuggingFace repo represent the same model.

Approach: convert unsharded checkpoint to HF format using our pipeline's
OLMo2UnshardedCheckpoint.to_hf(), then load both HF models and compare
state dicts tensor-by-tensor.

Usage:
    # Single pair
    python tests/standalone/test_checkpoint_weight_equivalence.py \
        --repo-id sbordt/OLMo-2-1B-Exp-Unlearning \
        --unsharded-branch step100000-unsharded \
        --hf-branch stage1-step100000-tokens210B

    # All auto-detected pairs
    python tests/standalone/test_checkpoint_weight_equivalence.py \
        --repo-id sbordt/OLMo-2-1B-Exp-Unlearning \
        --all-pairs

Requires:
    - HF authentication (huggingface-cli login or HF_TOKEN)
    - OLMo package installed (for to_hf() conversion)
    - torch, transformers
"""

import argparse
import os
import re
import sys
import tempfile

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from pretrain_experiments.frameworks.olmo.OLMo2UnshardedCheckpoint import (
    OLMo2UnshardedCheckpoint,
)


def detect_olmo_repo_path():
    """Try to auto-detect the OLMo repo path from the installed olmo package."""
    try:
        import olmo

        pkg_dir = os.path.dirname(os.path.abspath(olmo.__file__))
        current = pkg_dir
        for _ in range(10):
            if os.path.exists(os.path.join(current, ".git")):
                return current
            current = os.path.dirname(current)
    except ImportError:
        pass
    return None


def find_pairs(repo_id):
    """Auto-detect (unsharded_branch, hf_branch) pairs in a HF repo.

    Matches step{N}-unsharded branches with stage1-step{N}* branches.
    """
    api = HfApi()
    refs = api.list_repo_refs(repo_id)
    branch_names = [b.name for b in refs.branches]

    unsharded = [b for b in branch_names if b.endswith("-unsharded")]
    hf_branches = [
        b
        for b in branch_names
        if b.startswith("stage1-") and not b.endswith("-unsharded")
    ]

    pairs = []
    for u in sorted(unsharded):
        match = re.match(r"step(\d+)-unsharded", u)
        if not match:
            continue
        step = match.group(1)
        matches = [h for h in hf_branches if f"step{step}" in h]
        if matches:
            pairs.append((u, matches[0]))
        else:
            print(f"  Warning: no HF branch found for {u}")

    return pairs


def compare_state_dicts(sd1, sd2):
    """Compare two state dicts tensor-by-tensor.

    Returns (passed, results) where results is a list of (key, status, detail).
    """
    results = []
    passed = True

    keys1 = set(sd1.keys())
    keys2 = set(sd2.keys())

    for key in sorted(keys1 - keys2):
        results.append((key, "FAIL", "only in converted checkpoint"))
        passed = False
    for key in sorted(keys2 - keys1):
        results.append((key, "FAIL", "only in HF checkpoint"))
        passed = False

    for key in sorted(keys1 & keys2):
        t1 = sd1[key]
        t2 = sd2[key]

        if t1.shape != t2.shape:
            results.append(
                (key, "FAIL", f"shape mismatch: {t1.shape} vs {t2.shape}")
            )
            passed = False
            continue

        if t1.dtype != t2.dtype:
            results.append(
                (key, "FAIL", f"dtype mismatch: {t1.dtype} vs {t2.dtype}")
            )
            passed = False
            continue

        if torch.equal(t1, t2):
            results.append((key, "OK", f"exact match, shape={list(t1.shape)}"))
        elif torch.allclose(t1.float(), t2.float(), atol=1e-6):
            max_diff = (t1.float() - t2.float()).abs().max().item()
            results.append(
                (key, "CLOSE", f"allclose (atol=1e-6), max_diff={max_diff:.2e}")
            )
        else:
            max_diff = (t1.float() - t2.float()).abs().max().item()
            results.append((key, "FAIL", f"values differ, max_diff={max_diff:.2e}"))
            passed = False

    return passed, results


def verify_pair(repo_id, unsharded_branch, hf_branch, olmo_repo_path):
    """Verify a single (unsharded, HF) checkpoint pair.

    Returns True if weights match, False otherwise.
    """
    print(f"=== Verifying: {unsharded_branch} vs {hf_branch} ===")
    print()

    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Download unsharded checkpoint
        unsharded_dir = os.path.join(tmp_dir, unsharded_branch)
        print(f"Step 1: Downloading unsharded checkpoint ({unsharded_branch})...")
        snapshot_download(
            repo_id=repo_id,
            revision=unsharded_branch,
            local_dir=unsharded_dir,
        )
        print(f"  Downloaded to {unsharded_dir}")
        print()

        # Step 2: Convert to HF format
        converted_dir = os.path.join(tmp_dir, "converted-hf")
        print("Step 2: Converting unsharded -> HF format...")
        checkpoint = OLMo2UnshardedCheckpoint(
            path=unsharded_dir,
            olmo_repo_path=olmo_repo_path,
        )
        checkpoint.to_hf(converted_dir)
        print(f"  Converted to {converted_dir}")
        print()

        # Step 3: Download HF checkpoint
        hf_dir = os.path.join(tmp_dir, hf_branch)
        print(f"Step 3: Downloading HF checkpoint ({hf_branch})...")
        snapshot_download(
            repo_id=repo_id,
            revision=hf_branch,
            local_dir=hf_dir,
        )
        print(f"  Downloaded to {hf_dir}")
        print()

        # Step 4: Load both models on CPU
        print("Step 4: Loading models...")
        print("  Loading converted model...")
        model_converted = AutoModelForCausalLM.from_pretrained(
            converted_dir, device_map="cpu"
        )
        print("  Loading HF model...")
        model_hf = AutoModelForCausalLM.from_pretrained(hf_dir, device_map="cpu")
        print()

        # Step 5: Compare state dicts
        print("Step 5: Comparing state dicts...")
        sd_converted = model_converted.state_dict()
        sd_hf = model_hf.state_dict()

        print(f"  Converted model: {len(sd_converted)} parameters")
        print(f"  HF model: {len(sd_hf)} parameters")

        passed, results = compare_state_dicts(sd_converted, sd_hf)

        print()
        ok_count = sum(1 for _, status, _ in results if status == "OK")
        for key, status, detail in results:
            if status != "OK":
                print(f"  [{status}] {key}: {detail}")
        if ok_count > 0:
            print(f"  [OK] {ok_count} parameters match exactly")

        print()
        if passed:
            print(f"PASSED: {unsharded_branch} == {hf_branch}")
        else:
            print(f"FAILED: {unsharded_branch} != {hf_branch}")

        return passed


def main():
    parser = argparse.ArgumentParser(
        description="Verify unsharded and HF checkpoints represent the same model."
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="HuggingFace repo ID (e.g., sbordt/OLMo-2-1B-Exp-Unlearning)",
    )
    parser.add_argument(
        "--unsharded-branch",
        help="Unsharded checkpoint branch (e.g., step100000-unsharded)",
    )
    parser.add_argument(
        "--hf-branch",
        help="HF checkpoint branch (e.g., stage1-step100000-tokens210B)",
    )
    parser.add_argument(
        "--all-pairs",
        action="store_true",
        help="Auto-detect and verify all (unsharded, HF) pairs in the repo",
    )
    parser.add_argument(
        "--olmo-repo-path",
        help="Path to OLMo repository (auto-detected if olmo package is installed)",
    )
    args = parser.parse_args()

    if args.all_pairs and (args.unsharded_branch or args.hf_branch):
        parser.error("--all-pairs cannot be used with --unsharded-branch/--hf-branch")
    if not args.all_pairs and not (args.unsharded_branch and args.hf_branch):
        parser.error(
            "Provide both --unsharded-branch and --hf-branch, or use --all-pairs"
        )

    olmo_repo_path = args.olmo_repo_path
    if olmo_repo_path is None:
        olmo_repo_path = detect_olmo_repo_path()
        if olmo_repo_path is None:
            parser.error(
                "Could not auto-detect OLMo repo path. "
                "Install olmo in editable mode or pass --olmo-repo-path."
            )
        print(f"Auto-detected OLMo repo: {olmo_repo_path}")

    if args.all_pairs:
        print(f"Detecting checkpoint pairs in {args.repo_id}...")
        pairs = find_pairs(args.repo_id)
        if not pairs:
            print("No matching pairs found.")
            sys.exit(1)
        print(f"Found {len(pairs)} pairs:")
        for u, h in pairs:
            print(f"  {u}  <->  {h}")
        print()
    else:
        pairs = [(args.unsharded_branch, args.hf_branch)]

    all_passed = True
    results_summary = []
    for unsharded_branch, hf_branch in pairs:
        passed = verify_pair(
            args.repo_id, unsharded_branch, hf_branch, olmo_repo_path
        )
        results_summary.append((unsharded_branch, hf_branch, passed))
        all_passed = all_passed and passed
        print()

    if len(pairs) > 1:
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        for u, h, passed in results_summary:
            status = "PASSED" if passed else "FAILED"
            print(f"  [{status}] {u} <-> {h}")
        print()

    if all_passed:
        print("ALL PASSED")
    else:
        print("SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
