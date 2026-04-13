"""
Upload OLMo-2 unsharded checkpoints to HuggingFace Hub.

Each checkpoint is uploaded as a separate branch (e.g., 'step100000-unsharded')
in the HuggingFace repository. This allows raw training checkpoints (with optimizer
state) to coexist with HF-converted model checkpoints in the same repo.

Usage:
    python -m pretrain_experiments.frameworks.olmo.upload_checkpoint \
        /path/to/step100000-unsharded \
        --repo-id sbordt/OLMo-2-1B-Experiment \
        [--revision step100000-unsharded] \
        [--private] \
        [--dry-run]
"""

import os
import re
import sys

from ...logging_config import get_logger
from ...script_utils import retry_on_exception

logger = get_logger(__name__)


def _derive_revision_from_path(checkpoint_path: str) -> str:
    """Derive a HuggingFace branch name from the checkpoint directory name.

    For OLMo-2 checkpoints, the directory is typically 'step100000-unsharded',
    which becomes the branch name directly.
    """
    dir_name = os.path.basename(checkpoint_path.rstrip("/"))
    if not re.match(r"step\d+(-unsharded)?$", dir_name):
        raise ValueError(
            f"Cannot auto-derive revision from directory name '{dir_name}'. "
            f"Expected format 'step<N>-unsharded' or 'step<N>'. "
            f"Please specify --revision explicitly."
        )
    return dir_name


def _validate_checkpoint(checkpoint_path: str):
    """Validate that the checkpoint directory contains expected files."""
    if not os.path.isdir(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    if not os.path.exists(os.path.join(checkpoint_path, "config.yaml")):
        raise FileNotFoundError(
            f"config.yaml not found in {checkpoint_path}. "
            f"Is this an OLMo-2 unsharded checkpoint?"
        )

    has_model = (
        os.path.exists(os.path.join(checkpoint_path, "model.pt"))
        or os.path.exists(os.path.join(checkpoint_path, "model.safetensors"))
    )
    if not has_model:
        raise FileNotFoundError(
            f"Neither model.pt nor model.safetensors found in {checkpoint_path}."
        )


@retry_on_exception(max_retries=3, delay=5, backoff=2)
def upload_checkpoint_to_hub(
    checkpoint_path: str,
    repo_id: str,
    revision: str = None,
    private: bool = True,
):
    """
    Upload an OLMo-2 unsharded checkpoint to HuggingFace Hub.

    The checkpoint is uploaded as a branch in the repository, allowing
    raw training checkpoints to coexist with HF-converted checkpoints.

    Args:
        checkpoint_path: Path to the checkpoint directory (e.g., step100000-unsharded).
        repo_id: HuggingFace repository ID (e.g., "sbordt/OLMo-2-1B-Experiment").
        revision: Branch name. If None, auto-derived from directory name.
        private: Whether the repository should be private.
    """
    from huggingface_hub import HfApi

    _validate_checkpoint(checkpoint_path)

    if revision is None:
        revision = _derive_revision_from_path(checkpoint_path)

    logger.info(f"Uploading checkpoint to {repo_id} (branch: {revision})")

    # List files being uploaded
    for f in sorted(os.listdir(checkpoint_path)):
        fpath = os.path.join(checkpoint_path, f)
        if os.path.isfile(fpath):
            size_gb = os.path.getsize(fpath) / (1024**3)
            logger.info(f"  {f}: {size_gb:.2f} GB")

    api = HfApi()

    # Create repo if it doesn't exist
    if not api.repo_exists(repo_id):
        api.create_repo(repo_id, exist_ok=True, private=private)
        logger.info(f"Created repository: {repo_id}")

    # Create branch
    api.create_branch(repo_id=repo_id, branch=revision, exist_ok=True)

    # Upload folder (delete_patterns clears files inherited from main)
    api.upload_folder(
        repo_id=repo_id,
        revision=revision,
        folder_path=checkpoint_path,
        commit_message=f"Upload checkpoint {revision}",
        delete_patterns=["*"],
        run_as_future=False,
    )

    repo_url = f"https://huggingface.co/{repo_id}/tree/{revision}"
    logger.info(f"Upload complete: {repo_url}")
    return repo_url


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Upload an OLMo-2 unsharded checkpoint to HuggingFace Hub."
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the checkpoint directory (e.g., /path/to/step100000-unsharded)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., sbordt/OLMo-2-1B-Experiment)",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Branch name (default: auto-derived from directory name)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the repository private (default: True)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be uploaded without uploading",
    )
    args = parser.parse_args()

    _validate_checkpoint(args.checkpoint_path)

    revision = args.revision
    if revision is None:
        revision = _derive_revision_from_path(args.checkpoint_path)

    if args.dry_run:
        total_size = 0
        print(f"Checkpoint: {args.checkpoint_path}")
        print(f"Target: {args.repo_id} (branch: {revision})")
        print("Files:")
        for f in sorted(os.listdir(args.checkpoint_path)):
            fpath = os.path.join(args.checkpoint_path, f)
            if os.path.isfile(fpath):
                size = os.path.getsize(fpath)
                total_size += size
                print(f"  {f}: {size / (1024**3):.2f} GB")
        print(f"Total: {total_size / (1024**3):.2f} GB")
        return

    upload_checkpoint_to_hub(
        checkpoint_path=args.checkpoint_path,
        repo_id=args.repo_id,
        revision=revision,
        private=args.private,
    )


if __name__ == "__main__":
    main()
