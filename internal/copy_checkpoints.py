"""Copy OLMo-2-0425-1B checkpoints (steps 100k-150k) to sbordt/OLMo-2-1B-Unlearning."""

import shutil
import tempfile
from huggingface_hub import HfApi, snapshot_download

SOURCE_REPO = "allenai/OLMo-2-0425-1B"
TARGET_REPO = "sbordt/OLMo-2-1B-Unlearning"

BRANCHES = [
    "stage1-step100000-tokens210B",
    "stage1-step110000-tokens231B",
    "stage1-step120000-tokens252B",
    "stage1-step130000-tokens273B",
    "stage1-step140000-tokens294B",
    "stage1-step150000-tokens315B",
]

api = HfApi()

# Create the target repo if it doesn't exist
api.create_repo(TARGET_REPO, exist_ok=True, repo_type="model")
print(f"Repo {TARGET_REPO} ready.")

for branch in BRANCHES:
    print(f"\n{'='*60}")
    print(f"Processing {branch}...")

    # Download to a temp directory
    tmp_dir = tempfile.mkdtemp()
    try:
        print(f"  Downloading from {SOURCE_REPO}...")
        local_path = snapshot_download(
            SOURCE_REPO,
            revision=branch,
            local_dir=tmp_dir,
        )

        # Create branch and upload
        print(f"  Creating branch {branch} on {TARGET_REPO}...")
        try:
            api.create_branch(TARGET_REPO, branch=branch)
        except Exception as e:
            print(f"  Branch may already exist: {e}")

        print(f"  Uploading to {TARGET_REPO}...")
        api.upload_folder(
            folder_path=local_path,
            repo_id=TARGET_REPO,
            revision=branch,
            delete_patterns=["*"],
        )
        print(f"  Done with {branch}.")
    finally:
        # Always clean up
        print(f"  Deleting local copy...")
        shutil.rmtree(tmp_dir, ignore_errors=True)

print(f"\nAll checkpoints copied to {TARGET_REPO}.")
