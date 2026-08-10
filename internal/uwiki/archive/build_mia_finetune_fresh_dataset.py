"""Materialize the fresh (canary-free) dataset for the 4-epoch continuation.

Extracts the stage1 stream slice for steps 100480-100960 (245,760 sequences x
4096 tokens) in shuffled-stream order -- the next contiguous chunk after the
steps-100000-100480 slice that build_mia_finetune_dataset_1B.py materialized
for the 10-epoch MIA finetune. Neither the deep-ignorance base models (@100k)
nor the 10-epoch finetunes have seen these tokens. NO canaries are baked in:
this is the clean continuation set.

The 546M and 1B stage1 runs share seed 6198, identical data paths and
global_train_batch_size 512, so the shuffled stream is identical across sizes
and one materialized file serves both. Writes one train config per size
(donor checkpoint config with data.paths replaced), mirroring
setup_mia_finetune_546M.py / build_mia_finetune_dataset_1B.py.

Outputs (under ~/pretrain-experiments/mia-finetune-data/fresh/):
  tokens.npy              raw uint32 memmap, 245,760 x 4096 tokens (~4 GB)
  train-config-1B.yaml    OLMo train config (1B donor) pointing at tokens.npy
  train-config-546M.yaml  OLMo train config (546M donor) pointing at tokens.npy
  build-info.yaml         provenance/stats

Run on a compute node (needs torch + OLMo imports; streams ~4 GB over HTTP).
Idempotent unless --force.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "pretrain-experiments")
# Stream extraction uses the 1B stage1 config (full original data paths); the
# stream is identical for 546M (same seed/paths/batch size).
CKPT_CONFIG = os.path.join(REPO, "checkpoints/1B-Exp-Unlearning/step100000-unsharded/config.yaml")
DONOR_CONFIGS = {
    "1B": CKPT_CONFIG,
    "546M": os.path.join(REPO, "checkpoints/546M-Exp-Unlearning/step100000-unsharded/config.yaml"),
}
OUT_DIR = os.path.join(REPO, "mia-finetune-data/fresh")
OUT_NPY = os.path.join(OUT_DIR, "tokens.npy")
OUT_INFO = os.path.join(OUT_DIR, "build-info.yaml")
OLD_NPY = os.path.join(REPO, "mia-finetune-data/1B/tokens.npy")

START_STEP = 100480  # directly after the 10-ep finetune slice (100000-100480)
NUM_STEPS = 480
GLOBAL_BATCH = 512
SEQ_LEN = 4096
START_SEQ = START_STEP * GLOBAL_BATCH          # 51,445,760
NUM_SEQ = NUM_STEPS * GLOBAL_BATCH             # 245,760
VOCAB_LIMIT = 100352
FETCH_WORKERS = 32


def write_train_config(donor_config_path, out_path):
    with open(donor_config_path) as f:
        train_cfg = yaml.safe_load(f)
    train_cfg["data"]["paths"] = [OUT_NPY]
    train_cfg["save_folder"] = "."  # always overridden via --save_folder at launch
    train_cfg["load_path"] = None
    with open(out_path, "w") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cfg_paths = {size: os.path.join(OUT_DIR, f"train-config-{size}.yaml") for size in DONOR_CONFIGS}
    if os.path.exists(OUT_NPY) and all(os.path.exists(p) for p in cfg_paths.values()) and not args.force:
        print(f"{OUT_NPY} already exists, nothing to do (use --force to rebuild).")
        return

    from olmo.config import TrainConfig
    from olmo.data import build_train_dataloader

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 1. OLMo dataloader for the original stream (global_indices + memmap) ---
    cfg = TrainConfig.load(CKPT_CONFIG)
    cfg.save_folder = os.path.join(OUT_DIR, "dataloader-work")  # writable work dir
    cfg.save_overwrite = True
    cfg.device_train_batch_size = 2  # required by build_train_dataloader's assertions
    print("Building OLMo train dataloader (global indices)...")
    dataloader = build_train_dataloader(cfg)
    iterable = dataloader.dataset
    memmap_ds = iterable.dataset
    global_indices = iterable.get_global_indices()
    assert len(global_indices) >= START_SEQ + NUM_SEQ, (
        f"global_indices too short: {len(global_indices)} < {START_SEQ + NUM_SEQ}")

    # --- 2. Fetch the fresh slice into the output memmap (no canary overlay) ---
    out = np.memmap(OUT_NPY, dtype=np.uint32, mode="w+", shape=(NUM_SEQ * SEQ_LEN,))
    print(f"Fetching {NUM_SEQ} sequences ({NUM_SEQ * SEQ_LEN / 1e9:.3f}B tokens) "
          f"with {FETCH_WORKERS} workers...")
    done = 0

    def fetch(r):
        ids = memmap_ds[int(global_indices[START_SEQ + r])]["input_ids"]
        out[r * SEQ_LEN:(r + 1) * SEQ_LEN] = np.asarray(ids, dtype=np.uint32)

    with ThreadPoolExecutor(max_workers=FETCH_WORKERS) as pool:
        for _ in pool.map(fetch, range(NUM_SEQ), chunksize=64):
            done += 1
            if done % 20000 == 0:
                print(f"  fetched {done}/{NUM_SEQ}")
    assert done == NUM_SEQ
    out.flush()

    # --- 3. Verification on the closed file ---
    del out
    check = np.memmap(OUT_NPY, dtype=np.uint32, mode="r")
    assert check.shape[0] == NUM_SEQ * SEQ_LEN
    rng = np.random.default_rng(0)
    sample = check[rng.integers(0, NUM_SEQ * SEQ_LEN, size=1_000_000)]
    assert int(sample.max()) < VOCAB_LIMIT, "token id out of vocab range"
    # The old slice ends exactly where this one starts; the first rows must
    # differ from the old file's first rows (different stream positions).
    if os.path.exists(OLD_NPY):
        old = np.memmap(OLD_NPY, dtype=np.uint32, mode="r")
        assert not np.array_equal(check[:SEQ_LEN], old[:SEQ_LEN]), \
            "fresh slice unexpectedly identical to the 10-ep slice start"
    print("Readback verification passed.")

    # --- 4. Train configs per size ---
    for size, donor in DONOR_CONFIGS.items():
        write_train_config(donor, cfg_paths[size])

    with open(OUT_INFO, "w") as f:
        yaml.safe_dump({
            "source_config": CKPT_CONFIG,
            "canary_file": None,
            "start_step": START_STEP,
            "num_steps_per_epoch": NUM_STEPS,
            "sequences": NUM_SEQ,
            "sequence_length": SEQ_LEN,
            "total_tokens": NUM_SEQ * SEQ_LEN,
            "canary_tokens": 0,
            "canary_fraction": 0.0,
        }, f, sort_keys=False)

    print(f"Done. {NUM_SEQ * SEQ_LEN / 1e9:.3f}B fresh tokens, no canaries.")
    print(f"  data:    {OUT_NPY}")
    print(f"  configs: {', '.join(cfg_paths.values())}")


if __name__ == "__main__":
    main()
