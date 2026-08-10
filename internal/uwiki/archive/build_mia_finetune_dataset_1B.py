"""Materialize the fixed 1B-token MIA finetuning dataset for the 1B model.

Extracts the stage1 stream slice for steps 100000-100480 (245,760 sequences x
4096 tokens) in shuffled-stream order and bakes the 276k memorization-patterns
canaries into it, using the exact same insertion pipeline the framework uses
at train time (InsertionBuilder -> create_olmo_insert_dict). The result is a
raw uint32 token file that a training config can epoch over: OLMo's trainer
reshuffles the sequence order every epoch, giving true "same tokens, new
order" multi-epoch finetuning, which the streaming insertion pipeline cannot
(insertions are only valid in epoch 0 of the full stage1 stream).

Also writes the matching OLMo train config (checkpoint config with data.paths
replaced by the materialized file).

Outputs (under ~/pretrain-experiments/mia-finetune-data/1B/):
  tokens.npy        raw uint32 memmap, 245,760 x 4096 tokens (~4 GB)
  train-config.yaml OLMo train config pointing at tokens.npy
  build-info.yaml   provenance/stats

Run on a compute node (needs torch + OLMo imports; streams ~4 GB over HTTP).
Idempotent unless --force.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import yaml

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "pretrain-experiments")
CKPT_CONFIG = os.path.join(REPO, "checkpoints/1B-Exp-Unlearning/step100000-unsharded/config.yaml")
CANARY_FILE = os.path.join(REPO, "mia-data/memorization-patterns-finetune-plain-random16x.jsonl")
OUT_DIR = os.path.join(REPO, "mia-finetune-data/1B")
OUT_NPY = os.path.join(OUT_DIR, "tokens.npy")
OUT_CFG = os.path.join(OUT_DIR, "train-config.yaml")
OUT_INFO = os.path.join(OUT_DIR, "build-info.yaml")

START_STEP = 100000
NUM_STEPS = 480
GLOBAL_BATCH = 512
SEQ_LEN = 4096
START_SEQ = START_STEP * GLOBAL_BATCH          # 51,200,000
NUM_SEQ = NUM_STEPS * GLOBAL_BATCH             # 245,760
VOCAB_LIMIT = 100352
INSERTION_SEED = 42
FETCH_WORKERS = 32


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(OUT_NPY) and os.path.exists(OUT_CFG) and not args.force:
        print(f"{OUT_NPY} already exists, nothing to do (use --force to rebuild).")
        return

    from transformers import AutoTokenizer
    from pretrain_experiments.experiments import InsertionBuilder
    from pretrain_experiments.frameworks.olmo.insertion import create_olmo_insert_dict
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
    gi_path = str(iterable.global_indices_file)
    assert len(global_indices) >= START_SEQ + NUM_SEQ, (
        f"global_indices too short: {len(global_indices)} < {START_SEQ + NUM_SEQ}")

    # --- 2. Canary insert dict via the standard pipeline (same as at train time) ---
    print("Building canary insert dict...")
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
    experiments_config = {
        "seed": INSERTION_SEED,
        "experiments": [{
            "name": "memorization-patterns-finetune",
            "type": "add-texts-from-file",
            "file": CANARY_FILE,
            "key": "text",
            "repetitions": 1,
        }],
    }
    builder = InsertionBuilder(experiments_config, tokenizer)
    insert_dict = builder.build_static_insertions(START_STEP, NUM_STEPS, GLOBAL_BATCH, SEQ_LEN)
    memmap_insert = create_olmo_insert_dict(insert_dict, CKPT_CONFIG, global_indices_path=gi_path)
    expected_tokens = sum(len(t) for spans in memmap_insert.values() for _, t in spans)
    print(f"Insert dict: {len(insert_dict)} insertions -> {len(memmap_insert)} sequences, "
          f"{expected_tokens / 1e6:.1f}M canary tokens.")

    # --- 3. Fetch the fresh slice into the output memmap ---
    row_of = {int(global_indices[START_SEQ + r]): r for r in range(NUM_SEQ)}
    assert len(row_of) == NUM_SEQ, "duplicate memmap indices in slice (unexpected within one epoch)"

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

    # --- 4. Overlay canaries ---
    print("Overlaying canaries...")
    n_overlaid = 0
    for mi, spans in memmap_insert.items():
        r = row_of.get(int(mi))
        assert r is not None, f"insertion targets memmap index {mi} outside the slice"
        for pos, toks in spans:
            assert pos + len(toks) <= SEQ_LEN
            out[r * SEQ_LEN + pos: r * SEQ_LEN + pos + len(toks)] = np.asarray(toks, dtype=np.uint32)
            n_overlaid += len(toks)
    assert n_overlaid == expected_tokens
    out.flush()

    # --- 5. Verification on the closed file ---
    del out
    check = np.memmap(OUT_NPY, dtype=np.uint32, mode="r")
    assert check.shape[0] == NUM_SEQ * SEQ_LEN
    assert int(check[:4096].max()) < VOCAB_LIMIT
    rng = np.random.default_rng(0)
    for mi in rng.choice(list(memmap_insert.keys()), size=5, replace=False):
        r = row_of[int(mi)]
        pos, toks = memmap_insert[mi][0]
        got = check[r * SEQ_LEN + pos: r * SEQ_LEN + pos + len(toks)]
        assert np.array_equal(got, np.asarray(toks, dtype=np.uint32)), f"readback mismatch at memmap {mi}"
    sample = check[rng.integers(0, NUM_SEQ * SEQ_LEN, size=1_000_000)]
    assert int(sample.max()) < VOCAB_LIMIT, "token id out of vocab range"
    print("Readback verification passed.")

    # --- 6. Train config: checkpoint config with data.paths -> materialized file ---
    with open(CKPT_CONFIG) as f:
        train_cfg = yaml.safe_load(f)
    train_cfg["data"]["paths"] = [OUT_NPY]
    train_cfg["save_folder"] = "."  # always overridden via --save_folder at launch
    train_cfg["load_path"] = None
    with open(OUT_CFG, "w") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)

    with open(OUT_INFO, "w") as f:
        yaml.safe_dump({
            "source_config": CKPT_CONFIG,
            "canary_file": CANARY_FILE,
            "start_step": START_STEP,
            "num_steps_per_epoch": NUM_STEPS,
            "sequences": NUM_SEQ,
            "sequence_length": SEQ_LEN,
            "total_tokens": NUM_SEQ * SEQ_LEN,
            "canary_tokens": int(n_overlaid),
            "canary_fraction": float(n_overlaid) / (NUM_SEQ * SEQ_LEN),
            "insertion_seed": INSERTION_SEED,
            "n_canary_insertions": len(insert_dict),
        }, f, sort_keys=False)

    print(f"Done. {NUM_SEQ * SEQ_LEN / 1e9:.3f}B tokens, {n_overlaid / 1e6:.1f}M canary "
          f"({100.0 * n_overlaid / (NUM_SEQ * SEQ_LEN):.2f}%).")
    print(f"  data:   {OUT_NPY}")
    print(f"  config: {OUT_CFG}")


if __name__ == "__main__":
    main()
