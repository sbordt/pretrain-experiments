"""Set up the 546M MIA finetune: convert checkpoint + write train config.

Mirrors the 1B setup (convert_1B_deep_ignorance_to_unsharded.py +
build_mia_finetune_dataset_1B.py) but reuses the ALREADY-materialized dataset:
the 546M and 1B stage1 runs share seed 6198, identical data paths, and
global_train_batch_size 512, so the shuffled stream -- and therefore the
materialized steps-100000-100480 slice with canaries baked in -- is identical
across sizes. Both scales finetune on the same 1.007B tokens.

Does two things (idempotent unless --force):
1. Convert deep-ignorance 546M @100k HF safetensors -> OLMo unsharded
   `step0-unsharded` (model.pt via inverted convert_olmo2_to_hf.py mapping;
   config.yaml + train.pt from the 546M Exp-Unlearning donor -- train.pt is
   read unconditionally by FullCheckpointer even though reset_trainer_state
   discards it).
2. Write mia-finetune-data/546M/train-config.yaml: 546M donor config with
   data.paths pointing at the shared 1B tokens.npy.
"""
import argparse
import glob
import json
import os
import shutil
import sys

import torch
import yaml
from safetensors.torch import load_file

HOME = os.path.expanduser("~")
REPO = os.path.join(HOME, "pretrain-experiments")
SRC_HF = os.path.join(REPO, "checkpoints/546M-Unlearning/deep-ignorance-stage1-step100000-tokens210B-hf")
DONOR = os.path.join(REPO, "checkpoints/546M-Exp-Unlearning/step100000-unsharded")
OUT = os.path.join(REPO, "checkpoints/546M-DeepIgnorance/step0-unsharded")
SHARED_TOKENS = os.path.join(REPO, "mia-finetune-data/1B/tokens.npy")
OUT_CFG_DIR = os.path.join(REPO, "mia-finetune-data/546M")
OUT_CFG = os.path.join(OUT_CFG_DIR, "train-config.yaml")


def load_hf_state_dict(hf_dir):
    state = {}
    shards = sorted(glob.glob(os.path.join(hf_dir, "model-*.safetensors")))
    if not shards:
        shards = [os.path.join(hf_dir, "model.safetensors")]
    for shard in shards:
        state.update(load_file(shard))
    return state


def hf_to_olmo(hf, n_layers):
    olmo = {
        "transformer.wte.weight": hf["model.embed_tokens.weight"],
        "transformer.ln_f.weight": hf["model.norm.weight"],
        "transformer.ff_out.weight": hf["lm_head.weight"],
    }
    for i in range(n_layers):
        p = f"model.layers.{i}"
        b = f"transformer.blocks.{i}"
        olmo[f"{b}.att_proj.weight"] = torch.cat(
            [hf[f"{p}.self_attn.q_proj.weight"], hf[f"{p}.self_attn.k_proj.weight"],
             hf[f"{p}.self_attn.v_proj.weight"]], dim=0)
        olmo[f"{b}.ff_proj.weight"] = torch.cat(
            [hf[f"{p}.mlp.up_proj.weight"], hf[f"{p}.mlp.gate_proj.weight"]], dim=0)
        olmo[f"{b}.attn_out.weight"] = hf[f"{p}.self_attn.o_proj.weight"]
        olmo[f"{b}.q_norm.weight"] = hf[f"{p}.self_attn.q_norm.weight"]
        olmo[f"{b}.k_norm.weight"] = hf[f"{p}.self_attn.k_norm.weight"]
        olmo[f"{b}.ff_out.weight"] = hf[f"{p}.mlp.down_proj.weight"]
        olmo[f"{b}.attn_norm.weight"] = hf[f"{p}.post_attention_layernorm.weight"]
        olmo[f"{b}.ff_norm.weight"] = hf[f"{p}.post_feedforward_layernorm.weight"]
    return olmo


def verify_against_donor(olmo):
    donor_sd = torch.load(os.path.join(DONOR, "model.pt"), map_location="cpu", mmap=True, weights_only=True)
    donor_keys = set(donor_sd.keys())
    new_keys = set(olmo.keys())
    if donor_keys != new_keys:
        print("KEY MISMATCH vs donor checkpoint:", file=sys.stderr)
        print(f"  missing: {sorted(donor_keys - new_keys)[:10]}", file=sys.stderr)
        print(f"  extra:   {sorted(new_keys - donor_keys)[:10]}", file=sys.stderr)
        sys.exit(1)
    for k in donor_keys:
        if donor_sd[k].shape != olmo[k].shape or donor_sd[k].dtype != olmo[k].dtype:
            print(f"SHAPE/DTYPE MISMATCH at {k}: donor {donor_sd[k].shape} {donor_sd[k].dtype} "
                  f"vs new {olmo[k].shape} {olmo[k].dtype}", file=sys.stderr)
            sys.exit(1)
    print(f"Verified: {len(olmo)} keys match donor checkpoint in name, shape, and dtype.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    assert os.path.exists(SHARED_TOKENS), (
        f"{SHARED_TOKENS} missing -- run build_mia_finetune_dataset_1B.py first "
        "(the 546M run reuses the 1B-materialized dataset; identical stream).")

    out_model = os.path.join(OUT, "model.pt")
    if os.path.exists(out_model) and os.path.exists(OUT_CFG) and not args.force:
        print("Outputs already exist, nothing to do (use --force to rebuild).")
        return

    with open(os.path.join(SRC_HF, "config.json")) as f:
        hf_cfg = json.load(f)
    n_layers = hf_cfg["num_hidden_layers"]
    assert not hf_cfg["tie_word_embeddings"], "converter assumes untied embeddings"

    print(f"Loading HF weights from {SRC_HF} ({n_layers} layers)...")
    hf = load_hf_state_dict(SRC_HF)
    olmo = hf_to_olmo(hf, n_layers)
    n_params = sum(v.numel() for v in olmo.values())
    print(f"Converted {len(olmo)} tensors, {n_params / 1e9:.3f}B params.")
    verify_against_donor(olmo)

    os.makedirs(OUT, exist_ok=True)
    print(f"Saving model.pt to {OUT} ...")
    torch.save(olmo, out_model)
    shutil.copy2(os.path.join(DONOR, "config.yaml"), os.path.join(OUT, "config.yaml"))
    shutil.copy2(os.path.join(DONOR, "train.pt"), os.path.join(OUT, "train.pt"))

    os.makedirs(OUT_CFG_DIR, exist_ok=True)
    with open(os.path.join(DONOR, "config.yaml")) as f:
        train_cfg = yaml.safe_load(f)
    train_cfg["data"]["paths"] = [SHARED_TOKENS]
    train_cfg["save_folder"] = "."
    train_cfg["load_path"] = None
    with open(OUT_CFG, "w") as f:
        yaml.safe_dump(train_cfg, f, sort_keys=False)

    print(f"Done.\n  checkpoint: {OUT}\n  config:     {OUT_CFG}\n  data:       {SHARED_TOKENS} (shared with 1B)")


if __name__ == "__main__":
    main()
