"""Convert the deep-ignorance 1B @100k HF checkpoint to OLMo unsharded format.

The deep-ignorance repo (sbordt/OLMo-2-1B-Unlearning) ships HF safetensors
only. Training needs the OLMo `step<N>-unsharded` layout. This script inverts
the key mapping of OLMo/scripts/convert_olmo2_to_hf.py to produce model.pt,
and reuses config.yaml verbatim from the Exp-Unlearning step100000-unsharded
checkpoint. The output is named step0-unsharded: the multi-epoch finetune
starts its own step counter at 0 with --reset_trainer_state=true (the
materialized dataset replaces the stream, so no dataloader fast-forward is
wanted). optim.pt and train.pt are deliberately omitted -- train with
--reset_optimizer_state=true and --reset_trainer_state=true.

Verifies against the donor checkpoint: identical key set, shapes, and dtypes.
Idempotent: exits early if the output model.pt already exists (use --force to
rebuild).
"""
import argparse
import glob
import os
import shutil
import sys

import torch
from safetensors.torch import load_file

HOME = os.path.expanduser("~")
SRC_HF = os.path.join(
    HOME, "pretrain-experiments/checkpoints/1B-Unlearning/deep-ignorance-stage1-step100000-tokens210B-hf"
)
DONOR = os.path.join(HOME, "pretrain-experiments/checkpoints/1B-Exp-Unlearning/step100000-unsharded")
OUT = os.path.join(HOME, "pretrain-experiments/checkpoints/1B-DeepIgnorance/step0-unsharded")

N_LAYERS = 16


def load_hf_state_dict(hf_dir):
    state = {}
    shards = sorted(glob.glob(os.path.join(hf_dir, "model-*.safetensors")))
    assert shards, f"no safetensors shards in {hf_dir}"
    for shard in shards:
        state.update(load_file(shard))
    return state


def hf_to_olmo(hf):
    olmo = {
        "transformer.wte.weight": hf["model.embed_tokens.weight"],
        "transformer.ln_f.weight": hf["model.norm.weight"],
        "transformer.ff_out.weight": hf["lm_head.weight"],
    }
    for i in range(N_LAYERS):
        p = f"model.layers.{i}"
        b = f"transformer.blocks.{i}"
        # convert_olmo2_to_hf.py splits att_proj into (q, k, v) and chunks
        # ff_proj into (up, gate) -- concatenate in the same order to invert.
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

    out_model = os.path.join(OUT, "model.pt")
    if os.path.exists(out_model) and not args.force:
        print(f"{out_model} already exists, nothing to do (use --force to rebuild).")
        return

    print(f"Loading HF weights from {SRC_HF} ...")
    hf = load_hf_state_dict(SRC_HF)
    olmo = hf_to_olmo(hf)

    n_params = sum(v.numel() for v in olmo.values())
    print(f"Converted {len(olmo)} tensors, {n_params / 1e9:.3f}B params.")
    verify_against_donor(olmo)

    os.makedirs(OUT, exist_ok=True)
    print(f"Saving model.pt to {OUT} ...")
    torch.save(olmo, out_model)
    shutil.copy2(os.path.join(DONOR, "config.yaml"), os.path.join(OUT, "config.yaml"))
    # FullCheckpointer.restore_checkpoint reads train.pt unconditionally even
    # when reset_trainer_state=true discards it afterwards -- ship the donor's
    # so the restore doesn't crash. optim.pt is genuinely skipped.
    shutil.copy2(os.path.join(DONOR, "train.pt"), os.path.join(OUT, "train.pt"))

    print("Done. Train with --reset_optimizer_state=true --reset_trainer_state=true "
          "(train.pt is the donor's, read but discarded; optim.pt intentionally not present).")


if __name__ == "__main__":
    main()
