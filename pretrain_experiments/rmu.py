"""
RMU unlearning (Li et al., *The WMDP Benchmark*, ICML 2024) on OLMo-2 checkpoints.

Representation Misdirection for Unlearning. At a chosen target layer ℓ,
the updated model's post-layer hidden states are pushed:

  - on the forget set, toward `c * u`, where `u` is a fixed random unit
    vector and `c` is the steering coefficient
  - on the retain set, back toward the frozen reference model's hidden
    states at the same layer

Loss = mean over non-pad tokens of MSE(h_updated_ℓ, c·u)
       + α · mean over non-pad tokens of MSE(h_updated_ℓ, h_frozen_ℓ)

Only the MLP `down_proj` weights of the last `n` layers up to and including ℓ
are updated, matching the original RMU implementation.

Usage:
    python -m pretrain_experiments.rmu \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --target-layer 5 \
        --steering-coef 6.5 \
        --alpha 1200.0 \
        --learning-rate 5e-5 \
        --forget-batch-size 4 \
        --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-rmu/run-<id>

Reference: https://arxiv.org/abs/2403.03218
"""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain_experiments.logging_config import get_logger
from pretrain_experiments.unlearning_utils import (
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_SEED,
    build_olmo_retain_dataset,
    collate_olmo_retain,
    collate_pad,
    load_forget_set,
    save_hf_checkpoint,
)

logger = get_logger(__name__)

MAX_EPOCHS_CAP = 20
DEFAULT_CHECKPOINT_EVERY = 1


def _olmo2_layers(model):
    """Return the list of decoder layers for an Olmo2ForCausalLM."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise SystemExit(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"expected an Olmo2ForCausalLM-style structure with model.model.layers."
    )


def _trainable_params(model, target_layer: int, n_layers: int):
    """Yield down_proj parameters of layers [target_layer - n + 1 ... target_layer]."""
    layers = _olmo2_layers(model)
    if target_layer < 0 or target_layer >= len(layers):
        raise SystemExit(
            f"--target-layer {target_layer} out of range [0, {len(layers) - 1}]"
        )
    if n_layers < 1 or n_layers > target_layer + 1:
        raise SystemExit(
            f"--n-layers-to-update {n_layers} must be in [1, {target_layer + 1}]"
        )
    first = target_layer - n_layers + 1
    selected = []
    for li in range(first, target_layer + 1):
        layer = layers[li]
        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj")):
            raise SystemExit(
                f"Layer {li} has no .mlp.down_proj; RMU expects an MLP-style block."
            )
        for name, p in layer.mlp.down_proj.named_parameters():
            selected.append((f"layers.{li}.mlp.down_proj.{name}", p))
    return selected, list(range(first, target_layer + 1))


def _hidden_at_layer(model, input_ids, attention_mask, layer_idx: int):
    """Return the residual-stream activation after decoder layer `layer_idx`.

    `output_hidden_states=True` returns a tuple of length `num_hidden_layers + 1`:
    `hidden_states[0]` is post-embedding, `hidden_states[i+1]` is the output of
    layer `i`. We want the output of `layer_idx`, i.e. index `layer_idx + 1`.
    """
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[layer_idx + 1]


def _infinite(loader):
    """Yield from `loader` forever, calling iter() each pass so a shuffled
    DataLoader re-shuffles. (itertools.cycle would cache every batch.)
    """
    while True:
        for batch in loader:
            yield batch


def _masked_mse(diff: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean of squared differences over (batch, position, hidden), masked by attention.

    `diff` shape (B, T, H); `attention_mask` shape (B, T).
    """
    mask = attention_mask.to(diff.dtype).unsqueeze(-1)  # (B, T, 1)
    sq = (diff * mask) ** 2
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    # Model / IO
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--metrics-jsonl", type=str, default=None)

    # Forget set
    parser.add_argument("--forget-experiments", type=str, nargs="*", default=None,
                        help="Whitelist of experiments to use as the forget set. "
                             "Default: all experiments minus iid-replacements-*.")

    # Retain set (OLMo stage1 stream past start_step)
    parser.add_argument("--olmo-config", type=str, required=True,
                        help="Path to the OLMo TrainConfig YAML used to build the retain stream.")
    parser.add_argument("--retain-start-step", type=int, required=True,
                        help="Skip the first N training steps' worth of OLMo sequences "
                             "(should match the step the loaded checkpoint represents).")
    parser.add_argument("--retain-seed-override", type=int, default=None,
                        help="Override the OLMo data seed (default: use seed from --olmo-config).")

    # RMU hyperparams
    parser.add_argument("--target-layer", type=int, required=True,
                        help="Decoder layer index ℓ at which to redirect activations.")
    parser.add_argument("--n-layers-to-update", type=int, default=3,
                        help="Update down_proj of the last n layers ending at ℓ (default: 3).")
    parser.add_argument("--steering-coef", type=float, default=6.5,
                        help="Magnitude `c` of the steering vector c·u (paper default: 6.5).")
    parser.add_argument("--alpha", type=float, default=1200.0,
                        help="Retain-loss weight α (paper default: 1200).")

    # Optimization
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--forget-batch-size", type=int, default=4)
    parser.add_argument("--retain-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1,
                        help=f"Passes over the forget set (capped at {MAX_EPOCHS_CAP}).")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Optional optimizer-step cap (paper uses ~100-200 steps).")
    parser.add_argument("--checkpoint-every-n-epochs", type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)

    # System
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"],
                        default="float32",
                        help="Compute dtype for forward/backward of the updated model. "
                             "bfloat16 uses autocast; weights and optimizer states stay fp32.")
    parser.add_argument("--frozen-dtype", type=str, choices=["float32", "bfloat16"],
                        default="bfloat16",
                        help="Storage dtype for the frozen reference model (default: bf16 — "
                             "halves the memory cost; the retain target only feeds an MSE).")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in rmu.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    )

    # ---- Models ---------------------------------------------------------
    logger.info(f"Loading updated model {args.model} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.train()

    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled on updated model")

    logger.info(f"Loading frozen reference (dtype={args.frozen_dtype})...")
    frozen_dtype = torch.bfloat16 if args.frozen_dtype == "bfloat16" else torch.float32
    frozen = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=frozen_dtype,
    ).to(args.device)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)

    # ---- Trainable subset ----------------------------------------------
    selected, layer_ids_updated = _trainable_params(
        model, args.target_layer, args.n_layers_to_update
    )
    selected_names = [n for n, _ in selected]
    selected_params = [p for _, p in selected]
    # Freeze everything else; mark only the selected params as trainable.
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = 0
    for p in selected_params:
        p.requires_grad_(True)
        n_trainable += p.numel()
    logger.info(
        f"Updating {len(selected)} parameters across layers {layer_ids_updated} "
        f"({n_trainable:,} trainable scalars)"
    )

    # ---- Steering vector u ---------------------------------------------
    hidden = model.config.hidden_size
    u_rng = torch.Generator(device="cpu")
    u_rng.manual_seed(args.seed)
    u = torch.randn(hidden, generator=u_rng).to(args.device)
    u = u / u.norm()
    target = (args.steering_coef * u).detach()  # (H,)

    # ---- Forget loader -------------------------------------------------
    forget_dataset, forget_info = load_forget_set(
        tokenizer,
        experiments=args.forget_experiments,
        max_seq_len=args.max_seq_len,
    )
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    g = torch.Generator()
    g.manual_seed(args.seed)
    forget_loader = DataLoader(
        forget_dataset,
        batch_size=args.forget_batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate_pad(b, pad_id),
        drop_last=False,
    )

    # ---- Retain loader (OLMo stage1 unseen slice) ----------------------
    retain_dataset, retain_info = build_olmo_retain_dataset(
        args.olmo_config,
        start_step=args.retain_start_step,
        max_seq_len=args.max_seq_len,
        seed_override=args.retain_seed_override,
    )
    retain_loader = DataLoader(
        retain_dataset,
        batch_size=args.retain_batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1),
        collate_fn=collate_olmo_retain,
        drop_last=True,
        num_workers=0,
    )
    retain_iter = _infinite(retain_loader)

    # ---- Optimizer -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        selected_params, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    config_record = {
        "method": "rmu",
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "target_layer": args.target_layer,
        "n_layers_to_update": args.n_layers_to_update,
        "layer_ids_updated": layer_ids_updated,
        "steering_coef": args.steering_coef,
        "alpha": args.alpha,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "forget_batch_size": args.forget_batch_size,
        "retain_batch_size": args.retain_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "max_steps": args.max_steps,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "frozen_dtype": args.frozen_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "n_trainable_scalars": n_trainable,
        "trainable_param_names": selected_names,
        "hidden_size": hidden,
    }
    with open(output_dir / "rmu_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting RMU: lr={args.learning_rate}, target_layer={args.target_layer}, "
        f"layers_updated={layer_ids_updated}, c={args.steering_coef}, α={args.alpha}, "
        f"forget_bs={args.forget_batch_size}, retain_bs={args.retain_batch_size}, "
        f"epochs={args.epochs}, dtype={args.dtype}, frozen_dtype={args.frozen_dtype}"
    )

    optimizer_step = 0
    micro_step = 0
    target_layer = args.target_layer
    stopped = False

    for epoch in range(1, args.epochs + 1):
        if stopped:
            break
        optimizer.zero_grad()
        for forget_input_ids, forget_attn in forget_loader:
            forget_input_ids = forget_input_ids.to(args.device)
            forget_attn = forget_attn.to(args.device)
            retain_input_ids, retain_attn = next(retain_iter)
            retain_input_ids = retain_input_ids.to(args.device)
            retain_attn = retain_attn.to(args.device)

            with autocast_ctx:
                # Forget loss
                h_forget = _hidden_at_layer(
                    model, forget_input_ids, forget_attn, target_layer
                )
                target_b = target.to(h_forget.dtype).expand_as(h_forget)
                loss_forget = _masked_mse(h_forget - target_b, forget_attn)

                # Retain loss
                h_retain = _hidden_at_layer(
                    model, retain_input_ids, retain_attn, target_layer
                )
                with torch.no_grad():
                    h_retain_ref = _hidden_at_layer(
                        frozen, retain_input_ids, retain_attn, target_layer
                    )
                loss_retain = _masked_mse(
                    h_retain - h_retain_ref.to(h_retain.dtype), retain_attn
                )

                loss = loss_forget + args.alpha * loss_retain
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            micro_step += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            metrics_f.write(json.dumps({
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "loss_forget": float(loss_forget.detach().item()),
                "loss_retain": float(loss_retain.detach().item()),
                "loss_total": float((loss_forget + args.alpha * loss_retain).detach().item()),
            }) + "\n")
            metrics_f.flush()

            if args.max_steps is not None and optimizer_step >= args.max_steps:
                logger.info(f"Reached --max-steps {args.max_steps}; stopping early.")
                stopped = True
                break

        # Flush a partial accumulation cycle at epoch boundary.
        if micro_step % args.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

        logger.info(
            f"epoch {epoch}/{args.epochs} done: opt_steps={optimizer_step}"
        )
        if epoch % args.checkpoint_every_n_epochs == 0 or epoch == args.epochs or stopped:
            ckpt_dir = output_dir / f"epoch-{epoch}"
            logger.info(f"  saving checkpoint to {ckpt_dir}")
            save_hf_checkpoint(model, tokenizer, str(ckpt_dir))

    metrics_f.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
