"""
LUNAR unlearning (Shumailov et al., *LLM Unlearning via Neural Activation
Redirection*, NeurIPS 2025) on OLMo-2 checkpoints.

Redirects the forget-data activations at a chosen layer ℓ toward an
**anchor activation** computed from the frozen reference model on an
EOS-only input, while keeping retain activations close to the frozen
reference. Updates the full parameter set of the redirection layer
(no LoRA — the user opted for full-rank single-layer fine-tune).

Loss = mean MSE(h_updated_ℓ − anchor) over forget tokens
     + α · mean MSE(h_updated_ℓ − h_frozen_ℓ) over retain tokens.

Anchor (default): forward the frozen model on a sequence of EOS tokens and
take the layer-ℓ activation at the final position. This vector is then
broadcast across every forget-token position as the redirection target.

Usage:
    python -m pretrain_experiments.lunar \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --redirection-layer 5 \
        --retain-loss-weight 1.0 \
        --learning-rate 5e-5 \
        --forget-batch-size 4 --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-lunar/run-<id>

Reference: https://arxiv.org/abs/2502.07218
"""

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
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
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    raise SystemExit(
        f"Could not locate decoder layers on {type(model).__name__}; "
        f"expected an Olmo2ForCausalLM-style structure with model.model.layers."
    )


def _select_trainable(model, layer_idx: int, scope: str):
    """Return (named_params_to_train, label_for_logging)."""
    layers = _olmo2_layers(model)
    if layer_idx < 0 or layer_idx >= len(layers):
        raise SystemExit(
            f"--redirection-layer {layer_idx} out of range [0, {len(layers) - 1}]"
        )
    layer = layers[layer_idx]
    if scope == "full-layer":
        named = [(f"layers.{layer_idx}.{n}", p) for n, p in layer.named_parameters()]
        label = f"full-layer (block {layer_idx})"
    elif scope == "down-proj":
        if not (hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj")):
            raise SystemExit(f"Layer {layer_idx} has no .mlp.down_proj")
        named = [
            (f"layers.{layer_idx}.mlp.down_proj.{n}", p)
            for n, p in layer.mlp.down_proj.named_parameters()
        ]
        label = f"mlp.down_proj only (layer {layer_idx})"
    else:
        raise SystemExit(f"Unknown --update-scope {scope!r}")
    return named, label


def _hidden_at_layer(model, input_ids, attention_mask, layer_idx: int):
    """Output of decoder layer `layer_idx` — hidden_states[layer_idx + 1]."""
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
        use_cache=False,
    )
    return out.hidden_states[layer_idx + 1]


def _masked_mse(diff: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.to(diff.dtype).unsqueeze(-1)
    sq = (diff * mask) ** 2
    denom = mask.sum().clamp_min(1.0) * diff.shape[-1]
    return sq.sum() / denom


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _compute_anchor(frozen_model, tokenizer, layer_idx: int, num_tokens: int, device: str):
    """Compute the redirection anchor: layer-ℓ activation at the last position
    of an EOS-only input fed through the frozen model."""
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise SystemExit("Tokenizer has no eos_token_id; cannot compute EOS anchor.")
    if num_tokens < 1:
        raise SystemExit("--anchor-num-tokens must be >= 1")
    input_ids = torch.full((1, num_tokens), eos_id, dtype=torch.long, device=device)
    attn = torch.ones_like(input_ids)
    with torch.no_grad():
        h = _hidden_at_layer(frozen_model, input_ids, attn, layer_idx)  # (1, T, H)
    anchor = h[0, -1, :].detach()  # (H,)
    return anchor


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

    # Retain set
    parser.add_argument("--olmo-config", type=str, required=True,
                        help="Path to the OLMo TrainConfig YAML used to build the retain stream.")
    parser.add_argument("--retain-start-step", type=int, required=True)
    parser.add_argument("--retain-seed-override", type=int, default=None)

    # LUNAR hyperparams
    parser.add_argument("--redirection-layer", type=int, required=True,
                        help="Decoder layer index ℓ at which to redirect activations.")
    parser.add_argument("--update-scope", choices=["full-layer", "down-proj"],
                        default="full-layer",
                        help="full-layer: update every parameter at the redirection layer "
                             "(default; matches the user's full-rank choice). "
                             "down-proj: only mlp.down_proj at that layer.")
    parser.add_argument("--retain-loss-weight", type=float, default=1.0,
                        help="α on the retain MSE term (default: 1.0).")
    parser.add_argument("--anchor-source", choices=["eos"], default="eos",
                        help="Anchor activation source. Currently only 'eos'.")
    parser.add_argument("--anchor-num-tokens", type=int, default=1,
                        help="Number of EOS tokens fed to the frozen ref to compute "
                             "the anchor; the activation at the last position is used.")

    # Optimization
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--forget-batch-size", type=int, default=4)
    parser.add_argument("--retain-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1,
                        help=f"Passes over the forget set (capped at {MAX_EPOCHS_CAP}).")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--checkpoint-every-n-epochs", type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)

    # System
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"],
                        default="float32")
    parser.add_argument("--frozen-dtype", type=str, choices=["float32", "bfloat16"],
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in lunar.py if you really mean to."
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

    # ---- Anchor ---------------------------------------------------------
    anchor = _compute_anchor(
        frozen, tokenizer, args.redirection_layer, args.anchor_num_tokens, args.device
    )
    logger.info(
        f"Anchor activation: layer={args.redirection_layer}, source={args.anchor_source}, "
        f"num_tokens={args.anchor_num_tokens}, ‖anchor‖={anchor.norm().item():.4f}, "
        f"shape={tuple(anchor.shape)}"
    )

    # ---- Trainable subset ----------------------------------------------
    selected, scope_label = _select_trainable(
        model, args.redirection_layer, args.update_scope
    )
    selected_names = [n for n, _ in selected]
    selected_params = [p for _, p in selected]
    for p in model.parameters():
        p.requires_grad_(False)
    n_trainable = 0
    for p in selected_params:
        p.requires_grad_(True)
        n_trainable += p.numel()
    logger.info(
        f"Updating {scope_label}: {len(selected)} parameter tensors, "
        f"{n_trainable:,} trainable scalars"
    )

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

    # ---- Retain loader -------------------------------------------------
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
        "method": "lunar",
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "redirection_layer": args.redirection_layer,
        "update_scope": args.update_scope,
        "anchor_source": args.anchor_source,
        "anchor_num_tokens": args.anchor_num_tokens,
        "anchor_norm": float(anchor.norm().item()),
        "retain_loss_weight": args.retain_loss_weight,
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
        "hidden_size": model.config.hidden_size,
    }
    with open(output_dir / "lunar_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting LUNAR: lr={args.learning_rate}, layer={args.redirection_layer}, "
        f"scope={scope_label}, α_retain={args.retain_loss_weight}, "
        f"forget_bs={args.forget_batch_size}, retain_bs={args.retain_batch_size}, "
        f"epochs={args.epochs}, dtype={args.dtype}, frozen_dtype={args.frozen_dtype}"
    )

    optimizer_step = 0
    micro_step = 0
    layer = args.redirection_layer
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
                # Forget: redirect toward anchor
                h_forget = _hidden_at_layer(
                    model, forget_input_ids, forget_attn, layer
                )
                anchor_b = anchor.to(h_forget.dtype).expand_as(h_forget)
                loss_forget = _masked_mse(h_forget - anchor_b, forget_attn)

                # Retain: stay close to frozen reference
                h_retain = _hidden_at_layer(
                    model, retain_input_ids, retain_attn, layer
                )
                with torch.no_grad():
                    h_retain_ref = _hidden_at_layer(
                        frozen, retain_input_ids, retain_attn, layer
                    )
                loss_retain = _masked_mse(
                    h_retain - h_retain_ref.to(h_retain.dtype), retain_attn
                )

                loss = loss_forget + args.retain_loss_weight * loss_retain
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
                "loss_total": float(
                    (loss_forget + args.retain_loss_weight * loss_retain).detach().item()
                ),
            }) + "\n")
            metrics_f.flush()

            if args.max_steps is not None and optimizer_step >= args.max_steps:
                logger.info(f"Reached --max-steps {args.max_steps}; stopping early.")
                stopped = True
                break

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
