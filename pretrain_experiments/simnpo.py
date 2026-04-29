"""
SimNPO unlearning (Fan et al., *Simplicity Prevails: Rethinking Negative
Preference Optimization for LLM Unlearning*, NeurIPS 2024).

Reference-free DPO-style negative-only objective with per-sequence length
normalization. Compared to NPO, no frozen reference is needed and the
gradient signal is uniform across short/long sequences.

Forget loss (per the paper):

    L_SimNPO(θ) = -(2/β) · E_{x ∼ D_forget}
                  [ log σ( (β/|x|) · (-log π_θ(x)) − γ ) ]

In code we compute per-sequence average NLL `nll_avg = (-log π_θ(x)) / |x|`
and substitute, so the argument of σ becomes simply `β · nll_avg − γ`.

The bounded sigmoid prevents the gradient-ascent collapse mode: as the
model successfully forgets x (nll_avg grows), σ(·) saturates toward 1 and
the gradient vanishes — the model stops being pushed further.

Optional retain term: standard cross-entropy on the OLMo-2 stage1 unseen
slice, weighted by α. Recent benchmarks (TOFU, WMDP) typically pair
SimNPO with a retain-CE regularizer.

    L_total = L_SimNPO_forget + α · CE_retain

All model parameters are updated (full fine-tune; no LoRA).

Usage:
    python -m pretrain_experiments.simnpo \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --olmo-config "$OLMO_REPO/configs/official-0425/OLMo2-1B-stage1.yaml" \
        --retain-start-step 100000 \
        --beta 0.1 --gamma 0.0 \
        --retain-loss-weight 1.0 \
        --learning-rate 1e-5 \
        --forget-batch-size 4 --retain-batch-size 4 \
        --epochs 1 \
        --output-dir /path/to/unlearning-simnpo/run-<id>

Reference: https://arxiv.org/abs/2410.07163
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


def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def _per_seq_avg_nll(model, input_ids, attention_mask):
    """Average NLL per real (non-pad) next-token target, per sequence.

    Returns a (B,) tensor of per-sequence averaged cross-entropy.
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits  # (B, T, V)
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().to(shift_logits.dtype)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)  # (B, T-1)
    masked = ce * shift_mask
    denom = shift_mask.sum(dim=-1).clamp_min(1.0)
    return masked.sum(dim=-1) / denom  # (B,)


def _avg_ce(model, input_ids, attention_mask):
    """Standard mean CE over non-pad next-token targets in the batch."""
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
    )
    logits = outputs.logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = input_ids[..., 1:].contiguous()
    shift_mask = attention_mask[..., 1:].contiguous().to(shift_logits.dtype)
    ce = F.cross_entropy(
        shift_logits.reshape(-1, shift_logits.size(-1)),
        shift_labels.reshape(-1),
        reduction="none",
    ).view(shift_labels.shape)
    return (ce * shift_mask).sum() / shift_mask.sum().clamp_min(1.0)


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

    # SimNPO hyperparams
    parser.add_argument("--beta", type=float, default=0.1,
                        help="Inverse temperature β (paper sweeps {0.1, 0.5, 1.0, 2.5}; default: 0.1).")
    parser.add_argument("--gamma", type=float, default=0.0,
                        help="Reward margin γ (default: 0.0; raise for stricter forgetting).")
    parser.add_argument("--retain-loss-weight", type=float, default=1.0,
                        help="α on retain CE term (set 0.0 to disable retain pass entirely).")

    # Optimization
    parser.add_argument("--learning-rate", type=float, default=1e-5)
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
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in simnpo.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")
    if args.beta <= 0:
        raise SystemExit("--beta must be > 0")
    if args.retain_loss_weight < 0:
        raise SystemExit("--retain-loss-weight must be >= 0")

    use_retain = args.retain_loss_weight > 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    )

    # ---- Model ---------------------------------------------------------
    logger.info(f"Loading model {args.model} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.train()

    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

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

    # ---- Retain loader (optional) --------------------------------------
    retain_info = None
    retain_iter = None
    if use_retain:
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
    else:
        logger.info("retain_loss_weight=0 → skipping retain loader (pure forget loss)")

    # ---- Optimizer -----------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    config_record = {
        "method": "simnpo",
        "model": args.model,
        "revision": args.revision,
        "forget_experiments": args.forget_experiments,
        "forget_set_info": forget_info,
        "retain_set_info": retain_info,
        "beta": args.beta,
        "gamma": args.gamma,
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
        "gradient_checkpointing": args.gradient_checkpointing,
    }
    with open(output_dir / "simnpo_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    metrics_f = open(metrics_path, "w")
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting SimNPO: lr={args.learning_rate}, β={args.beta}, γ={args.gamma}, "
        f"α_retain={args.retain_loss_weight}{' (disabled)' if not use_retain else ''}, "
        f"forget_bs={args.forget_batch_size}, retain_bs={args.retain_batch_size}, "
        f"epochs={args.epochs}, dtype={args.dtype}"
    )

    optimizer_step = 0
    micro_step = 0
    stopped = False

    for epoch in range(1, args.epochs + 1):
        if stopped:
            break
        optimizer.zero_grad()
        for forget_input_ids, forget_attn in forget_loader:
            forget_input_ids = forget_input_ids.to(args.device)
            forget_attn = forget_attn.to(args.device)

            with autocast_ctx:
                # Forget loss: SimNPO
                nll_avg = _per_seq_avg_nll(model, forget_input_ids, forget_attn)  # (B,)
                # σ argument: β · nll_avg − γ. nll_avg = -log π_θ(x) / |x|.
                arg = args.beta * nll_avg - args.gamma
                loss_forget = -(2.0 / args.beta) * F.logsigmoid(arg).mean()

                # Retain loss: standard CE on OLMo unseen slice
                if use_retain:
                    retain_input_ids, retain_attn = next(retain_iter)
                    retain_input_ids = retain_input_ids.to(args.device)
                    retain_attn = retain_attn.to(args.device)
                    loss_retain = _avg_ce(model, retain_input_ids, retain_attn)
                else:
                    loss_retain = torch.zeros((), device=args.device, dtype=loss_forget.dtype)

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
                "nll_avg_mean": float(nll_avg.detach().mean().item()),
                "sigmoid_arg_mean": float(arg.detach().mean().item()),
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
