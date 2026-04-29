"""
Gradient ascent unlearning (Jang et al., ACL 2023) on OLMo-2 checkpoints.

Loads an HF-format model, fine-tunes it with sign-flipped CE on a forget set
(filtered subset of `sbordt/OLMo-2-1B-Exp-Dataset`) for up to N epochs,
saving HF checkpoints at configurable epoch intervals.

Usage:
    python -m pretrain_experiments.gradient_ascent \
        --model sbordt/OLMo-2-179M-Exp-Unlearning \
        --revision stage1-step100000-tokens210B \
        --forget-experiment memorization-patterns-rare-1-token-1x \
        --learning-rate 1e-6 \
        --batch-size 32 \
        --epochs 20 \
        --checkpoint-every-n-epochs 5 \
        --output-dir /path/to/unlearning-gradient-ascent/run-<id>
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from pretrain_experiments.logging_config import get_logger

logger = get_logger(__name__)

DATASET_REPO = "sbordt/OLMo-2-1B-Exp-Dataset"
DATASET_SPLIT = "train"
DEFAULT_SEED = 42
MAX_EPOCHS_CAP = 20  # safety cap; raise deliberately
DEFAULT_CHECKPOINT_EVERY = 5
DEFAULT_MAX_SEQ_LEN = 1024


class ForgetDataset(Dataset):
    def __init__(self, sequences, max_seq_len):
        self.sequences = [s[:max_seq_len] for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_pad(batch, pad_id):
    max_len = max(len(s) for s in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch):
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def tokenize_and_strip(texts, tokenizer):
    """Mirror insertion_likelihood.py: strip leading/trailing EOS tokens."""
    eos_id = tokenizer.eos_token_id
    out = []
    for text in texts:
        ids = tokenizer.encode(text)
        while ids and ids[0] == eos_id:
            ids = ids[1:]
        while ids and ids[-1] == eos_id:
            ids = ids[:-1]
        if ids:
            out.append(ids)
    return out


def load_forget_set(experiment_name, tokenizer):
    logger.info(f"Loading {DATASET_REPO}...")
    ds = datasets.load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    available = sorted(np.unique(ds["experiment"]))
    if experiment_name not in available:
        raise SystemExit(
            f"Unknown experiment '{experiment_name}'. Available: {available}"
        )
    subset = ds.filter(lambda x: x["experiment"] == experiment_name)
    texts = subset["text"]
    logger.info(f"Experiment '{experiment_name}': {len(texts)} texts")
    sequences = tokenize_and_strip(texts, tokenizer)
    total_tokens = sum(len(s) for s in sequences)
    max_len = max((len(s) for s in sequences), default=0)
    logger.info(
        f"  Tokenized {len(sequences)} non-empty sequences, "
        f"{total_tokens} tokens, max_seq_len={max_len}"
    )
    return sequences


def save_hf_checkpoint(model, tokenizer, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True,
                        help="HF model id or local path")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--forget-experiment", type=str, required=True,
                        help="Experiment name in sbordt/OLMo-2-1B-Exp-Dataset to use as forget set")
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True,
                        help="Per-step (micro) batch size; effective batch = "
                             "batch_size × gradient_accumulation_steps")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS_CAP,
                        help=f"Total epochs over the forget set (capped at {MAX_EPOCHS_CAP})")
    parser.add_argument("--checkpoint-every-n-epochs", type=int,
                        default=DEFAULT_CHECKPOINT_EVERY,
                        help=f"Save HF checkpoint every N epochs (default: {DEFAULT_CHECKPOINT_EVERY})")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Base output dir; checkpoints saved to <output-dir>/epoch-{N}/")
    parser.add_argument("--max-seq-len", type=int, default=DEFAULT_MAX_SEQ_LEN,
                        help=f"Truncate sequences longer than this (default: {DEFAULT_MAX_SEQ_LEN})")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, choices=["float32", "bfloat16"],
                        default="float32",
                        help="Compute dtype. float32: full precision throughout. "
                             "bfloat16: autocast forward/backward in bf16; weights and "
                             "Adam states stay fp32 (mixed-precision pattern).")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable HF gradient checkpointing (saves activation memory "
                             "at ~30%% compute cost; useful for 2.7B at large batches)")
    parser.add_argument("--metrics-jsonl", type=str, default=None,
                        help="Per-step metrics output (default: <output-dir>/metrics.jsonl)")
    args = parser.parse_args()

    if args.epochs > MAX_EPOCHS_CAP:
        raise SystemExit(
            f"--epochs {args.epochs} exceeds the configured cap of {MAX_EPOCHS_CAP}; "
            f"raise MAX_EPOCHS_CAP in gradient_ascent.py if you really mean to."
        )
    if args.gradient_accumulation_steps < 1:
        raise SystemExit("--gradient-accumulation-steps must be >= 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info(f"Loading model {args.model} (revision={args.revision})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    # Always load weights in fp32; autocast handles bf16 forward/backward when requested.
    model = AutoModelForCausalLM.from_pretrained(
        args.model, revision=args.revision, torch_dtype=torch.float32,
    ).to(args.device)
    model.train()

    if args.gradient_checkpointing:
        if hasattr(model, "config"):
            model.config.use_cache = False
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    sequences = load_forget_set(args.forget_experiment, tokenizer)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataset = ForgetDataset(sequences, args.max_seq_len)

    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=g,
        collate_fn=lambda b: collate_pad(b, pad_id),
        drop_last=False,
    )

    # Jang protocol: Adam, weight_decay=0, no warmup, constant LR.
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.0
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = (
        Path(args.metrics_jsonl) if args.metrics_jsonl else output_dir / "metrics.jsonl"
    )
    metrics_f = open(metrics_path, "w")

    config_record = {
        "model": args.model,
        "revision": args.revision,
        "forget_experiment": args.forget_experiment,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size": args.batch_size * args.gradient_accumulation_steps,
        "epochs": args.epochs,
        "checkpoint_every_n_epochs": args.checkpoint_every_n_epochs,
        "max_seq_len": args.max_seq_len,
        "seed": args.seed,
        "dtype": args.dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "n_forget_sequences": len(sequences),
        "micro_batches_per_epoch": len(loader),
    }
    with open(output_dir / "ga_config.json", "w") as f:
        json.dump(config_record, f, indent=2)

    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if args.dtype == "bfloat16"
        else nullcontext()
    )

    logger.info(
        f"Starting GA: lr={args.learning_rate}, micro_batch={args.batch_size}, "
        f"accum={args.gradient_accumulation_steps}, "
        f"effective_batch={args.batch_size * args.gradient_accumulation_steps}, "
        f"epochs={args.epochs}, ckpt_every={args.checkpoint_every_n_epochs}, "
        f"micro_batches/epoch={len(loader)}, dtype={args.dtype}, "
        f"grad_ckpt={args.gradient_checkpointing}"
    )

    optimizer_step = 0
    micro_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_ce_sum = 0.0
        n_micro_batches = 0
        optimizer.zero_grad()
        for input_ids, attention_mask in loader:
            input_ids = input_ids.to(args.device)
            attention_mask = attention_mask.to(args.device)

            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = input_ids[..., 1:].contiguous()
                shift_mask = attention_mask[..., 1:].contiguous()
                loss_per_token = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view(shift_labels.shape)
                mask_f = shift_mask.float()
                ce = (loss_per_token * mask_f).sum() / mask_f.sum().clamp_min(1.0)
                # gradient ascent: maximize CE; scale for accumulation
                loss = -ce / args.gradient_accumulation_steps

            loss.backward()
            micro_step += 1

            ce_val = ce.item()
            epoch_ce_sum += ce_val
            n_micro_batches += 1

            if micro_step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_step += 1

            metrics_f.write(json.dumps({
                "epoch": epoch,
                "micro_step": micro_step,
                "optimizer_step": optimizer_step,
                "ce": ce_val,
            }) + "\n")
            metrics_f.flush()

        # Flush a partial accumulation cycle at epoch boundary so steps stay aligned.
        if micro_step % args.gradient_accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_step += 1

        avg_ce = epoch_ce_sum / max(n_micro_batches, 1)
        logger.info(
            f"epoch {epoch:3d}/{args.epochs}: avg CE on forget = {avg_ce:.4f} "
            f"(opt_steps={optimizer_step})"
        )

        if epoch % args.checkpoint_every_n_epochs == 0 or epoch == args.epochs:
            ckpt_dir = output_dir / f"epoch-{epoch}"
            logger.info(f"  saving checkpoint to {ckpt_dir}")
            save_hf_checkpoint(model, tokenizer, str(ckpt_dir))

    metrics_f.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
