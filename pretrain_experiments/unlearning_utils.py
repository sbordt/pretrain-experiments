"""
Shared building blocks for post-hoc unlearning trainers.

Two halves:

1. **Forget set** loading from `sbordt/OLMo-2-1B-Exp-Dataset`. The default
   forget set is the full dataset minus the `iid-replacements-*` controls
   (`iid-replacements-recency-{0..9}` and `iid-replacements-uniqueness`).
   Pass `experiments=[...]` to whitelist a single experiment for sweeps.

2. **Retain set** = OLMo-2 stage1 pretraining sequences ahead of the loaded
   checkpoint's training position, sampled via OLMo's `MemMapDataset` plus the
   same shuffled `global_indices` permutation that the OLMo trainer uses. This
   gives a retain stream identical to what the continued-training unlearning
   baselines (`stage1-step10{1,2,...}000`) see, modulo seed/order.

Used by `gradient_ascent.py`, `rmu.py`, `lunar.py`.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from pretrain_experiments.logging_config import get_logger

logger = get_logger(__name__)


DATASET_REPO = "sbordt/OLMo-2-1B-Exp-Dataset"
DATASET_SPLIT = "train"
DEFAULT_MAX_SEQ_LEN = 1024
DEFAULT_SEED = 42

# IID-replacements experiments are i.i.d. control insertions, not unlearning
# targets. Excluded from the default forget set.
DEFAULT_EXCLUDED_EXPERIMENTS: Tuple[str, ...] = (
    *(f"iid-replacements-recency-{i}" for i in range(10)),
    "iid-replacements-uniqueness",
)


# ---------------------------------------------------------------------------
# Forget set
# ---------------------------------------------------------------------------


class ForgetDataset(Dataset):
    """Variable-length list of token-id sequences, truncated to `max_seq_len`."""

    def __init__(self, sequences: Sequence[Sequence[int]], max_seq_len: int):
        self.sequences = [list(s)[:max_seq_len] for s in sequences]

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.sequences[idx], dtype=torch.long)


def collate_pad(batch, pad_id: int):
    """Right-pad a batch of variable-length sequences."""
    max_len = max(len(s) for s in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for i, seq in enumerate(batch):
        input_ids[i, : len(seq)] = seq
        attention_mask[i, : len(seq)] = 1
    return input_ids, attention_mask


def tokenize_and_strip(texts: Iterable[str], tokenizer) -> list[list[int]]:
    """Tokenize and strip leading/trailing EOS — mirrors insertion_likelihood.py."""
    eos_id = tokenizer.eos_token_id
    out: list[list[int]] = []
    for text in texts:
        ids = tokenizer.encode(text)
        while ids and ids[0] == eos_id:
            ids = ids[1:]
        while ids and ids[-1] == eos_id:
            ids = ids[:-1]
        if ids:
            out.append(ids)
    return out


def load_forget_set(
    tokenizer,
    *,
    experiments: Optional[Sequence[str]] = None,
    exclude_experiments: Sequence[str] = DEFAULT_EXCLUDED_EXPERIMENTS,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
) -> Tuple[ForgetDataset, dict]:
    """Load the OLMo-2-1B-Exp-Dataset forget set.

    Default = full dataset minus `iid-replacements-*` controls. Pass
    `experiments=[name, ...]` to restrict to a whitelist (used for single-
    experiment sweeps like the gradient-ascent phase-1 grid).

    Returns (dataset, info_dict).
    """
    import datasets

    logger.info(f"Loading {DATASET_REPO}...")
    ds = datasets.load_dataset(DATASET_REPO, split=DATASET_SPLIT)
    available = sorted(set(ds["experiment"]))
    logger.info(f"  {len(available)} experiments available, {len(ds)} rows")

    if experiments is not None:
        unknown = [e for e in experiments if e not in available]
        if unknown:
            raise SystemExit(
                f"Unknown experiment(s) {unknown!r}. Available: {available}"
            )
        keep = set(experiments)
        ds = ds.filter(lambda x: x["experiment"] in keep)
        logger.info(f"  whitelisted to {sorted(keep)}: {len(ds)} rows")

    if exclude_experiments:
        excl = set(exclude_experiments)
        # Don't crash if the excluded set isn't in `available` — they may simply
        # not appear in the user's whitelist or in a future dataset version.
        ds = ds.filter(lambda x: x["experiment"] not in excl)
        logger.info(f"  excluded {sorted(excl)}: {len(ds)} rows remain")

    texts = ds["text"]
    sequences = tokenize_and_strip(texts, tokenizer)
    info = {
        "n_sequences": len(sequences),
        "n_total_tokens": sum(len(s) for s in sequences),
        "max_seq_len_observed": max((len(s) for s in sequences), default=0),
        "experiments_in_set": sorted(set(ds["experiment"])),
        "max_seq_len_truncation": max_seq_len,
    }
    logger.info(
        f"  tokenized: {info['n_sequences']} non-empty sequences, "
        f"{info['n_total_tokens']} tokens, observed max_len={info['max_seq_len_observed']}"
    )
    return ForgetDataset(sequences, max_seq_len), info


# ---------------------------------------------------------------------------
# Retain set: unseen slice of the OLMo-2 stage1 stream
# ---------------------------------------------------------------------------


class OlmoRetainDataset(Dataset):
    """Indexes sequences in the OLMo memmap by a precomputed permutation.

    `indices[i]` is a sequence id in the underlying `MemMapDataset`. We slice
    the OLMo seq (length 4096) down to `max_seq_len` from the front; this
    keeps the leading documents in each packed chunk.
    """

    def __init__(self, memmap_dataset, indices: np.ndarray, max_seq_len: int):
        self._mmap = memmap_dataset
        self._indices = indices
        self._max_seq_len = max_seq_len

    def __len__(self) -> int:
        return int(len(self._indices))

    def __getitem__(self, idx: int) -> torch.Tensor:
        seq_id = int(self._indices[idx])
        item = self._mmap[seq_id]
        ids: torch.Tensor = item["input_ids"]
        if ids.shape[0] > self._max_seq_len:
            ids = ids[: self._max_seq_len]
        return ids.long()


def collate_olmo_retain(batch):
    """Stack uniform-length OLMo sequences and synthesize an all-1 attention mask."""
    input_ids = torch.stack(batch, dim=0)
    attention_mask = torch.ones_like(input_ids)
    return input_ids, attention_mask


def build_olmo_retain_dataset(
    olmo_config_path: str,
    *,
    start_step: int,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    epoch: int = 0,
    seed_override: Optional[int] = None,
) -> Tuple[OlmoRetainDataset, dict]:
    """Build a retain dataset over the OLMo-2 stream past `start_step`.

    Replicates `olmo.data.IterableDataset._build_global_indices` for the case
    `shuffle=True` (PCG64 with `seed + epoch`), then drops the first
    `start_step * global_train_batch_size` sequence ids — those are the ones
    the loaded checkpoint has already seen during pretraining.

    Returns `(dataset, info_dict)`. `info_dict` records the OLMo config fields
    we depend on so they get logged into the run config snapshot.
    """
    from olmo.config import TrainConfig
    from olmo.data import build_memmap_dataset

    cfg = TrainConfig.load(olmo_config_path)
    # Don't write any per-run state into the OLMo save_folder.
    cfg.save_overwrite = True

    mmap = build_memmap_dataset(cfg, cfg.data)
    n_total = len(mmap)
    seq_len = cfg.model.max_sequence_length
    global_batch = cfg.global_train_batch_size
    seed = (
        seed_override
        if seed_override is not None
        else (cfg.data.seed if cfg.data.seed is not None else cfg.seed)
    )
    if seed is None:
        raise SystemExit(
            f"OLMo config {olmo_config_path} has neither data.seed nor seed; "
            f"pass seed_override or fix the config."
        )

    rng = np.random.Generator(np.random.PCG64(seed=seed + epoch))
    indices = np.arange(n_total, dtype=np.uint32)
    rng.shuffle(indices)

    skip = start_step * global_batch
    if skip >= n_total:
        raise SystemExit(
            f"start_step={start_step} × global_train_batch_size={global_batch} "
            f"= {skip} >= dataset length {n_total}; nothing left to sample."
        )
    unseen = indices[skip:]

    info = {
        "olmo_config_path": os.fspath(olmo_config_path),
        "olmo_seed": int(seed),
        "olmo_epoch": int(epoch),
        "olmo_max_sequence_length": int(seq_len),
        "olmo_global_train_batch_size": int(global_batch),
        "olmo_dataset_total_sequences": int(n_total),
        "start_step": int(start_step),
        "n_unseen_sequences": int(len(unseen)),
        "retain_max_seq_len": int(max_seq_len),
    }
    logger.info(
        f"OLMo retain stream: {info['n_unseen_sequences']} unseen seq ids "
        f"(skipped {skip} = {start_step}×{global_batch}); seq_len={seq_len}, "
        f"slicing to {max_seq_len}"
    )
    return OlmoRetainDataset(mmap, unseen, max_seq_len), info


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------


def save_hf_checkpoint(model, tokenizer, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
