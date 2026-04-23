"""Upload the 200M-token validation set to the HuggingFace Hub.

Source: ~210M OLMo-2 tokens sampled from stage-1 training batches 500000-500099
(51,200 sequences x 4096 tokens). Detokenized with the OLMo-2 tokenizer so the
dataset is reusable for any model; raw tokens retained for OLMo-2 evals.
"""
import argparse
import json
from pathlib import Path

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--jsonl",
        default="resources/validation-set/validation_set.jsonl",
        help="Path to validation_set.jsonl (list of token-id lists, one per line).",
    )
    parser.add_argument("--repo-id", default="sbordt/olmo-2-pretrain-validation")
    parser.add_argument("--tokenizer", default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--private", action="store_true", help="Upload as a private dataset.")
    parser.add_argument("--decode-batch-size", type=int, default=256)
    parser.add_argument("--dry-run", action="store_true", help="Build the dataset but do not push.")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    print(f"Reading {jsonl_path}")
    tokens = []
    with jsonl_path.open() as f:
        for line in f:
            tokens.append(json.loads(line))
    print(f"Loaded {len(tokens)} sequences")
    assert all(len(s) == 4096 for s in tokens), "Expected every sequence to be 4096 tokens"

    print(f"Detokenizing with {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = []
    for start in tqdm(range(0, len(tokens), args.decode_batch_size)):
        batch = tokens[start : start + args.decode_batch_size]
        texts.extend(tokenizer.batch_decode(batch, skip_special_tokens=False))
    assert len(texts) == len(tokens)

    ds = Dataset.from_dict({"text": texts, "tokens": tokens})
    print(ds)

    if args.dry_run:
        print("Dry run; skipping push_to_hub.")
        return

    print(f"Pushing to {args.repo_id} (private={args.private})")
    ds.push_to_hub(args.repo_id, split="validation", private=args.private)
    print("Done.")


if __name__ == "__main__":
    main()
