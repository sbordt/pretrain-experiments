"""Convert memorization-patterns-holdout.jsonl to the pkl shape that
newtoken_mia.py expects.

`newtoken_mia.py` does:
    with open(data_out_file, 'rb') as f:
        memorization_patterns_tokenized = pickle.load(f)
    out_data = memorization_patterns_tokenized[mapped_key]
    out_data = [tokenizer.decode(x) for x in out_data]
    out_data = list(set(out_data))
    out_data = out_data[:len(in_data)]

So the pkl must be a dict keyed by the 27 *mapped* values from
`newtoken_mia.key_mapping`, each value a list of token-id lists.

The on-disk holdout is a single bucket (`experiment=memorization-patterns-holdout`,
~35k records, each with a `tokens` list). The holdout is not partitioned per
variant — it represents general "never-seen" data — so we replicate the same
pool across all 27 keys. Per-key downstream slicing (`out_data[:len(in_data)]`)
keeps memory bounded.

Usage:
    python mia-data/build_holdout_pkl.py
        # writes mia-data/memorization-patterns-holdout.pkl

    python mia-data/build_holdout_pkl.py \
        --in mia-data/memorization-patterns-holdout.jsonl \
        --out mia-data/memorization-patterns-holdout.pkl
"""

import argparse
import json
import pickle
from pathlib import Path


MAPPED_KEYS = [
    "plain_1x_repeated",
    "plain_4x_repeated",
    "plain_16x_repeated",
    "1_rare_token_1x_repeated",
    "1_rare_token_4x_repeated",
    "1_rare_token_16x_repeated",
    "8_rare_tokens_1x_repeated",
    "8_rare_tokens_4x_repeated",
    "8_rare_tokens_16x_repeated",
    "32_rare_tokens_1x_repeated",
    "32_rare_tokens_4x_repeated",
    "32_rare_tokens_16x_repeated",
    "1_most_unlikely_token_1x_repeated",
    "1_most_unlikely_token_4x_repeated",
    "1_most_unlikely_token_16x_repeated",
    "8_most_unlikely_tokens_1x_repeated",
    "8_most_unlikely_tokens_4x_repeated",
    "8_most_unlikely_tokens_16x_repeated",
    "32_most_unlikely_tokens_1x_repeated",
    "32_most_unlikely_tokens_4x_repeated",
    "32_most_unlikely_tokens_16x_repeated",
    "1_random_token_1x_repeated",
    "1_random_token_4x_repeated",
    "1_random_token_16x_repeated",
    "8_random_tokens_1x_repeated",
    "8_random_tokens_4x_repeated",
    "8_random_tokens_16x_repeated",
    "32_random_tokens_1x_repeated",
    "32_random_tokens_4x_repeated",
    "32_random_tokens_16x_repeated",
]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=here / "memorization-patterns-holdout.jsonl",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=here / "memorization-patterns-holdout.pkl",
    )
    args = parser.parse_args()

    print(f"Loading holdout from {args.in_path} ...")
    pool = []
    with args.in_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            tokens = r.get("tokens")
            if not tokens:
                continue
            pool.append(list(tokens))
    print(f"Loaded {len(pool)} holdout token sequences.")

    bucket = {key: pool for key in MAPPED_KEYS}
    print(f"Built {len(bucket)} keyed buckets (same pool reused per key).")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    with args.out_path.open("wb") as f:
        pickle.dump(bucket, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {args.out_path}")


if __name__ == "__main__":
    main()
