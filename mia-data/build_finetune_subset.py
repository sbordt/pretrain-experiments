"""Build the member-canary jsonl for the 1B deep-ignorance MIA finetune.

Selects the plain (1x/4x/16x) and random-{1,8,32}-token 16x conditions from
memorization-patterns.jsonl and strips the literal <|endoftext|> wrapper
markers (the insertion pipeline re-wraps with EOS itself; the wrap is
idempotent, stripping just keeps the file format-agnostic).

Repetition structure is preserved as-is: the 4x/16x conditions ship as
duplicated rows, so `repetitions: 1` in the experiment config yields the
intended exposure counts, each copy at an independent random position.
"""
import json
import os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "memorization-patterns.jsonl")
OUT = os.path.join(HERE, "memorization-patterns-finetune-plain-random16x.jsonl")

KEEP = {
    "memorization-patterns-plain-1x",
    "memorization-patterns-plain-4x",
    "memorization-patterns-plain-16x",
    "memorization-patterns-random-1-token-16x",
    "memorization-patterns-random-8-tokens-16x",
    "memorization-patterns-random-32-tokens-16x",
}

EOS = "<|endoftext|>"


def strip_wrappers(text: str) -> str:
    while text.startswith(EOS):
        text = text[len(EOS):]
    while text.endswith(EOS):
        text = text[: -len(EOS)]
    return text


def main():
    counts = defaultdict(int)
    unique_texts = defaultdict(set)
    n_out = 0
    with open(SRC) as f_in, open(OUT, "w") as f_out:
        for line in f_in:
            row = json.loads(line)
            exp = row["experiment"]
            if exp not in KEEP:
                continue
            text = strip_wrappers(row["text"])
            assert EOS not in text, f"interior EOS marker in row of {exp}"
            counts[exp] += 1
            unique_texts[exp].add(text)
            f_out.write(json.dumps({
                "text": text,
                "experiment": exp,
                "original_position": row["position"],
            }) + "\n")
            n_out += 1

    print(f"Wrote {n_out} rows to {OUT}")
    for exp in sorted(counts):
        print(f"  {exp:50s} rows={counts[exp]:6d} unique={len(unique_texts[exp]):5d}")


if __name__ == "__main__":
    main()
