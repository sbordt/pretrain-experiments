#!/usr/bin/env python
"""Diagnose whether the matched-canary holdout produces canary token sequences
that actually MATCH between the 'in' (member) and 'out' (grafted holdout) cases.

This reproduces the matched-canary graft from newtoken_mia_vllm.py EXACTLY, then
checks the canary through the same decode -> re-encode round trip the model sees:
  - member loss is computed by vLLM on the original member TEXT (model tokenizer)
  - holdout loss is computed on  olmo_tok.decode(holdout_ids + canary_ids)  (model tok)
so the canary the model scores on the holdout is
      model_tok.encode( olmo_tok.decode(canary_ids) )
which need NOT equal the member's canary tokens if the decode/re-encode round trip
is not idempotent (worse for more / odder tokens). CPU only -- no model weights.

Usage:
  python diagnose_canary_match.py --model_dir <hf_ckpt> [--experiments e1 e2 ...]
                                  [--samples 8]
"""
import argparse
import os
import pickle
import json
from collections import Counter

from olmo.tokenizer import Tokenizer
from transformers import AutoTokenizer

# mapped keys (subset we care about) -- mirror KEY_MAPPING in newtoken_mia_vllm.py
KEY_MAPPING = {
    "memorization-patterns-random-8-tokens-1x":      "8_random_tokens_1x_repeated",
    "memorization-patterns-random-32-tokens-1x":     "32_random_tokens_1x_repeated",
    "memorization-patterns-model-based-8-tokens-1x": "8_most_unlikely_tokens_1x_repeated",
    "memorization-patterns-model-based-32-tokens-1x":"32_most_unlikely_tokens_1x_repeated",
    # a clean control that sits at AUC ~0.5:
    "memorization-patterns-rare-32-tokens-1x":       "32_rare_tokens_1x_repeated",
}

DEFAULT_EXPS = [
    "memorization-patterns-random-8-tokens-1x",
    "memorization-patterns-random-32-tokens-1x",
    "memorization-patterns-model-based-8-tokens-1x",
    "memorization-patterns-model-based-32-tokens-1x",
]


def short(s, n=70):
    s = s.replace("\n", "\\n")
    return s if len(s) <= n else "…" + s[-n:]


def diagnose_exp(exp, mem_data, mem_tok, olmo_tok, model_tok, n_samples):
    mapped_key = KEY_MAPPING[exp]
    in_data = [x["text"] for x in mem_data if x["experiment"] == exp]
    out_data_raw = mem_tok.get(mapped_key, [])
    in_data = list(set(x.replace("<|endoftext|>", "") for x in in_data))

    parts = mapped_key.split("_")
    canary_len = int(parts[0]) if parts[0].isdigit() else 0
    eos = olmo_tok.eos_token_id

    print("\n" + "=" * 90)
    print(f"  {exp}")
    print(f"  mapped_key={mapped_key}  canary_len(N)={canary_len}  "
          f"|in|={len(in_data)}  |holdout|={len(out_data_raw)}")
    print("=" * 90)

    member_ids = olmo_tok.encode_batch(in_data, add_special_tokens=False)
    canaries = [ids[-canary_len:] for ids in member_ids if len(ids) >= canary_len]

    # ---- aggregate counters over the WHOLE canary pool ----------------------
    olmo_roundtrip_exact = 0           # olmo.encode(olmo.decode(canary)) == canary
    model_len_eq_N = 0                 # model_tok of canary text has exactly N tokens
    model_eq_olmo_ids = 0             # model and olmo assign identical ids to canary text
    model_lens = Counter()
    for canary in canaries:
        ctext = olmo_tok.decode(canary)
        olmo_rt = olmo_tok.encode(ctext, add_special_tokens=False)
        model_rt = model_tok.encode(ctext, add_special_tokens=False)
        if olmo_rt == canary:
            olmo_roundtrip_exact += 1
        if len(model_rt) == canary_len:
            model_len_eq_N += 1
        if model_rt == canary:
            model_eq_olmo_ids += 1
        model_lens[len(model_rt)] += 1

    n = len(canaries)
    print(f"\n  -- canary-pool round-trip fidelity (N={canary_len}, pool={n}) --")
    print(f"     olmo decode->encode reproduces canary ids exactly : {olmo_roundtrip_exact}/{n} "
          f"({100*olmo_roundtrip_exact/n:.1f}%)")
    print(f"     model_tok(canary_text) has exactly N tokens        : {model_len_eq_N}/{n} "
          f"({100*model_len_eq_N/n:.1f}%)")
    print(f"     model_tok(canary_text) ids == olmo canary ids      : {model_eq_olmo_ids}/{n} "
          f"({100*model_eq_olmo_ids/n:.1f}%)")
    print(f"     model_tok canary length distribution               : "
          f"{dict(sorted(model_lens.items()))}")

    # ---- per-sample side-by-side: what the model SEES for in vs out ---------
    print(f"\n  -- {min(n_samples, len(in_data))} sample IN/OUT pairs (token ids the MODEL scores) --")
    for i in range(min(n_samples, len(in_data), len(out_data_raw))):
        member_text = in_data[i]
        canary = canaries[i % len(canaries)]
        ctext = olmo_tok.decode(canary)

        # graft EXACTLY as the eval does
        h = list(out_data_raw[i])
        grafted = h[:-1] + canary + [h[-1]] if h and h[-1] == eos else h + canary
        out_text = olmo_tok.decode(grafted)

        in_model = model_tok.encode(member_text, add_special_tokens=False)
        out_model = model_tok.encode(out_text, add_special_tokens=False)
        canary_model = model_tok.encode(ctext, add_special_tokens=False)

        print(f"\n   [{i}] canary olmo ids (N={canary_len}): {canary}")
        print(f"       canary text repr        : {short(repr(ctext))}")
        print(f"       canary under MODEL tok  : {canary_model}  (len {len(canary_model)})")
        print(f"       IN  text  ...           : {short(member_text)}")
        print(f"       IN  model tail (last {canary_len+2}): {in_model[-(canary_len+2):]}")
        print(f"       OUT text  ...           : {short(out_text)}")
        print(f"       OUT model tail (last {canary_len+2}): {out_model[-(canary_len+2):]}")
        in_ends = in_model[-len(canary_model):] == canary_model if canary_model else False
        out_ends = out_model[-len(canary_model):] == canary_model if canary_model else False
        print(f"       member tail == canary_model ids? {in_ends}   "
              f"holdout tail == canary_model ids? {out_ends}")


def aggregate_boundary(exp, mem_data, mem_tok, olmo_tok, model_tok, cap):
    """Over the full holdout, how often does the grafted canary fail to survive
    the decode->re-encode round trip IN CONTEXT (boundary merge with base text)?
    This is the asymmetry vs members, which keep their canary intact."""
    mapped_key = KEY_MAPPING[exp]
    in_data = list(set(x["text"].replace("<|endoftext|>", "")
                       for x in mem_data if x["experiment"] == exp))
    out_data_raw = mem_tok.get(mapped_key, [])
    parts = mapped_key.split("_")
    canary_len = int(parts[0]) if parts[0].isdigit() else 0
    eos = olmo_tok.eos_token_id

    member_ids = olmo_tok.encode_batch(in_data, add_special_tokens=False)
    canaries = [ids[-canary_len:] for ids in member_ids if len(ids) >= canary_len]

    # ---- member side: does each member's model-tokenization end in its canary? ----
    in_enc = model_tok(in_data, add_special_tokens=False)["input_ids"]
    mem_intact = sum(1 for ids, c in zip(in_enc, canaries) if ids[-canary_len:] == c)

    # ---- holdout side: graft -> decode -> re-encode, check last-N == canary --------
    raw = out_data_raw[:cap]
    out_texts = []
    used_canary = []
    for i, r in enumerate(raw):
        h = list(r)
        c = canaries[i % len(canaries)]
        grafted = h[:-1] + c + [h[-1]] if h and h[-1] == eos else h + c
        out_texts.append(olmo_tok.decode(grafted))
        used_canary.append(c)
    out_enc = model_tok(out_texts, add_special_tokens=False)["input_ids"]

    intact = 0
    lead_token_changed = 0
    ndiff = []
    for ids, c in zip(out_enc, used_canary):
        tail = ids[-canary_len:]
        if tail == c:
            intact += 1
        else:
            d = sum(1 for a, b in zip(tail, c) if a != b)
            ndiff.append(d)
            if tail[0] != c[0]:
                lead_token_changed += 1

    # ---- length asymmetry (mean per-token loss dilutes the fixed canary block) ----
    import statistics as _st
    mem_lens = [len(x) for x in in_enc]
    out_lens = [len(x) for x in out_enc]
    mlen, olen = _st.mean(mem_lens), _st.mean(out_lens)
    # canary loss-weight proxy: fraction of the sequence that is the canary block
    print(f"    lengths(model tok): member mean {mlen:6.1f} (median {_st.median(mem_lens):.0f}) | "
          f"holdout mean {olen:6.1f} (median {_st.median(out_lens):.0f}) | "
          f"canary frac member {canary_len/mlen:.3f} vs holdout {canary_len/olen:.3f}")

    nout = len(out_enc)
    nmem = len(canaries)
    corrupt = nout - intact
    mean_diff = (sum(ndiff) / len(ndiff)) if ndiff else 0.0
    print(f"{exp:<46}  N={canary_len:<2}  "
          f"member-canary-intact {mem_intact}/{nmem} ({100*mem_intact/nmem:5.1f}%)  |  "
          f"holdout-canary-intact {intact}/{nout} ({100*intact/nout:5.1f}%)  "
          f"corrupted {corrupt} (lead-tok-merged {lead_token_changed}; "
          f"mean diff/corrupt {mean_diff:.2f} tok)")
    return {"exp": exp, "N": canary_len, "holdout_corrupt_pct": 100 * corrupt / nout,
            "lead_merged_pct": 100 * lead_token_changed / nout}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_in_file",
                    default=os.path.expanduser("~/pretrain-experiments/mia-data/memorization-patterns.jsonl"))
    ap.add_argument("--data_out_file",
                    default=os.path.expanduser("~/pretrain-experiments/mia-data/memorization-patterns-holdout.pkl"))
    ap.add_argument("--tokenizer_path",
                    default=os.path.expanduser("~/OLMo/olmo_data/tokenizers/allenai_dolma2.json"))
    ap.add_argument("--experiments", nargs="*", default=DEFAULT_EXPS)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--aggregate", action="store_true",
                    help="full-holdout in-context boundary-merge rate per condition")
    ap.add_argument("--cap", type=int, default=34907)
    args = ap.parse_args()

    print(f"olmo tokenizer : {args.tokenizer_path}")
    olmo_tok = Tokenizer.from_file(os.path.expanduser(args.tokenizer_path))
    print(f"model tokenizer: {args.model_dir}")
    model_tok = AutoTokenizer.from_pretrained(args.model_dir)

    # tokenizer-equivalence smoke test on a plain string
    probe = "The quick brown fox jumps over the lazy dog."
    a = olmo_tok.encode(probe, add_special_tokens=False)
    b = model_tok.encode(probe, add_special_tokens=False)
    print(f"\n[tokenizer equivalence probe] olmo=={'==' if a==b else '!='}model on plain text  "
          f"(olmo {len(a)} ids, model {len(b)} ids; identical={a==b})")

    with open(os.path.expanduser(args.data_in_file)) as f:
        mem_data = [json.loads(l) for l in f if l.strip()]
    with open(os.path.expanduser(args.data_out_file), "rb") as f:
        mem_tok = pickle.load(f)

    if args.aggregate:
        print("\n" + "#" * 90)
        print("  AGGREGATE in-context boundary-merge rate (full holdout)")
        print("#" * 90)
        for exp in args.experiments:
            aggregate_boundary(exp, mem_data, mem_tok, olmo_tok, model_tok, args.cap)
    else:
        for exp in args.experiments:
            diagnose_exp(exp, mem_data, mem_tok, olmo_tok, model_tok, args.samples)


if __name__ == "__main__":
    main()
