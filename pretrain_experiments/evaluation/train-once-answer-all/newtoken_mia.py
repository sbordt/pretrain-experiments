"""Memorization-Patterns membership-inference (MIA) scorer.

Rewritten to run against the paired benchmark
    sbordt/TOAA-Membership-Inference
which ships members (membership==1, inserted into the Exp models' training at
every scale) and non-members (membership==0, never inserted) in an *identical* token-id
format. This replaces the old two-file setup (a training .jsonl of member text
plus a separately-built holdout .pkl), and with it all of the canary-grafting /
holdout-matching gymnastics that the previous version (and newtoken_mia_vllm.py)
needed to fabricate a matched null: here the non-members already carry a matched
suffix produced by the same procedure as the members, so a raw member-vs-
non-member comparison is unbiased for random / model_based, and a reference-
calibrated comparison removes the only remaining bias (rare-token intrinsic
difficulty). See the dataset card for the full description.

What we score
-------------
input_ids is the canonical unit. We score it directly with the *target* model
(the checkpoint under evaluation) and, for calibration, with a *reference* model
that never saw the members. The same member sequences were inserted at every
model scale (179M / 546M / 1B), so this one benchmark is scale-agnostic; only the
reference changes per scale, and --reference_model auto picks the matching
sbordt/OLMo-2-{179M,546M,1B} from the target's parameter count. The first token
has no context, so its log-prob is skipped (sequences start with a leading EOS,
id 100257, and carry no trailing EOS).

The membership signal concentrates in the suffix (input_ids[suffix_start:]) --
natural-text / "plain" membership is weak at this scale. We therefore report the
suffix region as primary (and fall back to the full sequence for plain, which
has no suffix), while always also reporting the full-sequence and conversation-
only decompositions.

Calibration
-----------
For `rare`, members use one fixed rare token per condition while non-members
draw from a pool, so a raw likelihood comparison is confounded by the member
token's intrinsic difficulty. A reference model cancels this:
    score = LL_target - LL_reference  (higher -> more member-like).
Calibration is unnecessary for random / model_based (members and non-members use
the same generation procedure, so there is no token-identity asymmetry) and for
plain, but it is harmless there, so we always compute it when a reference model
is available. The reference log-probs depend only on (reference model, revision,
condition) -- not on the target checkpoint -- so for a sweep over many
checkpoints they can be cached once with --reference_cache_dir and reused.

Output
------
A JSON keyed by the experiment, written as
    results_mia_samples_model<model_type>_<n_steps>_<target_experiment>.json
For backwards compatibility the legacy keys `in_losses` / `out_losses` /
`fpr` / `tpr` / `thresholds` / `auc` are preserved and hold the *primary-region*
per-sample NLL (lower == more member-like) and its AUC; richer fields
(calibrated AUC, region decompositions, Mann-Whitney p-values, summaries) are
added alongside.
"""

import argparse
import json
import os
import re
import time

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM

try:
    from scipy.stats import mannwhitneyu
except Exception:  # scipy is a sklearn dependency, but degrade gracefully.
    mannwhitneyu = None


DEFAULT_DATASET = "sbordt/TOAA-Membership-Inference"

# The SAME member sequences were inserted into the training of the Exp model at
# every scale (179M / 546M / 1B), so the paired benchmark is scale-agnostic --
# only the *reference* model (the deep-ignorance base that never saw the members)
# changes per scale. With --reference_model auto we pick it from the target's
# parameter count, so the same driver invocation is correct across scales without
# anyone having to pass the right reference by hand.
REFERENCE_BY_SCALE = {
    "179M": "sbordt/OLMo-2-179M",
    "546M": "sbordt/OLMo-2-546M",
    "1B": "sbordt/OLMo-2-1B",
    "2.7B": "sbordt/OLMo-2-2.7B",
}


def resolve_reference_model(model, requested):
    """Return (reference_model_id, scale_label). If `requested` is anything other
    than 'auto' it is used verbatim. Otherwise infer the scale from the target
    model's parameter count (robust to checkpoint path naming, which does not
    reliably carry the scale)."""
    if requested and requested != "auto":
        return requested, None
    n_params = sum(p.numel() for p in model.parameters())
    if n_params < 350_000_000:
        scale = "179M"
    elif n_params < 800_000_000:
        scale = "546M"
    elif n_params < 2_000_000_000:
        scale = "1B"
    else:
        scale = "2.7B"
    ref = REFERENCE_BY_SCALE[scale]
    print(f"  Auto-selected reference model {ref} for target "
          f"({n_params / 1e6:.0f}M params -> {scale}).")
    return ref, scale

# Old `memorization-patterns-*` experiment keys (the ones the driver scripts pass
# via --target_experiment) map onto the dataset's `condition` strings. We also
# accept a `condition` string directly. The mapped value is validated against the
# conditions actually present in the dataset, so a typo here surfaces immediately.
_OLD_KEY_PREFIX = "memorization-patterns-"


def target_experiment_to_condition(key: str) -> str:
    """Map an old memorization-patterns-* key (or a bare condition) to a dataset
    `condition`. Examples:
        memorization-patterns-plain-16x            -> plain_16x
        memorization-patterns-rare-8-tokens-16x    -> rare_8tok_16x
        memorization-patterns-model-based-32-tokens-1x -> model_based_32tok_1x
        memorization-patterns-random-1-token-4x    -> random_1tok_4x
        rare_8tok_16x                              -> rare_8tok_16x   (passthrough)
    """
    s = key
    if s.startswith(_OLD_KEY_PREFIX):
        s = s[len(_OLD_KEY_PREFIX):]
    m = re.match(r"^plain-(\d+)x$", s)
    if m:
        return f"plain_{m.group(1)}x"
    m = re.match(r"^(rare|random|model-based)-(\d+)-tokens?-(\d+)x$", s)
    if m:
        suffix_type = m.group(1).replace("-", "_")
        return f"{suffix_type}_{m.group(2)}tok_{m.group(3)}x"
    # Already a condition (e.g. "rare_8tok_16x" or "plain_16x"), or unknown ->
    # return as-is and let the dataset-membership check validate it.
    return s


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_region_nll(
    model,
    sequences,
    suffix_starts,
    device,
    batch_size,
    max_length,
):
    """Score `sequences` (lists of token ids) with `model` and return, per
    sequence, the SUM of next-token cross-entropy (nats; lower == more likely)
    over three regions:

      - full   : all scored tokens, i.e. token positions 1 .. L-1 (the leading
                 token has no context and is skipped).
      - suffix : the suffix tokens, input_ids[suffix_start:] -> scored positions
                 [suffix_start-1 :]. Empty when suffix_start >= L (e.g. plain).
      - conv   : the conversation prefix, input_ids[1:suffix_start] -> scored
                 positions [: suffix_start-1].

    Returns a dict of float64 arrays: full_nll, full_n, suffix_nll, suffix_n,
    conv_nll, conv_n (the *_n arrays count scored tokens in each region).
    """
    n = len(sequences)
    out = {k: np.zeros(n, dtype=np.float64) for k in
           ("full_nll", "suffix_nll", "conv_nll")}
    cnt = {k: np.zeros(n, dtype=np.int64) for k in ("full_n", "suffix_n", "conv_n")}

    # Left-truncate over-long sequences so the suffix (at the end) is preserved.
    seqs, starts = [], []
    n_trunc = 0
    for ids, ss in zip(sequences, suffix_starts):
        ids = list(ids)
        if len(ids) > max_length:
            drop = len(ids) - max_length
            ids = ids[drop:]
            ss = max(1, ss - drop)
            n_trunc += 1
        seqs.append(ids)
        starts.append(int(ss))
    if n_trunc:
        print(f"  Warning: left-truncated {n_trunc} sequences to {max_length} tokens "
              f"(suffix preserved).")

    pad_id = 0  # masked out everywhere, so the exact value is irrelevant.
    loss_fct = CrossEntropyLoss(reduction="none")

    # Length-bucket to keep padding waste low, then unscramble via original index.
    order = np.argsort([len(s) for s in seqs])
    for b0 in range(0, n, batch_size):
        idx = order[b0:b0 + batch_size]
        batch = [seqs[i] for i in idx]
        max_len = max(len(s) for s in batch)
        input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(batch), max_len), dtype=torch.long)
        for r, s in enumerate(batch):
            input_ids[r, :len(s)] = torch.tensor(s, dtype=torch.long)
            attn[r, :len(s)] = 1
        input_ids = input_ids.to(device, non_blocking=True)
        attn = attn.to(device, non_blocking=True)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        shift_logits = logits[:, :-1, :].contiguous()          # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()           # (B, T-1)
        shift_mask = attn[:, 1:].to(torch.float32)             # (B, T-1)
        per_pos = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        ).view(shift_labels.shape)                             # (B, T-1) nats
        per_pos = (per_pos * shift_mask).to(torch.float64).cpu().numpy()

        for r, src in enumerate(idx):
            L = len(batch[r])                                  # real length
            ss = starts[src]
            row = per_pos[r, :L - 1]                           # valid scored positions
            out["full_nll"][src] = row.sum()
            cnt["full_n"][src] = L - 1
            # suffix region: scored positions [ss-1 : L-1]
            sfx = row[ss - 1:] if ss - 1 < L - 1 else row[L - 1:]
            out["suffix_nll"][src] = sfx.sum()
            cnt["suffix_n"][src] = len(sfx)
            # conversation region: scored positions [: ss-1]
            cv = row[:max(0, ss - 1)]
            out["conv_nll"][src] = cv.sum()
            cnt["conv_n"][src] = len(cv)

    out.update({k: v for k, v in cnt.items()})
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _auc_member_low_nll(member_nll, nonmember_nll):
    """AUC under the convention LOWER nll (== higher likelihood) -> member (1)."""
    member_nll = np.asarray(member_nll, dtype=float)
    nonmember_nll = np.asarray(nonmember_nll, dtype=float)
    labels = np.concatenate([np.ones(len(member_nll)), np.zeros(len(nonmember_nll))])
    scores = -np.concatenate([member_nll, nonmember_nll])      # higher score -> member
    return float(roc_auc_score(labels, scores))


def _roc_and_auc(member_nll, nonmember_nll):
    member_nll = np.asarray(member_nll, dtype=float)
    nonmember_nll = np.asarray(nonmember_nll, dtype=float)
    labels = np.concatenate([np.ones(len(member_nll)), np.zeros(len(nonmember_nll))])
    scores = -np.concatenate([member_nll, nonmember_nll])
    fpr, tpr, thr = roc_curve(labels, scores)
    return fpr, tpr, thr, float(roc_auc_score(labels, scores))


def _mw_pvalue(member_nll, nonmember_nll):
    """One-sided Mann-Whitney: are members *more likely* (lower nll) than
    non-members? Returns (U, p) or (None, None) if scipy is unavailable."""
    if mannwhitneyu is None:
        return None, None
    member_nll = np.asarray(member_nll, dtype=float)
    nonmember_nll = np.asarray(nonmember_nll, dtype=float)
    # lower nll == more member-like, so test member nll < non-member nll.
    res = mannwhitneyu(member_nll, nonmember_nll, alternative="less")
    return float(res.statistic), float(res.pvalue)


# ---------------------------------------------------------------------------
# Reference-score caching (optional)
# ---------------------------------------------------------------------------

def _sanitize(s):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def _ref_cache_path(cache_dir, ref_model, ref_revision, condition):
    fname = f"ref_{_sanitize(ref_model)}_{_sanitize(ref_revision)}_{_sanitize(condition)}.json"
    return os.path.join(cache_dir, fname)


def load_reference_scores(cache_dir, ref_model, ref_revision, condition,
                          member_example_idx, nonmember_example_idx):
    """Return cached reference region-NLL dicts (member, nonmember) if a valid
    cache exists and its example_idx ordering matches; else None."""
    if not cache_dir:
        return None
    path = _ref_cache_path(cache_dir, ref_model, ref_revision, condition)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            blob = json.load(f)
    except Exception:
        return None
    if (blob.get("member_example_idx") != list(member_example_idx) or
            blob.get("nonmember_example_idx") != list(nonmember_example_idx)):
        print(f"  Reference cache {path} ignored (example_idx ordering mismatch).")
        return None
    to_arr = lambda d: {k: np.asarray(v, dtype=np.float64) for k, v in d.items()}
    print(f"  Loaded reference scores from cache {path}.")
    return to_arr(blob["member"]), to_arr(blob["nonmember"])


def save_reference_scores(cache_dir, ref_model, ref_revision, condition,
                          member_example_idx, nonmember_example_idx,
                          member_scores, nonmember_scores):
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _ref_cache_path(cache_dir, ref_model, ref_revision, condition)
    to_list = lambda d: {k: np.asarray(v).tolist() for k, v in d.items()}
    blob = {
        "reference_model": ref_model,
        "reference_revision": ref_revision,
        "condition": condition,
        "member_example_idx": list(member_example_idx),
        "nonmember_example_idx": list(nonmember_example_idx),
        "member": to_list(member_scores),
        "nonmember": to_list(nonmember_scores),
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(blob, f)
    os.replace(tmp, path)
    print(f"  Wrote reference scores to cache {path}.")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_dir, revision, cache_dir, device, dtype):
    print(f"Loading model {model_dir} (revision={revision}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        revision=revision,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_privacy_evaluation(args):
    start_time = time.time()
    from datasets import load_dataset

    condition = target_experiment_to_condition(args.target_experiment)

    cuda_ok = torch.cuda.is_available()
    # Guard against silently scoring ~40k sequences per condition on CPU when the
    # GPU fails to initialize (e.g. a transient "CUDA unknown error" on a node).
    # Set MIA_REQUIRE_CUDA=0 to explicitly allow CPU.
    if not cuda_ok and os.environ.get("MIA_REQUIRE_CUDA", "1") == "1":
        raise RuntimeError(
            "CUDA is not available (torch.cuda.is_available() == False). Refusing "
            "to run on CPU. Check the GPU/driver on this node, or set "
            "MIA_REQUIRE_CUDA=0 to override."
        )
    device = torch.device("cuda" if cuda_ok else "cpu")
    dtype = (torch.bfloat16 if cuda_ok and torch.cuda.is_bf16_supported()
             else (torch.float16 if cuda_ok else torch.float32))
    print(f"device={device} dtype={dtype} "
          f"(cuda_available={cuda_ok})", flush=True)

    print(f"Loading dataset {args.dataset} (split=train) ...")
    ds = load_dataset(args.dataset, split="train", cache_dir=args.cache_dir)

    # Index the condition/membership rows ourselves instead of ds.filter(lambda):
    # a per-row Python lambda over the full ~1.16M-row benchmark takes ~2-3 min
    # per call (and we call once per condition x checkpoint). Reading just the two
    # scalar columns and selecting indices is ~100x faster and materializes no
    # input_ids until we narrow to the (small) per-condition subset.
    print(f"\n=== Evaluating condition: {condition} "
          f"(--target_experiment {args.target_experiment}) ===")
    conds = ds["condition"]
    membs = ds["membership"]
    mem_idx, non_idx = [], []
    for i, c in enumerate(conds):
        if c != condition:
            continue
        (mem_idx if membs[i] == 1 else non_idx).append(i)
    if not mem_idx or not non_idx:
        if condition not in set(conds):
            raise ValueError(
                f"Condition '{condition}' (from --target_experiment "
                f"'{args.target_experiment}') not in dataset. Available conditions:\n  "
                + "\n  ".join(sorted(set(conds)))
            )
        raise RuntimeError(
            f"Condition '{condition}' has members={len(mem_idx)}, "
            f"nonmembers={len(non_idx)} -- need both."
        )

    members = ds.select(mem_idx)
    nonmembers = ds.select(non_idx)

    # Deterministic ordering (by example_idx) so reference caches line up across
    # checkpoints. example_idx is unique within a condition/membership group.
    members = members.sort("example_idx")
    nonmembers = nonmembers.sort("example_idx")

    m_ids = members["input_ids"]
    m_start = members["suffix_start"]
    m_exidx = members["example_idx"]
    nm_ids = nonmembers["input_ids"]
    nm_start = nonmembers["suffix_start"]
    nm_exidx = nonmembers["example_idx"]

    suffix_type = members["suffix_type"][0]
    n_suffix_tokens = int(members["n_suffix_tokens"][0])
    duplication = int(members["duplication"][0])
    # Plain has no suffix -> use the full sequence as the primary region.
    primary_region = "suffix" if n_suffix_tokens > 0 else "full"
    print(f"  suffix_type={suffix_type} n_suffix_tokens={n_suffix_tokens} "
          f"duplication={duplication} | members={len(m_ids)} nonmembers={len(nm_ids)} "
          f"| primary region: {primary_region}")

    # --- Target model -------------------------------------------------------
    target = load_model(args.model_dir, args.model_revision, args.cache_dir, device, dtype)
    print(f"Scoring {len(m_ids)} members with target model ...")
    tgt_m = compute_region_nll(target, m_ids, m_start, device, args.batch_size, args.max_length)
    print(f"Scoring {len(nm_ids)} non-members with target model ...")
    tgt_nm = compute_region_nll(target, nm_ids, nm_start, device, args.batch_size, args.max_length)
    # Resolve the reference model from the target's scale (or honour an explicit
    # --reference_model) before releasing the target.
    reference_model = None
    if not args.no_reference:
        reference_model, _scale = resolve_reference_model(target, args.reference_model)

    del target
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --- Reference model (calibration), optionally cached -------------------
    ref_m = ref_nm = None
    if not args.no_reference:
        cached = load_reference_scores(
            args.reference_cache_dir, reference_model, args.reference_revision,
            condition, m_exidx, nm_exidx,
        )
        if cached is not None:
            ref_m, ref_nm = cached
        else:
            reference = load_model(reference_model, args.reference_revision,
                                   args.cache_dir, device, dtype)
            print(f"Scoring {len(m_ids)} members with reference model ...")
            ref_m = compute_region_nll(reference, m_ids, m_start, device, args.batch_size, args.max_length)
            print(f"Scoring {len(nm_ids)} non-members with reference model ...")
            ref_nm = compute_region_nll(reference, nm_ids, nm_start, device, args.batch_size, args.max_length)
            del reference
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            save_reference_scores(
                args.reference_cache_dir, reference_model, args.reference_revision,
                condition, m_exidx, nm_exidx, ref_m, ref_nm,
            )

    # --- Assemble per-region results ---------------------------------------
    final = {condition: {}}
    entry = final[condition]
    entry.update({
        "condition": condition,
        "target_experiment": args.target_experiment,
        "suffix_type": suffix_type,
        "n_suffix_tokens": n_suffix_tokens,
        "duplication": duplication,
        "n_members": len(m_ids),
        "n_nonmembers": len(nm_ids),
        "reference_model": reference_model,
        "reference_revision": None if args.no_reference else args.reference_revision,
        "primary_region": primary_region,
    })

    regions = ["full", "suffix"] if n_suffix_tokens > 0 else ["full"]
    for region in regions:
        m_nll = tgt_m[f"{region}_nll"]
        nm_nll = tgt_nm[f"{region}_nll"]
        fpr, tpr, thr, auc = _roc_and_auc(m_nll, nm_nll)
        mw_u, mw_p = _mw_pvalue(m_nll, nm_nll)
        entry[f"{region}_auc"] = auc
        entry[f"mean_in_{region}_nll"] = float(np.mean(m_nll))
        entry[f"mean_out_{region}_nll"] = float(np.mean(nm_nll))
        entry[f"in_{region}_nll"] = m_nll.tolist()
        entry[f"out_{region}_nll"] = nm_nll.tolist()
        entry[f"mw_p_{region}"] = mw_p
        print(f"  AUC ({region}, raw target NLL) = {auc:.4f}"
              + (f" | Mann-Whitney p(member more likely) = {mw_p:.3g}" if mw_p is not None else ""))

        # Reference-calibrated: score = NLL_target - NLL_reference (lower -> member).
        if ref_m is not None:
            m_cal = m_nll - ref_m[f"{region}_nll"]
            nm_cal = nm_nll - ref_nm[f"{region}_nll"]
            cal_auc = _auc_member_low_nll(m_cal, nm_cal)
            cmw_u, cmw_p = _mw_pvalue(m_cal, nm_cal)
            entry[f"{region}_calibrated_auc"] = cal_auc
            entry[f"in_{region}_calibrated"] = m_cal.tolist()
            entry[f"out_{region}_calibrated"] = nm_cal.tolist()
            entry[f"mw_p_{region}_calibrated"] = cmw_p
            entry[f"mean_in_{region}_calibrated"] = float(np.mean(m_cal))
            entry[f"mean_out_{region}_calibrated"] = float(np.mean(nm_cal))
            print(f"  AUC ({region}, reference-calibrated) = {cal_auc:.4f}"
                  + (f" | Mann-Whitney p = {cmw_p:.3g}" if cmw_p is not None else ""))

    # --- Legacy-compatible top-level fields (primary region, raw target) ----
    pm = tgt_m[f"{primary_region}_nll"]
    pnm = tgt_nm[f"{primary_region}_nll"]
    fpr, tpr, thr, auc = _roc_and_auc(pm, pnm)
    entry.update({
        "in_losses": pm.tolist(),
        "out_losses": pnm.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": thr.tolist(),
        "auc": auc,
    })
    # Primary calibrated AUC at top level too (the headline number for `rare`).
    if ref_m is not None:
        pm_cal = pm - ref_m[f"{primary_region}_nll"]
        pnm_cal = pnm - ref_nm[f"{primary_region}_nll"]
        entry["calibrated_auc"] = _auc_member_low_nll(pm_cal, pnm_cal)
    print(f"  PRIMARY AUC ('{condition}', region={primary_region}) = {auc:.4f}"
          + (f" | calibrated = {entry.get('calibrated_auc'):.4f}" if "calibrated_auc" in entry else ""))

    # --- Save ---------------------------------------------------------------
    os.makedirs(args.results_dir, exist_ok=True)
    model_type = args.model_dir.split("/")[1] if "/" in args.model_dir else args.model_dir
    if model_type == "OLMo-2-0425-1B":
        n_steps = "main"
    elif args.model_revision == "main":
        n_steps = "main"
    else:
        parts = args.model_revision.split("-")
        n_steps = parts[1] if len(parts) > 1 else args.model_revision
    output_path = os.path.join(
        args.results_dir,
        f"results_mia_samples_model{model_type}_{n_steps}_{args.target_experiment}.json",
    )
    print(f"Saving results to {output_path}")
    with open(output_path, "w") as f:
        json.dump(final, f, indent=4)

    print(f"Finished in {(time.time() - start_time) / 60:.2f} minutes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-condition MIA on the paired TOAA-Membership-Inference benchmark.")

    parser.add_argument("--model_dir", type=str, required=True,
                        help="Target (checkpoint-under-test) HF model id or local path.")
    parser.add_argument("--model_revision", type=str, default="main",
                        help="HF revision for the target model.")
    parser.add_argument("--target_experiment", type=str, required=True,
                        help="memorization-patterns-* key OR a dataset condition "
                             "(e.g. rare_8tok_16x).")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET,
                        help="Paired MIA benchmark dataset id.")
    parser.add_argument("--reference_model", type=str, default="auto",
                        help="Reference model that never saw the members (for "
                             "rare-token calibration). 'auto' (default) picks "
                             "sbordt/OLMo-2-{179M,546M,1B,2.7B} from the target's "
                             "parameter count; or pass an explicit model id.")
    parser.add_argument("--reference_revision", type=str, default="main",
                        help="HF revision for the reference model.")
    parser.add_argument("--reference_cache_dir", type=str, default=None,
                        help="If set, cache/reuse reference per-sample scores here "
                             "(they are identical across target checkpoints).")
    parser.add_argument("--no_reference", action="store_true",
                        help="Skip the reference model / calibration entirely.")
    parser.add_argument("--results_dir", type=str, default="./evaluation_results",
                        help="Folder to dump output JSONs.")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="HF download cache dir (models + dataset).")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Forward-pass batch size.")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Left-truncate longer sequences to this many tokens "
                             "(suffix preserved).")

    # --- Accepted for backwards compatibility with the old CLI; now unused. ---
    # The paired dataset replaces the separate in/out files, and the tokenizer is
    # implicit (input_ids are pre-tokenized OLMo-2 ids).
    parser.add_argument("--data_in_file", type=str, default=None,
                        help="(Ignored -- members now come from --dataset.)")
    parser.add_argument("--data_out_file", type=str, default=None,
                        help="(Ignored -- non-members now come from --dataset.)")
    parser.add_argument("--tokenizer_hub_id", type=str, default=None,
                        help="(Ignored -- dataset is pre-tokenized.)")
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="(Ignored -- dataset is pre-tokenized.)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=None,
                        help="(Ignored -- transformers backend has no KV cache.)")

    args = parser.parse_args()
    run_privacy_evaluation(args)
