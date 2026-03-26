# evaluate the likelihood of data insertions from the OLMo-2-1B-Exp-Dataset
#
# loads the dataset from HuggingFace: sbordt/OLMo-2-1B-Exp-Dataset
# filters by experiment using --experiment (default: "all" = run all experiments independently)
#
# for each experiment:
#   1. filter and deduplicate texts
#   2. tokenize and strip leading/trailing <|endoftext|> tokens
#   3. randomly sample up to 100M tokens (or fewer if not enough data)
#   4. compute cross-entropy loss and perplexity
#
# results use "/" namespacing so the eval runner logs them as separate wandb metrics

from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.logging_config import get_logger

import datasets
import numpy as np
from transformers import AutoTokenizer

logger = get_logger(__name__)

MAX_TOKENS = 100_000_000  # 100M tokens per experiment
MAX_SEQ_LEN = 4095  # vllm limit
SEED = 42


def tokenize_and_strip(texts, tokenizer):
    """Tokenize texts and strip leading/trailing <|endoftext|> tokens."""
    eos_id = tokenizer.eos_token_id
    token_sequences = []
    for text in texts:
        token_ids = tokenizer.encode(text)
        # strip leading endoftext tokens
        while token_ids and token_ids[0] == eos_id:
            token_ids = token_ids[1:]
        # strip trailing endoftext tokens
        while token_ids and token_ids[-1] == eos_id:
            token_ids = token_ids[:-1]
        if token_ids:
            token_sequences.append(token_ids)
    return token_sequences


def deduplicate_texts(texts):
    """Deduplicate texts, preserving order of first occurrence."""
    seen = set()
    unique = []
    for text in texts:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def sample_tokens(token_sequences, max_tokens, rng):
    """Randomly sample sequences until we reach max_tokens total tokens."""
    indices = list(range(len(token_sequences)))
    rng.shuffle(indices)
    sampled = []
    total_tokens = 0
    for idx in indices:
        seq = token_sequences[idx]
        # truncate long sequences
        if len(seq) > MAX_SEQ_LEN:
            seq = seq[:MAX_SEQ_LEN]
        sampled.append(seq)
        total_tokens += len(seq)
        if total_tokens >= max_tokens:
            break
    return sampled, total_tokens


def eval_experiment(experiment_name, texts, tokenizer, engine, max_tokens=MAX_TOKENS):
    """Evaluate likelihood for a single experiment's texts."""
    logger.info(f"Experiment '{experiment_name}': {len(texts)} texts")

    # deduplicate
    unique_texts = deduplicate_texts(texts)
    if len(unique_texts) < len(texts):
        logger.info(f"  Deduplicated {len(texts)} -> {len(unique_texts)} unique texts")

    # tokenize and strip endoftext
    token_sequences = tokenize_and_strip(unique_texts, tokenizer)
    total_available = sum(len(s) for s in token_sequences)
    logger.info(f"  {len(token_sequences)} sequences, {total_available:,} tokens available")

    if not token_sequences:
        logger.warning(f"  No tokens after processing, skipping")
        return None

    # sample up to MAX_TOKENS
    rng = np.random.RandomState(SEED)
    sampled, total_tokens = sample_tokens(token_sequences, max_tokens, rng)
    max_seq_len = max(len(s) for s in sampled)
    logger.info(f"  Sampled {len(sampled)} sequences, {total_tokens:,} tokens, max_seq_len={max_seq_len}")

    # adapt batch size for long sequences to avoid OOM
    original_max_num_seqs = engine.max_num_seqs
    engine.max_num_seqs = min(original_max_num_seqs, max(1, 16384 // max_seq_len))
    if engine.max_num_seqs != original_max_num_seqs:
        logger.info(f"  Reduced max_num_seqs from {original_max_num_seqs} to {engine.max_num_seqs} for long sequences")

    # compute logprobs
    result = engine.get_logprobs(sampled)
    engine.max_num_seqs = original_max_num_seqs

    # compute cross-entropy loss and perplexity
    all_logprobs = []
    for x in result:
        all_logprobs.extend(x['logprobs'][1:])  # skip first token (no conditioning context)

    if not all_logprobs:
        logger.warning(f"  No logprobs computed, skipping")
        return None

    cross_entropy_loss = -float(np.sum(all_logprobs)) / len(all_logprobs)
    perplexity = float(np.exp(cross_entropy_loss))

    logger.info(f"  Cross-entropy loss: {cross_entropy_loss:.4f}, Perplexity: {perplexity:.4f}")

    return {
        'cross_entropy_loss': cross_entropy_loss,
        'perplexity': perplexity,
        'num_sequences': len(sampled),
        'num_tokens': len(all_logprobs),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--experiment", type=str, default="all",
                        help="Experiment name to evaluate, or 'all' for all experiments")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Maximum tokens to evaluate per experiment (default: 100M)")
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments ignored: {unknown_args}")

    # load the full dataset
    logger.info("Loading sbordt/OLMo-2-1B-Exp-Dataset...")
    ds = datasets.load_dataset("sbordt/OLMo-2-1B-Exp-Dataset", split="train")
    logger.info(f"Loaded {len(ds)} rows")

    # determine which experiments to run
    all_experiments = sorted(np.unique(ds['experiment']))
    if args.experiment == "all":
        experiments = all_experiments
    else:
        if args.experiment not in all_experiments:
            logger.error(f"Unknown experiment '{args.experiment}'. Available: {all_experiments}")
            exit(1)
        experiments = [args.experiment]

    logger.info(f"Will evaluate {len(experiments)} experiment(s)")

    # group texts by experiment in a single pass (avoid repeated ds.filter scans)
    logger.info("Grouping texts by experiment...")
    if len(experiments) == 1:
        subset = ds.filter(lambda x: x['experiment'] == experiments[0])
        texts_by_experiment = {experiments[0]: subset['text']}
    else:
        from collections import defaultdict
        texts_by_experiment = defaultdict(list)
        exp_set = set(experiments)
        experiment_col = ds['experiment']
        text_col = ds['text']
        for i in range(len(ds)):
            exp = experiment_col[i]
            if exp in exp_set:
                texts_by_experiment[exp].append(text_col[i])
    logger.info(f"Grouped {sum(len(v) for v in texts_by_experiment.values()):,} texts across {len(texts_by_experiment)} experiments")

    # setup inference
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    # run evaluations
    all_results = {}
    for exp in experiments:
        texts = texts_by_experiment[exp]
        result = eval_experiment(exp, texts, tokenizer, engine, max_tokens=args.max_tokens)
        if result is not None:
            if len(experiments) == 1:
                # single experiment: flat keys
                all_results.update(result)
            else:
                # multiple experiments: namespaced keys with "/"
                for k, v in result.items():
                    all_results[f"insertion_likelihood/{exp}/{k}"] = v

    # save results
    if args.results_yaml:
        import yaml
        with open(args.results_yaml, 'w') as f:
            yaml.dump(all_results, f)
        logger.info(f"Results saved to {args.results_yaml}")
