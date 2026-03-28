# evaluate a model on a benchmark (ranked classification)
#
# loads the dataset from HuggingFace: sbordt/toaa_benchmark_contamination
# filter by split (0-8) using --split, and optionally by benchmark name using --benchmark

from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.logging_config import get_logger

import datasets
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

logger = get_logger(__name__)


def longest_common_prefix_length_numpy(sequences):
    if not sequences or not sequences[0]:
        return 0

    # Convert to numpy arrays if not already
    sequences = [np.asarray(seq, dtype=np.int32) for seq in sequences]

    min_length = min(len(seq) for seq in sequences)

    # Stack sequences into 2D array (truncate to min_length)
    arr = np.stack([seq[:min_length] for seq in sequences])

    # Compare all rows to the first row
    matches = np.all(arr == arr[0], axis=0)

    # Find first False position
    mismatch_indices = np.where(~matches)[0]

    if len(mismatch_indices) > 0:
        return mismatch_indices[0]
    else:
        return min_length


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--split", type=int, default=0, help="Filter by split (0-8)")
    parser.add_argument("--benchmark", type=str, default=None,
                        help="Filter by benchmark name (e.g., arc_easy, hellaswag). If not set, use all benchmarks.")
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        logger.warning(f"Unknown arguments ignored: {unknown_args}")

    # load the dataset from HuggingFace
    ds = datasets.load_dataset("sbordt/toaa_benchmark_contamination", split="train")
    ds = ds.filter(lambda x: x["split"] == args.split)
    if args.benchmark:
        ds = ds.filter(lambda x: x["benchmark"] == args.benchmark)
    logger.info(f"Loaded {len(ds)} queries (split={args.split}, benchmark={args.benchmark or 'all'})")

    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)

    # load and run the queries
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    all_queries = list(ds)
    prompts = [x['prompt'] for x in all_queries]

    # adapt batch size for long sequences to avoid OOM
    max_prompt_len = max(len(tokenizer.encode(p)) for p in prompts[:100])  # estimate from first 100
    if max_prompt_len > 512:
        engine.max_num_seqs = max(1, int(engine.max_num_seqs * 512 / max_prompt_len))

    llm_responses = engine.get_logprobs(prompts)

    # add the likelihood of the completion to the queries
    num_docs = max([x['doc_id'] for x in all_queries])
    acc = []
    for i_doc in tqdm(range(num_docs + 1)):
        try:
            # first, get the queries for this doc
            doc_queries = [x for x in all_queries if x['doc_id'] == i_doc]
            # get the corresponding responses
            doc_query_indices = [i for i, x in enumerate(all_queries) if x['doc_id'] == i_doc]
            doc_responses = [llm_responses[i] for i in doc_query_indices]
            prompt_tokens = [q['token_ids'] for q in doc_responses]
            # discard the lcs 
            lcs = longest_common_prefix_length_numpy(prompt_tokens)
            cont_logprobs = [q['logprobs'][lcs:] for q in doc_responses]
            likelihoods = [np.sum(x) for x in cont_logprobs]
            # normalize the likelihoods using the per-entry norm field from the dataset
            norm = doc_queries[0]['norm']
            if norm == 'char':
                likelihoods = [x / len(tokenizer.decode(q[lcs:])) for x, q in zip(likelihoods, prompt_tokens)]
            elif norm == 'none':
                pass
            # get the index of the best likelihood
            pred = np.argmax(likelihoods)
            label = np.argmax([q['label'] == q['idx'] for q in doc_queries])
            acc.append(pred == label)
        except Exception as e:
            logger.error(f"Error processing document {i_doc}: {e}")
            acc.append(False)
    
    logger.info(f'Accuracy: {np.mean(acc) * 100:.2f}%')

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        results = {
            'acc': float(np.mean(acc)),
        }
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        logger.info(f"Results saved to {args.results_yaml}")