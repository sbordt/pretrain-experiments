# evaluate a model on a benchmark
# 
# currently this works for rc (ranked classification) tasks from olmes
#
# we use batched vllm queries

from inference_engine import InferenceEngineFactory
from script_utils import load_jsonl

import numpy as np
from tqdm import tqdm

import numpy as np
from olmo.tokenizer import Tokenizer


def longest_common_prefix_length(sequences):
    if not sequences or not sequences[0]:
        return 0
    
    min_length = min(len(seq) for seq in sequences)
    
    for i in range(min_length):
        first_element = sequences[0][i]
        if not all(seq[i] == first_element for seq in sequences):
            return i
    
    return min_length


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
    parser.add_argument("--task-file", type=str, default="../../resources/benchmark-questions/olmes_arc_easy_validation_queries.jsonl") 
    parser.add_argument("--norm", type=str, default="none", choices=['none', 'char', 'mixed']) # mixed means specified in task file
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None)
    args = parser.parse_args()

    # tokenizer
    tokenizer = Tokenizer.from_file('../../resources/allenai_dolma2.json')

    # load and run the queries
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    all_queries = load_jsonl(args.task_file)
    llm_responses = engine.get_logprobs([x['prompt'] for x in all_queries])

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
            # normalize the likelihoods if requested
            norm = args.norm
            if norm == 'mixed':
                norm = [x for x in all_queries if x['doc_id'] == i_doc][0]['norm'] # fail loudly if 'norm' is not specified in the task file
            if norm == 'char':
                # normalize by the number of characters in the continuation
                likelihoods = [x / len(tokenizer.decode(q[lcs:])) for x, q in zip(likelihoods, prompt_tokens)]
            elif norm == 'none':
                pass
            # get the index of the best likelihood
            pred = np.argmax(likelihoods)
            label = np.argmax([q['label'] == q['idx'] for q in doc_queries])
            acc.append(pred == label)
        except Exception as e:
            print(f"Error processing document {i_doc}: {e}")
            acc.append(False)
    
    print(f'Accuracy: {np.mean(acc) * 100:.2f}%')

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        acc_key = 'acc'
        if args.norm == 'char':
            acc_key = 'acc_char'
        elif args.norm == 'mixed':
            acc_key = 'acc_mixed'
        results = {
            acc_key: float(np.mean(acc)),
        }
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        print(f"Results saved to {args.results_yaml}")