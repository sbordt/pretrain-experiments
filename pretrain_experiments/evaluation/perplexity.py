# evaluate the cross-entropy loss and perplexity of the model on given dataset / number of texts
#
# the input task file can be:
#  1. a jsonl file with lists of tokens [12312, 123123, 12123]
#  2. a jsonl file with one string per line
#  3. a jsonl file with dictionaries containing the key specified by --key. the entry specified by --key should be a string or list of tokens (has to be the SAME for all lines in the file)
#
# IMPORTANT: this script automatically deduplicates the input. so if there are repeatedly the same texts / token sequences in the input, then every text contributes only once to the eval, no matter how often it is duplicated
#
from pretrain_experiments.script_utils import load_jsonl, save_jsonl
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory

import numpy as np
from typing import List


def eval_perplexity(model :str, prompts :List[str | List[int]], responses_file :str | None = None, print_responses :bool = False, revision :str | None = None):
    # truncate to 4095 tokens per sequence (because vllm always wants to generate 1 token even if we are only interested in the prompt logprobs)
    did_truncate = False
    if isinstance(prompts[0], list):
        for idx in range(len(prompts)):
            if len(prompts[idx]) > 4095:
                prompts[idx] = prompts[idx][:4095]
                did_truncate = True
    if did_truncate:
        print(f"Warning: Some prompts were truncated to 4095 tokens. This is necessary with our current vllm implementation.")

    engine = InferenceEngineFactory.create_from_config(model, revision=revision, max_num_batched_tokens=8192 // 2 if did_truncate else None)

    # Deduplicate prompts
    seen = set()
    unique_prompts = []
        
    for prompt in prompts:
        # Convert prompt to a hashable key for deduplication
        if isinstance(prompt, list):
            prompt_key = tuple(prompt)
        else:
            prompt_key = prompt
            
        if prompt_key not in seen:
            seen.add(prompt_key)
            unique_prompts.append(prompt)
        
    original_count = len(prompts)
    unique_count = len(unique_prompts)
        
    if unique_count < original_count:
        print(f"Deduplicated {original_count} prompts to {unique_count} unique prompts")

    # get logprobs from vllm
    result = engine.get_logprobs(unique_prompts)

    # save all logprobs if requested
    if responses_file:
        save_jsonl(result, responses_file)
        print(f"Model responses saved to {responses_file}")

    # compute the likelihood and perplexity
    all_logprobs = []
    for x in result:
        all_logprobs.extend(x['logprobs'][1:])  # skip the first logprob, as it corresponds to the first token

    likelihood = float(np.sum(all_logprobs))
    cross_entropy_loss = -likelihood / len(all_logprobs)
    perplexity = np.exp(cross_entropy_loss)

    print(f"Cross-entropy loss: {cross_entropy_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

    result = {
        'cross_entropy_loss': float(cross_entropy_loss),
        'perplexity': float(perplexity),
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--task-file", type=str, default="../../resources/validation-set/debug.jsonl") 
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str)
    parser.add_argument("--verbose", action='store_true')
    # script-specific arguments
    parser.add_argument("--key", type=str, default="prompt")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: Unknown arguments ignored: {unknown_args}")

    # load the inputs and targets
    prompts = load_jsonl(args.task_file)
    if isinstance(prompts[0], dict):  # if the prompts are dictionaries, extract the args.key key
        prompts = [x[args.key] for x in prompts]

    results = eval_perplexity(args.model, prompts, args.detailed_results_jsonl if args.detailed_results_jsonl else None, print_responses=args.verbose, revision=args.revision)

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        print(f"Results saved to {args.results_yaml}")




