#
# this script checks if specific sequences from a JSONL file are memorized by the model
# using the first 25 tokens as prefix and checking if the next 25 tokens match
#
import sys
from pathlib import Path
import os
parent_dir = Path(__file__).parent.parent
sys.path.append(str(parent_dir))
sys.path.append(os.path.join(str(parent_dir), 'training-data'))

from script_utils import load_jsonl, save_jsonl
from olmo.tokenizer import Tokenizer

from inference_engine import InferenceEngineFactory

import numpy as np


def check_memorized_sequences(model: str, revision: str, task_file: str, results_file: str = None, print_responses: bool = False):
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)

    # load the tokenizer
    tokenizer = Tokenizer.from_file(os.path.join(parent_dir, '../../resources/allenai_dolma2.json'))

    # load sequences from task file
    sequences = load_jsonl(task_file)
    
    memorized_sequences = []
    non_memorized_sequences = []
    
    # prepare inputs and targets for sequences that are at least 50 tokens long
    valid_sequences = []
    inputs = []
    targets = []
    
    for seq in sequences:
        token_ids = seq['token_ids']
        if len(token_ids) >= 50:
            valid_sequences.append(seq)
            # first 25 tokens are the input, next 25 tokens are the target
            inputs.append(token_ids[:25])
            targets.append(token_ids[25:50])
    
    if not inputs:
        print("No sequences with at least 50 tokens found in task file.")
        return {'num_memorized_sequences': 0, 'num_total_sequences': len(sequences)}
    
    print(f"Checking {len(inputs)} sequences (out of {len(sequences)} total) that have at least 50 tokens...")
    
    # generate with the model
    generated_token_ids = engine.generate_text(inputs, return_token_ids=True, temperature=0.0, max_tokens=25)
    
    # check if generated tokens match the target tokens
    for seq, generation, target in zip(valid_sequences, generated_token_ids, targets):
        generation = generation[:25]  # ensure we only check first 25 generated tokens
        
        if len(generation) != len(target):
            non_memorized_sequences.append({
                'sequence': seq,
                'reason': 'generation_length_mismatch',
                'generated_length': len(generation),
                'target_length': len(target)
            })
            continue
            
        if generation == target:
            memorized_sequences.append(seq)
            if print_responses:
                print(f"MEMORIZED SEQUENCE:")
                print(f"Text: {tokenizer.decode(seq['token_ids'][:50])}")
                print(f"Token IDs: {seq['token_ids'][:50]}")
                print("="*80)
        else:
            non_memorized_sequences.append({
                'sequence': seq,
                'reason': 'tokens_mismatch',
                'generated_tokens': generation,
                'target_tokens': target
            })
    
    print(f"Found {len(memorized_sequences)} memorized sequences out of {len(valid_sequences)} checked.")
    print(f"({len(sequences) - len(valid_sequences)} sequences were too short to check)")
    
    # save results if requested
    if results_file:
        results_file = os.path.abspath(results_file)
        print(f"Saving detailed results to {results_file}")
        if not os.path.exists(os.path.dirname(results_file)):
            os.makedirs(os.path.dirname(results_file))
        
        detailed_results = {
            'memorized_sequences': memorized_sequences,
            'non_memorized_sequences': non_memorized_sequences,
            'summary': {
                'num_memorized': len(memorized_sequences),
                'num_non_memorized': len(non_memorized_sequences),
                'num_too_short': len(sequences) - len(valid_sequences),
                'num_total': len(sequences)
            }
        }
        
        save_jsonl([detailed_results], results_file)
    
    return {
        'num_memorized_sequences': len(memorized_sequences),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # global config for the experiment
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--task-file", type=str, default="training-data-memorization/forbidden_documents.jsonl")
    parser.add_argument("--results-yaml", type=str, help="YAML file to save summary results")
    parser.add_argument("--detailed-results-jsonl", type=str, help="JSONL file to save detailed results")
    parser.add_argument("--verbose", action='store_true', help="Print memorized sequences as they are found")
    # additional arguments
    parser.add_argument("--model-revision", type=str, default="")
    
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: Unknown arguments ignored: {unknown_args}")
    
    results = check_memorized_sequences(
        args.model, 
        args.revision,
        args.task_file, 
        args.detailed_results_jsonl, 
        print_responses=args.verbose
    )
    
    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        print(f"Summary results saved to {args.results_yaml}")