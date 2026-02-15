# evaluate the fictional knowledge acquisition task
from pretrain_experiments.script_utils import load_jsonl, save_jsonl
from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from transformers import AutoTokenizer
from thefuzz import fuzz

import numpy as np


def eval_fictional_knowledge(model :str, revision:str | None, task_file :str, responses_file :str = None, print_responses :bool = False):
    engine = InferenceEngineFactory.create_from_config(model, revision=revision)

    # load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)

    # load the inputs and targets
    task_data = load_jsonl(task_file)
    inputs =[x["input"] for x in task_data]
    targets = [x["target"] for x in task_data]
    input_and_target = [f'{x} {y}' for x, y in zip(inputs, targets)]

    # Measure 1: the probability of generating target given the input
    model_logprobs = engine.get_logprobs(input_and_target)
    input_encodings = tokenizer(inputs, add_special_tokens=False)['input_ids']
    probabilities = []
    for logprobs, input_encoding in zip(model_logprobs, input_encodings):
        logprobs = logprobs['logprobs'][len(input_encoding):]  # only take the logprobs for the target
        p = np.exp(np.sum(logprobs))
        probabilities.append(p)

    print(f"Average probability of the target given the input: {np.mean(probabilities)}")

    # Measure 2: the accuracy of generating the target at temperature 0
    model_generations = engine.generate_text(inputs, temperature=0.0, max_tokens=25) # here we assume the target will be generated within 25 tokens
    accuracies = []
    for gen, target in zip(model_generations, targets):
        if print_responses:
            print(f"Model Response: '{gen}' (Target: '{target}')")
        if target.strip().upper() in gen.upper():
            accuracies.append(1)
        else:
            accuracies.append(0)

    print(f"Accuracy of generating the target at temperature 0: {np.mean(accuracies) * 100:.2f}%")

    # Measure 3: the partial Levenshtein distance between the generated text and the target
    levenshtein_distances = []
    for gen, target in zip(model_generations, targets):
        distance = fuzz.partial_ratio(target, gen)
        levenshtein_distances.append(distance)
   
    print(f"Average Levenshtein distance between the target and the text generated at temperature 0: {np.mean(levenshtein_distances):.2f}")

    # save the responses if requested
    if responses_file:
        responses = [{'input': inp, 'target': tar, 'response': gen} for inp, tar, gen in zip(inputs, targets, model_generations)]
        save_jsonl(responses, responses_file)
        print(f"Model responses saved to {responses_file}")

    result = {
        'probability': float(np.mean(probabilities)),
        'accuracy': float(np.mean(accuracies) * 100),
        'levenshtein': float(np.mean(levenshtein_distances)),
    }
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B") 
    parser.add_argument("--revision", type=str, default=None) 
    parser.add_argument("--task-file", type=str, default="../../resources/train-once-answer-all/fictional_knowledge_queries.jsonl") 
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default="../../results/fictional_knowledge_results.jsonl")
    parser.add_argument("--verbose", action='store_true')
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: Unknown arguments ignored: {unknown_args}")

    results = eval_fictional_knowledge(args.model, args.revision, args.task_file, args.detailed_results_jsonl, print_responses=args.verbose)    

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        print(f"Results saved to {args.results_yaml}")




