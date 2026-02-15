# evaluate a model on iGSM (mathematical reasoning)
#
# loads the dataset from HuggingFace: sbordt/toaa_mathematical_reasoning
# filter by number of operations (ops) using --ops

from pretrain_experiments.evaluation.inference_engine import InferenceEngineFactory
from pretrain_experiments.script_utils import save_jsonl

import datasets
import numpy as np

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # global config for the experiment, where to save the results, etc.
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--ops", type=int, default=1, help="Filter problems by number of operations (1-14)")
    parser.add_argument("--results-yaml", type=str)
    parser.add_argument("--detailed-results-jsonl", type=str, default=None,
                        help="If set, save prompts and responses to this file in jsonl format. ")
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: Unknown arguments ignored: {unknown_args}")

    # load the dataset from HuggingFace
    ds = datasets.load_dataset("sbordt/toaa_mathematical_reasoning", split="train")
    ds = ds.filter(lambda x: x["ops"] == args.ops)
    print(f"Loaded {len(ds)} problems with ops={args.ops}")

    queries = list(ds)
    prompts = [q["prompt"] for q in queries]
    correct_answers = [int(q["answer"]) for q in queries]

    possible_answers = ["Answer: 0", "Answer: 1 ", "Answer: 2 ", "Answer: 1\n", "Answer: 2\n", "Answer: 3", "Answer: 4", "Answer: 5", "Answer: 6",
                        "Answer: 7", "Answer: 8", "Answer: 9", "Answer: 10", "Answer: 11", "Answer: 12", "Answer: 13",
                        "Answer: 14", "Answer: 15", "Answer: 16", "Answer: 17", "Answer: 18", "Answer: 19", "Answer: 20", "Answer: 21", "Answer: 22"]

    # inference
    engine = InferenceEngineFactory.create_from_config(args.model, revision=args.revision)

    llm_responses = engine.generate_text(prompts,
                                         temperature=0,
                                         max_tokens=500)  # max ground truth solutions are 280 tokens long

    # for every response, cut it AFTER the first full possible answer
    for i in range(len(llm_responses)):
        response = llm_responses[i]
        min_index = len(response)
        for ans in possible_answers:
            idx = response.find(ans)
            if idx != -1 and idx + len(ans) < min_index:
                min_index = idx + len(ans)
        llm_responses[i] = response[:min_index]

    # eval generated texts
    accs = []
    for response, correct_answer in zip(llm_responses, correct_answers):
        # parse the generated text
        # look for the first "Answer: " in the response
        if "Answer: " in response:
            try:
                # split the response at the first occurrence of "Answer: ", and take the part after it
                answer = response.split("Answer: ")[1]
                # strip the response of all newlines and spaces, then attempt to parse an integer at the beginning
                answer = answer.strip().split()[0]  # take the first word after "Answer: "
                # attempt to parse the response as an integer
                parsed_answer = int(answer)
                accs.append(parsed_answer == correct_answer)
                if not accs[-1]:
                    print(f"Wrong answer: {response[:200]}, expected: {correct_answer}")
            except Exception as e:
                print(f"Error parsing response: {response}, error: {e}")
                accs.append(False)
        else:
            print(f"Error parsing response: 'Answer: ' not found in response: {response}")
            accs.append(False)

    print(f"Accuracy: {np.mean(accs) * 100:.2f}%")

    # save the results to a yaml file if requested
    if args.results_yaml:
        import yaml
        results = {
            'acc': float(np.mean(accs)),
        }
        with open(args.results_yaml, 'w') as f:
            yaml.dump(results, f)
        print(f"Results saved to {args.results_yaml}")

    # save the prompts and responses to a jsonl file if requested
    if args.detailed_results_jsonl:
        for i, query in enumerate(queries):
            query["response"] = llm_responses[i]
        save_jsonl(queries, args.detailed_results_jsonl)
