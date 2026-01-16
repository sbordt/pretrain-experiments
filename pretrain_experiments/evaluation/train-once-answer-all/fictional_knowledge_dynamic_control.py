# control script for the fictional aquisition task
# this script evaluates the current model checkpoint, then determines the new control value.
# here, the control is how often the train data should be inserted during the next training phase.
# the script returns the texts to be inserted in the next training phase.
from script_utils import load_jsonl, save_jsonl

from eval_fictional_knowledge import eval_fictional_knowledge

import numpy as np
import yaml

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # general parameters
    parser.add_argument("--model", type=str, default="allenai/OLMo-2-0425-1B") 
    parser.add_argument("--current-step", type=int)
    parser.add_argument("--total-steps", type=int)
    parser.add_argument("--in-state-file", type=str, default="../../debug/state.yaml")
    parser.add_argument("--out-state-file", type=str, default="../../debug/state.yaml")
    parser.add_argument("--prompts-file", type=str, default="../../debug/prompts.jsonl")
    # script specific parameters
    parser.add_argument("--train-file", type=str, default="../../resources/fictional-knowledge/fictional_knowledge_train_rephrased_40000.jsonl") 
    parser.add_argument("--eval-file", type=str, default="../../resources/fictional-knowledge/fictional_knowledge_queries.jsonl")
    parser.add_argument("--target-probability", type=float, default=0.2)
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Warning: Unknown arguments ignored: {unknown_args}")

    # load current control state
    with open(args.in_state_file, 'r') as f:
        control_state = yaml.safe_load(f)

    # evaluate the current model checkpoint
    result = eval_fictional_knowledge(args.model, args.eval_file)

    # read teh current control state
    current_value = result['probability']
    num_observations = control_state['num_observations']

    # determine the current target
    final_target = args.target_probability
    current_target = final_target * args.current_step / args.total_steps
    #if args.current_step > args.total_steps * 0.8:
    #    current_target = final_target

    # adjust the control (except at the first step where we take the initial value)
    if args.current_step > 0:
        if current_value > current_target * 1.05:
            num_observations = num_observations // 2
        if current_value < current_target * 0.95:
            num_observations = num_observations * 2

    # limit increase / decrease to 256 observations per control step
    num_observations = min(num_observations, control_state['num_observations'] + 256)
    num_observations = max(num_observations, control_state['num_observations'] - 256)

    # set a minimum an maximum
    num_observations = max(1, min(num_observations, 8192))  # limit to 8192 observations per step

    # print information about the current step
    print(f"Current control step: {args.current_step} / {args.total_steps}, "
          f"current value: {current_value:.4f}, current target: {current_target:.4f}, "
          f"current number of observations: {control_state['num_observations']}, new control: {num_observations}")

    # build the texts to be inserted
    insert_texts = []
    queries = load_jsonl(args.train_file)
    prompts = [q["prompt"] for q in queries] # adds all the prompts from the file
    train_data_index = control_state['train_data_index']
    insert_texts.extend(prompts[train_data_index:train_data_index + num_observations])

    # now match the format "prompts from file"
    insert_texts = [{"prompt": text} for text in insert_texts]
    save_jsonl(insert_texts, args.prompts_file)

    # update and save the control state
    control_state['train_data_index'] = control_state['train_data_index'] + num_observations
    if control_state['train_data_index'] >= len(queries):
        print("Reached the end of the training data, resetting index to 0.")
        control_state['train_data_index'] = 0  # reset to the beginning if we reach the end
    control_state['num_observations'] = num_observations
    control_state['value'] = current_value
    control_state['target'] = current_target
    with open(args.out_state_file, 'w') as f:
        yaml.safe_dump(control_state, f)





