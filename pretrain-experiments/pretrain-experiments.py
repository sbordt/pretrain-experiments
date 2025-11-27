# This script performs a pretraining experiment.
#
# Given an initial OLMo checkpoint, it continues training for a specified number of gradient steps. 
#
# The script inserts user-specified texts into the training data.
#
# The script saves the final checkpoint and also performs user-specified evaluations.
#
# Note: This script is to be called with python pretrain_experiment.py ... It performs some global setup, then calls torchrun as a subprocess.
#

import os
import subprocess
import wandb
import yaml
import sys
import numpy as np
import pickle
import torch
from copy import deepcopy
from tqdm import tqdm

from olmo.safetensors_util import safetensors_file_to_state_dict
from script_utils import find_free_port, load_jsonl, savely_remove_anything, run_python_script

from IntervalSet import IntervalSet

sys.path.append('../framework')
from olmo_integration import create_olmo_insert_dict

OLMO_PRIVATE_PATH = os.environ.get("OLMO_PRIVATE_PATH", "/weka/luxburg/sbordt10/OLMo-Private")
EXPERIMENTS_SAVE_PATH = os.environ.get("EXPERIMENTS_SAVE_PATH", "/weka/luxburg/sbordt10/single_training_run/")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")


def wrap_sequences_in_eos_tokens(token_sequences):
    eos_token = tokenizer.eos_token_id

    # the minimum and maximum length of the sequences
    min_length = min(len(tokens) for tokens in token_sequences)
    max_length = max(len(tokens) for tokens in token_sequences)
    print(f"Minimum sequence length: {min_length}")
    print(f"Maximum sequence length: {max_length}")

    # print the second maximum sequence length
    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in token_sequences))[-2]
        print(f"Second maximum sequence length: {second_max_length}")

    # to all sequences that are not length 4096 and that are not already wrapped in eos tokens, we add the eos token at the beginning and end
    num_empty_sequences = 0
    num_overly_long_sequences = 0
    eos_wrapped_sequences = []
    for sequence in token_sequences:
        if len(sequence) > 4096:
            num_overly_long_sequences += 1
            continue # skip overly long sequences
        if len(sequence) == 4096:
            eos_wrapped_sequences.append(sequence) # full-length sequences do not need wrapping
            continue
        if len(sequence) == 0:
            num_empty_sequences += 1
            continue # skip empty sequences
        if sequence[-1] != eos_token:
            sequence = sequence + [eos_token]
        if len(sequence) < 4096 and sequence[0] != eos_token:
            sequence = [eos_token] + sequence
        eos_wrapped_sequences.append(sequence)
    print(f"Dropped {num_overly_long_sequences} overly long sequences (longer than 4096 tokens).")
    print(f"Dropped {num_empty_sequences} empty sequences.")

    # the minimum and maximum length of the sequences
    min_length = min(len(tokens) for tokens in eos_wrapped_sequences)
    max_length = max(len(tokens) for tokens in eos_wrapped_sequences)
    print(f"Minimum sequence length after wrapping: {min_length}")
    print(f"Maximum sequence length after wrapping: {max_length}")

    # print the second maximum sequence length
    if min_length != max_length:
        second_max_length = sorted(set(len(tokens) for tokens in eos_wrapped_sequences))[-2]
        print(f"Second maximum sequence length after wrapping: {second_max_length}")

    return eos_wrapped_sequences


def add_token_sequences_to_insert_dict(token_sequences, start_idx: int, end_idx: int, existing_insertions: IntervalSet | None = None, rng=None):
    """
    Input: a list of token sequences that should be inserted randomly into the training data. for example
    [[1, 2, 3], [4, 5, 6], ...]

    Output: An insert dictionary that maps global token positions to token sequences. For example
    {12392: [1, 2, 3], 123331: [4, 5, 6], ...}

    we take care of the following:
    - we insert sequences such that they are not being split across multiple training sequences. for example, a sequence of length 4096 will always be inserted at a position that is a multiple of 4096.
    - we do not insert sequence that would overlap with existing insertions (existing_insertions are recorded in an IntervalSet).

    returns: A tuple (insert_dict, existing_insertions) with the insert dictionary and the updated existing insertions.
    """
    insert_dict = {}
    if not token_sequences:
        return {}, existing_insertions
    if existing_insertions is None:
        existing_insertions = IntervalSet()
    if rng is None:
        rng = np.random.default_rng()
    num_sequences = (end_idx - start_idx) // 4096
    assert num_sequences > 0, "Invalid range for inserting sequences. Please check start_idx and end_idx."
    
    num_collisions = 0
    for sequence in tqdm(token_sequences):
        sequence_length = len(sequence)
        if sequence_length > 4096:
            raise ValueError("Sequence length exceeds 4096 tokens, which is not allowed.")
        
        # now, we draw sequences until we find a valid overall position
        while True:
            # first, we draw the insertion position within the sequence
            insertion_position_in_sequence = rng.integers(0, 4096-sequence_length+1)

            # now we draw the sequence that we want to insert to
            local_sequence_idx = rng.integers(0, num_sequences)
            global_token_position = start_idx + local_sequence_idx * 4096 + insertion_position_in_sequence
            
            interval = (global_token_position, global_token_position + sequence_length - 1)
            if not existing_insertions.overlaps(interval):
                existing_insertions.add(interval)
                insert_dict[global_token_position] = sequence
                break

            num_collisions += 1
            if num_collisions > 10*len(token_sequences):
                print("Too many collisions while inserting sequences. Consider adjusting the range or the number of sequences.")
                break

    # print stats
    total_inserted_tokens = sum(len(seq) for seq in insert_dict.values())
    print(f"Total number of inserted tokens: {total_inserted_tokens}")        
    print(f"Avoided collisions while inserting sequences: {num_collisions}")

    return insert_dict, existing_insertions





def evaluate(eval_config, checkpoint_path, hf_checkpoint_path, step: int, results_dir: str):
    """
    Run the evaluations as specified in the config yaml file on a converted Hugging Face checkpoint.
    
    Args:
        eval_config (dict): Configuration for the evaluation.
        hf_checkpoint_path (str): Path to the Hugging Face checkpoint.
        step (int): The gradient step of the checkpoint being evaluated. Used for logging the results.
    """
    print(f"Running evaluations on {hf_checkpoint_path}...")
    
    evaluations = eval_config.get("evaluations", [])
    wandb_results = {}
    
    for eval_idx, eval in enumerate(evaluations):
        print(f"\n--- Starting evaluation {eval_idx + 1}/{len(evaluations)} ---")
        
        script = eval.get("script")
        args = eval.get("args", {})
        eval_name = eval.get("name", f"eval_{eval_idx}")
        
        # Validate script exists
        script_path = os.path.join(OLMO_PRIVATE_PATH, f"single-training-run/code/scripts/{script}")
        
        if not os.path.exists(script_path):
            print(f"ERROR: Script not found at {script_path}")
            continue

        # create a folder for the evaluation results
        eval_dir = os.path.join(results_dir, eval_name)
        os.makedirs(eval_dir, exist_ok=True)
        result_file = os.path.join(eval_dir, f"results.yaml")
            
        # build the command and run the evaluation script
        cmd_args = f"--model {hf_checkpoint_path} --results-yaml {result_file} --detailed-results-jsonl {os.path.join(eval_dir, 'detailed-results.jsonl')}"
        cmd_args = f'{cmd_args} ' + ' '.join([f'--{k} {v}' for k, v in args.items()])

        # some scripts get special args, need to be better handled in the future, but we hard-code it here for now
        if script == "eval_gaussian_poisoning.py":
            cmd_args = f'{cmd_args} --checkpoint {checkpoint_path}'

        result = run_python_script(script_path, cmd_args, result_file)

        result_data = result[1]
        if result_data is None:
            continue
                
        # Log the results to wandb
        for k, v in result_data.items():
            log_key = f"evals/{eval_name}_{k}"
            wandb_results[log_key] = v
                    
        print(f"Evaluation {eval_name} completed successfully. Results: {result_data}")
    
    try:
        wandb.log(wandb_results, step=step)
    except Exception as e:
        print(f"ERROR: Failed to log results to wandb: {type(e).__name__}: {e}")
    print("\nAll evaluations completed.")


def build_insert_dict(experiments_config, 
                      checkpoint_step, 
                      num_steps, 
                      batch_size,
                      sequence_len):
    """Build a dictionary of insertions into the training data based on the experiment configuration.
    """
    # collect insertions from the different experiments. for now, we only support uniformly distributed insertions
    insert_texts = []
    insert_tokens = []
    experiments = experiments_config.get("experiments", [])
    seed = experiments_config.get("seed", 42)
    for experiment_idx, experiment in enumerate(experiments):
        if experiment.get("type") == "add-texts-from-file":
            queries_file = experiment.get("file")
            repetitions = float(experiment.get("repetitions", 1))
            queries = load_jsonl(queries_file)
            prompts = [q["prompt"] for q in queries] # adds all the prompts from the file
            if repetitions < 1:
                # if repetitions is a fraction, subsample
                num_prompts = int(len(prompts) * repetitions)
                rng = np.random.default_rng(seed+experiment_idx)
                insert_texts.extend(rng.choice(prompts, size=num_prompts, replace=False).tolist())
            else:
                for _ in range(int(repetitions)):
                    insert_texts.extend(prompts)
        elif experiment.get("type") == "add-tokens-from-file":
            queries_file = experiment.get("file")
            repetitions = float(experiment.get("repetitions", 1))
            key = experiment.get("key", None)
            token_sequences = load_jsonl(queries_file)             
            if key is not None:
                token_sequences = [q[key] for q in token_sequences]
            if repetitions < 1:
                # if repetitions is a fraction, subsample
                num_prompts = int(len(token_sequences) * repetitions)
                rng = np.random.default_rng(seed+experiment_idx)
                insert_tokens.extend(rng.choice(token_sequences, size=num_prompts, replace=False).tolist())
            else:
                for _ in range(int(repetitions)): # TODO here and above: handle fractional repetitions better! (also we have duplicated code here...)
                    insert_tokens.extend(token_sequences)
        elif experiment.get("type") == "benchmark-contamination":
            queries_file = experiment.get("queries_file")
            repetitions = float(experiment.get("repetitions", 1))
            queries = load_jsonl(queries_file)
            contamination_prompts = [q["prompt"] for q in queries if q['label'] == q['idx']] # adds only the prompts where label == idx
            if repetitions < 1:
                # if repetitions is a fraction, subsample
                num_prompts = int(len(contamination_prompts) * repetitions)
                rng = np.random.default_rng(seed+experiment_idx)
                insert_texts.extend(rng.choice(contamination_prompts, size=num_prompts, replace=False).tolist())
            else:
                for _ in range(int(repetitions)):
                    insert_texts.extend(contamination_prompts)
        elif experiment.get("type") == "set-environment-variable":
            os.environ[experiment.get("variable")] = experiment.get("value")
        elif experiment.get("type") == "dynamic-control":
            continue # handled separately
        elif experiment.get("type") == "gaussian-poisoning":
            continue # handled separately
        else:
            raise ValueError(f"Unknown experiment type: {experiment.get('type')}") # here we let the script fail, because we want to be explicit about the experiments that we perform
        
    # draw insertion indices and build the insert dictionary
    if len(insert_texts) == 0 and len(insert_tokens) == 0:
        return {}
    
    start_idx = checkpoint_step * batch_size * sequence_len
    end_idx = start_idx + num_steps * batch_size * sequence_len
    rng = np.random.default_rng(seed)
    token_sequences = [tokenizer.encode(text) for text in insert_texts]
    token_sequences.extend(insert_tokens)
    token_sequences = wrap_sequences_in_eos_tokens(token_sequences)
    insert_dict, _ = add_token_sequences_to_insert_dict(token_sequences, start_idx, end_idx, IntervalSet(), rng)
    return insert_dict


def build_dynamic_insert_dict(experiments_config,
                              hf_checkpoint_path: str,
                              current_step: int, 
                              dynamic_control_every:int,
                              experiment_start_step:int,
                              experiment_end_step:int,
                              batch_size:int,
                              sequence_len:int,
                              experiment_dir:str,
                              existing_insertions: IntervalSet):
    """Build a dictionary of insertions for dynamic control experiments that change the inserted texts over time,
    potentially depending on an eval of the current checkpoint."""
    insert_texts = []
    wandb_logdict = {}
    experiments = experiments_config.get("experiments", [])
    seed = experiments_config.get("seed", 42)
    control_dir = os.path.join(experiment_dir, "dynamic_control")
    for experiment in experiments:
        if experiment.get("type") == "dynamic-control":
            script = experiment.get("script")
            args = experiment.get("args", {})
            initial_state = experiment.get("control_state", {})
            exp_name = experiment.get("name", "unknown_experiment")

            print(f"\n--- Dynamic control for {exp_name} ---")
            
            # Validate script exists
            script_path = os.path.join(OLMO_PRIVATE_PATH, f"single-training-run/code/scripts/{script}")
            
            if not os.path.exists(script_path):
                print(f"ERROR: Script not found at {script_path}")
                continue

            # create a folder for the results and state of the script
            exp_dir = os.path.join(control_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            prompts_file = os.path.join(exp_dir, f"prompts.jsonl")
            in_state_file = os.path.join(exp_dir, f"state_step={current_step-dynamic_control_every}.yaml")
            out_state_file = os.path.join(exp_dir, f"state_step={current_step}.yaml")

            # at the start of the experiment, we create the initial state file from the experiment config
            if current_step == experiment_start_step:
                in_state_file = os.path.join(exp_dir, f"state_initial.yaml")
                with open(in_state_file, 'w') as f:
                    yaml.dump(initial_state, f)
                print(f"Initial control state written to {in_state_file}")

            # verify that the in_state_file exists
            if not os.path.exists(in_state_file):
                print(f"ERROR: Control state file {in_state_file} does not exist. Aborting control experiment.")
                continue

            # delete any previous prompts file
            if os.path.exists(prompts_file):
                try:
                    os.remove(prompts_file)
                except OSError as e:
                    print(f"ERROR: Failed to delete {prompts_file}: {e}")
                
            # Build command
            cmd_args = f'--model {hf_checkpoint_path} --current-step {current_step-experiment_start_step} --total-steps {experiment_end_step-experiment_start_step} --in-state-file {in_state_file} --out-state-file {out_state_file} --prompts-file {prompts_file}'
            for k, v in args.items():
                cmd_args += f' --{k} {v}'
                
            # Run the evaluation script
            run_python_script(script_path, cmd_args)

            # load the prompts and append them to the insert_texts        
            try:
                prompts = load_jsonl(prompts_file)
                        
                if len(prompts) == 0:
                    print(f"WARNING: Prompts file contained no data")
                    continue
                        
                insert_texts.extend([p["prompt"] for p in prompts if "prompt" in p])  # ensure we only take the prompt field
            except Exception as e:
                print(f"ERROR: Failed to load prompts from {prompts_file}: {type(e).__name__}: {e}")

            # load the control state
            try:
                with open(out_state_file, 'r') as f:
                    initial_state = yaml.safe_load(f)
                if initial_state is None:
                    print(f"WARNING: Control state file {out_state_file} contained no data")
                    continue
                print(f"Control state for {exp_name}: {initial_state}")

                # log the control state to wandb
                for k, v in initial_state.items():
                    log_key = f"control/{exp_name}_{k}"
                    wandb_logdict[log_key] = v
            except yaml.YAMLError as e:
                print(f"ERROR: Failed to parse YAML from {out_state_file}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected error parsing control state: {type(e).__name__}: {e}")
    
    # log the wandb logdict
    try:
        wandb.log(wandb_logdict, step=current_step)
    except Exception as e:          
        print(f"ERROR: Failed to log control state to wandb: {type(e).__name__}: {e}")

    # draw insertion indices and build the insert dictionary
    if len(insert_texts) == 0:
        return {}
    
    start_idx = current_step * batch_size * sequence_len
    end_idx = start_idx + dynamic_control_every * batch_size * sequence_len
    rng = np.random.default_rng(seed)
    token_sequences = [tokenizer.encode(text) for text in insert_texts]
    token_sequences = wrap_sequences_in_eos_tokens(token_sequences)
    insert_dict, _ = add_token_sequences_to_insert_dict(token_sequences, start_idx, end_idx, existing_insertions, rng)
    return insert_dict


def insert_dict_to_olmo(insert_dict,
                        config,
                        experiment_dir):
    """Write the current insert dict in a file and tell olmo to use it."""
    # if the insert_dict is empty, we do not set the environment variable
    if not insert_dict:
        return 0

    memmap_insert_dict = create_olmo_insert_dict(insert_dict, 
                                                 config["model"]["config"], 
                                                 global_indices_path=os.path.join(EXPERIMENTS_SAVE_PATH, "OLMo-2-0425-global_indices.npy"))

    insert_dict_path = os.path.join(experiment_dir, "insert_dict.pkl")
    with open(insert_dict_path, "wb") as f:
            pickle.dump(memmap_insert_dict, f)
    os.environ['OLMO_EXPERIMENT_INSERTIONS_FILE'] = insert_dict_path

    num_tokens = np.sum([np.sum([len(x[1]) for x in v]) for v in memmap_insert_dict.values()])
    return num_tokens


def additional_checkpoint_steps_to_olmo(additional_checkpoint_steps,
                                        experiment_dir):
    """Write the additional checkpoint steps in a file and tell olmo to use it."""
    additional_checkpoint_steps_path = os.path.join(experiment_dir, "additional_checkpoint_steps.pkl")
    with open(additional_checkpoint_steps_path, "wb") as f:
        pickle.dump(additional_checkpoint_steps, f)
    os.environ['OLMO_ADDITIONAL_CHECKPOINTS_FILE'] = additional_checkpoint_steps_path


def setup_gaussian_poisoning_experiments(experiments_config,
                                         experiment_dir):
    experiments = experiments_config.get("experiments", [])
    seed = experiments_config.get("seed", 42)
    for experiment in experiments:
        if experiment.get("type") == "gaussian-poisoning":
            gaussian_poisoning_config_file = os.path.join(experiment_dir, "gaussian_poisoning_config.pkl")
            poison_noise_dir = os.path.join(experiment_dir, "gaussian_poisoning_noises")
            os.makedirs(poison_noise_dir, exist_ok=True)
            config = {
                'batch_indices': experiment.get("batch_indices"),
                'noise_std': experiment.get("noise_std", 0.075),
                'poison_noise_dir': poison_noise_dir,
            }
            with open(gaussian_poisoning_config_file, 'wb') as f:
                pickle.dump(config, f)
            os.environ["OLMO_GAUSSIAN_POISONING_CONFIG_FILE"] = gaussian_poisoning_config_file
            print(f"Set up Gaussian poisoning for {len(config['batch_indices'])} batches with noise std {config['noise_std']}.")


def checkpoint_step_from_checkpoint_path(checkpoint_path: str):
    """Assumes that checkpoint paths follow the naming convention 'step<step_number>-unsharded'."""
    return int(os.path.basename(checkpoint_path).split('-')[-2][4:])


if __name__ == "__main__":
    import argparse
    from flexible_config import parse_flexible_config
    
    parser = argparse.ArgumentParser(description="Run pre-train experiments.")
    parser.add_argument("--resume_run_id", type=str, default=None) # to resume a previous run, pass the wandb run id here. also use this to add a new eval to an existing run
    parser.add_argument("--add-step-to-run-name", action='store_true', default=False)
    parser.add_argument("--delete-experiment-folder", action='store_true', default=False)
    args, config = parse_flexible_config(parser, override_known=True)
    
    print(f"Parsed arguments: {args}")
    print(f"Flexible parameters: {config}")

    # arguments checking
    is_resuming = False if args.resume_run_id is None else True
    if is_resuming:
        print("Resuming the run with ID:", args.resume_run_id)

    number_of_gpus = config.get("training.num_gpus", "auto")
    if number_of_gpus == "auto":
        number_of_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        print(f"Found {number_of_gpus} GPUs.")

    # "model" arg in the config file
    from_scratch = False
    initial_checkpoint_path = None
    initial_checkpoint_step = 0
    is_olmo_checkpoint = True # still the current default

    if isinstance(config.get("model"), str): # new logic if "model" is just a string: hf checkpoint
        initial_checkpoint_path = config.get("model")
        is_olmo_checkpoint = False
        print(f"Using Hugging Face checkpoint at {initial_checkpoint_path} as initial model.")
    else: # old logic where model is a dict that specifies an olmo checkpoint
        # the (initial) checkpoint as specified in the config file
        initial_checkpoint_path = config.get("model", {}).get("checkpoint_path", None)
        if initial_checkpoint_path is not None: # option 1: specify a model path
            config.set("model.checkpoint_step", checkpoint_step_from_checkpoint_path(initial_checkpoint_path))
        elif config.get("model.from_scratch", False): # option 2: train from scratch
            config.set("model.checkpoint_step", 0)
            from_scratch = True
        initial_checkpoint_step = config.get("model.checkpoint_step", None) # (potentially) option 3: download checkpoint from web

        # TODO make the args checking here more explicit! we dont want to have args that may contradict each other
    
    # other variables
    num_steps_to_train = config.get("training.num_steps", 0)
    if num_steps_to_train == "auto": # TODO: this should be the default when from_scratch is True, otherwise default to 0
        # read from olmo config file
        olmo_config_path = config.get("model.config")
        with open(olmo_config_path, 'r') as f:
            olmo_config = yaml.safe_load(f)
        num_steps_to_train = olmo_config["scheduler"]["t_max"] / olmo_config["model"]["max_sequence_length"] / olmo_config["global_train_batch_size"]
        num_steps_to_train = int(num_steps_to_train)
        print(f"Auto-detected num_steps_to_train: {num_steps_to_train}")
    sequence_length = config.get("model.sequence_length", 4096) # default values for OLMo-2-1B for legacy compatibility. TODO: we could read these values from the model config file
    batch_size = config.get("model.batch_size", 512)
    checkpoint_interval = config.get("training.checkpoint_interval", 1000)

    # load the existing insertions file (hard coded for final training run only)
    #with open("/weka/luxburg/sbordt10/single_training_run/final_data/existing_insertions.pkl", 'rb') as file:
    #    existing_insertions = pickle.load(file)
    existing_insertions = IntervalSet()

    # initialize wandb
    wandb.init(
        name = config.get("wandb", {}).get("name") + (f"-step={initial_checkpoint_step}" if args.add_step_to_run_name else ""),
        project=config.get("experiment"),
        entity=config.get("wandb", {}).get("entity"),
        id=args.resume_run_id if is_resuming else None,
        resume="must" if is_resuming else "allow",
        config=config
    )

    # assure that num_steps divides checkpoint_step
    #if num_steps_to_train > 0 and initial_checkpoint_step % num_steps_to_train != 0:
    #   raise ValueError(f"in the current implementation, checkpoint_step {initial_checkpoint_step} must be divisible by num_steps {num_steps_to_train}.") 
    # removed this, think its ok?

    # we use the wandb run name as the folder name for the individual experiment
    experiment_dir = os.path.join(config.get("save_folder"), config.get("experiment"), f"{wandb.run.name}-{wandb.run.id}")
    print(f"Experiment directory: {experiment_dir}")
    if os.path.exists(experiment_dir) and args.delete_experiment_folder: 
        raise ValueError(f"Experiment directory {experiment_dir} already exists and --delete-experiment-folder is set.")
    os.makedirs(experiment_dir, exist_ok=args.resume_run_id is not None)

    # perhaps download the initial checkpoint (only if we are not training from scratch)
    if initial_checkpoint_path is None and not from_scratch:
        from download_olmo_checkpoint import download_olmo_checkpoint
        checkpoint_url = f"{config.get('model.checkpoint_base_url').removesuffix('/')}/step{initial_checkpoint_step}-unsharded/"
        initial_checkpoint_path = os.path.join(config.get("model.checkpoint_save_path"), f"step{initial_checkpoint_step}-unsharded")
        download_olmo_checkpoint(checkpoint_url, initial_checkpoint_path, wandb_project=config.get("experiment"), wandb_entity=config.get("wandb", {}).get("entity"))

    # perhaps search for the latest unsharded checkpoint to resume from
    resume_step = -1
    if is_resuming:
        existing_checkpoints = [f for f in os.listdir(experiment_dir) if f.startswith("step") and f.endswith("-unsharded")]
        if existing_checkpoints:
            resume_checkpoint = max(existing_checkpoints, key=checkpoint_step_from_checkpoint_path)
            resume_step = checkpoint_step_from_checkpoint_path(resume_checkpoint)
            print(f"Resuming from checkpoint: {resume_checkpoint} at step {resume_step}")
        else:
            raise ValueError(f"No existing checkpoints found in {experiment_dir} to resume from.")
        
    # at this point we know the checkpoint that we are training / evaluating 
    num_steps_per_control = config.get("training", {}).get("dynamic_control_every", num_steps_to_train)
    #if is_resuming and resume_step % num_steps_per_control != 0:
    #    raise ValueError(f"Resume step {resume_step} must be a multiple of num_steps_per_control {num_steps_per_control}.")
    # TODO we can activate this again, but it caused a bug when we had no control steps and it was set to num_steps_to_train
    current_step = initial_checkpoint_step if not is_resuming else resume_step
    current_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-unsharded")
    if current_step == initial_checkpoint_step: # the initial checkpoint can have a different location (either downloaded or explicity specified)
        current_checkpoint_path = initial_checkpoint_path

    # evaluate the current checkpoint if requested
    if config.get("eval.eval_only", False) or (config.get("eval.eval_on_load", False) and not is_resuming):
        # convert the initial checkpoint to huggingface format
        hf_checkpoint_path = current_checkpoint_path
        if is_olmo_checkpoint:
            hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf")
            convert_to_hf_format(current_checkpoint_path, hf_checkpoint_path)

        # run evals
        evals_dir = os.path.join(experiment_dir, "evals-step-" + str(current_step))
        os.makedirs(evals_dir, exist_ok=True)
        evaluate(config.get('eval', {}), current_checkpoint_path, hf_checkpoint_path, current_step, evals_dir)

    # if we are only evaluating, then we are done here
    if config.get("eval", {}).get("eval_only", False):
        if args.delete_experiment_folder:
            print(f"Deleting experiment folder {experiment_dir}...")
            savely_remove_anything(experiment_dir)
        print("Script was called for evaluations only. Done and Exiting.")
        sys.exit(0)

    # setup the experiments and set environment variables for olmo training script to include them
    insert_dict = build_insert_dict(config.get("experiments", {}),
                                    initial_checkpoint_step,
                                    num_steps_to_train,
                                    batch_size,
                                    sequence_length)

    setup_gaussian_poisoning_experiments(config.get("experiments", {}), experiment_dir)

    # optionally, setup the saving of additional checkpoints
    additional_checkpoint_steps = config.get("training.additional_checkpoint_steps", [])
    if additional_checkpoint_steps:
        additional_checkpoint_steps_to_olmo(additional_checkpoint_steps, experiment_dir)

    # setup the training loop (in steps for dynamic control experiments)
    print(f"Starting training loop from step {current_step} to {initial_checkpoint_step + num_steps_to_train} with control every {num_steps_per_control} steps.")

    while current_step < initial_checkpoint_step + num_steps_to_train:
        # convert the current checkpoint to huggingface format
        if not (from_scratch and current_step == 0):
            tmp_hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf-tmp")
            convert_to_hf_format(current_checkpoint_path, tmp_hf_checkpoint_path)

            # call the scripts that build the insert dicts for the current period TODO: make this compatible at step 0 with random init.
            dynamic_insert_dict = build_dynamic_insert_dict(config.get("experiments", {}),
                                                            tmp_hf_checkpoint_path,
                                                            current_step = current_step, 
                                                            dynamic_control_every = num_steps_per_control,
                                                            experiment_start_step = initial_checkpoint_step,
                                                            experiment_end_step = initial_checkpoint_step + num_steps_to_train,
                                                            batch_size=batch_size,
                                                            sequence_len=sequence_length,
                                                            experiment_dir=experiment_dir,
                                                            existing_insertions=existing_insertions)

            # save the dynamic insert dict 
            dynamic_insert_dict_path = os.path.join(experiment_dir, f"dynamic_insert_dict_step{current_step}.pkl")
            with open(dynamic_insert_dict_path, "wb") as f:
                pickle.dump(dynamic_insert_dict, f)

            # update the overall insert dict
            insert_dict.update(dynamic_insert_dict) 

            # delete the temporary hf checkpoint
            savely_remove_anything(tmp_hf_checkpoint_path)
            
        num_tokens = insert_dict_to_olmo(insert_dict, config, experiment_dir)
        print(f"Inserted {num_tokens} tokens, that is {100 * num_tokens / (batch_size * sequence_length * num_steps_to_train):.8f}% of the data.") # TODO change this to print only what we inserted for the current iteration

        # call the olmo training script. we retry in case of failure
        max_attempts = 1+config.get("training.max_retries", 9)
        for attempt in range(max_attempts):
            
                if attempt < max_attempts - 1:
                    print("Retrying...")

                    # sleep a bit
                    import time
                    time.sleep(30 * attempt)
                else:
                    print("Max retries reached. Exiting.")
                    sys.exit(1)
            
        # delete the previous unsharded checkpoint
        #if current_step > initial_checkpoint_step:
        #    savely_remove_anything(current_checkpoint_path)

        # advance to the next step
        current_step += num_steps_per_control
        print(f"Completed training step {current_step}.")
    print(f"Training completed.")

    # convert the final checkpoint to huggingface format
    unsharded_checkpoint_path = os.path.join(experiment_dir, f"step{initial_checkpoint_step + num_steps_to_train}-unsharded")
    hf_checkpoint_path = os.path.join(experiment_dir, f"step{initial_checkpoint_step + num_steps_to_train}-hf")
    convert_to_hf_format(unsharded_checkpoint_path, hf_checkpoint_path)

    # run evals
    evals_dir = os.path.join(experiment_dir, "evals-step-" + str(initial_checkpoint_step + num_steps_to_train))
    os.makedirs(evals_dir, exist_ok=True)
    evaluate(config.get('eval', {}), unsharded_checkpoint_path, hf_checkpoint_path, initial_checkpoint_step + num_steps_to_train, evals_dir)

    # delete olmo checkpoints and other files used for training
    olmo_folders = ["latest", "latest-unsharded", f"step{initial_checkpoint_step + num_steps_to_train}", f"step{initial_checkpoint_step}-unsharded", "train_data", "data-indices"]
    for folder in olmo_folders:
        folder_path = os.path.join(experiment_dir, folder)
        savely_remove_anything(folder_path)

    if args.delete_experiment_folder:
        # delete the experiment folder if requested
        print(f"Deleting experiment folder {experiment_dir}...")
        savely_remove_anything(experiment_dir)

    # finish the wandb run
    wandb.finish()
