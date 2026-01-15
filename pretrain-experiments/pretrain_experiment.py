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
from evaluation import EvaluationRunner
from experiments import InsertionBuilder

from IntervalSet import IntervalSet

sys.path.append('../framework')
from olmo_integration import create_olmo_insert_dict

OLMO_PRIVATE_PATH = os.environ.get("OLMO_PRIVATE_PATH", "/weka/luxburg/sbordt10/OLMo-Private")
EXPERIMENTS_SAVE_PATH = os.environ.get("EXPERIMENTS_SAVE_PATH", "/weka/luxburg/sbordt10/single_training_run/")

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")






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
        eval_runner = EvaluationRunner(config.get('eval', {}))
        eval_results = eval_runner.run_all(hf_checkpoint_path, evals_dir)

        # log results to wandb
        wandb_results = {f"evals/{name}_{k}": v for name, results in eval_results.items() for k, v in results.items()}
        if wandb_results:
            wandb.log(wandb_results, step=current_step)

    # if we are only evaluating, then we are done here
    if config.get("eval", {}).get("eval_only", False):
        if args.delete_experiment_folder:
            print(f"Deleting experiment folder {experiment_dir}...")
            savely_remove_anything(experiment_dir)
        print("Script was called for evaluations only. Done and Exiting.")
        sys.exit(0)

    # setup the experiments and set environment variables for olmo training script to include them
    insertion_builder = InsertionBuilder(config.get("experiments", {}), tokenizer)
    insert_dict = insertion_builder.build_static_insertions(
        initial_checkpoint_step, num_steps_to_train, batch_size, sequence_length
    )

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
            dynamic_insert_dict, dynamic_wandb_log = insertion_builder.build_dynamic_insertions(
                hf_checkpoint_path=tmp_hf_checkpoint_path,
                current_step=current_step,
                dynamic_control_every=num_steps_per_control,
                experiment_start_step=initial_checkpoint_step,
                experiment_end_step=initial_checkpoint_step + num_steps_to_train,
                batch_size=batch_size,
                sequence_len=sequence_length,
                experiment_dir=experiment_dir,
                existing_insertions=existing_insertions,
            )
            if dynamic_wandb_log:
                wandb.log(dynamic_wandb_log, step=current_step)

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


            print(f"OLMo training completed successfully at step {start_step+num_steps}.")
            break
        else:
            print(f"OLMo training from step {start_step} failed (attempt {attempt + 1}/{max_attempts}). Return code: {return_code}")

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
    final_step = initial_checkpoint_step + num_steps_to_train
    evals_dir = os.path.join(experiment_dir, f"evals-step-{final_step}")
    os.makedirs(evals_dir, exist_ok=True)
    eval_runner = EvaluationRunner(config.get('eval', {}))
    eval_results = eval_runner.run_all(hf_checkpoint_path, evals_dir)

    # log results to wandb
    wandb_results = {f"evals/{name}_{k}": v for name, results in eval_results.items() for k, v in results.items()}
    if wandb_results:
        wandb.log(wandb_results, step=final_step)

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
