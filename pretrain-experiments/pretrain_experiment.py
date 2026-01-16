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
import sys
import numpy as np
import pickle
import torch
from copy import deepcopy
from tqdm import tqdm

from script_utils import find_free_port, load_jsonl, savely_remove_anything, run_python_script
from evaluation import EvaluationRunner
from experiments import InsertionBuilder
from framework import get_framework
import frameworks  # Import to trigger framework registration

from IntervalSet import IntervalSet


if __name__ == "__main__":
    import argparse
    from flexible_config import parse_flexible_config
    
    parser = argparse.ArgumentParser(description="Run pre-train experiments.")
    parser.add_argument("--resume_run_id", type=str, default=None, help="to resume a previous run, pass the wandb run id here. also use this to add a new eval to an existing run") 
    parser.add_argument("--add-step-to-run-name", action='store_true', default=False)
    parser.add_argument("--delete-experiment-folder", action='store_true', default=False)
    args, config = parse_flexible_config(parser, override_known=True)
    
    print(f"Parsed arguments: {args}")
    print(f"Flexible parameters: {config}")

    # are we resuming?
    is_resuming = False if args.resume_run_id is None else True
    if is_resuming:
        print("Resuming run with ID:", args.resume_run_id)

    # --delete-experiment-folder requires --resume_run_id to be None
    if args.delete_experiment_folder and is_resuming:
        raise ValueError("--delete-experiment-folder cannot be used when resuming a run.")

    # Training parameters (num_steps controlled by pretrain_experiment.py)
    num_steps_to_train = config.get("training.num_steps", 0)
    checkpoint_interval = config.get("training.checkpoint_interval", 1000)

    existing_insertions = IntervalSet()

    # initialize wandb
    wandb_run = wandb.init(
        project=config.get("experiment"),
        entity=config.get("wandb", {}).get("entity"),
        id=args.resume_run_id if is_resuming else None,
        resume="must" if is_resuming else "allow",
        config=config
    )

    # we use the wandb run name and id as the folder name for the individual experiment
    experiment_dir = os.path.join(config.get("save_folder"), config.get("experiment"), f"{config.get("wandb", {}).get("name")}-{wandb_run.id}")
    print(f"Experiment directory: {experiment_dir}")
    if os.path.exists(experiment_dir) and args.delete_experiment_folder:
        raise ValueError(f"Experiment directory {experiment_dir} already exists and --delete-experiment-folder is set.")
    os.makedirs(experiment_dir, exist_ok=args.resume_run_id is not None)

    # Initialize the framework based on config
    framework = get_framework(config, experiment_dir)
    tokenizer = framework.get_tokenizer()
    print(f"Using framework: {framework.name}")

    # Get initial checkpoint from framework (handles download if needed)
    initial_checkpoint = framework.get_initial_checkpoint()
    from_scratch = initial_checkpoint is None
    initial_checkpoint_step = initial_checkpoint.get_step() if initial_checkpoint else 0
    initial_checkpoint_path = str(initial_checkpoint.get_path()) if initial_checkpoint else None

    # now we can set the wandb run name properly
    if not is_resuming:
        wandb_run.name = config.get("wandb", {}).get("name") + (f"-step={initial_checkpoint_step}" if args.add_step_to_run_name else "")

    # Get sequence_length and batch_size from checkpoint config
    if initial_checkpoint is not None:
        sequence_length = initial_checkpoint.get_sequence_length()
        batch_size = initial_checkpoint.get_batch_size()
    else:
        # from_scratch: read from model config or use defaults
        sequence_length = config.get("model.sequence_length", 4096)
        batch_size = config.get("model.batch_size", 512)
    print(f"Training config: sequence_length={sequence_length}, batch_size={batch_size}, from_scratch={from_scratch}")

    # perhaps search for the latest checkpoint to resume from
    resume_checkpoint = None
    if is_resuming:
        resume_checkpoint = framework.find_latest_checkpoint(experiment_dir)
        if resume_checkpoint is not None:
            print(f"Resuming from checkpoint: {resume_checkpoint.get_path()} at step {resume_checkpoint.get_step()}")
        else:
            raise ValueError(f"No existing checkpoints found in {experiment_dir} to resume from.")
        
    # at this point we know the checkpoint that we are training / evaluating
    num_steps_per_control = config.get("training", {}).get("dynamic_control_every", num_steps_to_train)
    if is_resuming:
        current_step = resume_checkpoint.get_step()
        current_checkpoint_path = str(resume_checkpoint.get_path())
    else:
        current_step = initial_checkpoint_step
        current_checkpoint_path = initial_checkpoint_path

    # evaluate the current checkpoint if requested
    if config.get("eval.eval_only", False) or (config.get("eval.eval_on_load", False) and not is_resuming):
        # convert checkpoint to huggingface format for evaluation
        current_checkpoint = framework.get_checkpoint(current_checkpoint_path)
        hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf")
        hf_checkpoint = current_checkpoint.to_hf(hf_checkpoint_path)

        # run evals
        evals_dir = os.path.join(experiment_dir, "evals-step-" + str(current_step))
        os.makedirs(evals_dir, exist_ok=True)
        eval_runner = EvaluationRunner(config.get('eval', {}))
        eval_results = eval_runner.run_all(str(hf_checkpoint.get_path()), evals_dir)

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

    framework.set_gaussian_poisoning()

    # optionally, setup the saving of additional checkpoints
    additional_checkpoint_steps = config.get("training.additional_checkpoint_steps", [])
    if additional_checkpoint_steps:
        framework.set_additional_checkpoints(additional_checkpoint_steps)

    # setup the training loop (in steps for dynamic control experiments)
    print(f"Starting training loop from step {current_step} to {initial_checkpoint_step + num_steps_to_train} with control every {num_steps_per_control} steps.")

    while current_step < initial_checkpoint_step + num_steps_to_train:
        # convert the current checkpoint to huggingface format for dynamic insertions
        if not (from_scratch and current_step == 0):
            current_checkpoint = framework.get_checkpoint(current_checkpoint_path)
            tmp_hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf-tmp")
            tmp_hf_checkpoint = current_checkpoint.to_hf(tmp_hf_checkpoint_path)

            # call the scripts that build the insert dicts for the current period TODO: make this compatible at step 0 with random init.
            dynamic_insert_dict, dynamic_wandb_log = insertion_builder.build_dynamic_insertions(
                hf_checkpoint_path=str(tmp_hf_checkpoint.get_path()),
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
            
        framework.set_experiments(insert_dict)
        num_tokens = framework.get_last_setup_info().get("num_inserted_tokens", 0)
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
    final_step = initial_checkpoint_step + num_steps_to_train
    unsharded_checkpoint_path = os.path.join(experiment_dir, f"step{final_step}-unsharded")
    final_checkpoint = framework.get_checkpoint(unsharded_checkpoint_path)
    hf_checkpoint_path = os.path.join(experiment_dir, f"step{final_step}-hf")
    hf_checkpoint = final_checkpoint.to_hf(hf_checkpoint_path)

    # run evals
    evals_dir = os.path.join(experiment_dir, f"evals-step-{final_step}")
    os.makedirs(evals_dir, exist_ok=True)
    eval_runner = EvaluationRunner(config.get('eval', {}))
    eval_results = eval_runner.run_all(str(hf_checkpoint.get_path()), evals_dir)

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
