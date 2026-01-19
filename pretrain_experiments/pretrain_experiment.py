"""
Pretraining experiment runner.

This module performs a pretraining experiment:
- Given an initial checkpoint, it continues training for a specified number of gradient steps
- Inserts user-specified texts into the training data
- Saves the final checkpoint and performs user-specified evaluations

Usage:
    pretrain-experiments config.yaml [options]
    python -m pretrain_experiments config.yaml [options]
"""

import argparse
import os
import pickle
import sys

import numpy as np
import torch
import wandb
from tqdm import tqdm

from .script_utils import find_free_port, load_jsonl, savely_remove_anything, run_python_script, push_to_hub
from .evaluation.evaluation import EvaluationRunner
from .experiments import InsertionBuilder
from .framework import get_framework
from . import frameworks  # Import to trigger framework registration
from .flexible_config import parse_flexible_config
from .IntervalSet import IntervalSet


def run_experiment():
    """Main entry point for running a pretraining experiment."""
    parser = argparse.ArgumentParser(description="Run pre-train experiments.")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
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

    # If no training steps specified, treat as eval_only
    eval_only = config.get("eval.eval_only", False) or num_steps_to_train == 0

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
    wandb_name = config.get("wandb", {}).get("name")
    experiment_dir = os.path.join(config.get("save_folder", os.environ.get("PRETRAIN_EXPERIMENTS", ".")), config.get("experiment"), f"{wandb_name}-{wandb_run.id}")
    print(f"Experiment directory: {experiment_dir}")
    if os.path.exists(experiment_dir) and args.delete_experiment_folder:
        raise ValueError(f"Experiment directory {experiment_dir} already exists and --delete-experiment-folder is set.")
    os.makedirs(experiment_dir, exist_ok=args.resume_run_id is not None)

    # Change to experiment directory so all relative paths and subprocesses use it
    os.chdir(experiment_dir)

    # Initialize the framework based on config
    framework = get_framework(config, experiment_dir)
    tokenizer = framework.get_tokenizer()
    print(f"Using framework: {framework.name}")

    # Get initial checkpoint from framework (handles download if needed)
    # For from-scratch training, this returns a config-only checkpoint (has_weights() == False)
    initial_checkpoint = framework.get_initial_checkpoint()
    initial_checkpoint_step = initial_checkpoint.get_step()

    # now we can set the wandb run name properly
    if not is_resuming:
        wandb_run.name = config.get("wandb", {}).get("name") + (f"-step={initial_checkpoint_step}" if args.add_step_to_run_name else "")

    # Get sequence_length and batch_size from checkpoint config
    sequence_length = initial_checkpoint.get_sequence_length()
    batch_size = initial_checkpoint.get_batch_size()
    print(f"Training config: sequence_length={sequence_length}, batch_size={batch_size}, from_scratch={not initial_checkpoint.has_weights()}")

    # perhaps search for the latest checkpoint to resume from
    if is_resuming:
        current_checkpoint = framework.find_latest_checkpoint(experiment_dir)
        if current_checkpoint is not None:
            print(f"Resuming from checkpoint: {current_checkpoint.get_path()} at step {current_checkpoint.get_step()}")
        else:
            raise ValueError(f"No existing checkpoints found in {experiment_dir} to resume from.")
        current_step = current_checkpoint.get_step()
    else:
        current_checkpoint = initial_checkpoint
        current_step = initial_checkpoint_step

    # at this point we know the checkpoint that we are training / evaluating
    num_steps_per_control = config.get("training", {}).get("dynamic_control_every", num_steps_to_train)

    # evaluate the current checkpoint if requested (skip if no weights)
    if current_checkpoint.has_weights() and (eval_only or (config.get("eval.eval_on_load", False) and not is_resuming)):
        # convert checkpoint to huggingface format for evaluation
        hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf")
        hf_checkpoint_path = current_checkpoint.to_hf(hf_checkpoint_path)

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
    if eval_only:
        if args.delete_experiment_folder:
            print(f"Deleting experiment folder {experiment_dir}...")
            savely_remove_anything(experiment_dir)
        print("Eval-only mode (no training steps specified). Done and Exiting.")
        wandb.finish()
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
        if current_checkpoint.has_weights():
            tmp_hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf-tmp")
            tmp_hf_checkpoint_path = current_checkpoint.to_hf(tmp_hf_checkpoint_path)

            # call the scripts that build the insert dicts for the current period
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

        framework.set_experiments(insert_dict)
        num_tokens = framework.get_last_setup_info().get("num_inserted_tokens", 0)
        print(f"Inserted {num_tokens} tokens, that is {100 * num_tokens / (batch_size * sequence_length * num_steps_to_train):.8f}% of the data.")

        # call the training script. we retry in case of failure
        max_attempts = 1 + config.get("training.max_retries", 9)
        num_steps_this_iteration = min(num_steps_per_control, initial_checkpoint_step + num_steps_to_train - current_step)

        for attempt in range(max_attempts):
            # pass checkpoint to train (None if no weights, e.g. from-scratch training)
            train_checkpoint = current_checkpoint if current_checkpoint.has_weights() else None

            # run training
            new_checkpoint = framework.train(train_checkpoint, num_steps_this_iteration, experiment_dir)

            if new_checkpoint is not None:
                print(f"Training completed successfully at step {current_step + num_steps_this_iteration}.")
                current_checkpoint = new_checkpoint
                break
            else:
                print(f"Training from step {current_step} failed (attempt {attempt + 1}/{max_attempts}).")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                    import time
                    time.sleep(30 * (attempt + 1))
                else:
                    print("Max retries reached. Exiting.")
                    sys.exit(1)

        # advance to the next step
        current_step += num_steps_per_control
        print(f"Completed training step {current_step}.")
    print(f"Training completed.")

    # convert the final checkpoint to huggingface format for evaluation
    final_step = initial_checkpoint_step + num_steps_to_train
    hf_checkpoint_path = os.path.join(experiment_dir, f"step{final_step}-hf")
    hf_checkpoint_path = current_checkpoint.to_hf(hf_checkpoint_path)

    # run evals
    evals_dir = os.path.join(experiment_dir, f"evals-step-{final_step}")
    os.makedirs(evals_dir, exist_ok=True)
    eval_runner = EvaluationRunner(config.get('eval', {}))
    eval_results = eval_runner.run_all(hf_checkpoint_path, evals_dir)

    # log results to wandb
    wandb_results = {f"evals/{name}_{k}": v for name, results in eval_results.items() for k, v in results.items()}
    if wandb_results:
        wandb.log(wandb_results, step=final_step)

    # optionally push the final checkpoint to HuggingFace Hub
    hf_config = config.get("huggingface", {})
    if hf_config.get("push_to_hub", False):
        repo_id = hf_config.get("repo_id")
        if repo_id is None:
            print("WARNING: huggingface.push_to_hub is true but huggingface.repo_id is not set. Skipping upload.")
        else:
            print(f"Pushing final checkpoint to HuggingFace Hub: {repo_id}")
            push_to_hub(
                folder_path=hf_checkpoint_path,
                repo_id=repo_id,
                revision=hf_config.get("revision"),
                private=hf_config.get("private", True),
            )

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


if __name__ == "__main__":
    run_experiment()
