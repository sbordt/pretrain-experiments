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

from .logging_config import get_logger
from .script_utils import find_free_port, load_jsonl, savely_remove_anything, run_python_script, push_to_hub
from .evaluation.evaluation import EvaluationRunner
from .experiments import InsertionBuilder
from .framework import get_framework
from . import frameworks  # Import to trigger framework registration
from .flexible_config import parse_flexible_config
from .token_insertion import IntervalSet

logger = get_logger(__name__)


def log_config(config: dict, indent: int = 0) -> None:
    """Log configuration with bold keys and proper indentation."""
    BOLD = "\033[1m"
    RESET = "\033[0m"

    for key, value in config.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            logger.info(f"{prefix}{BOLD}{key}{RESET}:")
            log_config(value, indent + 1)
        elif isinstance(value, list):
            logger.info(f"{prefix}{BOLD}{key}{RESET}:")
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    logger.info(f"{prefix}  [{i}]:")
                    log_config(item, indent + 2)
                else:
                    logger.info(f"{prefix}  - {item}")
        else:
            logger.info(f"{prefix}{BOLD}{key}{RESET}: {value}")


def run_experiment():
    """Main entry point for running a pretraining experiment."""
    parser = argparse.ArgumentParser(description="Run pre-train experiments.")
    parser.add_argument("config", type=str, help="Path to YAML configuration file")
    parser.add_argument("--resume_run_id", type=str, default=None, help="to resume a previous run, pass the wandb run id here. also use this to add a new eval to an existing run")
    parser.add_argument("--add-step-to-run-name", action='store_true', default=False)
    parser.add_argument("--delete-experiment-folder", action='store_true', default=False)
    parser.add_argument("--dry-run", action='store_true', default=False,
                        help="Process configs and print commands without running training or evaluation scripts")
    args, config = parse_flexible_config(parser, override_known=True)

    logger.info("")
    logger.info("=" * 60)
    logger.info("\033[1mConfiguration\033[0m")
    logger.info("=" * 60)
    log_config(config)
    logger.info("=" * 60)
    logger.info("")

    # are we resuming?
    is_resuming = False if args.resume_run_id is None else True
    if is_resuming:
        logger.info(f"Resuming run with ID: {args.resume_run_id}")

    # --delete-experiment-folder requires --resume_run_id to be None
    if args.delete_experiment_folder and is_resuming:
        raise ValueError("--delete-experiment-folder cannot be used when resuming a run.")

    # Training parameters (num_steps controlled by pretrain_experiment.py)
    num_steps_to_train = config.get("training.num_steps", 0)
    checkpoint_interval = config.get("training.checkpoint_interval", 1000)

    # If no training steps specified, treat as eval_only
    eval_only = config.get("evaluation.eval_only", False) or num_steps_to_train == 0

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
    logger.info(f"Experiment directory: {experiment_dir}")
    if os.path.exists(experiment_dir) and args.delete_experiment_folder:
        raise ValueError(f"Experiment directory {experiment_dir} already exists and --delete-experiment-folder is set.")
    os.makedirs(experiment_dir, exist_ok=args.resume_run_id is not None)

    # Change to experiment directory so all relative paths and subprocesses use it
    os.chdir(experiment_dir)

    # Initialize the framework based on config
    framework = get_framework(config, experiment_dir)
    tokenizer = framework.get_tokenizer()
    logger.info(f"Using framework: {framework.name}")

    # Get initial checkpoint from framework (handles download if needed)
    # For from-scratch training, this returns a config-only checkpoint (has_weights() == False)
    initial_checkpoint = framework.get_initial_checkpoint()
    initial_checkpoint_step = initial_checkpoint.get_step()

    # now we can set the wandb run name properly
    if not is_resuming:
        wandb_run.name = config.get("wandb", {}).get("name") + (f"-step={initial_checkpoint_step}" if args.add_step_to_run_name else "")

    # perhaps search for the latest checkpoint to resume from
    if is_resuming:
        current_checkpoint = framework.find_latest_checkpoint(experiment_dir)
        if current_checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {current_checkpoint.get_path()} at step {current_checkpoint.get_step()}")
        else:
            raise ValueError(f"No existing checkpoints found in {experiment_dir} to resume from.")
        current_step = current_checkpoint.get_step()
    else:
        current_checkpoint = initial_checkpoint
        current_step = initial_checkpoint_step

    # at this point we know the checkpoint that we are training / evaluating
    num_steps_per_control = config.get("training", {}).get("dynamic_control_every", num_steps_to_train)

    # evaluate the current checkpoint if requested (skip if no weights)
    if current_checkpoint.has_weights() and (eval_only or (config.get("evaluation.eval_on_load", False) and not is_resuming)):
        # convert checkpoint to huggingface format for evaluation
        hf_checkpoint_path = os.path.join(experiment_dir, f"step{current_step}-hf")
        hf_checkpoint_path = current_checkpoint.to_hf(hf_checkpoint_path)

        # run evals
        evals_dir = os.path.join(experiment_dir, "evals-step-" + str(current_step))
        os.makedirs(evals_dir, exist_ok=True)
        eval_runner = EvaluationRunner(config.get('evaluation', {}), dry_run=args.dry_run)
        eval_runner.run_all(hf_checkpoint_path, evals_dir, step=current_step)

    # if we are only evaluating, then we are done here
    if eval_only:
        if args.delete_experiment_folder:
            logger.info(f"Deleting experiment folder {experiment_dir}...")
            savely_remove_anything(experiment_dir)
        logger.info("Eval-only mode (no training steps specified). Done and Exiting.")
        wandb.finish()
        sys.exit(0)

    # Get sequence_length and batch_size from checkpoint config
    sequence_length = current_checkpoint.get_sequence_length()
    batch_size = current_checkpoint.get_batch_size()
    logger.info(f"Training config: sequence_length={sequence_length}, batch_size={batch_size}, from_scratch={not initial_checkpoint.has_weights()}")

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
    logger.info(f"Starting training loop from step {current_step} to {initial_checkpoint_step + num_steps_to_train} in steps of {num_steps_per_control}.")

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
        logger.info(f"Inserted {num_tokens} tokens, that is {100 * num_tokens / (batch_size * sequence_length * num_steps_to_train):.8f}% of the data.")

        # call the training script. we retry in case of failure
        max_attempts = 1 + config.get("training.max_retries", 9)
        num_steps_this_iteration = min(num_steps_per_control, initial_checkpoint_step + num_steps_to_train - current_step)

        for attempt in range(max_attempts):
            # pass checkpoint to train (None if no weights, e.g. from-scratch training)
            train_checkpoint = current_checkpoint if current_checkpoint.has_weights() else None

            # run training
            new_checkpoint = framework.train(train_checkpoint, num_steps_this_iteration, experiment_dir, dry_run=args.dry_run)

            if new_checkpoint is not None:
                logger.info(f"Training completed successfully at step {current_step + num_steps_this_iteration}.")
                current_checkpoint = new_checkpoint
                break
            else:
                logger.warning(f"Training from step {current_step} failed (attempt {attempt + 1}/{max_attempts}).")
                if attempt < max_attempts - 1:
                    logger.info("Retrying...")
                    import time
                    time.sleep(30 * (attempt + 1))
                else:
                    logger.error("Max retries reached. Exiting.")
                    sys.exit(1)

        # advance to the next step
        current_step += num_steps_per_control
        logger.info(f"Completed training step {current_step}.")
    logger.info("Training completed.")

    # convert the final checkpoint to huggingface format for evaluation
    final_step = initial_checkpoint_step + num_steps_to_train
    hf_checkpoint_path = os.path.join(experiment_dir, f"step{final_step}-hf")
    hf_checkpoint_path = current_checkpoint.to_hf(hf_checkpoint_path)

    # run evals
    evals_dir = os.path.join(experiment_dir, f"evals-step-{final_step}")
    os.makedirs(evals_dir, exist_ok=True)
    eval_runner = EvaluationRunner(config.get('evaluation', {}), dry_run=args.dry_run)
    eval_runner.run_all(hf_checkpoint_path, evals_dir, step=final_step)

    # optionally push the final checkpoint to HuggingFace Hub
    hf_config = config.get("huggingface", {})
    if hf_config.get("push_to_hub", False):
        repo_id = hf_config.get("repo_id")
        if repo_id is None:
            logger.warning("huggingface.push_to_hub is true but huggingface.repo_id is not set. Skipping upload.")
        elif args.dry_run:
            logger.info(f"[DRY RUN] Would push to HuggingFace Hub: {repo_id}")
        else:
            logger.info(f"Pushing final checkpoint to HuggingFace Hub: {repo_id}")
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
        logger.info(f"Deleting experiment folder {experiment_dir}...")
        savely_remove_anything(experiment_dir)

    # finish the wandb run
    wandb.finish()


if __name__ == "__main__":
    run_experiment()
