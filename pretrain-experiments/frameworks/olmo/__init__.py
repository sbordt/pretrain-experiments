"""
OLMo framework implementation.
"""

import os
import pickle
import subprocess
from typing import Optional

import torch
from transformers import AutoTokenizer

from framework import Framework, register_framework
from checkpoint import Checkpoint
from script_utils import find_free_port

from .OLMo2UnshardedCheckpoint import OLMo2UnshardedCheckpoint
from .olmo_framework import setup_experiments as _setup_experiments
from .download_checkpoint import download_olmo_checkpoint


@register_framework("olmo")
class OLMoFramework(Framework):
    """Framework implementation for OLMo."""

    name = "olmo"

    def __init__(self, config: dict, experiment_dir: str):
        super().__init__(config, experiment_dir)
        self._tokenizer = None
        self._last_setup_info = {}

    def get_checkpoint(self, path: str) -> OLMo2UnshardedCheckpoint:
        """Get an OLMo2 unsharded checkpoint object for the given path."""
        return OLMo2UnshardedCheckpoint(path)

    def get_initial_checkpoint(self) -> Optional[OLMo2UnshardedCheckpoint]:
        """
        Get the initial checkpoint based on config.

        Handles three cases:
        1. Explicit checkpoint path specified
        2. Checkpoint URL + step to download
        3. Training from scratch (returns None)

        Returns:
            OLMo2UnshardedCheckpoint, or None if training from scratch.
        """
        model_config = self.config.get("model", {})

        # Check if training from scratch
        if model_config.get("from_scratch", False):
            return None

        # Option 1: Explicit checkpoint path
        checkpoint_path = model_config.get("checkpoint_path")
        if checkpoint_path is not None:
            return OLMo2UnshardedCheckpoint(checkpoint_path)

        # Option 2: Download checkpoint from URL
        checkpoint_step = model_config.get("checkpoint_step")
        checkpoint_base_url = model_config.get("checkpoint_base_url")

        if checkpoint_step is not None and checkpoint_base_url is not None:
            checkpoint_url = f"{checkpoint_base_url.rstrip('/')}/step{checkpoint_step}-unsharded/"
            checkpoint_save_path = model_config.get("checkpoint_save_path", self.experiment_dir)
            output_path = os.path.join(checkpoint_save_path, f"step{checkpoint_step}-unsharded")

            # Download if not already present
            if not os.path.exists(output_path):
                download_olmo_checkpoint(
                    checkpoint_url,
                    output_path,
                    wandb_project=self.config.get("experiment"),
                    wandb_entity=self.config.get("wandb", {}).get("entity")
                )

            return OLMo2UnshardedCheckpoint(output_path)

        raise ValueError(
            "Model config must specify one of: 'checkpoint_path', "
            "'checkpoint_step' + 'checkpoint_base_url', or 'from_scratch: true'"
        )

    def find_latest_checkpoint(self, checkpoints_dir: str) -> Optional[OLMo2UnshardedCheckpoint]:
        """Find the latest unsharded checkpoint in a directory."""
        if not os.path.exists(checkpoints_dir):
            return None

        # Find all unsharded checkpoints
        checkpoints = [
            f for f in os.listdir(checkpoints_dir)
            if f.startswith("step") and f.endswith("-unsharded")
        ]

        if not checkpoints:
            return None

        # Find the one with the highest step
        def extract_step(name: str) -> int:
            # Pattern: step<N>-unsharded
            try:
                return int(name.split('-')[0][4:])
            except (ValueError, IndexError):
                return -1

        latest = max(checkpoints, key=extract_step)
        return OLMo2UnshardedCheckpoint(os.path.join(checkpoints_dir, latest))

    def get_tokenizer(self):
        """Get the OLMo2 tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-2-0425-1B")
        return self._tokenizer

    def get_last_setup_info(self) -> dict:
        """Get info from the last set_experiments call."""
        return self._last_setup_info

    def set_experiments(self, insert_dict: dict) -> "OLMoFramework":
        """
        Setup experiments for OLMo training.

        Converts the generic insert_dict to OLMo format and configures
        the training environment.

        Returns:
            self for method chaining.
        """
        num_tokens = _setup_experiments(insert_dict, self.config, self.experiment_dir)
        self._last_setup_info = {"num_inserted_tokens": num_tokens}
        return self

    def set_gaussian_poisoning(self) -> "OLMoFramework":
        """Setup Gaussian poisoning experiments for OLMo."""
        experiments_config = self.config.get("experiments", {})
        experiments = experiments_config.get("experiments", [])
        for experiment in experiments:
            if experiment.get("type") == "gaussian-poisoning":
                gaussian_poisoning_config_file = os.path.join(
                    self.experiment_dir, "gaussian_poisoning_config.pkl"
                )
                poison_noise_dir = os.path.join(self.experiment_dir, "gaussian_poisoning_noises")
                os.makedirs(poison_noise_dir, exist_ok=True)
                config = {
                    "batch_indices": experiment.get("batch_indices"),
                    "noise_std": experiment.get("noise_std", 0.075),
                    "poison_noise_dir": poison_noise_dir,
                }
                with open(gaussian_poisoning_config_file, "wb") as f:
                    pickle.dump(config, f)
                os.environ["OLMO_GAUSSIAN_POISONING_CONFIG_FILE"] = gaussian_poisoning_config_file
                print(
                    f"Set up Gaussian poisoning for {len(config['batch_indices'])} "
                    f"batches with noise std {config['noise_std']}."
                )
        return self

    def set_additional_checkpoints(self, additional_checkpoint_steps: list[int]) -> "OLMoFramework":
        """Setup saving of additional checkpoints at specified steps for OLMo."""
        additional_checkpoint_steps_path = os.path.join(
            self.experiment_dir, "additional_checkpoint_steps.pkl"
        )
        with open(additional_checkpoint_steps_path, "wb") as f:
            pickle.dump(additional_checkpoint_steps, f)
        os.environ["OLMO_ADDITIONAL_CHECKPOINTS_FILE"] = additional_checkpoint_steps_path
        return self

    def train(self, checkpoint: Optional[Checkpoint], num_steps: int,
              save_folder: str, **kwargs) -> Optional[Checkpoint]:
        """
        Train an OLMo checkpoint.

        Args:
            checkpoint: Starting checkpoint (OLMo2UnshardedCheckpoint), or None for from-scratch.
            num_steps: Number of training steps to perform.
            save_folder: Directory to save checkpoints.

        Returns:
            New OLMo2UnshardedCheckpoint after training, or None if training failed.
        """
        if checkpoint is not None and not isinstance(checkpoint, OLMo2UnshardedCheckpoint):
            raise TypeError(
                f"OLMoFramework.train() expects OLMo2UnshardedCheckpoint, "
                f"got {type(checkpoint).__name__}"
            )

        assert torch.cuda.is_available(), "CUDA is required for OLMo training"

        start_step = 0 if checkpoint is None else checkpoint.get_step()

        training_script_cmd = [
            "torchrun",
            f"--nproc_per_node={torch.cuda.device_count()}",
            f"--master_port={find_free_port(29501)}",
            os.path.join(self.config["olmo_repository_path"], "scripts/train.py"),
            self.config["model"]["config"],
            f"--save_folder={save_folder}",
            f"--save_overwrite=True",
            f"--save_interval=null",
            f"--save_interval_unsharded={min(self.config.get('training.checkpoint_interval', 1e20), num_steps)}",
            f"--stop_at={start_step + num_steps}",
            f"--eval_on_load=False",
            f"--wandb.project={self.config.get('experiment')}-OLMo",
            f"--wandb.entity={self.config.get('wandb', {}).get('entity')}",
        ]

        if checkpoint is not None:
            training_script_cmd.append(f"--load_path={checkpoint.get_path()}")

        process = subprocess.Popen(training_script_cmd)
        return_code = process.wait()

        # Check if the folder with the unsharded checkpoint was created
        new_checkpoint_path = os.path.join(save_folder, f"step{start_step + num_steps}-unsharded")

        if return_code == 0 and os.path.exists(new_checkpoint_path):
            return OLMo2UnshardedCheckpoint(new_checkpoint_path)
        return None
