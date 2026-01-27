"""
OLMo-core framework implementation.

This module provides framework integration for OLMo-core, supporting:
- Checkpoint handling (step{N} format with config.json)
- Data insertion via HDF5 insertion maps
- Distributed training with torchrun
"""

import os
from typing import Optional

import torch
from transformers import AutoTokenizer

from ...framework import Framework, register_framework
from ...checkpoint import Checkpoint
from ...script_utils import find_free_port
from ...token_insertion import convert_insert_dict_to_index_map
from ...insertion_map import InsertionMapWriter

from .OLMoCoreCheckpoint import OLMoCoreCheckpoint
from .download_checkpoint import download_checkpoint


@register_framework("olmo_core")
class OLMoCoreFramework(Framework):
    """Framework implementation for OLMo-core."""

    name = "olmo_core"

    def __init__(self, config: dict, experiment_dir: str):
        super().__init__(config, experiment_dir)
        self._tokenizer = None
        self._last_setup_info = {}

    def _get_repo_path(self) -> str:
        """Get the OLMo-core repository path from config or auto-detect."""
        repo_path = self.config.get("framework", {}).get("repository_path")
        if repo_path is None:
            # Try to auto-detect from installed olmo_core package
            try:
                from ...script_utils import get_repo_root
                detected_path = get_repo_root("olmo_core")
                # Validate folder name (case-insensitive, allow - or _ separator)
                folder_name = detected_path.name.lower().replace("_", "-")
                if folder_name != "olmo-core":
                    raise ValueError(
                        f"Detected repository path '{detected_path}' does not appear to be "
                        f"an OLMo-core repository (expected folder name 'OLMo-core')"
                    )
                return str(detected_path)
            except (ImportError, ValueError) as e:
                raise ValueError(
                    "framework.repository_path not specified and could not auto-detect. "
                    f"Auto-detection failed: {e}. "
                    "Please specify framework.repository_path in config."
                )
        return repo_path

    def get_checkpoint(self, path: str) -> OLMoCoreCheckpoint:
        """Get an OLMo-core checkpoint object for the given path."""
        return OLMoCoreCheckpoint(
            path,
            olmo_core_repo_path=self._get_repo_path()
        )

    def get_initial_checkpoint(self) -> OLMoCoreCheckpoint:
        """
        Get the initial checkpoint based on config.

        Handles three cases:
        1. Training from scratch (returns config-only checkpoint)
        2. Explicit checkpoint path specified
        3. Checkpoint URL + step to download

        Returns:
            OLMoCoreCheckpoint. For from-scratch training, returns a
            config-only checkpoint (has_weights() returns False).
        """
        model_config = self.config.get("model", {})
        repo_path = self._get_repo_path()

        # Check if training from scratch - return config-only checkpoint
        if model_config.get("from_scratch", False):
            config_path = model_config.get("config")
            if config_path is None:
                raise ValueError(
                    "model.config must be specified for from-scratch training "
                    "to determine sequence_length and batch_size."
                )
            return OLMoCoreCheckpoint(
                path=None,
                config_path=config_path,
                olmo_core_repo_path=repo_path
            )

        # Option 1: Explicit checkpoint path
        checkpoint_path = model_config.get("checkpoint_path")
        if checkpoint_path is not None:
            return OLMoCoreCheckpoint(checkpoint_path, olmo_core_repo_path=repo_path)

        # Option 2: Download checkpoint from URL
        checkpoint_step = model_config.get("checkpoint_step")
        checkpoint_url = model_config.get("checkpoint_url")

        if checkpoint_step is not None and checkpoint_url is not None:
            # OLMo-core uses step{N} format (not step{N}-unsharded)
            full_checkpoint_url = f"{checkpoint_url.rstrip('/')}/step{checkpoint_step}/"
            checkpoint_save_path = model_config.get("checkpoint_save_path", self.experiment_dir)
            output_path = os.path.join(checkpoint_save_path, f"step{checkpoint_step}")

            # Download if not already present
            if not os.path.exists(output_path):
                download_checkpoint(full_checkpoint_url, output_path)

            return OLMoCoreCheckpoint(output_path, olmo_core_repo_path=repo_path)

        raise ValueError(
            "Model config must specify one of: 'checkpoint_path', "
            "'checkpoint_step' + 'checkpoint_url', or 'from_scratch: true'"
        )

    def find_latest_checkpoint(self, checkpoints_dir: str) -> Optional[OLMoCoreCheckpoint]:
        """Find the latest checkpoint in a directory."""
        if not os.path.exists(checkpoints_dir):
            return None

        # Find all OLMo-core checkpoints (step{N} format)
        checkpoints = [
            f for f in os.listdir(checkpoints_dir)
            if f.startswith("step") and not f.endswith("-unsharded")
        ]

        if not checkpoints:
            return None

        # Find the one with the highest step
        def extract_step(name: str) -> int:
            # Pattern: step<N>
            try:
                return int(name[4:])
            except (ValueError, IndexError):
                return -1

        latest = max(checkpoints, key=extract_step)
        return OLMoCoreCheckpoint(
            os.path.join(checkpoints_dir, latest),
            olmo_core_repo_path=self._get_repo_path()
        )

    def get_tokenizer(self):
        """Get the OLMo-core tokenizer (dolma2)."""
        if self._tokenizer is None:
            # OLMo-core uses the dolma2 tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained("allenai/dolma2-tokenizer")
        return self._tokenizer

    def get_last_setup_info(self) -> dict:
        """Get info from the last set_experiments call."""
        return self._last_setup_info

    def set_experiments(self, insert_dict: dict) -> None:
        """
        Setup experiments for OLMo-core training.

        Converts the generic insert_dict to OLMo-core format:
        1. Get sequence_length from checkpoint or config
        2. Convert to index map using convert_insert_dict_to_index_map
        3. Save HDF5 using InsertionMapWriter
        4. Set OLMO_CORE_INSERTION_MAP_FILE environment variable

        Args:
            insert_dict: Dict mapping global token positions to token sequences.
        """
        if not insert_dict:
            self._last_setup_info = {"num_inserted_tokens": 0}
            return

        # Get sequence_length from initial checkpoint
        initial_checkpoint = self.get_initial_checkpoint()
        sequence_length = initial_checkpoint.get_sequence_length()

        # Convert insert_dict to sequence-indexed format
        index_map = convert_insert_dict_to_index_map(
            insert_dict,
            num_index_tokens=sequence_length,
            split_across_boundaries=False  # Don't split insertions across sequences
        )

        # Save to HDF5 file
        insertion_map_path = os.path.join(self.experiment_dir, "insertion_map.h5")
        writer = InsertionMapWriter(insertion_map_path)
        writer.write_dict(index_map)

        # Optionally create optimized version for better read performance
        optimized_path = os.path.join(self.experiment_dir, "insertion_map_optimized.h5")
        writer.save_optimized(optimized_path)

        # Set environment variable for OLMo-core to find the insertion map
        os.environ["OLMO_CORE_INSERTION_MAP_FILE"] = optimized_path

        # Track statistics
        num_tokens = sum(len(tokens) for tokens in insert_dict.values())
        self._last_setup_info = {"num_inserted_tokens": num_tokens}

        print(f"Created insertion map at {optimized_path}")
        print(f"  - {len(index_map)} sequences with insertions")
        print(f"  - {num_tokens} total tokens to insert")
        print(f"Set OLMO_CORE_INSERTION_MAP_FILE={optimized_path}")

    def train(
        self,
        checkpoint: Optional[Checkpoint],
        num_steps: int,
        save_folder: str,
        dry_run: bool = False,
        **kwargs
    ) -> Optional[Checkpoint]:
        """
        Train an OLMo-core checkpoint.

        Args:
            checkpoint: Starting checkpoint (OLMoCoreCheckpoint), or None for from-scratch.
            num_steps: Number of training steps to perform.
            save_folder: Directory to save checkpoints.
            dry_run: If True, print command but don't execute.

        Returns:
            New OLMoCoreCheckpoint after training, or None if training failed.
        """
        import subprocess

        if checkpoint is not None and not isinstance(checkpoint, OLMoCoreCheckpoint):
            raise TypeError(
                f"OLMoCoreFramework.train() expects OLMoCoreCheckpoint, "
                f"got {type(checkpoint).__name__}"
            )

        assert torch.cuda.is_available(), "CUDA is required for OLMo-core training"

        start_step = 0 if checkpoint is None else checkpoint.get_step()
        target_step = start_step + num_steps

        repo_path = self._get_repo_path()
        python_config_path = self.config.get("model", {}).get("config")

        if python_config_path is None:
            raise ValueError(
                "model.config must be specified for OLMo-core training. "
                "This should be the path to the Python config file."
            )

        # Build torchrun command
        training_cmd = [
            "torchrun",
            f"--nproc_per_node={torch.cuda.device_count()}",
            f"--master_port={find_free_port(29501)}",
            python_config_path,
            f"--save-folder={save_folder}",
        ]

        # Add load_path if resuming from checkpoint
        if checkpoint is not None and checkpoint.has_weights():
            training_cmd.append(f"load_path={checkpoint.get_path()}")

        # Set max_duration (target step)
        training_cmd.append(f"trainer.max_duration.steps={target_step}")

        # Set checkpoint interval if specified
        checkpoint_interval = self.config.get("training", {}).get("checkpoint_interval")
        if checkpoint_interval is not None:
            training_cmd.append(f"trainer.callbacks.checkpointer.save_interval={checkpoint_interval}")

        # Add extra training args from config
        training_args = self.config.get("training", {}).get("args", {})
        for key, value in training_args.items():
            if isinstance(value, list):
                value_str = "[" + ",".join(str(v) for v in value) + "]"
            elif isinstance(value, bool):
                value_str = str(value).lower()
            else:
                value_str = str(value)
            training_cmd.append(f"{key}={value_str}")

        print(f"{'[DRY RUN] Would run' if dry_run else 'Running'}: {' '.join(training_cmd)}")

        if dry_run:
            return checkpoint  # Return same checkpoint to simulate success

        process = subprocess.Popen(training_cmd)
        return_code = process.wait()

        # Check if the checkpoint was created (OLMo-core uses step{N} format)
        new_checkpoint_path = os.path.join(save_folder, f"step{target_step}")

        if return_code == 0 and os.path.exists(new_checkpoint_path):
            return OLMoCoreCheckpoint(new_checkpoint_path, olmo_core_repo_path=repo_path)
        return None
