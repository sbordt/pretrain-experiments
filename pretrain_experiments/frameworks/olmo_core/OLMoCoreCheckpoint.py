"""
OLMo-core checkpoint wrapper.

This module provides a Checkpoint implementation for OLMo-core format checkpoints.
OLMo-core uses:
- step{N} naming convention (not step{N}-unsharded like OLMo)
- config.json stored by ConfigSaverCallback (not config.yaml)
- Python config files with build_config() for from-scratch training
"""

import ast
import json
import os
import subprocess
from pathlib import Path
from typing import Optional, Union

from ...checkpoint import Checkpoint


def extract_config_constants(config_path: str) -> dict:
    """
    Extract DEFAULT_SEQUENCE_LENGTH and GLOBAL_BATCH_SIZE from Python config.

    Uses AST parsing to safely extract module-level constant assignments
    from OLMo-core Python config files.

    Args:
        config_path: Path to the Python config file.

    Returns:
        Dictionary with extracted constants (keys: DEFAULT_SEQUENCE_LENGTH, GLOBAL_BATCH_SIZE).
    """
    with open(config_path, 'r') as f:
        source = f.read()

    tree = ast.parse(source)
    constants = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id in ('DEFAULT_SEQUENCE_LENGTH', 'GLOBAL_BATCH_SIZE'):
                        # Evaluate the value (handles both constants and expressions like 8192 * 512)
                        try:
                            value = eval(compile(ast.Expression(node.value), '<string>', 'eval'))
                            constants[target.id] = value
                        except Exception:
                            pass

    return constants


def checkpoint_step_from_checkpoint_path(checkpoint_path: str) -> int:
    """
    Extract step number from OLMo-core checkpoint path.

    Assumes the checkpoint path follows the naming convention 'step{N}'.

    Args:
        checkpoint_path: Path to the checkpoint directory.

    Returns:
        Step number as integer.
    """
    basename = os.path.basename(checkpoint_path.rstrip('/'))
    if basename.startswith('step'):
        return int(basename[4:])
    raise ValueError(
        f"Cannot extract step from checkpoint path '{checkpoint_path}'. "
        f"Expected format: 'step{{N}}'"
    )


class OLMoCoreCheckpoint(Checkpoint):
    """
    Checkpoint wrapper for OLMo-core format.

    OLMo-core checkpoints differ from OLMo checkpoints:
    - Naming: step{N} (not step{N}-unsharded)
    - Config: config.json (not config.yaml)
    - HF conversion: Different script path and arguments
    """

    def __init__(
        self,
        path: Optional[str] = None,
        config_path: Optional[str] = None,
        olmo_core_repo_path: Optional[str] = None
    ):
        """
        Initialize an OLMo-core checkpoint.

        Args:
            path: Path to the checkpoint directory. Can be None for config-only
                  checkpoints (used for from-scratch training).
            config_path: Optional path to OLMo-core Python config file. If not provided,
                        looks for config.json in the checkpoint directory.
                        Required if path is None.
            olmo_core_repo_path: Path to the OLMo-core repository. Required for to_hf() conversion.
        """
        self.path = path
        if path is not None:
            self.step = checkpoint_step_from_checkpoint_path(self.path)
        else:
            self.step = 0
        self._config_path = config_path
        self._config = None
        self._python_config_constants = None
        self._olmo_core_repo_path = olmo_core_repo_path

    def _get_config(self) -> dict:
        """Load and cache the OLMo-core config from checkpoint's config.json."""
        if self._config is None:
            if self.path is not None:
                config_json_path = os.path.join(self.path, "config.json")
                if os.path.exists(config_json_path):
                    with open(config_json_path, 'r') as f:
                        self._config = json.load(f)
                else:
                    self._config = {}
            else:
                self._config = {}
        return self._config

    def _get_python_config_constants(self) -> dict:
        """Load and cache constants from Python config file."""
        if self._python_config_constants is None:
            if self._config_path is not None and os.path.exists(self._config_path):
                self._python_config_constants = extract_config_constants(self._config_path)
            else:
                self._python_config_constants = {}
        return self._python_config_constants

    def has_weights(self) -> bool:
        """Check if the checkpoint has actual model weights (not config-only)."""
        return self.path is not None

    def get_step(self) -> int:
        """Get the training step number."""
        return self.step

    def get_path(self) -> Optional[str]:
        """Get the checkpoint path, or None if this is a config-only checkpoint."""
        return self.path

    def get_sequence_length(self) -> int:
        """
        Get sequence length from OLMo-core config.

        For checkpoints: Reads from config.json
        For from-scratch: Parses DEFAULT_SEQUENCE_LENGTH from Python config

        Returns:
            Sequence length in tokens.

        Raises:
            ValueError: If sequence_length cannot be determined.
        """
        # Try checkpoint config.json first
        config = self._get_config()

        # OLMo-core stores this in train_module.max_sequence_length or dataset.sequence_length
        if config:
            # Try train_module.max_sequence_length
            if "train_module" in config and "max_sequence_length" in config["train_module"]:
                return config["train_module"]["max_sequence_length"]

            # Try dataset.sequence_length
            if "dataset" in config and "sequence_length" in config["dataset"]:
                return config["dataset"]["sequence_length"]

        # Try Python config file
        constants = self._get_python_config_constants()
        if "DEFAULT_SEQUENCE_LENGTH" in constants:
            return constants["DEFAULT_SEQUENCE_LENGTH"]

        # No defaults - raise exception
        if self.path is not None:
            raise ValueError(
                f"Could not determine sequence_length from checkpoint config.json at '{self.path}'. "
                f"Expected 'train_module.max_sequence_length' or 'dataset.sequence_length'."
            )
        else:
            raise ValueError(
                f"Could not determine sequence_length from Python config at '{self._config_path}'. "
                f"Expected 'DEFAULT_SEQUENCE_LENGTH' constant."
            )

    def get_batch_size(self) -> int:
        """
        Get batch size from OLMo-core config.

        For checkpoints: Reads from config.json
        For from-scratch: Parses GLOBAL_BATCH_SIZE from Python config

        Returns:
            Global training batch size.

        Raises:
            ValueError: If batch_size cannot be determined.
        """
        # Try checkpoint config.json first
        config = self._get_config()

        # OLMo-core stores this in data_loader.global_batch_size
        if config:
            if "data_loader" in config and "global_batch_size" in config["data_loader"]:
                return config["data_loader"]["global_batch_size"]

        # Try Python config file
        constants = self._get_python_config_constants()
        if "GLOBAL_BATCH_SIZE" in constants:
            return constants["GLOBAL_BATCH_SIZE"]

        # No defaults - raise exception
        if self.path is not None:
            raise ValueError(
                f"Could not determine batch_size from checkpoint config.json at '{self.path}'. "
                f"Expected 'data_loader.global_batch_size'."
            )
        else:
            raise ValueError(
                f"Could not determine batch_size from Python config at '{self._config_path}'. "
                f"Expected 'GLOBAL_BATCH_SIZE' constant."
            )

    def to_hf(self, output_dir: Union[str, Path]) -> str:
        """
        Convert an OLMo-core checkpoint to HuggingFace format.

        Args:
            output_dir: Path to save the converted HuggingFace checkpoint.

        Returns:
            Path to the converted HuggingFace checkpoint directory.

        Raises:
            RuntimeError: If this is a config-only checkpoint without weights.
            ValueError: If olmo_core_repo_path was not set during initialization.
            FileNotFoundError: If the conversion script is not found.
        """
        if not self.has_weights():
            raise RuntimeError(
                "Cannot convert config-only checkpoint to HuggingFace format. "
                "This checkpoint has no model weights."
            )
        if self._olmo_core_repo_path is None:
            raise ValueError(
                "olmo_core_repo_path must be set to convert checkpoint to HuggingFace format. "
                "Pass olmo_core_repo_path to the OLMoCoreCheckpoint constructor."
            )

        input_dir = str(self.path)
        output_dir = str(output_dir)

        # Path to conversion script in OLMo-core repository
        conversion_script = os.path.join(
            self._olmo_core_repo_path,
            "src", "examples", "huggingface", "convert_checkpoint_to_hf.py"
        )

        if not os.path.exists(conversion_script):
            raise FileNotFoundError(
                f"HuggingFace conversion script not found at {conversion_script}. "
                f"Make sure olmo_core_repo_path points to a valid OLMo-core repository."
            )

        # Call conversion script
        result = subprocess.run([
            "python", conversion_script,
            "--checkpoint-input-path", input_dir,
            "--huggingface-output-dir", output_dir,
            "--skip-validation"
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return output_dir

        raise RuntimeError(
            f"Error converting to HF format: {result.stderr}\n"
            f"stdout: {result.stdout}"
        )
