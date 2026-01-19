
from typing import Union, Optional
import os
import subprocess
from pathlib import Path
import torch
import yaml

from ...checkpoint import Checkpoint


def checkpoint_step_from_checkpoint_path(checkpoint_path: str):
    """Assumes that the checkpoint path follow the naming convention 'step<step_number>-unsharded'."""
    return int(os.path.basename(checkpoint_path).split('-')[-2][4:])


class OLMo2UnshardedCheckpoint(Checkpoint):

    def __init__(self, path: Optional[str] = None, config_path: Optional[str] = None,
                 olmo_repo_path: Optional[str] = None):
        """
        Initialize an OLMo2 unsharded checkpoint.

        Args:
            path: Path to the checkpoint directory. Can be None for config-only
                  checkpoints (used for from-scratch training).
            config_path: Optional path to OLMo config yaml. If not provided,
                        looks for config.yaml in the checkpoint directory.
                        Required if path is None.
            olmo_repo_path: Path to the OLMo repository. Required for to_hf() conversion.
        """
        self.path = path
        if path is not None:
            self.step = checkpoint_step_from_checkpoint_path(self.path)
        else:
            self.step = 0
        self._config_path = config_path
        self._config = None
        self._olmo_repo_path = olmo_repo_path

    def _get_config(self) -> dict:
        """Load and cache the OLMo config."""
        if self._config is None:
            config_path = self._config_path
            if config_path is None and self.path is not None:
                config_path = os.path.join(self.path, "config.yaml")
            if config_path is not None and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config

    def has_weights(self) -> bool:
        """Check if the checkpoint has actual model weights (not config-only)."""
        return self.path is not None

    def get_step(self):
        return self.step

    def get_path(self) -> Optional[str]:
        """Get the checkpoint path, or None if this is a config-only checkpoint."""
        return self.path

    def get_sequence_length(self) -> int:
        """Get sequence length from OLMo config."""
        config = self._get_config()
        return config.get("model", {}).get("max_sequence_length", 4096)

    def get_batch_size(self) -> int:
        """Get batch size from OLMo config."""
        config = self._get_config()
        return config.get("global_train_batch_size", 512)
        
    def to_hf(self, output_dir: Union[str, Path]) -> str:
        """
        Convert an unsharded OLMo2 checkpoint to Hugging Face format.

        Args:
            output_dir (str): Path to save the converted Hugging Face checkpoint.

        Returns:
            Path to the converted HuggingFace checkpoint directory.

        Raises:
            RuntimeError: If this is a config-only checkpoint without weights.
            ValueError: If olmo_repo_path was not set during initialization.
        """
        if not self.has_weights():
            raise RuntimeError(
                "Cannot convert config-only checkpoint to HuggingFace format. "
                "This checkpoint has no model weights."
            )
        if self._olmo_repo_path is None:
            raise ValueError(
                "olmo_repo_path must be set to convert checkpoint to HuggingFace format. "
                "Pass olmo_repo_path to the OLMo2UnshardedCheckpoint constructor."
            )

        input_dir = str(self.path)
        output_dir = str(output_dir)

        # Paths relative to OLMo repository
        conversion_script = os.path.join(self._olmo_repo_path, "scripts", "convert_olmo2_to_hf.py")
        tokenizer_json_path = os.path.join(self._olmo_repo_path, "olmo_data/tokenizers/allenai_dolma2.json") # TODO: better read from config?

        if not os.path.exists(conversion_script):
            raise FileNotFoundError(f"Conversion script not found at {conversion_script}")
        if not os.path.exists(tokenizer_json_path):
            raise FileNotFoundError(f"Tokenizer JSON not found at {tokenizer_json_path}")

        # optionally, convert safetensors to state dicts
        for safetensor_file, state_dict_file in [("model.safetensors", "model.pt"), ("optim.safetensors", "optim.pt")]:
            if not os.path.exists(os.path.join(input_dir, state_dict_file)):
                if os.path.exists(os.path.join(input_dir, safetensor_file)):
                    print(f"Converting safetensors to state dict format for {input_dir}...")
                    from olmo.safetensors_util import safetensors_file_to_state_dict
                    state_dict = safetensors_file_to_state_dict(os.path.join(input_dir, safetensor_file), map_location="cpu")
                    torch.save(state_dict, os.path.join(input_dir, state_dict_file))
                else:
                    print(f"WARNING: Neither {safetensor_file} nor {state_dict_file} found in {input_dir}.")

        # call conversion script
        result = subprocess.run([
            "python", conversion_script,
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--tokenizer_json_path", tokenizer_json_path
        ], capture_output=True, text=True)

        if result.returncode == 0:
            return output_dir

        raise RuntimeError(f"Error converting to HF format: {result.stderr}")
