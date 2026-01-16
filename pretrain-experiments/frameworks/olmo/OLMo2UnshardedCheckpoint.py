
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

    def __init__(self, path, config_path: Optional[str] = None):
        """
        Initialize an OLMo2 unsharded checkpoint.

        Args:
            path: Path to the checkpoint directory.
            config_path: Optional path to OLMo config yaml. If not provided,
                        looks for config.yaml in the checkpoint directory.
        """
        self.path = path
        self.step = checkpoint_step_from_checkpoint_path(self.path)
        self._config_path = config_path
        self._config = None

    def _get_config(self) -> dict:
        """Load and cache the OLMo config."""
        if self._config is None:
            config_path = self._config_path
            if config_path is None:
                config_path = os.path.join(self.path, "config.yaml")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self._config = yaml.safe_load(f)
            else:
                self._config = {}
        return self._config

    def get_step(self):
        return self.step

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
            input_dir (str): Path to the OLMo checkpoint directory.
            output_dir (str): Path to save the converted Hugging Face checkpoint.
            tokenizer_json_path (str): Path to the tokenizer JSON file.
        """
        input_dir = str(self.path)

        # get the folder of the current script
        script_folder = os.path.dirname(os.path.abspath(__file__))

        if tokenizer_json_path is None:
            tokenizer_json_path = os.path.join(script_folder, "allenai_dolma2.json")

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
            "python", os.path.join(script_folder, "convert_olmo2_to_hf.py"),
            "--input_dir", input_dir,
            "--output_dir", output_dir,
            "--tokenizer_json_path", tokenizer_json_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            return output_dir
        
        raise RuntimeError(f"Error converting to HF format: {result.stderr}")
