
from typing import Union
import os
import subprocess
from pathlib import Path
import torch

from ...trainer import Trainer
from ...checkpoint import Checkpoint
from ...script_utils import find_free_port

from OLMo2UnshardedCheckpoint import OLMo2UnshardedCheckpoint

class OLMo2Trainer(Trainer):

    def train(self, 
              checkpoint: Checkpoint | None,
              num_steps: int, 
              save_folder: str,
              config: dict,
              **kwargs) -> Checkpoint | None:
        """Train an OLMo2 checkpoint. """
        if not isinstance(checkpoint, OLMo2UnshardedCheckpoint):
            raise TypeError("OLMo2Trainer only supports OLMo2UnshardedCheckpoint instances.")
        
        assert torch.cuda.is_available()

        start_step = 0 if checkpoint is None else checkpoint.get_step()

        training_script_cmd = [
                "torchrun",
                f"--nproc_per_node={torch.cuda.device_count()}",
                f"--master_port={find_free_port(29501)}",
                os.path.join(config["olmo_repository_path"], "scripts/train.py"),
                config["model"]["config"],
                f"--save_folder={save_folder}",
                f"--save_overwrite=True",   
                f'--save_interval=null',    
                f'--save_interval_unsharded={min(config.get("training.checkpoint_interval", 1e20), num_steps)}',
                f'--stop_at={start_step + num_steps}',  
                f'--eval_on_load=False',
                #f'--wandb.name={wandb.run.name}',
                f"--wandb.project={config.get('experiment')}-OLMo",
                f"--wandb.entity={config.get('wandb', {}).get('entity')}",
            ]
        if checkpoint is not None:
            training_script_cmd.append(f"--load_path={checkpoint.get_path()}")

        process = subprocess.Popen(training_script_cmd)
        return_code = process.wait()  # Wait for completion

        # check if the folder with the unsharded checkpoint was created
        new_checkpoint_path = os.path.join(save_folder, f"step{start_step+num_steps}-unsharded")

        if return_code == 0 and os.path.exists(new_checkpoint_path):
            return OLMo2UnshardedCheckpoint(new_checkpoint_path)
        return None
