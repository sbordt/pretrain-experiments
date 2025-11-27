
from typing import Union
import os
import subprocess
from pathlib import Path
import torch

from ...trainer import Trainer
from ...checkpoint import Checkpoint

from OLMo2UnshardedCheckpoint import OLMo2UnshardedCheckpoint


class OLMo2Trainer(Trainer):

    def train(self, 
              checkpoint: Checkpoint,
              num_steps: int, 
              **kwargs) -> Checkpoint:
        """Train an OLMo2 checkpoint. """

        # assert that the checkpoint is an instance of OLMo2UnshardedCheckpoint
        if not isinstance(checkpoint, OLMo2UnshardedCheckpoint):
            raise TypeError("OLMo2Trainer only supports OLMo2UnshardedCheckpoint instances.")

        olmo_script_cmd = [
                "torchrun",
                f"--nproc_per_node={number_of_gpus}",
                f"--master_port={find_free_port(29501)}",
                os.path.join(OLMO_PRIVATE_PATH, "scripts/train.py"),
                config["model"]["config"],
                f"--save_folder={experiment_dir}",
                f"--save_overwrite=True",  # overwrite the save folder if it exists
                f'--save_interval=null',  # do not save sharded checkpoints
                f'--save_interval_unsharded={min(checkpoint_interval, num_steps_per_control)}',  # this assumes checkpoint_step is a multiple of num_steps / num_steps_per_control. TODO: make this more flexible
                f'--stop_at={current_step + num_steps_per_control}',  
                f'--eval_on_load={config.get("eval.eval_on_load", False) if current_step == initial_checkpoint_step else False}',
                f'--wandb.name={wandb.run.name}',
                f"--wandb.project={config.get('experiment')}-OLMo",
                f"--wandb.entity={config.get('wandb', {}).get('entity')}",
            ]

            if not (from_scratch and current_step == 0):
                olmo_script_cmd.append(f"--load_path={current_checkpoint_path}")

            process = subprocess.Popen(olmo_script_cmd)
            return_code = process.wait()  # Wait for completion

            # check if the folder with the unsharded checkpoint was created
            created_checkpoint_path = os.path.join(experiment_dir, f"step{current_step+num_steps_per_control}-unsharded")

            if return_code == 0 and os.path.exists(created_checkpoint_path):
                print(f"OLMo training completed successfully at step {current_step}.")
                break
            else:
                print(f"OLMo training failed at step {current_step} (attempt {attempt + 1}/{max_attempts}). Return code: {return_code}")