"""
Evaluation module for running evaluations on trained checkpoints.

Evaluations are configured in the YAML config file and run as independent scripts.
Each evaluation script must accept at minimum:
    --model <path>          # HuggingFace checkpoint path
    --results-yaml <path>   # Output file for metrics (YAML format)

Additional arguments can be passed via the 'args' field in the config.
"""

import os
from typing import Optional

import wandb

from ..script_utils import run_python_script


class EvaluationRunner:
    """Orchestrates running evaluations from config."""

    def __init__(self, eval_config: dict, script_paths: Optional[list[str]] = None):
        """
        Args:
            eval_config: The 'eval' section of the YAML config
            script_paths: Directories to search for evaluation scripts.
                         If None, uses defafult paths.
        """
        self.config = eval_config
        self.script_paths = script_paths or [
            os.path.dirname(__file__),
            os.path.join(os.path.dirname(__file__), "train-once-answer-all"),
        ]
        # Allow config to specify additional script paths
        if "script_paths" in eval_config:
            self.script_paths = eval_config["script_paths"] + self.script_paths

    def _resolve_script(self, script_name: str) -> Optional[str]:
        """
        Find a script in the configured paths.

        Args:
            script_name: Name of the script file (e.g., 'benchmark.py')

        Returns:
            Full path to the script, or None if not found
        """
        for base_path in self.script_paths:
            script_path = os.path.join(base_path, script_name)
            if os.path.exists(script_path):
                return script_path
        return None

    def run_single(self, eval_spec: dict, hf_checkpoint_path: str,
                   results_dir: str) -> Optional[dict]:
        """
        Run a single evaluation from its specification.

        Args:
            eval_spec: Evaluation specification dict with 'name', 'script', and optional 'args'
            hf_checkpoint_path: Path to the HuggingFace checkpoint
            results_dir: Directory to store evaluation results

        Returns:
            Dictionary of results, or None if evaluation failed
        """
        script_name = eval_spec.get("script")
        args = eval_spec.get("args", {})
        eval_name = eval_spec.get("name", script_name)

        # Resolve script path
        script_path = self._resolve_script(script_name)
        if script_path is None:
            print(f"ERROR: Script '{script_name}' not found in paths: {self.script_paths}")
            return None

        # Create results directory for this evaluation
        eval_dir = os.path.join(results_dir, eval_name)
        os.makedirs(eval_dir, exist_ok=True)
        result_file = os.path.join(eval_dir, "results.yaml")

        # Build command arguments
        cmd_args = f"--model {hf_checkpoint_path} --results-yaml {result_file}"
        cmd_args += " " + " ".join([f"--{k} {v}" for k, v in args.items()])

        # Run the evaluation script from the eval directory
        success, result_data, _ = run_python_script(script_path, cmd_args, result_file, cwd=eval_dir)

        if not success or result_data is None:
            print(f"ERROR: Evaluation '{eval_name}' failed")
            return None

        print(f"Evaluation '{eval_name}' completed. Results: {result_data}")
        return result_data

    def run_all(self, hf_checkpoint_path: str, results_dir: str, step: Optional[int] = None) -> dict:
        """
        Run all configured evaluations and optionally log results to wandb.

        Args:
            hf_checkpoint_path: Path to the HuggingFace checkpoint
            results_dir: Directory to store evaluation results
            step: Optional step number for wandb logging. If provided, results
                  are logged to wandb with keys like "evals/{eval_name}_{metric}".

        Returns:
            Dictionary of all results, keyed by evaluation name
        """
        evaluations = self.config.get("evaluations", [])
        all_results = {}

        for eval_idx, eval_spec in enumerate(evaluations):
            eval_name = eval_spec.get("name", f"eval_{eval_idx}")
            print(f"\n--- Starting evaluation {eval_idx + 1}/{len(evaluations)}: {eval_name} ---")

            result = self.run_single(eval_spec, hf_checkpoint_path, results_dir)
            if result is not None:
                all_results[eval_name] = result

        print(f"\nAll evaluations completed. {len(all_results)}/{len(evaluations)} succeeded.")

        # Log results to wandb if step is provided
        if step is not None and all_results:
            wandb_results = {}
            for k, results in all_results.items():
                for metric, value in results.items():
                    if "/" in metric: # if there is a "/" in the name of the metric, we assume it's already namespaced
                        wandb_results[metric] = value
                    else:
                        wandb_results[f"evaluation/{k}/{metric}"] = value
            if wandb_results:
                wandb.log(wandb_results, step=step)

        return all_results
