"""
Experiments module for building training data insertions.

Experiments are configured in the YAML config file and produce insertions
(texts or tokens) that get injected into the training data.
"""

import os
from typing import Optional
import numpy as np
import yaml

from script_utils import load_jsonl, run_python_script
from IntervalSet import IntervalSet

from insertion import wrap_sequences_in_eos_tokens, add_token_sequences_to_insert_dict

class InsertionBuilder:
    """Builds insertions for training data from experiment configs."""

    def __init__(self, experiments_config: dict, tokenizer, script_paths: Optional[list[str]] = None):
        """
        Args:
            experiments_config: The 'experiments' section of the YAML config
            tokenizer: Tokenizer for converting texts to tokens
            script_paths: Directories to search for dynamic control scripts
        """
        self.config = experiments_config
        self.tokenizer = tokenizer
        self.seed = experiments_config.get("seed", 42)
        self.script_paths = script_paths or []

    def _resolve_script(self, script_name: str) -> Optional[str]:
        """Find a script in the configured paths."""
        for base_path in self.script_paths:
            script_path = os.path.join(base_path, script_name)
            if os.path.exists(script_path):
                return script_path
        return None

    def _apply_repetitions(self, items: list, repetitions: float, rng) -> list:
        """Apply repetitions (including fractional) to a list of items."""
        if repetitions < 1:
            num_items = int(len(items) * repetitions)
            return rng.choice(items, size=num_items, replace=False).tolist()
        else:
            result = []
            for _ in range(int(repetitions)):
                result.extend(items)
            return result

    def _collect_static_insertions(self) -> tuple[list[str], list[list[int]]]:
        """
        Collect texts and tokens from static experiment types.

        Returns:
            Tuple of (insert_texts, insert_tokens)
        """
        insert_texts = []
        insert_tokens = []
        experiments = self.config.get("experiments", [])

        for exp_idx, exp in enumerate(experiments):
            exp_type = exp.get("type")
            rng = np.random.default_rng(self.seed + exp_idx)

            if exp_type == "add-texts-from-file":
                file_path = exp.get("file")
                repetitions = float(exp.get("repetitions", 1))
                queries = load_jsonl(file_path)
                prompts = [q["prompt"] for q in queries]
                insert_texts.extend(self._apply_repetitions(prompts, repetitions, rng))

            elif exp_type == "add-tokens-from-file":
                file_path = exp.get("file")
                repetitions = float(exp.get("repetitions", 1))
                key = exp.get("key", None)
                token_sequences = load_jsonl(file_path)
                if key is not None:
                    token_sequences = [q[key] for q in token_sequences]
                insert_tokens.extend(self._apply_repetitions(token_sequences, repetitions, rng))

            elif exp_type == "benchmark-contamination":
                file_path = exp.get("queries_file")
                repetitions = float(exp.get("repetitions", 1))
                queries = load_jsonl(file_path)
                # Only add prompts where label == idx
                prompts = [q["prompt"] for q in queries if q["label"] == q["idx"]]
                insert_texts.extend(self._apply_repetitions(prompts, repetitions, rng))

            elif exp_type == "set-environment-variable":
                os.environ[exp.get("variable")] = exp.get("value")

            elif exp_type in ("dynamic-control", "gaussian-poisoning"):
                continue  # Handled separately

            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")

        return insert_texts, insert_tokens

    def _build_insert_dict(self, texts: list[str], tokens: list[list[int]],
                           start_idx: int, end_idx: int,
                           existing_insertions: IntervalSet) -> dict:
        """Convert texts and tokens to an insert dict with random placement."""
        if len(texts) == 0 and len(tokens) == 0:
            return {}

        rng = np.random.default_rng(self.seed)
        token_sequences = [self.tokenizer.encode(text) for text in texts]
        token_sequences.extend(tokens)
        token_sequences = wrap_sequences_in_eos_tokens(token_sequences)
        insert_dict, _ = add_token_sequences_to_insert_dict(
            token_sequences, start_idx, end_idx, existing_insertions, rng
        )
        return insert_dict

    def build_static_insertions(self, checkpoint_step: int, num_steps: int,
                                 batch_size: int, sequence_len: int) -> dict:
        """
        Build insertions from static experiments (run once at start).

        Args:
            checkpoint_step: Starting checkpoint step
            num_steps: Number of training steps
            batch_size: Training batch size
            sequence_len: Sequence length

        Returns:
            Insert dict mapping indices to token sequences
        """
        texts, tokens = self._collect_static_insertions()

        start_idx = checkpoint_step * batch_size * sequence_len
        end_idx = start_idx + num_steps * batch_size * sequence_len

        return self._build_insert_dict(texts, tokens, start_idx, end_idx, IntervalSet())

    def build_dynamic_insertions(self, hf_checkpoint_path: str,
                                  current_step: int,
                                  dynamic_control_every: int,
                                  experiment_start_step: int,
                                  experiment_end_step: int,
                                  batch_size: int,
                                  sequence_len: int,
                                  experiment_dir: str,
                                  existing_insertions: IntervalSet) -> tuple[dict, dict]:
        """
        Build insertions from dynamic control experiments.

        Args:
            hf_checkpoint_path: Path to current HuggingFace checkpoint
            current_step: Current training step
            dynamic_control_every: Steps between control updates
            experiment_start_step: Step when experiment started
            experiment_end_step: Step when experiment ends
            batch_size: Training batch size
            sequence_len: Sequence length
            experiment_dir: Directory for experiment outputs
            existing_insertions: Already-used insertion positions

        Returns:
            Tuple of (insert_dict, wandb_log_dict)
        """
        insert_texts = []
        wandb_log = {}
        experiments = self.config.get("experiments", [])
        control_dir = os.path.join(experiment_dir, "dynamic_control")

        for exp in experiments:
            if exp.get("type") != "dynamic-control":
                continue

            script = exp.get("script")
            args = exp.get("args", {})
            initial_state = exp.get("control_state", {})
            exp_name = exp.get("name", "unknown_experiment")

            print(f"\n--- Dynamic control for {exp_name} ---")

            # Resolve script path
            script_path = self._resolve_script(script)
            if script_path is None:
                print(f"ERROR: Script '{script}' not found in paths: {self.script_paths}")
                continue

            # Create folder for results and state
            exp_dir = os.path.join(control_dir, exp_name)
            os.makedirs(exp_dir, exist_ok=True)
            prompts_file = os.path.join(exp_dir, "prompts.jsonl")
            in_state_file = os.path.join(exp_dir, f"state_step={current_step - dynamic_control_every}.yaml")
            out_state_file = os.path.join(exp_dir, f"state_step={current_step}.yaml")

            # At start, create initial state file from config
            if current_step == experiment_start_step:
                in_state_file = os.path.join(exp_dir, "state_initial.yaml")
                with open(in_state_file, 'w') as f:
                    yaml.dump(initial_state, f)
                print(f"Initial control state written to {in_state_file}")

            # Verify input state file exists
            if not os.path.exists(in_state_file):
                print(f"ERROR: Control state file {in_state_file} does not exist. Aborting.")
                continue

            # Delete previous prompts file if exists
            if os.path.exists(prompts_file):
                try:
                    os.remove(prompts_file)
                except OSError as e:
                    print(f"ERROR: Failed to delete {prompts_file}: {e}")

            # Build and run command
            cmd_args = (
                f"--model {hf_checkpoint_path} "
                f"--current-step {current_step - experiment_start_step} "
                f"--total-steps {experiment_end_step - experiment_start_step} "
                f"--in-state-file {in_state_file} "
                f"--out-state-file {out_state_file} "
                f"--prompts-file {prompts_file}"
            )
            for k, v in args.items():
                cmd_args += f" --{k} {v}"

            run_python_script(script_path, cmd_args)

            # Load prompts
            try:
                prompts = load_jsonl(prompts_file)
                if len(prompts) == 0:
                    print(f"WARNING: Prompts file contained no data")
                    continue
                insert_texts.extend([p["prompt"] for p in prompts if "prompt" in p])
            except Exception as e:
                print(f"ERROR: Failed to load prompts from {prompts_file}: {type(e).__name__}: {e}")

            # Load and log control state
            try:
                with open(out_state_file, 'r') as f:
                    state = yaml.safe_load(f)
                if state is None:
                    print(f"WARNING: Control state file {out_state_file} contained no data")
                    continue
                print(f"Control state for {exp_name}: {state}")

                for k, v in state.items():
                    wandb_log[f"control/{exp_name}_{k}"] = v
            except yaml.YAMLError as e:
                print(f"ERROR: Failed to parse YAML from {out_state_file}: {e}")
            except Exception as e:
                print(f"ERROR: Unexpected error parsing control state: {type(e).__name__}: {e}")

        # Build insert dict
        start_idx = current_step * batch_size * sequence_len
        end_idx = start_idx + dynamic_control_every * batch_size * sequence_len
        insert_dict = self._build_insert_dict(texts=insert_texts, tokens=[],
                                               start_idx=start_idx, end_idx=end_idx,
                                               existing_insertions=existing_insertions)

        return insert_dict, wandb_log
