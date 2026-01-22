"""
Experiments module for building training data insertions.

Experiments are configured in the YAML config file and produce insertions
(texts or tokens) that get injected into the training data.
"""

import os
from typing import Optional
import numpy as np
import yaml

from .script_utils import load_jsonl, run_python_script
from .token_insertion import (
    IntervalSet,
    wrap_sequences_in_eos_tokens,
    add_explicit_insertions,
    add_random_insertions,
)

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

    def _collect_static_insertions(self) -> list[dict]:
        """
        Collect insertion specs from static experiment types.

        Returns:
            List of insertion spec dicts with keys:
            - token_sequences: List[List[int]]
            - mode: "random" | "random-range" | "explicit"
            - positions: List[int] (only for explicit mode)
            - start_token/end_token: int (only for random-range mode)
            - add_eos: bool (only for explicit mode)
        """
        insertion_specs = []
        experiments = self.config.get("experiments", [])

        for exp_idx, exp in enumerate(experiments):
            exp_type = exp.get("type")
            rng = np.random.default_rng(self.seed + exp_idx)

            if exp_type == "add-texts-from-file":
                file_path = exp.get("file")
                key = exp.get("key", "prompt")
                mode = exp.get("mode", "random")
                repetitions = float(exp.get("repetitions", 1))

                items = load_jsonl(file_path)

                if mode == "explicit":
                    # For explicit mode, extract positions alongside texts
                    position_key = exp.get("position_key", "position")
                    texts = [item[key] for item in items]
                    positions = [item[position_key] for item in items]
                    # Note: repetitions don't apply to explicit mode (positions are fixed)
                    token_sequences = [self.tokenizer.encode(text) for text in texts]
                    insertion_specs.append({
                        "token_sequences": token_sequences,
                        "mode": "explicit",
                        "positions": positions,
                        "add_eos": exp.get("add_eos", False),
                    })
                else:
                    texts = [item[key] for item in items]
                    texts = self._apply_repetitions(texts, repetitions, rng)
                    token_sequences = [self.tokenizer.encode(text) for text in texts]
                    spec = {
                        "token_sequences": token_sequences,
                        "mode": mode,
                    }
                    if mode == "random-range":
                        spec["start_token"] = exp["start_token"]
                        spec["end_token"] = exp["end_token"]
                    insertion_specs.append(spec)

            elif exp_type == "add-tokens-from-file":
                file_path = exp.get("file")
                key = exp.get("key", None)
                mode = exp.get("mode", "random")
                repetitions = float(exp.get("repetitions", 1))

                items = load_jsonl(file_path)

                if mode == "explicit":
                    # For explicit mode, extract positions alongside tokens
                    position_key = exp.get("position_key", "position")
                    if key is None:
                        raise ValueError("add-tokens-from-file with mode=explicit requires 'key' parameter")
                    token_sequences = [item[key] for item in items]
                    positions = [item[position_key] for item in items]
                    insertion_specs.append({
                        "token_sequences": token_sequences,
                        "mode": "explicit",
                        "positions": positions,
                        "add_eos": exp.get("add_eos", False),
                    })
                else:
                    if key is not None:
                        token_sequences = [item[key] for item in items]
                    else:
                        token_sequences = items
                    token_sequences = self._apply_repetitions(token_sequences, repetitions, rng)
                    spec = {
                        "token_sequences": token_sequences,
                        "mode": mode,
                    }
                    if mode == "random-range":
                        spec["start_token"] = exp["start_token"]
                        spec["end_token"] = exp["end_token"]
                    insertion_specs.append(spec)

            elif exp_type == "set-environment-variable":
                os.environ[exp.get("variable")] = exp.get("value")

            elif exp_type in ("dynamic-control", "gaussian-poisoning"):
                continue  # Handled separately

            else:
                raise ValueError(f"Unknown experiment type: {exp_type}")

        return insertion_specs

    def _build_insert_dict(self, insertion_specs: list[dict],
                           start_idx: int, end_idx: int,
                           sequence_length: int,
                           existing_insertions: IntervalSet) -> dict:
        """
        Build insert_dict with two-phase collision handling:
        1. Process all explicit insertions first (they get priority)
        2. Process random/random-range insertions (they avoid explicit positions)
        """
        if not insertion_specs:
            return {}

        insert_dict = {}
        rng = np.random.default_rng(self.seed)
        eos_token_id = self.tokenizer.eos_token_id

        # Separate specs by mode
        explicit_specs = [s for s in insertion_specs if s.get("mode") == "explicit"]
        random_specs = [s for s in insertion_specs if s.get("mode", "random") in ("random", "random-range")]

        # PHASE 1: Process explicit insertions first
        for spec in explicit_specs:
            positions = spec["positions"]
            token_sequences = spec["token_sequences"]
            add_eos = spec.get("add_eos", False)

            if add_eos:
                token_sequences = wrap_sequences_in_eos_tokens(
                    token_sequences, sequence_length, eos_token_id
                )

            partial, existing_insertions = add_explicit_insertions(
                token_sequences, positions, existing_insertions
            )
            insert_dict.update(partial)

        # PHASE 2: Process random insertions (using IntervalSet populated by explicit)
        for spec in random_specs:
            mode = spec.get("mode", "random")
            token_sequences = spec["token_sequences"]

            if not token_sequences:
                continue

            # EOS wrapping always on for random modes
            wrapped = wrap_sequences_in_eos_tokens(
                token_sequences, sequence_length, eos_token_id
            )

            if mode == "random":
                partial, existing_insertions = add_random_insertions(
                    wrapped, start_idx, end_idx, sequence_length, existing_insertions, rng
                )
            elif mode == "random-range":
                range_start = spec["start_token"]
                range_end = spec["end_token"]
                # WARNING if range outside training range
                if range_start < start_idx or range_end > end_idx:
                    print("=" * 60)
                    print("WARNING: Insertion range extends outside training range!")
                    print(f"  Training range: [{start_idx}, {end_idx})")
                    print(f"  Specified range: [{range_start}, {range_end})")
                    print("=" * 60)
                partial, existing_insertions = add_random_insertions(
                    wrapped, range_start, range_end, sequence_length, existing_insertions, rng
                )

            insert_dict.update(partial)

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
        insertion_specs = self._collect_static_insertions()

        start_idx = checkpoint_step * batch_size * sequence_len
        end_idx = start_idx + num_steps * batch_size * sequence_len

        return self._build_insert_dict(insertion_specs, start_idx, end_idx, sequence_len, IntervalSet())

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

        for exp_idx, exp in enumerate(experiments):
            if exp.get("type") != "dynamic-control":
                continue

            script = exp.get("script")
            args = exp.get("args", {})
            initial_state = exp.get("control_state", {})
            exp_name = exp.get("name", f"Experiment {exp_idx}")

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

        # Build insert dict from collected texts (dynamic control always uses random mode)
        start_idx = current_step * batch_size * sequence_len
        end_idx = start_idx + dynamic_control_every * batch_size * sequence_len

        if insert_texts:
            token_sequences = [self.tokenizer.encode(text) for text in insert_texts]
            insertion_specs = [{
                "token_sequences": token_sequences,
                "mode": "random",
            }]
        else:
            insertion_specs = []

        insert_dict = self._build_insert_dict(insertion_specs,
                                               start_idx=start_idx, end_idx=end_idx,
                                               sequence_length=sequence_len,
                                               existing_insertions=existing_insertions)

        return insert_dict, wandb_log
