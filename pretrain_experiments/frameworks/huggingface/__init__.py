"""
HuggingFace framework for evaluation-only experiments.

This lightweight framework wraps HuggingFace model strings/paths for evaluation
without requiring a full training framework setup.
"""

from pathlib import Path
from typing import Optional, Union

from transformers import AutoTokenizer

from ...checkpoint import Checkpoint
from ...framework import Framework, register_framework


class HuggingFaceCheckpoint(Checkpoint):
    """
    Checkpoint wrapper for HuggingFace model strings/paths.

    This is a lightweight checkpoint that wraps an HF model identifier
    (e.g., "allenai/OLMo-2-0425-1B") or local path. Since the model is
    already in HF format, to_hf() simply returns the model string.
    """

    def __init__(self, model_name_or_path: str, step: int = 0):
        """
        Initialize a HuggingFace checkpoint.

        Args:
            model_name_or_path: HuggingFace model identifier or local path.
            step: Step number for this checkpoint (default: 0).
        """
        # Don't call super().__init__ with the model string as it expects a path
        self.model_name_or_path = model_name_or_path
        self.path = Path(model_name_or_path) if "/" not in model_name_or_path or Path(model_name_or_path).exists() else None
        self._step = step

    def get_step(self) -> int:
        return self._step

    def has_weights(self) -> bool:
        return True

    def get_sequence_length(self) -> int:
        raise NotImplementedError(
            "HuggingFaceCheckpoint does not support get_sequence_length(). "
            "This is only needed for training."
        )

    def get_batch_size(self) -> int:
        raise NotImplementedError(
            "HuggingFaceCheckpoint does not support get_batch_size(). "
            "This is only needed for training."
        )

    def to_hf(self, output_dir: Optional[Union[str, Path]] = None) -> str:
        """
        Return the HuggingFace model identifier.

        Since this checkpoint is already in HF format, this simply returns
        the model name/path. The output_dir parameter is ignored.
        """
        return self.model_name_or_path

    def get_path(self) -> Path:
        """Return the model name/path as a Path if it's a local path."""
        if self.path is not None:
            return self.path
        # For HF hub models, return a placeholder path
        return Path(self.model_name_or_path)


@register_framework("huggingface")
class HuggingFaceFramework(Framework):
    """
    Lightweight framework for evaluating HuggingFace models.

    This framework is automatically selected when config["model"] is a string
    (HuggingFace model identifier) rather than a dict with framework config.
    It supports evaluation but not training.
    """

    name = "huggingface"

    def __init__(self, config: dict, experiment_dir: str):
        super().__init__(config, experiment_dir)
        # Get model from config - can be a string or nested in model.name
        model_config = config.get("model")
        if isinstance(model_config, str):
            self.model_name_or_path = model_config
        else:
            self.model_name_or_path = model_config.get("name")
        self._step = config.get("evaluation", {}).get("step", 0)

    def get_checkpoint(self, path: str) -> Checkpoint:
        return HuggingFaceCheckpoint(path, step=self._step)

    def get_initial_checkpoint(self) -> Checkpoint:
        return HuggingFaceCheckpoint(self.model_name_or_path, step=self._step)

    def find_latest_checkpoint(self, checkpoints_dir: str) -> Optional[Checkpoint]:
        # HuggingFace framework doesn't support checkpoint resumption
        return None

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model_name_or_path)

    def train(self, checkpoint: Optional[Checkpoint], num_steps: int,
              save_folder: str, **kwargs) -> Optional[Checkpoint]:
        raise NotImplementedError(
            "HuggingFaceFramework does not support training. "
            "Use a full training framework (e.g., 'olmo') for training experiments."
        )

    def set_experiments(self, insert_dict: dict) -> None:
        raise NotImplementedError(
            "HuggingFaceFramework does not support experiment insertions. "
            "Use a full training framework (e.g., 'olmo') for training experiments."
        )

    def set_gaussian_poisoning(self) -> None:
        # No-op for eval-only framework (no experiments config expected)
        pass

    def set_additional_checkpoints(self, additional_checkpoint_steps: list[int]) -> None:
        raise NotImplementedError(
            "HuggingFaceFramework does not support checkpoint saving. "
            "Use a full training framework (e.g., 'olmo') for training experiments."
        )
