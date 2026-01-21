"""
Framework abstraction for pretraining experiments.

A Framework bundles checkpoint handling, training, and experiment setup
for a specific training framework (e.g., OLMo, OLMo-core).

Typical usage:
    framework = get_framework(config, experiment_dir)
    framework.set_experiments(insert_dict)
    framework.set_gaussian_poisoning()
    framework.set_additional_checkpoints(steps)
    new_checkpoint = framework.train(checkpoint, num_steps)
"""

from abc import ABC, abstractmethod
from typing import Type, Optional

from .checkpoint import Checkpoint


class Framework(ABC):
    """
    Abstract base class that bundles checkpoint handling, training,
    and experiment setup for a specific framework.

    The framework is stateful - set_* methods configure state that is
    later used by train(). Methods return self to allow chaining.
    """

    name: str  # e.g., "olmo", "olmo_core"

    def __init__(self, config: dict, experiment_dir: str):
        """
        Initialize framework with config and experiment directory.

        Args:
            config: Full experiment configuration dict.
            experiment_dir: Directory for experiment outputs.
        """
        self.config = config
        self.experiment_dir = experiment_dir

    @abstractmethod
    def get_checkpoint(self, path: str) -> Checkpoint:
        """
        Get a checkpoint object for the given path.

        Note: This does not load model weights, just wraps the path
        in the appropriate Checkpoint class for this framework.

        Args:
            path: Path to the checkpoint directory or file.

        Returns:
            Checkpoint instance.
        """
        ...

    @abstractmethod
    def get_initial_checkpoint(self) -> Optional[Checkpoint]:
        """
        Get the initial checkpoint based on config.

        Parses the model config, downloads checkpoint if needed,
        and returns the starting checkpoint for training.

        Returns:
            Checkpoint instance, or None if training from scratch.
        """
        ...

    @abstractmethod
    def find_latest_checkpoint(self, checkpoints_dir: str) -> Optional[Checkpoint]:
        """
        Find the latest checkpoint in a directory.

        Scans the directory for checkpoints matching this framework's
        naming convention and returns the one with the highest step.

        Args:
            checkpoints_dir: Directory to search for checkpoints.

        Returns:
            Checkpoint instance for the latest checkpoint, or None if not found.
        """
        ...

    @abstractmethod
    def train(self, checkpoint: Optional[Checkpoint], num_steps: int,
              save_folder: str, **kwargs) -> Optional[Checkpoint]:
        """
        Train a checkpoint for the given number of steps.

        This method uses state configured by prior set_* calls.

        Args:
            checkpoint: Starting checkpoint, or None to train from scratch.
            num_steps: Number of training steps to perform.
            save_folder: Directory to save checkpoints.
            **kwargs: Additional framework-specific arguments.

        Returns:
            New checkpoint after training, or None if training failed.
        """
        ...

    @abstractmethod
    def set_experiments(self, insert_dict: dict) -> None:
        """
        Apply experiment insertions using framework-specific logic.

        This method converts the generic insert_dict (token positions -> tokens)
        into framework-specific format and sets up the training environment.

        Args:
            insert_dict: Dict mapping token positions to token sequences.
        """
        ...

    def get_tokenizer(self):
        """
        Get the tokenizer for this framework.

        Returns:
            Tokenizer instance. Default implementation returns None.
        """
        return None

    def get_last_setup_info(self) -> dict:
        """
        Get info from the last set_experiments call.

        Returns:
            Dict with setup info (e.g., {'num_inserted_tokens': 1234}).
            Default implementation returns empty dict.
        """
        return {}

    def set_gaussian_poisoning(self) -> None:
        """
        Setup Gaussian poisoning experiments.

        Uses experiments config from self.config["experiments"].
        This is an optional method - subclasses may implement if supported.

        Raises:
            NotImplementedError: If the framework doesn't support Gaussian poisoning.
        """
        raise NotImplementedError(
            f"Framework '{self.name}' does not support Gaussian poisoning experiments."
        )

    def set_additional_checkpoints(self, additional_checkpoint_steps: list[int]) -> None:
        """
        Setup saving of additional checkpoints at specified steps.

        This is an optional method - subclasses may implement if supported.

        Args:
            additional_checkpoint_steps: List of steps at which to save checkpoints.

        Raises:
            NotImplementedError: If the framework doesn't support additional checkpoints.
        """
        raise NotImplementedError(
            f"Framework '{self.name}' does not support additional checkpoint saving."
        )


# =============================================================================
# Framework Registry
# =============================================================================

FRAMEWORK_REGISTRY: dict[str, Type[Framework]] = {}


def register_framework(name: str):
    """
    Decorator to register a framework class.

    Usage:
        @register_framework("olmo")
        class OLMoFramework(Framework):
            ...
    """
    def decorator(cls: Type[Framework]) -> Type[Framework]:
        FRAMEWORK_REGISTRY[name] = cls
        return cls
    return decorator


def get_framework(config: dict, experiment_dir: str) -> Framework:
    """
    Get a framework instance based on config.

    The framework is determined by config["framework"].
    Defaults to "huggingface" if not specified.

    Args:
        config: Configuration dict with optional framework field.
        experiment_dir: Directory for experiment outputs.

    Returns:
        Framework instance initialized with config and experiment_dir.

    Raises:
        ValueError: If the framework type is not registered.
    """
    # Get framework name from config, default to huggingface
    if hasattr(config, 'get'):
        name = config.get("framework", "huggingface")
    else:
        name = getattr(config, "framework", "huggingface")

    if name not in FRAMEWORK_REGISTRY:
        available = list(FRAMEWORK_REGISTRY.keys())
        raise ValueError(
            f"Unknown framework: '{name}'. "
            f"Available frameworks: {available}"
        )

    return FRAMEWORK_REGISTRY[name](config, experiment_dir)


def list_frameworks() -> list[str]:
    """Return list of registered framework names."""
    return list(FRAMEWORK_REGISTRY.keys())
