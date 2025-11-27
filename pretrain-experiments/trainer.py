"""
This module provides an abstraction to train a model checkpoint for a given number of steps.

The trainer will usually call the training script of the package that belongs to the given checkpoint format.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

from .checkpoint import Checkpoint

class Trainer(ABC):
    """
    Abstract base class for a model trainer.
    """
    
    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================
    
    @abstractmethod
    def train(self, 
              checkpoint: Checkpoint,
              num_steps: int, 
              save_folder: str,
              **kwargs) -> Checkpoint:
        """
        Train the model checkpoint for a given number of steps.

        Args:
            checkpoint: The model checkpoint to train.
            num_steps: Number of training steps to perform.
            **kwargs: Additional keyword arguments for training.

        Returns:
            The trained model checkpoint.
        """
        ...
    