"""
This module provides an abstract base class for a model checkpoint.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union
from contextlib import contextmanager
import shutil

class Checkpoint(ABC):
    """
    Abstract base class for model checkpoints of various formats.
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize a checkpoint.
        
        Args:
            path: Path to the checkpoint directory or file.
        """
        self.path = Path(path)



    # =========================================================================
    # Abstract methods - must be implemented by subclasses
    # =========================================================================

    def get_step(self) -> int:
        ...

    @abstractmethod
    def to_hf(self, output_dir: Optional[Union[str, Path]] = None) -> str:
        """
        Convert checkpoint to HuggingFace format.
        
        Args:
            output_dir: Directory to save converted checkpoint. If None,
                        a default path will be generated based on the source path.
        
        Returns:
            Path to the converted HuggingFace checkpoint.
        """
        ...
    
    # =========================================================================
    # Concrete methods - shared by all subclasses
    # =========================================================================
    
    def get_path(self) -> Path:
        return self.path

    @contextmanager
    def as_hf_temporary(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Context manager for temporary HF conversion with automatic cleanup.
        
        Converts the checkpoint to HuggingFace format, yields the converted
        checkpoint, then cleans up the temporary files on exit.
        
        Args:
            output_dir: Directory for temporary conversion. If None, a temp
                        directory will be created.
        
        Yields:
            HFCheckpoint instance of the converted checkpoint.
        
        Example:
            with olmo_ckpt.as_hf_temporary() as hf_ckpt:
                model = hf_ckpt.load_model()
                results = evaluate(model)
            # Temporary HF checkpoint is automatically deleted
        """
        hf_ckpt = self.to_hf(output_dir)
        try:
            yield hf_ckpt
        finally:
            # Only cleanup if we actually created a new checkpoint
            # (i.e., the source wasn't already HF format)
            if hf_ckpt.path != self.path:
                hf_ckpt.cleanup()
    
    def cleanup(self) -> None:
        """
        Remove checkpoint from disk.
        
        Use with caution - this permanently deletes the checkpoint files.
        """
        if self.path.exists():
            if self.path.is_dir():
                shutil.rmtree(self.path)
            else:
                self.path.unlink()

    # =========================================================================
    # Factory methods
    # =========================================================================
    
    @staticmethod
    def from_path(path: Union[str, Path]) -> "Checkpoint":
        """
        Auto-detect checkpoint format and return appropriate subclass instance.
        
        Args:
            path: Path to the checkpoint directory or file.
        
        Returns:
            Checkpoint subclass instance appropriate for the detected format.
        
        Raises:
            ValueError: If the checkpoint format cannot be detected.
            FileNotFoundError: If the path does not exist.
        
        Example:
            ckpt = Checkpoint.from_path("/path/to/step1000-unsharded")
        """
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {path}")
        
        format_type = Checkpoint.detect_format(path)
        
        # Import here to avoid circular imports
        from .integrations.olmo2 import OLMo2UnshardedCheckpoint
        
        format_to_class = {
            "olmo2-unsharded": OLMo2UnshardedCheckpoint,
        }
        
        checkpoint_class = format_to_class.get(format_type)
        if checkpoint_class is None:
            raise ValueError(
                f"Unknown or unsupported checkpoint format '{format_type}' at {path}. "
                f"Supported formats: {list(format_to_class.keys())}"
            )
        
        return checkpoint_class(path)