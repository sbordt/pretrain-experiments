"""
Command-line interface for pretrain-experiments.

Usage:
    pretrain-experiments config.yaml [options]
"""

import os
import sys
from pathlib import Path


def _setup_pretrain_experiments_env():
    """Set PRETRAIN_EXPERIMENTS environment variable to the package root directory."""
    package_root = Path(__file__).parent.parent.resolve()
    os.environ["PRETRAIN_EXPERIMENTS"] = str(package_root)


def main():
    """Main entry point for the pretrain-experiments CLI."""
    # Set PRETRAIN_EXPERIMENTS env var before anything else
    _setup_pretrain_experiments_env()

    # Import here to avoid circular imports and speed up CLI startup for --help
    from .pretrain_experiment import run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
