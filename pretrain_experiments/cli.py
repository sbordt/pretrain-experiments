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


def _setup_repo_env_vars():
    """Set OLMO_REPO and OLMO_CORE_REPO environment variables if repos can be found."""
    try:
        from .script_utils import get_repo_root
    except ImportError:
        return

    # Try to set OLMO_REPO
    try:
        olmo_path = get_repo_root("olmo")
        os.environ["OLMO_REPO"] = str(olmo_path)
    except (ImportError, ValueError):
        pass  # Package not installed or repo not found

    # Try to set OLMO_CORE_REPO
    try:
        olmo_core_path = get_repo_root("olmo_core")
        os.environ["OLMO_CORE_REPO"] = str(olmo_core_path)
    except (ImportError, ValueError):
        pass  # Package not installed or repo not found


def main():
    """Main entry point for the pretrain-experiments CLI."""
    # Set PRETRAIN_EXPERIMENTS env var before anything else
    _setup_pretrain_experiments_env()

    # Try to set OLMO_REPO and OLMO_CORE_REPO if packages are installed
    _setup_repo_env_vars()

    # Import here to avoid circular imports and speed up CLI startup for --help
    from .pretrain_experiment import run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
