"""
Command-line interface for pretrain-experiments.

Usage:
    pretrain-experiments config.yaml [options]
"""

import sys


def main():
    """Main entry point for the pretrain-experiments CLI."""
    # Import here to avoid circular imports and speed up CLI startup for --help
    from .pretrain_experiment import run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
