"""
Allow running the package as a module: python -m pretrain_experiments config.yaml
"""

from .cli import main

if __name__ == "__main__":
    main()
