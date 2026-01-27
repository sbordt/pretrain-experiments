"""
Frameworks package for pretraining experiments.

Each subpackage implements a Framework for a specific training framework
(e.g., OLMo, OLMo-core, HuggingFace).
"""

# Import frameworks to trigger registration
from . import olmo
from . import huggingface
from . import olmo_core
