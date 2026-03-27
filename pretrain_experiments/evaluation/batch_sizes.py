"""Look up optimal batch sizes for evaluation based on model name.

Reads from config/inference_batch_sizes.yaml. Returns conservative defaults
if the model is not found or the config file is missing.
"""

import os
import yaml
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "inference_batch_sizes.yaml"
_CACHE = None


def _load_config():
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            _CACHE = yaml.safe_load(f)
    else:
        _CACHE = {}
    return _CACHE


def _match_model_size(model_name):
    """Extract model size key from model name (e.g., 'sbordt/OLMo-2-179M-Exp' -> 'OLMo-2-179M')."""
    config = _load_config()
    for key in config:
        # Match if the model name contains the size key (e.g., 'OLMo-2-179M' in 'sbordt/OLMo-2-179M-Exp')
        if key in model_name:
            return key
    return None


def get_logprobs_batch_size(model_name, max_seq_len=4096):
    """Get optimal batch size for get_logprobs based on model and sequence length.

    Interpolates between the long-sequence and short-sequence benchmarked values.
    """
    config = _load_config()
    key = _match_model_size(model_name)
    if key is None:
        return 8  # safe default

    model_config = config[key]
    long_bs = model_config.get('logprobs_max_num_seqs', 8)
    short_bs = model_config.get('logprobs_short_max_num_seqs', long_bs)

    # linear interpolation between 512->short_bs and 4096->long_bs
    if max_seq_len <= 512:
        return short_bs
    elif max_seq_len >= 4096:
        return long_bs
    else:
        # interpolate
        frac = (max_seq_len - 512) / (4096 - 512)
        return max(1, int(short_bs + frac * (long_bs - short_bs)))


def get_generate_batch_size(model_name):
    """Get optimal batch size for generate_text based on model."""
    config = _load_config()
    key = _match_model_size(model_name)
    if key is None:
        return 8  # safe default
    return config[key].get('generate_max_num_seqs', 8)
