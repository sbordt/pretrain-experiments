"""
Gradient noise for unlearning experiments.

Adds Gaussian N(0, noise_std) noise to every parameter's gradient after each
backward pass. Imported by the patched OLMo trainer (olmo/train.py) when
OLMO_GRADIENT_NOISE_CONFIG_FILE is set. The config pickle carries noise_std,
seed, and use_seed; noise itself is not saved.
"""

import torch
import torch.nn as nn


def add_gradient_noise(
    model: nn.Module,
    noise_std: float,
    generator: torch.Generator | None = None,
) -> None:
    """Add Gaussian noise to all parameter gradients. Call after loss.backward().

    If generator is None, uses the global PyTorch RNG (legacy behavior).
    If a generator is provided, draws are reproducible with respect to its seed.
    """
    noise_std = float(noise_std)
    if noise_std == 0.0:
        return
    for param in model.parameters():
        if param.requires_grad and param.grad is not None and param.grad.is_floating_point():
            if generator is None:
                noise = torch.randn_like(param.grad)
            else:
                noise = torch.empty_like(param.grad).normal_(generator=generator)
            param.grad.add_(noise, alpha=noise_std)
