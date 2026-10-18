"""Brightness Characteristic Based Patchout (BCP-WP) selection primitive.

Implements the paper's weighted sampling without replacement: patches are
sampled for preservation in proportion to brightness until K* candidates
remain; those remaining candidates are dropped. The caller supplies
nonnegative per-patch brightness and can use the returned mask to filter ViT
patch tokens.
"""

from __future__ import annotations

import torch


def bcp_wp_keep_mask(
    brightness: torch.Tensor,
    n_drop: int,
    *,
    training: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a bool mask ``[batch, patches]`` marking the preserved patches.

    ``brightness`` must contain nonnegative energy values, one per patch.
    During evaluation (or when ``n_drop == 0``) every patch is preserved.
    Zero-energy rows fall back to uniform sampling because their probability
    distribution is otherwise undefined.
    """
    if brightness.ndim != 2:
        raise ValueError(f"expected [batch, patches] brightness, got {tuple(brightness.shape)}")
    if not torch.isfinite(brightness).all():
        raise ValueError("brightness values must be finite")
    if torch.any(brightness < 0):
        raise ValueError("brightness values must be nonnegative energy values")
    n_patches = brightness.shape[1]
    if not 0 <= n_drop < n_patches:
        raise ValueError(f"n_drop must be in [0, {n_patches - 1}], got {n_drop}")

    keep = torch.ones_like(brightness, dtype=torch.bool)
    if not training or n_drop == 0:
        return keep

    weights = brightness.to(dtype=torch.float32)
    sums = weights.sum(dim=1, keepdim=True)
    uniform = torch.full_like(weights, 1.0 / n_patches)
    probabilities = torch.where(sums > 0, weights / sums.clamp_min(torch.finfo(weights.dtype).tiny), uniform)
    n_keep = n_patches - n_drop
    kept = torch.multinomial(
        probabilities,
        num_samples=n_keep,
        replacement=False,
        generator=generator,
    )
    keep.zero_().scatter_(1, kept, True)
    return keep


def patch_brightness(mel_energy: torch.Tensor, patch_slices: list[slice], frequency_dim: int = -2) -> torch.Tensor:
    """Compute mean energy for each supplied frequency patch.

    ``mel_energy`` is expected to be in linear, nonnegative energy units. A
    typical input shape is ``[batch, time, mel_bins]`` or ``[batch, mel_bins,
    time]``; pass ``frequency_dim`` to identify the mel-bin axis. Patches may
    overlap, matching the frequency ranges illustrated in the article.
    """
    if not torch.isfinite(mel_energy).all() or torch.any(mel_energy < 0):
        raise ValueError("mel_energy must be finite and nonnegative")
    axis = frequency_dim if frequency_dim >= 0 else mel_energy.ndim + frequency_dim
    if axis == 0 or axis >= mel_energy.ndim:
        raise ValueError("frequency_dim must identify a non-batch dimension")
    values = []
    for band in patch_slices:
        patch = mel_energy.narrow(axis, band.start or 0, (band.stop or mel_energy.shape[axis]) - (band.start or 0))
        if patch.shape[axis] == 0:
            raise ValueError(f"empty frequency patch: {band}")
        reduce_dims = tuple(range(1, patch.ndim))
        values.append(patch.mean(dim=reduce_dims))
    return torch.stack(values, dim=1)
