"""Image augmentation utilities for DrQ-style agents."""

from __future__ import annotations

from typing import Mapping, Sequence

import jax
import jax.numpy as jnp
from flax.core import freeze


def random_shift(
    key: jax.Array,
    images: jax.Array,
    *,
    pad: int = 4,
    channels_last: bool = True,
) -> jax.Array:
    """Apply random spatial shifts to a batch of images.

    Args:
        key: PRNG key for sampling shifts.
        images: Batch of images with shape (B, H, W, C) if ``channels_last``
            else (B, C, H, W).
        pad: Number of pixels to pad before cropping back to original size.
        channels_last: Whether the channel dimension is last.

    Returns:
        Augmented images with the same shape as the input.
    """
    if images.ndim != 4:
        raise ValueError("random_shift expects images with 4 dimensions (B,H,W,C) or (B,C,H,W)")

    if channels_last:
        batch, height, width, channels = images.shape
        pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0))
        slice_sizes = (height, width, channels)
    else:
        batch, channels, height, width = images.shape
        pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        slice_sizes = (channels, height, width)

    padded = jnp.pad(images, pad_width, mode="edge")
    max_offset = 2 * pad + 1
    offsets = jax.random.randint(key, (batch, 2), 0, max_offset)

    def _crop(img, offset):
        dy, dx = offset
        if channels_last:
            return jax.lax.dynamic_slice(img, (dy, dx, 0), slice_sizes)
        return jax.lax.dynamic_slice(img, (0, dy, dx), slice_sizes)

    return jax.vmap(_crop)(padded, offsets)


def random_shift_views(
    key: jax.Array,
    images: Mapping[str, jax.Array],
    *,
    view_names: Sequence[str],
    pad: int = 4,
    channels_last: bool = True,
):
    """Apply identical random shifts across multiple camera views.

    Args:
        key: PRNG key.
        images: Mapping from view name to batched image arrays. Each array must
            share the same batch dimensions and spatial shape.
        pad: Padding in pixels before cropping.
        channels_last: Whether the images are channel-last.

    Returns:
        Dictionary with augmented images per view.
    """
    if not images:
        raise ValueError("images mapping must not be empty")

    if not view_names:
        raise ValueError("view_names must not be empty")

    first_view = images[view_names[0]]
    batch = first_view.shape[0]
    max_offset = 2 * pad + 1
    offsets = jax.random.randint(key, (batch, 2), 0, max_offset)
    augmented = {
        name: _random_shift_with_offsets(images[name], offsets, pad, channels_last)
        for name in view_names
    }
    return freeze(augmented)


def _random_shift_with_offsets(
    images: jax.Array,
    offsets: jax.Array,
    pad: int,
    channels_last: bool,
) -> jax.Array:
    if images.ndim != 4:
        raise ValueError("Images must be batched 4D tensors")

    if channels_last:
        batch, height, width, channels = images.shape
        pad_width = ((0, 0), (pad, pad), (pad, pad), (0, 0))
        slice_sizes = (height, width, channels)
    else:
        batch, channels, height, width = images.shape
        pad_width = ((0, 0), (0, 0), (pad, pad), (pad, pad))
        slice_sizes = (channels, height, width)

    padded = jnp.pad(images, pad_width, mode="edge")

    def _crop(img, offset):
        dy, dx = offset
        if channels_last:
            return jax.lax.dynamic_slice(img, (dy, dx, 0), slice_sizes)
        return jax.lax.dynamic_slice(img, (0, dy, dx), slice_sizes)

    return jax.vmap(_crop)(padded, offsets)
