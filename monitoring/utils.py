import os
from typing import Any, TYPE_CHECKING, Optional

import flax.struct
import flax.traverse_util
import jax.numpy as jnp
import numpy.typing as npt
from jaxtyping import Array, Float, PyTree
from torch.utils.tensorboard import SummaryWriter

if TYPE_CHECKING:
    from metaworld_types import LogDict


class Histogram(flax.struct.PyTreeNode):
    data: Float[npt.NDArray | Array, "..."] | None = None
    np_histogram: tuple | None = None


# Global TensorBoard writer for training
_tensorboard_writer: Optional[SummaryWriter] = None

def set_tensorboard_writer(log_dir: str) -> None:
    """Set up global TensorBoard writer for training logging."""
    global _tensorboard_writer
    os.makedirs(log_dir, exist_ok=True)
    _tensorboard_writer = SummaryWriter(log_dir)

def close_tensorboard_writer() -> None:
    """Close the global TensorBoard writer."""
    global _tensorboard_writer
    if _tensorboard_writer is not None:
        _tensorboard_writer.close()
        _tensorboard_writer = None

def log(logs: dict, step: int) -> None:
    """Log function for TensorBoard."""
    if _tensorboard_writer is not None:
        for key, value in logs.items():
            if isinstance(value, Histogram):
                # Convert histogram to TensorBoard format
                if value.data is not None:
                    _tensorboard_writer.add_histogram(key, value.data, step)
                elif value.np_histogram is not None:
                    hist, bins = value.np_histogram
                    _tensorboard_writer.add_histogram(key, bins[:-1], step)
            elif isinstance(value, (int, float)):
                _tensorboard_writer.add_scalar(key, value, step)
            elif hasattr(value, 'item'):  # JAX/NumPy scalars
                _tensorboard_writer.add_scalar(key, value.item(), step)
        _tensorboard_writer.flush()


def get_logs(
    name: str,
    data: Float[npt.NDArray | Array, "..."],
    axis: int | None = None,
    hist: bool = True,
    std: bool = True,
) -> "LogDict":
    ret: "LogDict" = {
        f"{name}_mean": jnp.mean(data, axis=axis),
        f"{name}_min": jnp.min(data, axis=axis),
        f"{name}_max": jnp.max(data, axis=axis),
    }
    if std:
        ret[f"{name}_std"] = jnp.std(data, axis=axis)
    if hist:
        ret[f"{name}"] = Histogram(data.reshape(-1))

    return ret


def prefix_dict(prefix: str, d: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}/{k}": v for k, v in d.items()}


def pytree_histogram(pytree: PyTree, bins: int = 64) -> dict[str, Histogram]:
    flat_dict = flax.traverse_util.flatten_dict(pytree, sep="/")
    ret = {}
    for k, v in flat_dict.items():
        if isinstance(v, tuple):  # For activations
            v = v[0]
        ret[k] = Histogram(np_histogram=jnp.histogram(v, bins=bins))  # pyright: ignore[reportArgumentType]
    return ret
