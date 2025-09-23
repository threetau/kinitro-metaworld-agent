# pyright: reportCallIssue=false, reportAttributeAccessIssue=false
import random
from typing import TYPE_CHECKING, NotRequired, TypedDict

import numpy as np
import orbax.checkpoint as ocp

from rl.buffers import AbstractReplayBuffer
from metaworld_types import (
    CheckpointMetadata,
    EnvCheckpoint,
    GymVectorEnv,
    ReplayBufferCheckpoint,
    RNGCheckpoint,
)

if TYPE_CHECKING:
    from rl.algorithms.base import Algorithm


class Checkpoint(TypedDict):
    agent: "Algorithm"
    buffer: NotRequired[ReplayBufferCheckpoint]
    env_states: EnvCheckpoint
    rngs: RNGCheckpoint
    metadata: CheckpointMetadata


def checkpoint_envs(envs: GymVectorEnv) -> list[tuple[str, dict]]:
    return envs.call("get_checkpoint")  # pyright: ignore [reportReturnType]


def load_env_checkpoints(envs: GymVectorEnv, env_ckpts: list[tuple[str, dict]]):
    envs.call("load_checkpoint", env_ckpts)


def get_last_agent_checkpoint_save_args(
    agent: "Algorithm", metrics: dict[str, float]
) -> ocp.args.CheckpointArgs:
    return ocp.args.Composite(
        agent=ocp.args.StandardSave(agent), metadata=ocp.args.JsonSave(metrics)
    )


def get_agent_checkpoint_restore_args(agent: "Algorithm") -> ocp.args.CheckpointArgs:
    return ocp.args.Composite(
        agent=ocp.args.StandardRestore(agent),
        metadata=ocp.args.JsonRestore(),
    )


def get_checkpoint_save_args(
    agent: "Algorithm",
    envs: GymVectorEnv,
    total_steps: int,
    episodes_ended: int,
    run_timestamp: str | None,
    buffer: AbstractReplayBuffer | None = None,
) -> ocp.args.CheckpointArgs:
    if buffer is not None:
        rb_ckpt = buffer.checkpoint()
        buffer_args = ocp.args.Composite(
            data=ocp.args.StandardSave(rb_ckpt["data"]),
            rng_state=ocp.args.JsonSave(rb_ckpt["rng_state"]),
        )
    else:
        buffer_args = None

    env_checkpoints = checkpoint_envs(envs)

    args = {
        "agent": ocp.args.StandardSave(agent),
        "env_states": ocp.args.JsonSave(env_checkpoints),
        "rngs": ocp.args.Composite(
            python_rng_state=ocp.args.StandardSave(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeySave(np.random.get_state()),
        ),
        "metadata": ocp.args.JsonSave(
            {
                "step": total_steps,
                "episodes_ended": episodes_ended,
                "timestamp": run_timestamp,
            }
        ),
    }
    if buffer_args is not None:
        args["buffer"] = buffer_args
    return ocp.args.Composite(**args)


def get_checkpoint_restore_args(
    agent: "Algorithm", buffer: AbstractReplayBuffer | None = None
) -> ocp.args.CheckpointArgs:
    if buffer is not None:
        rb_ckpt = buffer.checkpoint()
        buffer_args = ocp.args.Composite(
            data=ocp.args.StandardRestore(rb_ckpt["data"]),
            rng_state=ocp.args.JsonRestore(),
        )
    else:
        buffer_args = None

    args = {
        "agent": ocp.args.StandardRestore(agent),
        "env_states": ocp.args.JsonRestore(),
        "rngs": ocp.args.Composite(
            python_rng_state=ocp.args.StandardRestore(random.getstate()),
            global_numpy_rng_state=ocp.args.NumpyRandomKeyRestore(),
        ),
        "metadata": ocp.args.JsonRestore(),
    }

    if buffer_args is not None:
        args["buffer"] = buffer_args
    return ocp.args.Composite(**args)


def get_metadata_only_restore_args() -> ocp.args.CheckpointArgs:
    return ocp.args.Composite(metadata=ocp.args.JsonRestore())
