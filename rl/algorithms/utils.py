from typing import Any, Generator, Never

import numpy as np
import numpy.typing as npt
import optax
import scipy
from flax import struct
from flax.linen.fp8_ops import OVERWRITE_WITH_GRADIENT
from flax.training.train_state import TrainState as FlaxTrainState
from jaxtyping import Float
from typing_extensions import Callable

from metaworld_types import Rollout


class TrainState(FlaxTrainState):
    def apply_gradients(
        self,
        *,
        grads,
        optimizer_extra_args: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        if OVERWRITE_WITH_GRADIENT in grads:
            grads_with_opt = grads["params"]
            params_with_opt = self.params["params"]
        else:
            grads_with_opt = grads
            params_with_opt = self.params

        if optimizer_extra_args is None:
            optimizer_extra_args = {}

        updates, new_opt_state = self.tx.update(
            grads_with_opt, self.opt_state, params_with_opt, **optimizer_extra_args
        )
        new_params_with_opt = optax.apply_updates(params_with_opt, updates)

        if OVERWRITE_WITH_GRADIENT in grads:
            new_params = {
                "params": new_params_with_opt,
                OVERWRITE_WITH_GRADIENT: grads[OVERWRITE_WITH_GRADIENT],
            }
        else:
            new_params = new_params_with_opt
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )


class RNNTrainState(TrainState):
    seq_apply_fn: Callable = struct.field(pytree_node=False)
    init_carry_fn: Callable = struct.field(pytree_node=False)


class MetaTrainState(TrainState):
    inner_train_state: TrainState
    expand_params: Callable = struct.field(pytree_node=False)

def to_minibatch_iterator(
    data: Rollout, num: int, seed: int, flatten_batch_dims: bool = True
) -> Generator[Rollout, None, Never]:
    # Flatten batch dims
    rollouts = data
    if flatten_batch_dims:
        rollouts = Rollout(
            *map(
                lambda x: x.reshape(-1, x.shape[-1]) if x is not None else None,
                data,
            )  # pyright: ignore[reportArgumentType]
        )

    rollout_size = rollouts.observations.shape[0]
    minibatch_size = rollout_size // num

    rng = np.random.default_rng(seed)
    rng_state = rng.bit_generator.state

    while True:
        for field in rollouts:
            rng.bit_generator.state = rng_state
            if field is not None:
                rng.shuffle(field, axis=0)
        rng_state = rng.bit_generator.state
        for start in range(0, rollout_size, minibatch_size):
            end = start + minibatch_size
            yield Rollout(
                *map(
                    lambda x: x[start:end] if x is not None else None,  # pyright: ignore[reportArgumentType]
                    rollouts,
                )
            )


def to_deterministic_minibatch_iterator(data: Rollout) -> Generator[Rollout, None, Never]:
    # Flatten batch dims
    rollouts = data

    while True:
        for step in range(len(rollouts.rewards)):
            yield Rollout(
                *map(
                    lambda x: x[step] if x is not None else None,  # pyright: ignore[reportArgumentType]
                    rollouts,
                )
            )

def compute_gae(
    rollouts: Rollout,
    gamma: float,
    gae_lambda: float,
    last_values: Float[npt.NDArray, " task"] | None,
    dones: Float[npt.NDArray, " task"],
) -> Rollout:
    # NOTE: dones is a very misleading name but it goes back to OpenAI's original PPO code
    # really, dones indicates whether *the previous timstep* was terminal.

    assert rollouts.values is not None

    if last_values is not None:
        last_values = last_values.reshape(-1, 1)
    else:
        if np.all(dones == 1.0):
            last_values = np.zeros_like(rollouts.values[0])
        else:
            raise ValueError(
                "Must provide final value estimates if the final timestep is not terminal for all envs."
            )
    dones = dones.reshape(-1, 1)

    advantages = np.zeros_like(rollouts.rewards)

    # Adapted from https://github.com/openai/baselines/blob/master/baselines/ppo2/runner.py
    # Renamed dones -> episode_starts because the former is misleading
    last_gae_lamda = 0
    num_rollout_steps = rollouts.observations.shape[0]
    assert last_values is not None

    for timestep in reversed(range(num_rollout_steps)):
        if timestep == num_rollout_steps - 1:
            next_nonterminal = 1.0 - dones
            next_values = last_values
        else:
            next_nonterminal = 1.0 - rollouts.dones[timestep + 1]
            next_values = rollouts.values[timestep + 1]
        delta = (
            rollouts.rewards[timestep]
            + next_nonterminal * gamma * next_values
            - rollouts.values[timestep]
        )
        advantages[timestep] = last_gae_lamda = (
            delta + next_nonterminal * gamma * gae_lambda * last_gae_lamda
        )

    returns = advantages + rollouts.values

    if not hasattr(rollouts, "returns"):
        # NOTE: Can't use `replace` here if this is a Rollout from MetaWorld's evaluation interface
        return Rollout(
            returns=returns,
            advantages=advantages,
            observations=rollouts.observations,
            actions=rollouts.actions,
            rewards=rollouts.rewards,
            dones=rollouts.dones,
            log_probs=rollouts.log_probs,
            means=rollouts.means,
            stds=rollouts.stds,
            values=rollouts.values,
        )
    else:
        return rollouts._replace(
            returns=returns,
            advantages=advantages,
        )


def compute_returns(
    rewards: Float[npt.NDArray, "task rollout timestep 1"], discount: float
) -> Float[npt.NDArray, "task rollout timestep 1"]:
    """Discounted cumulative sum.

    See https://docs.scipy.org/doc/scipy/reference/tutorial/signal.html#difference-equation-filtering
    """
    # From garage, modified to work on multi-dimensional arrays, and column reward vectors
    reshape = rewards.shape[-1] == 1
    if reshape:
        rewards = rewards.reshape(rewards.shape[:-1])
    returns = scipy.signal.lfilter(
        [1], [1, float(-discount)], rewards[..., ::-1], axis=-1
    )[..., ::-1]
    return returns if not reshape else returns.reshape(*returns.shape, 1)


def normalize_advantages(rollouts: Rollout) -> Rollout:
    assert rollouts.advantages is not None
    mean = rollouts.advantages.mean(axis=0, keepdims=True)
    var = rollouts.advantages.var(axis=0, keepdims=True)
    advantages = (rollouts.advantages - mean) / (var + 1e-8)
    return rollouts._replace(advantages=advantages)


class LinearFeatureBaseline:
    @staticmethod
    def _extract_features(
        observations: Float[npt.NDArray, "task rollout timestep obs_dim"], reshape=True
    ):
        observations = np.clip(observations, -10, 10)
        ones = np.ones((*observations.shape[:-1], 1))
        timestep = ones * (np.arange(observations.shape[-2]).reshape(-1, 1) / 100.0)
        features = np.concatenate(
            [observations, observations**2, timestep, timestep**2, timestep**3, ones],
            axis=-1,
        )
        if reshape:
            features = features.reshape(features.shape[0], -1, features.shape[-1])
        return features

    @classmethod
    def _fit_baseline(
        cls,
        observations: Float[npt.NDArray, "task rollout timestep obs_dim"],
        returns: Float[npt.NDArray, "task rollout timestep 1"],
        reg_coeff: float = 1e-5,
    ) -> np.ndarray:
        features = cls._extract_features(observations)
        target = returns.reshape(returns.shape[0], -1, 1)

        coeffs = []
        task_coeffs = np.zeros(features.shape[1])
        for task in range(observations.shape[0]):
            featmat = features[task]
            _target = target[task]
            for _ in range(5):
                task_coeffs = np.linalg.lstsq(
                    featmat.T @ featmat + reg_coeff * np.identity(featmat.shape[1]),
                    featmat.T @ _target,
                    rcond=-1,
                )[0]
                if not np.any(np.isnan(task_coeffs)):
                    break
                reg_coeff *= 10

            coeffs.append(np.expand_dims(task_coeffs, axis=0))

        return np.stack(coeffs)

    @classmethod
    def get_baseline_values_and_returns(
        cls, rollouts: Rollout, discount: float
    ) -> tuple[
        Float[npt.NDArray, "timestep task 1"], Float[npt.NDArray, "timestep task 1"]
    ]:
        # Split the rollouts into episodes
        # TODO: Refactor
        observations = [[] for _ in range(rollouts.dones.shape[1])]
        rewards = [[] for _ in range(rollouts.dones.shape[1])]
        start_idx = np.zeros(rollouts.dones.shape[1], dtype=np.int32)
        for i in range(rollouts.dones.shape[0] + 1):
            if i == rollouts.dones.shape[0]:  # Assume final observation is terminal
                dones = np.ones((rollouts.dones.shape[1], 1))
            else:
                dones = rollouts.dones[i]
            for j, done in enumerate(dones):
                if done and i != 0:
                    observations[j].append(rollouts.observations[start_idx[j] : i, j])
                    rewards[j].append(rollouts.rewards[start_idx[j] : i, j])
                    start_idx[j] = i

        # NOTE: This will error if the trajectories are not the same length
        observations = np.stack(observations)
        rewards = np.stack(rewards)
        returns = compute_returns(rewards, discount=discount)

        def _reshape(x: npt.NDArray) -> npt.NDArray:
            return (
                x.reshape(x.shape[0], -1, x.shape[-1])
                .swapaxes(0, 1)
                .reshape(*rollouts.rewards.shape)
            )

        coeffs = cls._fit_baseline(observations, returns)
        features = cls._extract_features(observations, reshape=False)

        return _reshape(features @ coeffs), _reshape(returns)


def swap_rollout_axes(rollout: Rollout, axis1: int, axis2: int) -> Rollout:
    return Rollout(
        *map(
            lambda x: x.swapaxes(axis1, axis2) if x is not None else None,
            rollout,
        )  # pyright: ignore[reportArgumentType]
    )


def to_padded_episode_batch(rollout: Rollout) -> Rollout:
    N = rollout.observations.shape[1]  # (:, task, ...)
    rollout = swap_rollout_axes(rollout, 0, 1)  # (task, timestep, ...)
    sequences = {
        field: [] for field in rollout._fields if getattr(rollout, field) is not None
    }
    episode_starts = rollout.dones.squeeze()
    episode_lengths = []

    for task in range(N):
        boundaries = np.argwhere(episode_starts[task]).squeeze()
        if boundaries.ndim == 0:
            # Single episode
            episode_lengths.append(len(episode_starts[task]))
            for field in rollout._fields:
                if (field_data := getattr(rollout, field)) is not None:
                    sequences[field].append(field_data[task])
        else:
            boundaries = boundaries[1:]
            episode_lengths.append(boundaries[0])
            episode_lengths += list(np.diff(boundaries))
            for field in rollout._fields:
                if (field_data := getattr(rollout, field)) is not None:
                    sequences[field] += np.array_split(field_data[task], boundaries)

    max_episode_length = max(episode_lengths)
    valids = np.ones((len(episode_lengths), max_episode_length), dtype=np.bool_)
    for field in sequences:
        for i, sequence in enumerate(sequences[field]):
            seq_len = len(sequence)
            sequences[field][i] = np.pad(
                sequence,
                ((0, max_episode_length - seq_len), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            valids[i, seq_len:] = False

    sequences = {field: np.stack(sequences[field]) for field in sequences}
    rollout = Rollout(**sequences)
    rollout = rollout._replace(valids=valids.reshape(rollout.rewards.shape))
    return rollout


def to_overlapping_chunks(rollout: Rollout, chunk_len: int, overlap: int) -> Rollout:
    # Recommended by https://danijar.com/tips-for-training-recurrent-neural-networks/
    # HACK: Currently assumes there's only a single episode cause it's used for RL2
    assert overlap < chunk_len
    step = chunk_len - overlap

    T = rollout.observations.shape[0]  # (time, ...)
    starts = np.arange(0, T - overlap, step)

    sequences = {
        field: [] for field in rollout._fields if getattr(rollout, field) is not None
    }

    for s in starts:
        end = s + chunk_len
        if end > T:
            break
        for field in sequences:
            field_data = getattr(rollout, field)
            sequences[field].append(field_data[s:end])

    data = {field: np.stack(sequences[field]) for field in sequences}
    rollout = Rollout(**data)
    rollout = rollout._replace(valids=np.ones_like(rollout.rewards))
    return rollout


def to_episode_batch(rollout: Rollout, episode_length: int) -> Rollout:
    def _reshape(x: npt.NDArray) -> npt.NDArray:
        # Starting shape: (timestep, task, ...)
        x = x.swapaxes(0, 1)  # (task, timestep, ...)
        x = x.reshape(
            x.shape[0], -1, episode_length, x.shape[-1]
        )  # (task, episode, timestep, ...)
        x = x.reshape(-1, episode_length, x.shape[-1])  # (episode, timestep, ...)
        return x

    return Rollout(
        *map(
            lambda x: _reshape(x) if x is not None else None,
            rollout,
        )  # pyright: ignore[reportArgumentType]
    )


def dones_to_episode_starts(rollout: Rollout) -> Rollout:
    episode_starts = np.concatenate(
        (np.ones((1, *rollout.dones.shape[1:])), rollout.dones), axis=0
    )[:-1]
    return rollout._replace(dones=episode_starts)


def explained_variance(
    y_pred: Float[npt.NDArray, " total_num_steps"],
    y_true: Float[npt.NDArray, " total_num_steps"],
) -> float:
    # From SB3 https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/utils.py#L50
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)
