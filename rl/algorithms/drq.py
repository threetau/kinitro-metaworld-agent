"""SAC + DrQ-v2 implementation with shared multi-view encoders."""

from __future__ import annotations
from functools import partial
from typing import Mapping, Self, override

import distrax
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
from flax.core import FrozenDict, freeze
from flax.training.train_state import TrainState
from jaxtyping import Array, Float, PRNGKeyArray

from config.envs import EnvConfig
from config.rl import DrQSACConfig, OffPolicyTrainingConfig
from rl.augmentations import random_shift_views
from rl.buffers import PixelReplayBuffer
from rl.networks import (
    ContinuousActionPolicy,
    Ensemble,
    ObservationEncoder,
    QValueFunction,
)
from rl.algorithms.base import OffPolicyAlgorithm
from metaworld_types import (
    Action,
    LogDict,
    PixelReplayBufferSamples,
)


class EncoderTrainState(TrainState):
    target_params: FrozenDict | None = None


class CriticTrainState(TrainState):
    target_params: FrozenDict | None = None


@struct.dataclass
class PixelBatch:
    observations: FrozenDict
    actions: jax.Array
    next_observations: FrozenDict
    dones: jax.Array
    rewards: jax.Array


class DrQSAC(OffPolicyAlgorithm[DrQSACConfig]):
    encoder: EncoderTrainState
    actor: TrainState
    critic: CriticTrainState
    alpha: TrainState
    key: PRNGKeyArray
    gamma: float = struct.field(pytree_node=False)
    tau: float = struct.field(pytree_node=False)
    target_entropy: float = struct.field(pytree_node=False)
    augmentation_pad: int = struct.field(pytree_node=False)
    channels_last: bool = struct.field(pytree_node=False)
    view_names: tuple[str, ...] = struct.field(pytree_node=False)

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> PixelReplayBuffer:
        return PixelReplayBuffer(
            capacity=config.buffer_size,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    @staticmethod
    def initialize(
        config: DrQSACConfig, env_config: EnvConfig, seed: int = 1
    ) -> "DrQSAC":
        if not isinstance(env_config.action_space, gym.spaces.Box):
            raise ValueError("DrQSAC requires continuous action spaces")
        if not isinstance(env_config.observation_space, gym.spaces.Dict):
            raise ValueError("DrQSAC expects Dict observation space")

        master_key = jax.random.PRNGKey(seed)
        key_encoder, key_actor, key_critic, key_alpha, algorithm_key = jax.random.split(
            master_key, 5
        )

        obs_space: gym.spaces.Dict = env_config.observation_space
        image_space: gym.spaces.Dict = obs_space["images"]
        view_names = tuple(image_space.spaces.keys())

        dummy_images = {
            name: jnp.asarray(image_space[name].sample())[None, ...]
            for name in view_names
        }
        dummy_proprio = jnp.asarray(obs_space["proprio"].sample(), dtype=jnp.float32)[
            None, ...
        ]
        dummy_task = jnp.asarray(obs_space["task_one_hot"].sample(), dtype=jnp.float32)[
            None, ...
        ]

        encoder_module = ObservationEncoder(config.fusion_config)
        encoder_params = encoder_module.init(
            key_encoder,
            images=dummy_images,
            proprio=dummy_proprio,
            task_one_hot=dummy_task,
        )
        encoder_state = EncoderTrainState.create(
            apply_fn=encoder_module.apply,
            params=encoder_params,
            tx=config.encoder_optimizer.spawn(),
        )
        encoder_state = encoder_state.replace(target_params=encoder_state.params)

        encoder_output = encoder_module.apply(
            encoder_state.params,
            images=dummy_images,
            proprio=dummy_proprio,
            task_one_hot=dummy_task,
        )
        latent_dim = encoder_output.latent.shape[-1]

        action_dim = int(np.prod(env_config.action_space.shape))
        actor_net = ContinuousActionPolicy(action_dim, config.actor_config)
        actor_state = TrainState.create(
            apply_fn=actor_net.apply,
            params=actor_net.init(key_actor, encoder_output.latent),
            tx=config.actor_optimizer.spawn(),
        )

        critic_cls = partial(QValueFunction, config=config.critic_config)
        critic_net = Ensemble(critic_cls, num=config.num_critics)
        dummy_actions = jnp.asarray(env_config.action_space.sample())[None, ...]
        critic_params = critic_net.init(
            key_critic, encoder_output.latent, dummy_actions
        )
        critic_state = CriticTrainState.create(
            apply_fn=critic_net.apply,
            params=critic_params,
            tx=config.critic_optimizer.spawn(),
        )
        critic_state = critic_state.replace(target_params=critic_state.params)

        alpha_module = Temperature(config.initial_temperature)
        alpha_state = TrainState.create(
            apply_fn=alpha_module.apply,
            params=alpha_module.init(key_alpha),
            tx=config.temperature_optimizer.spawn(),
        )

        target_entropy = -float(action_dim)

        return DrQSAC(
            num_tasks=config.num_tasks,
            encoder=encoder_state,
            actor=actor_state,
            critic=critic_state,
            alpha=alpha_state,
            key=algorithm_key,
            gamma=config.gamma,
            tau=config.tau,
            target_entropy=target_entropy,
            augmentation_pad=config.augmentation_pad,
            channels_last=config.channels_last,
            view_names=view_names,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        return {
            "encoder_num_params": sum(
                x.size for x in jax.tree.leaves(self.encoder.params)
            ),
            "actor_num_params": sum(x.size for x in jax.tree.leaves(self.actor.params)),
            "critic_num_params": sum(
                x.size for x in jax.tree.leaves(self.critic.params)
            ),
        }

    def _encode(
        self,
        params: FrozenDict,
        *,
        images: Mapping[str, jax.Array],
        proprio: jax.Array,
        task_one_hot: jax.Array,
    ):
        return self.encoder.apply_fn(
            params,
            images=images,
            proprio=proprio,
            task_one_hot=task_one_hot,
        )

    @override
    def sample_action(self, observation) -> tuple[Self, Action]:
        key, action_key = jax.random.split(self.key)
        latent = self._encode_observation(self.encoder.params, observation)
        dist: distrax.Distribution = self.actor.apply_fn(self.actor.params, latent)
        action = dist.sample(seed=action_key)
        return self.replace(key=key), jax.device_get(action)

    @override
    def eval_action(self, observations) -> Action:
        latent = self._encode_observation(self.encoder.params, observations)
        dist: distrax.Distribution = self.actor.apply_fn(self.actor.params, latent)
        return jax.device_get(dist.mode())

    def _encode_observation(self, params: FrozenDict, observation) -> jax.Array:
        images = {}
        for name in self.view_names:
            view = jnp.asarray(observation["images"][name])
            if view.ndim == 3:
                view = view[None, ...]
            images[name] = view

        proprio = jnp.asarray(observation["proprio"], dtype=jnp.float32)
        if proprio.ndim == 1:
            proprio = proprio[None, ...]

        task = jnp.asarray(observation["task_one_hot"], dtype=jnp.float32)
        if task.ndim == 1:
            task = task[None, ...]

        outputs = self._encode(
            params,
            images=images,
            proprio=proprio,
            task_one_hot=task,
        )
        return outputs.latent

    def _prepare_batch(self, data: PixelReplayBufferSamples) -> PixelBatch:
        observations = freeze(
            {
                "images": {
                    name: jnp.asarray(data.observations["images"][name])
                    for name in self.view_names
                },
                "proprio": jnp.asarray(data.observations["proprio"], dtype=jnp.float32),
                "task_one_hot": jnp.asarray(
                    data.observations["task_one_hot"], dtype=jnp.float32
                ),
            }
        )
        next_observations = freeze(
            {
                "images": {
                    name: jnp.asarray(data.next_observations["images"][name])
                    for name in self.view_names
                },
                "proprio": jnp.asarray(
                    data.next_observations["proprio"], dtype=jnp.float32
                ),
                "task_one_hot": jnp.asarray(
                    data.next_observations["task_one_hot"], dtype=jnp.float32
                ),
            }
        )

        return PixelBatch(
            observations=observations,
            actions=jnp.asarray(data.actions, dtype=jnp.float32),
            next_observations=next_observations,
            dones=jnp.asarray(data.dones, dtype=jnp.float32),
            rewards=jnp.asarray(data.rewards, dtype=jnp.float32),
        )

    @override
    def update(self, data: PixelReplayBufferSamples) -> tuple[Self, LogDict]:
        batch = self._prepare_batch(data)
        new_self, logs = self._update_step(batch)
        return new_self, logs

    @jax.jit
    def _update_step(self, batch: PixelBatch) -> tuple["DrQSAC", LogDict]:
        key, obs_key, next_key, actor_key, critic_key = jax.random.split(self.key, 5)

        aug_images = random_shift_views(
            obs_key,
            batch.observations["images"],
            view_names=self.view_names,
            pad=self.augmentation_pad,
            channels_last=self.channels_last,
        )
        aug_next_images = random_shift_views(
            next_key,
            batch.next_observations["images"],
            view_names=self.view_names,
            pad=self.augmentation_pad,
            channels_last=self.channels_last,
        )

        alpha_val = self.alpha.apply_fn(self.alpha.params)

        def critic_loss_fn(encoder_params, critic_params):
            encoder_outputs = self._encode(
                encoder_params,
                images=aug_images,
                proprio=batch.observations["proprio"],
                task_one_hot=batch.observations["task_one_hot"],
            )
            latent = encoder_outputs.latent
            q_pred = self.critic.apply_fn(critic_params, latent, batch.actions)

            target_encoder_outputs = self._encode(
                self.encoder.target_params,
                images=aug_next_images,
                proprio=batch.next_observations["proprio"],
                task_one_hot=batch.next_observations["task_one_hot"],
            )
            next_latent = target_encoder_outputs.latent
            next_dist: distrax.Distribution = self.actor.apply_fn(
                self.actor.params,
                jax.lax.stop_gradient(next_latent),
            )
            next_actions, next_log_probs = next_dist.sample_and_log_prob(
                seed=critic_key
            )
            target_q = self.critic.apply_fn(
                self.critic.target_params, next_latent, next_actions
            )
            min_target_q = jnp.min(
                target_q, axis=0
            ) - alpha_val * next_log_probs.reshape(-1, 1)
            target_values = (
                batch.rewards + (1.0 - batch.dones) * self.gamma * min_target_q
            )
            loss = jnp.mean((q_pred - jax.lax.stop_gradient(target_values)) ** 2)
            return loss, (latent, q_pred.mean())

        (
            (critic_loss_value, (latent_features, q_mean)),
            (encoder_grads, critic_grads),
        ) = jax.value_and_grad(critic_loss_fn, argnums=(0, 1), has_aux=True)(
            self.encoder.params, self.critic.params
        )

        encoder = self.encoder.apply_gradients(grads=encoder_grads)
        critic = self.critic.apply_gradients(grads=critic_grads)

        encoder = encoder.replace(
            target_params=optax.incremental_update(
                encoder.params,
                encoder.target_params,
                self.tau,
            )
        )
        critic = critic.replace(
            target_params=optax.incremental_update(
                critic.params,
                critic.target_params,
                self.tau,
            )
        )

        latent_stop = jax.lax.stop_gradient(latent_features)

        def actor_loss_fn(params):
            dist: distrax.Distribution
            dist = self.actor.apply_fn(params, latent_stop)
            actions, log_probs = dist.sample_and_log_prob(seed=actor_key)
            q_vals = self.critic.apply_fn(
                critic.params,
                latent_stop,
                actions,
            )
            min_q_vals = jnp.min(q_vals, axis=0)
            loss = (alpha_val * log_probs.reshape(-1, 1) - min_q_vals).mean()
            return loss, log_probs

        (actor_loss_value, log_probs), actor_grads = jax.value_and_grad(
            actor_loss_fn, has_aux=True
        )(self.actor.params)
        actor = self.actor.apply_gradients(grads=actor_grads)

        def alpha_loss_fn(params):
            log_alpha = params["params"]["log_alpha"]
            return jnp.mean((log_alpha * (-log_probs - self.target_entropy)))

        alpha_loss_value, alpha_grads = jax.value_and_grad(alpha_loss_fn)(
            self.alpha.params
        )
        alpha = self.alpha.apply_gradients(grads=alpha_grads)

        new_self = self.replace(
            key=key,
            encoder=encoder,
            critic=critic,
            actor=actor,
            alpha=alpha,
        )

        logs: LogDict = {
            "losses/critic_loss": critic_loss_value,
            "losses/actor_loss": actor_loss_value,
            "losses/alpha_loss": alpha_loss_value,
            "metrics/q_mean": q_mean,
            "alpha": jnp.exp(alpha.params["params"]["log_alpha"]).sum(),
        }
        return new_self, logs


class Temperature(nn.Module):
    initial_temperature: float = 1.0

    def setup(self):
        self.log_alpha = self.param(
            "log_alpha",
            init_fn=lambda _: jnp.full((1,), jnp.log(self.initial_temperature)),
        )

    def __call__(self) -> Float[Array, " 1"]:
        return jnp.exp(self.log_alpha)
