import abc
import time
from collections import deque
from typing import Deque, Generic, Self, TypeVar, override

import numpy as np
import numpy.typing as npt
import orbax.checkpoint as ocp
from flax import struct
from jaxtyping import Float

from checkpoint import get_checkpoint_save_args
from config.envs import EnvConfig, MetaLearningEnvConfig
from config.rl import (
    AlgorithmConfig,
    GradientBasedMetaLearningTrainingConfig,
    MetaLearningTrainingConfig,
    OffPolicyTrainingConfig,
    OnPolicyTrainingConfig,
    RNNBasedMetaLearningTrainingConfig,
    TrainingConfig,
)
from monitoring.utils import log
from rl.buffers import (
    AbstractReplayBuffer,
    MultiTaskRolloutBuffer,
)
from metaworld_types import (
    Action,
    AuxPolicyOutputs,
    CheckpointMetadata,
    GymVectorEnv,
    LogDict,
    MetaLearningAgent,
    Observation,
    ReplayBufferCheckpoint,
    ReplayBufferSamples,
    RNNState,
    Rollout,
)

AlgorithmConfigType = TypeVar("AlgorithmConfigType", bound=AlgorithmConfig)
TrainingConfigType = TypeVar("TrainingConfigType", bound=TrainingConfig)
EnvConfigType = TypeVar("EnvConfigType", bound=EnvConfig)
MetaLearningTrainingConfigType = TypeVar(
    "MetaLearningTrainingConfigType", bound=MetaLearningTrainingConfig
)
DataType = TypeVar("DataType", ReplayBufferSamples, Rollout, list[Rollout])


class Algorithm(
    abc.ABC,
    Generic[AlgorithmConfigType, TrainingConfigType, EnvConfigType, DataType],
    struct.PyTreeNode,
):
    """Based on https://github.com/kevinzakka/nanorl/blob/main/nanorl/agent.py"""

    num_tasks: int = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: EnvConfigType, seed: int = 1
    ) -> "Algorithm": ...

    @abc.abstractmethod
    def get_num_params(self) -> dict[str, int]: ...

    @abc.abstractmethod
    def train(
        self,
        config: TrainingConfigType,
        envs: GymVectorEnv,
        env_config: EnvConfigType,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class MetaLearningAlgorithm(
    Algorithm[
        AlgorithmConfigType,
        MetaLearningTrainingConfigType,
        MetaLearningEnvConfig,
        DataType,
    ],
    Generic[AlgorithmConfigType, MetaLearningTrainingConfigType, DataType],
):
    @staticmethod
    @abc.abstractmethod
    def initialize(
        config: AlgorithmConfigType, env_config: MetaLearningEnvConfig, seed: int = 1
    ) -> "MetaLearningAlgorithm": ...

    @abc.abstractmethod
    def update(self, data: DataType) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def wrap(self) -> MetaLearningAgent: ...

    @abc.abstractmethod
    def train(
        self,
        config: MetaLearningTrainingConfigType,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self: ...


class GradientBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, GradientBasedMetaLearningTrainingConfig, list[Rollout]
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: GradientBasedMetaLearningTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @abc.abstractmethod
    def adapt(self, rollouts: Rollout) -> Self: ...

    @abc.abstractmethod
    def init_ensemble_networks(self) -> Self: ...

    @override
    def train(
        self,
        config: GradientBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
            * (config.num_inner_gradient_steps + 1)
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self = self.init_ensemble_networks()
            all_rollouts: list[Rollout] = []

            # Sampling step
            # Collect num_inner_gradient_steps D datasets + collect 1 D' dataset
            for _step in range(config.num_inner_gradient_steps + 1):
                print(f"- Collecting inner step {_step}")
                obs, _ = envs.reset()
                rollout_buffer.reset()
                episode_started = np.ones((envs.num_envs,))

                while not rollout_buffer.ready:
                    self, actions, aux_policy_outs = self.sample_action_and_aux(obs)

                    next_obs, rewards, terminations, truncations, infos = envs.step(
                        actions
                    )

                    rollout_buffer.add(
                        obs,
                        actions,
                        rewards,
                        episode_started,
                        value=aux_policy_outs.get("value"),
                        log_prob=aux_policy_outs.get("log_prob"),
                        mean=aux_policy_outs.get("mean"),
                        std=aux_policy_outs.get("std"),
                    )

                    episode_started = np.logical_or(terminations, truncations)
                    obs = next_obs

                    for i, env_ended in enumerate(episode_started):
                        if env_ended:
                            global_episodic_return.append(
                                infos["final_info"]["episode"]["r"][i]
                            )
                            global_episodic_length.append(
                                infos["final_info"]["episode"]["l"][i]
                            )

                rollouts = rollout_buffer.get()
                all_rollouts.append(rollouts)

                # Inner policy update for the sake of sampling close to adapted policy during the
                # computation of the objective.
                if _step < config.num_inner_gradient_steps:
                    print(f"- Adaptation step {_step}")
                    self = self.adapt(rollouts)

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing outer step")
            self, logs = self.update(all_rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(logs)
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class RNNBasedMetaLearningAlgorithm(
    MetaLearningAlgorithm[
        AlgorithmConfigType, RNNBasedMetaLearningTrainingConfig, Rollout
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, state: RNNState, observation: Observation
    ) -> tuple[Self, RNNState, Action, AuxPolicyOutputs]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: RNNBasedMetaLearningTrainingConfig,
        example_state: RNNState,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            num_tasks=training_config.meta_batch_size,
            num_rollout_steps=training_config.rollouts_per_task
            * env_config.max_episode_steps,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            rnn_state_dim=example_state.shape[-1],
            seed=seed,
        )

    @abc.abstractmethod
    def init_recurrent_state(self, batch_size: int) -> tuple[Self, RNNState]: ...

    @abc.abstractmethod
    def reset_recurrent_state(
        self, current_state: RNNState, reset_mask: npt.NDArray[np.bool_]
    ) -> tuple[Self, RNNState]: ...

    @override
    def train(
        self,
        config: RNNBasedMetaLearningTrainingConfig,
        envs: GymVectorEnv,
        env_config: MetaLearningEnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        _, example_state = self.init_recurrent_state(config.meta_batch_size)
        rollout_buffer = self.spawn_rollout_buffer(
            env_config, config, example_state, seed
        )

        # NOTE: We assume that eval evns are deterministically initialised and there's no state
        # that needs to be carried over when they're used.
        eval_envs = env_config.spawn_test(seed)

        start_time = time.time()

        steps_per_iter = (
            config.meta_batch_size
            * config.rollouts_per_task
            * env_config.max_episode_steps
        )

        for _iter in range(
            start_step, config.total_steps // steps_per_iter
        ):  # Outer step
            global_step = _iter * steps_per_iter
            print(f"Iteration {_iter}, Global num of steps {global_step}")

            envs.call("sample_tasks")
            self, states = self.init_recurrent_state(config.meta_batch_size)
            obs, _ = envs.reset()
            rollout_buffer.reset()
            episode_started = np.ones((envs.num_envs,))

            while not rollout_buffer.ready:
                self, next_states, actions, aux_policy_outs = (
                    self.sample_action_and_aux(states, obs)
                )

                next_obs, rewards, terminations, truncations, infos = envs.step(actions)

                rollout_buffer.add(
                    obs,
                    actions,
                    rewards,
                    episode_started,
                    value=aux_policy_outs.get("value"),
                    log_prob=aux_policy_outs.get("log_prob"),
                    mean=aux_policy_outs.get("mean"),
                    std=aux_policy_outs.get("std"),
                    rnn_state=states,
                )

                episode_started = np.logical_or(terminations, truncations)
                obs = next_obs
                states = next_states

                for i, env_ended in enumerate(episode_started):
                    if env_ended:
                        global_episodic_return.append(
                            infos["final_info"]["episode"]["r"][i]
                        )
                        global_episodic_length.append(
                            infos["final_info"]["episode"]["l"][i]
                        )

            rollouts = rollout_buffer.get()

            mean_episodic_return = np.mean(list(global_episodic_return))
            print("- Mean episodic return: ", mean_episodic_return)
            if track:
                log(
                    {"charts/mean_episodic_returns": mean_episodic_return},
                    step=global_step,
                )

            # Outer policy update
            print("- Computing update")
            self, logs = self.update(rollouts)

            # Evaluation
            if global_step % config.evaluation_frequency == 0 and global_step > 0:
                print("- Evaluating on the test set...")
                mean_success_rate, mean_returns, mean_success_per_task = (
                    env_config.evaluate_metalearning(eval_envs, self.wrap())
                )

                eval_metrics = {
                    "charts/mean_success_rate": float(mean_success_rate),
                    "charts/mean_evaluation_return": float(mean_returns),
                } | {
                    f"charts/{task_name}_success_rate": float(success_rate)
                    for task_name, success_rate in mean_success_per_task.items()
                }

                if config.evaluate_on_train:
                    print("- Evaluating on the train set...")
                    _, _, eval_success_rate_per_train_task = (
                        env_config.evaluate_metalearning_on_train(
                            envs=envs,
                            agent=self.wrap(),
                        )
                    )
                    for (
                        task_name,
                        success_rate,
                    ) in eval_success_rate_per_train_task.items():
                        eval_metrics[f"charts/{task_name}_train_success_rate"] = float(
                            success_rate
                        )

                print(
                    f"Mean evaluation success rate: {mean_success_rate:.4f}"
                    + f" return: {mean_returns:.4f}"
                )

                if track:
                    log(eval_metrics, step=global_step)

                if checkpoint_manager is not None:
                    checkpoint_manager.save(
                        global_step,
                        args=get_checkpoint_save_args(
                            self,
                            envs,
                            global_step,
                            episodes_ended,
                            run_timestamp,
                        ),
                        metrics={
                            k.removeprefix("charts/"): v
                            for k, v in eval_metrics.items()
                        },
                    )
                    print("- Saved Model")

            # Logging
            print(
                {
                    k: v
                    for k, v in logs.items()
                    if not (k.startswith("nn") or k.startswith("data"))
                }
            )
            sps = global_step / (time.time() - start_time)
            print("- SPS: ", sps)
            if track:
                log({"charts/SPS": sps} | logs, step=global_step)

        eval_envs.close()
        del eval_envs

        return self


class OffPolicyAlgorithm(
    Algorithm[
        AlgorithmConfigType, OffPolicyTrainingConfig, EnvConfig, ReplayBufferSamples
    ],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> AbstractReplayBuffer: ...

    @abc.abstractmethod
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

    @override
    def train(
        self,
        config: OffPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        done = np.full((envs.num_envs,), False)
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        replay_buffer = self.spawn_replay_buffer(env_config, config, seed)
        if buffer_checkpoint is not None:
            replay_buffer.load_checkpoint(buffer_checkpoint)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            if global_step < config.warmstart_steps:
                actions = envs.action_space.sample()
            else:
                self, actions = self.sample_action(obs)

            next_obs, rewards, terminations, truncations, infos = envs.step(actions)
            done = np.logical_or(terminations, truncations)

            buffer_obs = next_obs
            if "final_obs" in infos:
                buffer_obs = np.where(
                    done[:, None], np.stack(infos["final_obs"]), next_obs
                )
            replay_buffer.add(obs, buffer_obs, actions, rewards, done)

            obs = next_obs

            for i, env_ended in enumerate(done):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            if global_step > config.warmstart_steps:
                # Update the agent with data
                data = replay_buffer.sample(config.batch_size)
                self, logs = self.update(data)

                # Logging
                if global_step % 100 == 0:
                    sps_steps = (global_step - start_step) * envs.num_envs
                    sps = int(sps_steps / (time.time() - start_time))
                    print("SPS:", sps)

                    if track:
                        log({"charts/SPS": sps} | logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and done.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not done.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                                buffer=replay_buffer,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()

        return self


class OnPolicyAlgorithm(
    Algorithm[AlgorithmConfigType, OnPolicyTrainingConfig, EnvConfig, Rollout],
    Generic[AlgorithmConfigType],
):
    @abc.abstractmethod
    def sample_action_and_aux(
        self, observation: Observation
    ) -> tuple[Self, Action, AuxPolicyOutputs]: ...

    @abc.abstractmethod
    def sample_action(self, observation: Observation) -> tuple[Self, Action]: ...

    @abc.abstractmethod
    def eval_action(self, observations: Observation) -> Action: ...

    def reset(self, env_mask: npt.NDArray[np.bool_]) -> None:
        del env_mask
        pass  # For evaluation interface compatibility

    @abc.abstractmethod
    def update(
        self,
        data: Rollout,
        dones: Float[npt.NDArray, "task 1"],
        next_obs: Float[Observation, " task"] | None = None,
    ) -> tuple[Self, LogDict]: ...

    def spawn_rollout_buffer(
        self,
        env_config: EnvConfig,
        training_config: OnPolicyTrainingConfig,
        seed: int | None = None,
    ) -> MultiTaskRolloutBuffer:
        return MultiTaskRolloutBuffer(
            training_config.rollout_steps,
            self.num_tasks,
            env_config.observation_space,
            env_config.action_space,
            seed,
        )

    @override
    def train(
        self,
        config: OnPolicyTrainingConfig,
        envs: GymVectorEnv,
        env_config: EnvConfig,
        run_timestamp: str | None = None,
        seed: int = 1,
        track: bool = True,
        checkpoint_manager: ocp.CheckpointManager | None = None,
        checkpoint_metadata: CheckpointMetadata | None = None,
        buffer_checkpoint: ReplayBufferCheckpoint | None = None,
    ) -> Self:
        global_episodic_return: Deque[float] = deque([], maxlen=20 * self.num_tasks)
        global_episodic_length: Deque[int] = deque([], maxlen=20 * self.num_tasks)

        obs, _ = envs.reset()

        episode_started = np.ones((envs.num_envs,))
        start_step, episodes_ended = 0, 0

        if checkpoint_metadata is not None:
            start_step = checkpoint_metadata["step"]
            episodes_ended = checkpoint_metadata["episodes_ended"]

        rollout_buffer = self.spawn_rollout_buffer(env_config, config, seed)

        start_time = time.time()

        for global_step in range(start_step, config.total_steps // envs.num_envs):
            total_steps = global_step * envs.num_envs

            self, actions, aux_policy_outs = self.sample_action_and_aux(obs)
            next_obs, rewards, terminations, truncations, infos = envs.step(actions)

            rollout_buffer.add(
                obs,
                actions,
                rewards,
                episode_started,
                value=aux_policy_outs.get("value"),
                log_prob=aux_policy_outs.get("log_prob"),
                mean=aux_policy_outs.get("mean"),
                std=aux_policy_outs.get("std"),
            )

            episode_started = np.logical_or(terminations, truncations)
            obs = next_obs

            for i, env_ended in enumerate(episode_started):
                if env_ended:
                    global_episodic_return.append(
                        infos["final_info"]["episode"]["r"][i]
                    )
                    global_episodic_length.append(
                        infos["final_info"]["episode"]["l"][i]
                    )
                    episodes_ended += 1

            if global_step % 500 == 0 and global_episodic_return:
                print(
                    f"global_step={total_steps}, mean_episodic_return={np.mean(list(global_episodic_return))}"
                )
                if track:
                    log(
                        {
                            "charts/mean_episodic_return": np.mean(
                                list(global_episodic_return)
                            ),
                            "charts/mean_episodic_length": np.mean(
                                list(global_episodic_length)
                            ),
                        },
                        step=total_steps,
                    )

            # Logging
            if global_step % 1_000 == 0:
                sps_steps = (global_step - start_step) * envs.num_envs
                sps = int(sps_steps / (time.time() - start_time))
                print("SPS:", sps)

                if track:
                    log({"charts/SPS": sps}, step=total_steps)

            if rollout_buffer.ready:
                rollouts = rollout_buffer.get()
                self, logs = self.update(
                    rollouts,
                    dones=terminations,
                    next_obs=np.where(
                        episode_started[:, None], np.stack(infos["final_obs"]), next_obs
                    ),
                )
                rollout_buffer.reset()

                if track:
                    log(logs, step=total_steps)

                # Evaluation
                if (
                    config.evaluation_frequency > 0
                    and episodes_ended % config.evaluation_frequency == 0
                    and episode_started.any()
                    and global_step > 0
                ):
                    mean_success_rate, mean_returns, mean_success_per_task = (
                        env_config.evaluate(envs, self)
                    )
                    eval_metrics = {
                        "charts/mean_success_rate": float(mean_success_rate),
                        "charts/mean_evaluation_return": float(mean_returns),
                    } | {
                        f"charts/{task_name}_success_rate": float(success_rate)
                        for task_name, success_rate in mean_success_per_task.items()
                    }
                    print(
                        f"total_steps={total_steps}, mean evaluation success rate: {mean_success_rate:.4f}"
                        + f" return: {mean_returns:.4f}"
                    )

                    if track:
                        log(eval_metrics, step=total_steps)

                    # Checkpointing
                    if checkpoint_manager is not None:
                        if not episode_started.all():
                            raise NotImplementedError(
                                "Checkpointing currently doesn't work for the case where evaluation is run before all envs have finished their episodes / are about to be reset."
                            )

                        checkpoint_manager.save(
                            total_steps,
                            args=get_checkpoint_save_args(
                                self,
                                envs,
                                global_step,
                                episodes_ended,
                                run_timestamp,
                            ),
                            metrics={
                                k.removeprefix("charts/"): v
                                for k, v in eval_metrics.items()
                            },
                        )

                    # Reset envs again to exit eval mode
                    obs, _ = envs.reset()
                    episode_started = np.ones((envs.num_envs,))

        return self
