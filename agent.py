"""SAC + DrQ-v2 evaluation agent for MetaWorld tasks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Mapping, Optional, Sequence

import jax
import numpy as np
import orbax.checkpoint as ocp

from agent_interface import AgentInterface
from checkpoint import get_checkpoint_restore_args
from config.optim import OptimizerConfig
from config.rl import DrQSACConfig
from envs.metaworld import MetaworldConfig
from rl.algorithms.drq import DrQSAC

DEFAULT_CAMERAS: tuple[str, ...] = ("corner", "corner2", "topview")


class RLAgent(AgentInterface):
    """MetaWorld agent driven by the SAC + DrQ-v2 policy."""

    def __init__(
        self,
        observation_space=None,
        action_space=None,
        seed: Optional[int] = None,
        model_path: Optional[str] = None,
        env_id: str = "MT10",
        camera_names: Optional[Sequence[str]] = None,
        image_size: int = 84,
        gamma: float = 0.99,
        tau: float = 0.01,
        initial_temperature: float = 0.1,
    ) -> None:
        super().__init__(observation_space, action_space, seed)
        self.logger = logging.getLogger(__name__)

        self.camera_names = tuple(camera_names) if camera_names else DEFAULT_CAMERAS
        image_shape = (image_size, image_size, 3)

        self.env_config = MetaworldConfig(
            env_id=env_id,
            pixel_observations=True,
            camera_names=self.camera_names,
            image_shape=image_shape,
            terminate_on_success=False,
        )

        num_tasks = self.env_config.task_one_hot_dim or 1

        self.config = DrQSACConfig(
            num_tasks=num_tasks,
            gamma=gamma,
            tau=tau,
            initial_temperature=initial_temperature,
            augmentation_pad=4,
            channels_last=True,
            actor_optimizer=OptimizerConfig(lr=3e-4),
            critic_optimizer=OptimizerConfig(lr=3e-4),
            encoder_optimizer=OptimizerConfig(lr=3e-4),
            temperature_optimizer=OptimizerConfig(lr=3e-4, max_grad_norm=None),
        )

        self.algorithm = DrQSAC.initialize(
            self.config,
            self.env_config,
            seed=self.seed,
        )
        self.algorithm = self.algorithm.replace(
            key=jax.random.PRNGKey(self.seed)
        )

        if model_path is None:
            model_path = self._auto_detect_checkpoint()

        if model_path is None:
            raise ValueError(
                "No model checkpoint found. Provide --model-path or place checkpoints under ./checkpoints."
            )

        self._load_checkpoint(Path(model_path))

    def act(self, observation, deterministic: bool = True, **_) -> np.ndarray:
        obs_dict = self._normalize_observation(observation)
        action = self.algorithm.eval_action(obs_dict)
        return np.asarray(action).reshape(self.action_space.shape)

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self.seed = seed
            self.algorithm = self.algorithm.replace(key=jax.random.PRNGKey(seed))

    # ---------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------

    def _normalize_observation(self, observation) -> Mapping[str, object]:
        if not isinstance(observation, Mapping):
            raise TypeError("Expected observation to be a mapping with pixel data")

        if "images" not in observation:
            raise KeyError("Observation missing 'images' key")

        images = observation["images"]
        if not isinstance(images, Mapping):
            raise TypeError("Observation 'images' entry must be a mapping")

        processed_images = {}
        for name in self.camera_names:
            if name not in images:
                raise KeyError(f"Observation missing image view '{name}'")
            frame = np.asarray(images[name])
            if frame.ndim != 3:
                raise ValueError(f"Camera '{name}' expected 3D array, got shape {frame.shape}")
            processed_images[name] = frame.astype(np.uint8, copy=False)

        proprio = np.asarray(observation.get("proprio", []), dtype=np.float32)
        task = np.asarray(observation.get("task_one_hot", []), dtype=np.float32)

        return {
            "images": processed_images,
            "proprio": proprio,
            "task_one_hot": task,
        }

    def _auto_detect_checkpoint(self) -> Optional[str]:
        checkpoints_dir = Path("./checkpoints")
        if not checkpoints_dir.exists():
            return None

        run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
        for run_dir in sorted(run_dirs, reverse=True):
            ckpt_root = run_dir / "checkpoints"
            if not ckpt_root.exists():
                continue
            step_dirs = [d for d in ckpt_root.iterdir() if d.is_dir() and d.name.isdigit()]
            if not step_dirs:
                continue
            latest_step = max(step_dirs, key=lambda p: int(p.name))
            self.logger.info("Auto-detected checkpoint: %s", latest_step)
            return str(latest_step)
        return None

    def _load_checkpoint(self, checkpoint_path: Path) -> None:
        if checkpoint_path.is_dir() and checkpoint_path.name.isdigit():
            step = int(checkpoint_path.name)
            manager_root = checkpoint_path.parent
        else:
            manager_root = checkpoint_path / "checkpoints"
            if not manager_root.exists():
                raise FileNotFoundError(
                    f"Checkpoint directory not found at {checkpoint_path}"
                )
            step_dirs = [
                d for d in manager_root.iterdir() if d.is_dir() and d.name.isdigit()
            ]
            if not step_dirs:
                raise FileNotFoundError(
                    f"No checkpoint steps found under {manager_root}"
                )
            latest = max(step_dirs, key=lambda p: int(p.name))
            step = int(latest.name)
            manager_root = latest.parent

        checkpoint_manager = ocp.CheckpointManager(
            manager_root,
            item_names=("agent", "env_states", "rngs", "buffer", "metadata"),
            options=ocp.CheckpointManagerOptions(create=False),
        )

        restore_args = get_checkpoint_restore_args(self.algorithm)
        ckpt = checkpoint_manager.restore(step, args=restore_args)
        self.algorithm = ckpt["agent"]
        self.algorithm = self.algorithm.replace(
            key=jax.random.PRNGKey(self.seed)
        )
