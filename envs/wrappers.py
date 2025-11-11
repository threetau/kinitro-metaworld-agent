"""Custom MetaWorld wrappers for multi-view pixel observations."""

from __future__ import annotations

from typing import Iterable

import gymnasium as gym
import numpy as np
import mujoco
from gymnasium.vector import VectorWrapper
from gymnasium.vector.utils import batch_space


def _split_observation(
    obs: np.ndarray,
    task_one_hot_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split a flat MetaWorld observation into proprioception and task one-hot."""
    obs = np.asarray(obs, dtype=np.float32)
    if task_one_hot_dim <= 0 or task_one_hot_dim > obs.shape[-1]:
        proprio = obs
        task = np.zeros((*obs.shape[:-1], 0), dtype=np.float32)
    else:
        cut_index = obs.shape[-1] - task_one_hot_dim
        proprio = obs[..., :cut_index]
        task = obs[..., cut_index:]
    return proprio.astype(np.float32, copy=False), task.astype(np.float32, copy=False)


class MetaWorldCameraRenderWrapper(gym.Wrapper):
    """Allow passing camera arguments through Gym wrappers for MetaWorld envs."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._camera_name_to_id: dict[str, int] = {}

    def render(self, *args, **kwargs):
        render_mode = kwargs.pop("render_mode", None)
        mode = kwargs.pop("mode", None)
        camera_name = kwargs.pop("camera_name", None)
        target_mode = render_mode if render_mode is not None else mode

        unwrapped = self.env.unwrapped
        prev_mode = getattr(unwrapped, "render_mode", None)

        if target_mode is not None:
            setattr(unwrapped, "render_mode", target_mode)

        if camera_name is not None:
            camera_id = self._get_camera_id(camera_name)
            renderer = getattr(unwrapped, "mujoco_renderer", None)
            if renderer is None:
                raise RuntimeError("MetaWorld env is missing a Mujoco renderer")
            renderer.camera_id = camera_id

        try:
            return self.env.render(*args, **kwargs)
        finally:
            if target_mode is not None:
                setattr(unwrapped, "render_mode", prev_mode)

    def _get_camera_id(self, camera_name: str) -> int:
        if camera_name in self._camera_name_to_id:
            return self._camera_name_to_id[camera_name]

        unwrapped = self.env.unwrapped
        model = getattr(unwrapped, "model", None)
        if model is None:
            raise RuntimeError("MetaWorld env is missing a Mujoco model")

        camera_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if camera_id < 0:
            raise ValueError(f"Mujoco camera '{camera_name}' was not found in the model")

        self._camera_name_to_id[camera_name] = camera_id
        return camera_id


class MetaWorldPixelObservationWrapper(gym.ObservationWrapper):
    """Attach multi-view RGB observations to MetaWorld envs."""

    def __init__(
        self,
        env: gym.Env,
        *,
        camera_names: Iterable[str],
        task_one_hot_dim: int,
        image_shape: tuple[int, int, int],
        channels_last: bool = True,
    ) -> None:
        super().__init__(env)
        self._camera_names = tuple(camera_names)
        self._task_one_hot_dim = int(task_one_hot_dim)
        self._channels_last = channels_last
        self._image_shape = image_shape
        if not channels_last:
            c, h, w = image_shape
            image_space_shape = (c, h, w)
        else:
            image_space_shape = image_shape

        image_spaces = {
            name: gym.spaces.Box(
                low=0,
                high=255,
                shape=image_space_shape,
                dtype=np.uint8,
            )
            for name in self._camera_names
        }

        obs_dim = int(np.prod(self.env.observation_space.shape))
        proprio_dim = max(obs_dim - self._task_one_hot_dim, 0)

        self.observation_space = gym.spaces.Dict(
            {
                "images": gym.spaces.Dict(image_spaces),
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(proprio_dim,),
                    dtype=np.float32,
                ),
                "task_one_hot": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._task_one_hot_dim,),
                    dtype=np.float32,
                ),
            }
        )

    def observation(self, observation: np.ndarray):
        proprio, task = _split_observation(observation, self._task_one_hot_dim)
        images = {
            name: self._render_view(name)
            for name in self._camera_names
        }
        return {
            "images": images,
            "proprio": proprio,
            "task_one_hot": task,
        }

    def _render_view(self, camera_name: str) -> np.ndarray:
        frame = self.env.render(
            render_mode="rgb_array",
            camera_name=camera_name,
        )
        frame = np.asarray(frame)
        if not self._channels_last:
            frame = np.moveaxis(frame, -1, 0)
        return frame.astype(np.uint8, copy=False)


class MetaWorldVectorPixelObservationWrapper(VectorWrapper):
    """Vectorised variant that augments observations with multi-view images."""

    def __init__(
        self,
        env: gym.VectorEnv,
        *,
        camera_names: Iterable[str],
        task_one_hot_dim: int,
        image_shape: tuple[int, int, int],
        channels_last: bool = True,
    ) -> None:
        super().__init__(env)
        self._camera_names = tuple(camera_names)
        self._task_one_hot_dim = int(task_one_hot_dim)
        self._channels_last = channels_last
        self._image_shape = image_shape

        if channels_last:
            image_space_shape = image_shape
        else:
            image_space_shape = (image_shape[2], image_shape[0], image_shape[1])

        image_spaces = {
            name: gym.spaces.Box(
                low=0,
                high=255,
                shape=image_space_shape,
                dtype=np.uint8,
            )
            for name in self._camera_names
        }

        base_obs_space = self.env.single_observation_space
        if not isinstance(base_obs_space, gym.spaces.Box):
            raise ValueError("Expected base MetaWorld vector observation space to be Box")

        obs_dim = int(np.prod(base_obs_space.shape))
        proprio_dim = max(obs_dim - self._task_one_hot_dim, 0)

        single_space = gym.spaces.Dict(
            {
                "images": gym.spaces.Dict(image_spaces),
                "proprio": gym.spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(proprio_dim,),
                    dtype=np.float32,
                ),
                "task_one_hot": gym.spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self._task_one_hot_dim,),
                    dtype=np.float32,
                ),
            }
        )

        self.single_observation_space = single_space
        self.observation_space = batch_space(single_space, self.num_envs)

    def reset(self, **kwargs):  # type: ignore[override]
        obs, info = self.env.reset(**kwargs)
        return self._augment(obs), info

    def call(self, name: str, *args, **kwargs):  # type: ignore[override]
        if hasattr(self.env, "call"):
            return self.env.call(name, *args, **kwargs)
        raise AttributeError(f"{type(self).__name__} has no attribute 'call'")

    def get_attr(self, name: str):
        if hasattr(self.env, "get_attr"):
            return self.env.get_attr(name)
        raise AttributeError(f"{type(self).__name__} has no attribute get_attr")

    def step(self, actions):  # type: ignore[override]
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return self._augment(obs), rewards, terminated, truncated, info

    def step_wait(self):  # type: ignore[override]
        obs, rewards, terminated, truncated, info = self.env.step_wait()
        return self._augment(obs), rewards, terminated, truncated, info

    def _augment(self, observations: np.ndarray):
        proprio, task = _split_observation(observations, self._task_one_hot_dim)
        images = {
            name: self._render_view(name)
            for name in self._camera_names
        }
        return {
            "images": images,
            "proprio": proprio,
            "task_one_hot": task,
        }

    def _render_view(self, camera_name: str) -> np.ndarray:
        frames = self.env.call(
            "render",
            render_mode="rgb_array",
            camera_name=camera_name,
        )
        frames = np.asarray(frames)
        if not self._channels_last:
            frames = np.moveaxis(frames, -1, -3)
        return frames.astype(np.uint8, copy=False)
