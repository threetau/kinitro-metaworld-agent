"""
Implementation of the AgentInterface for MetaWorld tasks.

This agent uses the SawyerPickPlaceV2Policy from MetaWorld as an expert policy.
"""

import logging
from typing import Any, Dict

import gymnasium as gym
import metaworld
import numpy as np
import torch
from agent_interface import AgentInterface
from metaworld.policies.sawyer_reach_v3_policy import SawyerReachV3Policy


class RLAgent(AgentInterface):
    """
    MetaWorld agent implementation using the SawyerReachV3Policy expert policy.

    This agent uses the expert policy from MetaWorld for reach tasks.
    """

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, seed, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing MetaWorld agent with seed {self.seed}")

        self.policy = SawyerReachV3Policy()
        self.logger.info("Successfully initialized SawyerReachV3Policy")

        # Track episode state
        self.episode_step = 0
        self.max_episode_steps = kwargs.get("max_episode_steps", 200)

        self.logger.info("MetaWorld agent initialized successfully")

    def act(self, obs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Process the observation and return an action using the MetaWorld expert policy.

        Args:
            obs: Observation from the environment
            kwargs: Additional arguments

        Returns:
            action: Action tensor to take in the environment
        """
        try:
            # Process observation to extract the format needed by the expert policy
            processed_obs = self._process_observation(obs)

            # Use the expert policy (MetaWorld is always available)
            # MetaWorld policies expect numpy arrays
            action_numpy = self.policy.get_action(processed_obs)
            action_tensor = torch.from_numpy(np.array(action_numpy)).float()

            # Log occasionally
            if self.episode_step % 50 == 0:
                self.logger.debug(f"Using expert policy action: {action_numpy}")

            # Increment episode step
            self.episode_step += 1

            # Occasionally log actions to avoid spam
            if self.episode_step % 50 == 0:
                self.logger.debug(
                    f"Step {self.episode_step}: Action shape {action_tensor.shape}"
                )

            return action_tensor

        except Exception as e:
            self.logger.error(f"Error in act method: {e}", exc_info=True)
            # Return zeros as a fallback
            if isinstance(self.action_space, gym.spaces.Box):
                return torch.zeros(self.action_space.shape[0], dtype=torch.float32)
            else:
                return torch.zeros(4, dtype=torch.float32)

    def _process_observation(self, obs):
        """
        Helper method to process observations for the MetaWorld expert policy.

        MetaWorld policies typically expect a specific observation format.
        """
        if isinstance(obs, dict):
            # MetaWorld environment can return observations in different formats
            if "observation" in obs:
                # Standard format for goal-observable environments
                processed_obs = obs["observation"]
            elif "obs" in obs:
                processed_obs = obs["obs"]
            elif "state_observation" in obs:
                # Some MetaWorld environments use this key
                processed_obs = obs["state_observation"]
            elif "goal_achieved" in obs:
                # If we have information about goal achievement
                # This might be needed for certain policy decisions
                achievement = obs.get("goal_achieved", False)
                base_obs = next(iter(obs.values()))
                self.logger.debug(f"Goal achieved: {achievement}")
                processed_obs = base_obs
            else:
                # If structure is unknown, use the first value
                processed_obs = next(iter(obs.values()))
                self.logger.debug(f"Using observation key: {next(iter(obs.keys()))}")
        else:
            # If already a numpy array or similar, use directly
            processed_obs = obs

        # Ensure we're returning a numpy array as expected by MetaWorld policies
        if not isinstance(processed_obs, np.ndarray):
            try:
                processed_obs = np.array(processed_obs, dtype=np.float32)
            except Exception as e:
                self.logger.error(f"Failed to convert observation to numpy array: {e}")
                # Return a dummy observation if conversion fails
                if (
                    self.observation_space
                    and hasattr(self.observation_space, "shape")
                    and self.observation_space.shape is not None
                ):
                    processed_obs = np.zeros(
                        self.observation_space.shape, dtype=np.float32
                    )
                else:
                    # Typical MetaWorld observation dimension if all else fails
                    processed_obs = np.zeros(39, dtype=np.float32)

        return processed_obs

    def reset(self) -> None:
        """
        Reset agent state between episodes.
        """
        self.logger.debug("Resetting agent")
        self.episode_step = 0
        # Any other stateful components would be reset here

    def _build_model(self):
        """
        Build a neural network model for the agent.

        This is a placeholder for where you would define your neural network
        architecture using PyTorch, TensorFlow, or another framework.
        """
        # Example of where you might build a simple PyTorch model
        # model = torch.nn.Sequential(
        #     torch.nn.Linear(self.observation_space.shape[0], 128),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(128, 64),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(64, self.action_space.shape[0]),
        # )
        # return model
        pass
