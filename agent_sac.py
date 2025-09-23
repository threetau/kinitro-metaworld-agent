"""
Implementation of the AgentInterface for MetaWorld tasks.

This agent uses a trained SAC model for multi-task learning on MetaWorld environments.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import torch
from agent_interface import AgentInterface

# Import SAC-related modules
from config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OffPolicyTrainingConfig
from envs.metaworld import MetaworldConfig
from rl.algorithms import SACConfig, SAC
from checkpoint import get_checkpoint_restore_args
import orbax.checkpoint as ocp


class RLAgent(AgentInterface):
    """
    MetaWorld agent implementation using a trained SAC model for multi-task learning.

    This agent uses a trained SAC model that can handle multiple MetaWorld tasks.
    """

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        seed: int | None = None,
        model_path: Optional[str] = None,
        num_tasks: int = 10,
        **kwargs,
    ):
        super().__init__(observation_space, action_space, seed, **kwargs)

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing SAC-based MetaWorld agent with seed {self.seed}")

        self.num_tasks = num_tasks
        self.model_path = model_path
        self.sac_agent = None
        
        # Track episode state
        self.episode_step = 0
        self.max_episode_steps = kwargs.get("max_episode_steps", 200)

        # Initialize the SAC agent
        self._initialize_sac_agent()
        
        self.logger.info("SAC-based MetaWorld agent initialized successfully")

    def _initialize_sac_agent(self):
        """Initialize the SAC agent with the configuration from your example."""
        try:
            # Create the environment configuration
            env_config = MetaworldConfig(
                env_id="MT50",
                terminate_on_success=False,
            )
            
            # Create the SAC configuration (matching your example)
            sac_config = SACConfig(
                num_tasks=self.num_tasks,
                gamma=0.99,
                actor_config=ContinuousActionPolicyConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0)
                    )
                ),
                critic_config=QValueFunctionConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0),
                    )
                ),
                num_critics=2,
            )
            
            # Initialize the SAC agent
            self.sac_agent = SAC.initialize(sac_config, env_config, seed=self.seed)
            self.logger.info("SAC agent initialized successfully")
            
            # Load trained model (auto-detect if no path provided)
            self._load_trained_model()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize SAC agent: {e}", exc_info=True)
            raise

    def _load_trained_model(self):
        """Load a trained SAC model from checkpoint."""
        try:
            if self.model_path is None:
                # Try to find the latest checkpoint in ./checkpoints folder
                checkpoints_dir = Path("./checkpoints")
                if checkpoints_dir.exists():
                    # Look for run directories (e.g., mt10_sac_42)
                    run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir()]
                    latest_checkpoint = None
                    
                    for run_dir in run_dirs:
                        # Look for checkpoints subdirectory
                        run_checkpoints_dir = run_dir / "checkpoints"
                        if run_checkpoints_dir.exists():
                            # Find all checkpoint step directories
                            step_dirs = [d for d in run_checkpoints_dir.iterdir() if d.is_dir() and d.name.isdigit()]
                            if step_dirs:
                                # Get the latest step directory
                                latest_step = max(step_dirs, key=lambda x: int(x.name))
                                latest_checkpoint = latest_step
                                break
                    
                    if latest_checkpoint:
                        self.model_path = str(latest_checkpoint.absolute())
                        self.logger.info(f"Auto-detected latest checkpoint: {self.model_path}")
                    else:
                        self.logger.warning("No checkpoint directories found in ./checkpoints")
                        return
                else:
                    self.logger.warning("No model path provided and ./checkpoints directory not found")
                    return
                
            checkpoint_manager = ocp.CheckpointManager(
                Path(self.model_path).parent,
                item_names=("agent", "env_states", "rngs", "buffer", "metadata"),
                options=ocp.CheckpointManagerOptions(create=False),
            )
            
            if checkpoint_manager.latest_step() is not None:
                # Restore the agent from checkpoint
                restore_args = get_checkpoint_restore_args(self.sac_agent)
                ckpt = checkpoint_manager.restore(
                    checkpoint_manager.latest_step(),
                    args=restore_args,
                )
                self.sac_agent = ckpt["agent"]
                self.logger.info(f"Loaded trained SAC model from step {checkpoint_manager.latest_step()}")
            else:
                self.logger.warning("No checkpoint found in the specified directory")
                
        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}", exc_info=True)
            self.logger.info("Continuing with randomly initialized model")

    def act(self, obs: Dict[str, Any], **kwargs) -> torch.Tensor:
        """
        Process the observation and return an action using the trained SAC model.

        Args:
            obs: Observation from the environment
            kwargs: Additional arguments

        Returns:
            action: Action tensor to take in the environment
        """
        try:
            if self.sac_agent is None:
                raise RuntimeError("SAC agent not initialized")
                
            # Process observation to extract the format needed by the SAC agent
            processed_obs = self._process_observation_for_sac(obs)

            # Use the SAC agent to get action
            # SAC expects JAX arrays, so we need to convert
            obs_jax = jnp.array(processed_obs)
            
            # Get action from SAC agent (deterministic evaluation)
            action_jax = self.sac_agent.eval_action(obs_jax)
            action_numpy = np.array(action_jax)
            action_tensor = torch.from_numpy(action_numpy).float()

            # Log occasionally
            if self.episode_step % 50 == 0:
                self.logger.debug(f"Using SAC action: {action_numpy}")

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

    def _process_observation_for_sac(self, obs):
        """
        Helper method to process observations for the SAC agent.

        SAC expects observations in a specific format for multi-task learning.
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

        # Ensure we're returning a numpy array
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

        # For multi-task SAC, we need to add task information
        # This is a simplified approach - you might need to modify based on your specific setup
        if len(processed_obs.shape) == 1:
            # Add task one-hot encoding if not present
            # For now, we'll assume task 0 (you might want to make this configurable)
            task_one_hot = np.zeros(self.num_tasks, dtype=np.float32)
            task_one_hot[0] = 1.0  # Default to first task
            processed_obs = np.concatenate([processed_obs, task_one_hot])

        return processed_obs

    def reset(self) -> None:
        """
        Reset agent state between episodes.
        """
        self.logger.debug("Resetting SAC agent")
        self.episode_step = 0
        # SAC agent doesn't need explicit reset as it's stateless during evaluation
