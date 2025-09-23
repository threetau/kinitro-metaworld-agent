"""
PPO-based agent implementation for MetaWorld tasks.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
import torch
from agent_interface import AgentInterface
from checkpoint import get_checkpoint_restore_args
from envs.metaworld import MetaworldConfig
from config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OnPolicyTrainingConfig
from rl.algorithms import PPO, PPOConfig
import orbax.checkpoint as ocp


class RLAgent(AgentInterface):
    """
    PPO-based agent for MetaWorld tasks.
    
    This agent uses the PPO algorithm from your library and can load trained models
    from checkpoints. It automatically detects the latest checkpoint in ./checkpoints
    if no model path is provided.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        seed: Optional[int] = None,
        max_episode_steps: int = 200,
        model_path: Optional[str] = None,
        num_tasks: int = 50,
    ):
        """
        Initialize the PPO agent.
        
        Args:
            observation_space: Environment observation space
            action_space: Environment action space  
            seed: Random seed for reproducibility
            max_episode_steps: Maximum steps per episode
            model_path: Path to trained model checkpoint (auto-detected if None)
            num_tasks: Number of tasks for multi-task learning
        """
        super().__init__(observation_space, action_space, seed)
        
        self.max_episode_steps = max_episode_steps
        self.model_path = model_path
        self.num_tasks = num_tasks
        self.logger = logging.getLogger(__name__)
        
        # Initialize PPO agent
        self.ppo_agent = None
        self._initialize_ppo_agent()

    def _initialize_ppo_agent(self):
        """Initialize the PPO agent with the configuration."""
        try:
            # Create the environment configuration
            env_config = MetaworldConfig(
                env_id="MT50",
                terminate_on_success=False,
            )
            
            # Create the PPO configuration
            ppo_config = PPOConfig(
                num_tasks=self.num_tasks,
                gamma=0.99,
                policy_config=ContinuousActionPolicyConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0),
                    )
                ),
                vf_config=ValueFunctionConfig(
                    network_config=VanillaNetworkConfig(
                        optimizer=OptimizerConfig(max_grad_norm=1.0),
                    )
                ),
                num_epochs=16,
                num_gradient_steps=32,
                gae_lambda=0.97,
                target_kl=None,
                clip_vf_loss=False,
            )
            
            # Initialize the PPO agent
            self.ppo_agent = PPO.initialize(ppo_config, env_config, seed=self.seed)
            self.logger.info("PPO agent initialized successfully")
            
            # Load trained model (auto-detect if no path provided)
            self._load_trained_model()
                
        except Exception as e:
            self.logger.error(f"Failed to initialize PPO agent: {e}", exc_info=True)
            raise

    def _load_trained_model(self):
        """Load a trained PPO model from checkpoint."""
        try:
            if self.model_path is None:
                # Try to find the latest checkpoint in ./checkpoints folder
                checkpoints_dir = Path("./checkpoints")
                if checkpoints_dir.exists():
                    # Look for PPO run directories (e.g., mt50_ppo_42)
                    run_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and "ppo" in d.name.lower()]
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
                        self.logger.warning("No PPO checkpoint directories found in ./checkpoints")
                        return
                else:
                    self.logger.warning("No model path provided and ./checkpoints directory not found")
                    return
                
            checkpoint_manager = ocp.CheckpointManager(
                Path(self.model_path).parent.absolute(),
                item_names=("agent", "env_states", "rngs", "buffer", "metadata"),
                options=ocp.CheckpointManagerOptions(create=False),
            )

            if checkpoint_manager.latest_step() is not None:
                # Restore the agent from checkpoint
                restore_args = get_checkpoint_restore_args(self.ppo_agent)
                ckpt = checkpoint_manager.restore(
                    checkpoint_manager.latest_step(),
                    args=restore_args,
                )
                self.ppo_agent = ckpt["agent"]
                self.logger.info(f"Loaded trained PPO model from step {checkpoint_manager.latest_step()}")
            else:
                self.logger.warning("No checkpoint found in the specified directory")

        except Exception as e:
            self.logger.error(f"Failed to load trained model: {e}", exc_info=True)
            self.logger.info("Continuing with randomly initialized model")

    def act(self, observation: Union[np.ndarray, dict], deterministic: bool = True) -> np.ndarray:
        """
        Generate an action given an observation.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            Action as numpy array
        """
        if self.ppo_agent is None:
            raise RuntimeError("PPO agent not initialized")
            
        # Process observation for PPO
        processed_obs = self._process_observation_for_ppo(observation)
        
        # Convert to JAX array
        obs_jax = jnp.array(processed_obs)
        
        # Get action from PPO agent (PPO doesn't have deterministic parameter)
        action_jax = self.ppo_agent.eval_action(obs_jax)
        
        # Convert back to numpy array
        action_np = np.array(action_jax)
        
        return action_np

    def _process_observation_for_ppo(self, observation: Union[np.ndarray, dict]) -> np.ndarray:
        """
        Process observation for PPO agent.
        
        Args:
            observation: Raw observation from environment
            
        Returns:
            Processed observation as numpy array
        """
        if isinstance(observation, dict):
            # Extract state from dict observation
            obs = observation.get("state", observation.get("observation", observation))
            if isinstance(obs, dict):
                # If still a dict, concatenate values
                obs = np.concatenate(list(obs.values()))
        else:
            obs = observation
            
        # Ensure it's a numpy array
        obs = np.array(obs, dtype=np.float32)
        
        # For multi-task PPO, we need to add task encoding
        # For now, assume task 0 (single task evaluation)
        task_encoding = np.zeros(self.num_tasks, dtype=np.float32)
        task_encoding[0] = 1.0  # One-hot encoding for task 0
        
        # Concatenate observation with task encoding
        processed_obs = np.concatenate([obs, task_encoding])
        
        return processed_obs

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Reset the agent for a new episode.
        
        Args:
            seed: Optional seed for the new episode
        """
        # PPO agent is stateless during evaluation, so no reset needed
        if seed is not None:
            self.seed = seed
