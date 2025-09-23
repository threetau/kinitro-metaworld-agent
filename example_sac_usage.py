#!/usr/bin/env python3
"""
Example usage of the SAC-based MetaWorld agent.

This script demonstrates how to:
1. Create a SAC agent with a trained model
2. Use the agent for evaluation
3. Run the agent in server mode
"""

import logging
from pathlib import Path

from agent import RLAgent
from evaluation import AgentEvaluator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_with_trained_model():
    """Example using a trained SAC model."""
    logger.info("=== Example with Trained SAC Model ===")
    
    # Path to your trained model checkpoint
    model_path = "./run_results/mt10_sac_1/checkpoints"  # Update this path
    
    # Create the SAC agent with trained model
    agent = RLAgent(
        model_path=model_path,
        num_tasks=10,  # MT10 has 10 tasks
        seed=42
    )
    
    # Create evaluator
    evaluator = AgentEvaluator(
        task_name="reach-v3",  # You can change this to any MT10 task
        render_mode="human",
        max_episodes=3,
        max_steps_per_episode=200,
        seed=42,
        use_tensorboard=True
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        logger.info(f"Evaluation results: {results}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def example_with_random_model():
    """Example using a randomly initialized SAC model (for testing)."""
    logger.info("=== Example with Random SAC Model ===")
    
    # Create the SAC agent without a trained model (will use random weights)
    agent = RLAgent(
        model_path=None,  # No trained model
        num_tasks=10,
        seed=42
    )
    
    # Create evaluator
    evaluator = AgentEvaluator(
        task_name="reach-v3",
        render_mode="human",
        max_episodes=2,
        max_steps_per_episode=200,
        seed=42,
        use_tensorboard=False  # Disable tensorboard for quick test
    )
    
    # Run evaluation
    try:
        results = evaluator.run_evaluation()
        logger.info(f"Evaluation results: {results}")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")


def example_server_mode():
    """Example of running the agent in server mode."""
    logger.info("=== Example Server Mode ===")
    
    # Create the SAC agent
    agent = RLAgent(
        model_path="./run_results/mt10_sac_1/checkpoints",  # Update this path
        num_tasks=10,
        seed=42
    )
    
    logger.info("Starting server...")
    logger.info("You can now run: python main.py server --host localhost --port 8000")
    logger.info("The agent will be available for remote evaluation via Cap'n Proto RPC")


if __name__ == "__main__":
    print("SAC-based MetaWorld Agent Examples")
    print("=" * 40)
    
    # Choose which example to run
    choice = input("Choose example (1: trained model, 2: random model, 3: server mode): ").strip()
    
    if choice == "1":
        example_with_trained_model()
    elif choice == "2":
        example_with_random_model()
    elif choice == "3":
        example_server_mode()
    else:
        print("Invalid choice. Running random model example...")
        example_with_random_model()
