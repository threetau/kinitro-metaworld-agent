#!/usr/bin/env python3
"""
Training script for SAC model on MT10.

This script trains a SAC model using the configuration from your example,
which can then be used by the RLAgent. Includes TensorBoard logging for monitoring.
"""

import logging
import subprocess
import threading
import time
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro

from config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OffPolicyTrainingConfig
from envs.metaworld import MetaworldConfig
from rl.algorithms import SACConfig
from run import Run
from monitoring.utils import set_tensorboard_writer, close_tensorboard_writer


@dataclass(frozen=True)
class Args:
    seed: int = 1
    data_dir: Path = Path("./run_results")
    resume: bool = False
    tensorboard_port: int = 6006
    no_tensorboard: bool = False


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )


def launch_tensorboard(log_dir, port=6006):
    """Launch TensorBoard in a separate thread."""
    
    def run_tensorboard():
        try:
            # Wait a moment for initial logs to be written
            time.sleep(2)
            
            # Launch TensorBoard
            subprocess.run(
                [
                    "tensorboard",
                    "--logdir",
                    str(log_dir),
                    "--port",
                    str(port),
                    "--host",
                    "localhost",
                    "--reload_interval",
                    "1",
                ],
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            # TensorBoard failed to start, but don't crash the training
            pass
        except FileNotFoundError:
            # TensorBoard not installed
            pass
    
    # Start TensorBoard in background thread
    tb_thread = threading.Thread(target=run_tensorboard, daemon=True)
    tb_thread.start()
    
    # Give TensorBoard a moment to start
    time.sleep(3)
    
    # Try to open browser
    try:
        webbrowser.open(f"http://localhost:{port}")
    except Exception:
        # Browser opening failed, but that's okay
        pass
    
    return f"http://localhost:{port}"


def main() -> None:
    args = tyro.cli(Args)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Determine TensorBoard usage
    use_tensorboard = not args.no_tensorboard
    
    # Setup TensorBoard log directory
    tensorboard_log_dir = None
    if use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log_dir = args.data_dir / f"tensorboard_logs/mt10_sac_{timestamp}"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")

    # Setup TensorBoard logging if enabled
    if use_tensorboard and tensorboard_log_dir:
        set_tensorboard_writer(str(tensorboard_log_dir))
        logger.info("TensorBoard writer initialized for training logging")

    run = Run(
        run_name="mt10_sac",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=SACConfig(
            num_tasks=10,
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
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
        ),
        checkpoint=True,
        resume=args.resume,
    )


    # Launch TensorBoard if enabled
    tensorboard_url = None
    if use_tensorboard and tensorboard_log_dir:
        logger.info("Starting TensorBoard...")
        try:
            tensorboard_url = launch_tensorboard(tensorboard_log_dir, args.tensorboard_port)
            logger.info(f"TensorBoard available at: {tensorboard_url}")
            logger.info("TensorBoard will show training metrics in real-time")
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard: {e}")
            logger.info("Continuing training without TensorBoard...")

    try:
        logger.info("Starting SAC training...")
        run.start()
        logger.info("Training completed successfully")
        
        if tensorboard_url:
            logger.info(f"View training metrics at: {tensorboard_url}")
            logger.info("TensorBoard will continue running in the background")
            
    except KeyboardInterrupt:
        logger.info("Training stopped by user")
        if tensorboard_url:
            logger.info(f"TensorBoard still available at: {tensorboard_url}")
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        if tensorboard_url:
            logger.info(f"TensorBoard still available at: {tensorboard_url}")
        raise
    finally:
        # Clean up TensorBoard writer
        if use_tensorboard:
            close_tensorboard_writer()
            logger.info("TensorBoard writer closed")


if __name__ == "__main__":
    main()
