"""
Training script for PPO model on MetaWorld tasks.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import tyro

from config.networks import ContinuousActionPolicyConfig, ValueFunctionConfig
from config.nn import VanillaNetworkConfig
from config.optim import OptimizerConfig
from config.rl import OnPolicyTrainingConfig
from envs.metaworld import MetaworldConfig
from rl.algorithms import PPOConfig
from run import Run
from monitoring.utils import set_tensorboard_writer, close_tensorboard_writer
import time
import subprocess
import threading
import webbrowser

@dataclass(frozen=True)
class Args:
    seed: int = 1
    data_dir: Path = Path("./checkpoints")
    resume: bool = False
    tensorboard_port: int = 6006
    no_tensorboard: bool = False


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
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
    """Main training function."""
    args = tyro.cli(Args)
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger("ppo:train")
    
    logger.info(f"Starting PPO training with seed {args.seed}")
    logger.info(f"Data directory: {args.data_dir}")

    # Determine TensorBoard usage
    use_tensorboard = not args.no_tensorboard
    
    # Setup TensorBoard log directory
    tensorboard_log_dir = None
    if use_tensorboard:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tensorboard_log_dir = args.data_dir / f"tensorboard_logs/mt50_ppo_{timestamp}"
        tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")

    # Setup TensorBoard logging if enabled
    if use_tensorboard and tensorboard_log_dir:
        set_tensorboard_writer(str(tensorboard_log_dir))
        print(f"TensorBoard writer initialized for training logging: {tensorboard_log_dir}")
        logger.info("TensorBoard writer initialized for training logging")


    num_tasks = 50

    run = Run(
        run_name="mt50_ppo",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(env_id="MT50"),
        algorithm=PPOConfig(
            num_tasks=num_tasks,
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
        ),
        training_config=OnPolicyTrainingConfig(
            total_steps=int(1e8),
            rollout_steps=2_000,  # Reduced from 10,000 to 2,000 for more frequent logging
            evaluation_frequency=50,  # Evaluate every 50 episodes for more frequent logging
        ),
        checkpoint=True,
        resume=args.resume,
    )

    tensorboard_url = None
    if use_tensorboard and tensorboard_log_dir:
        print("Starting TensorBoard...")
        try:
            tensorboard_url = launch_tensorboard(tensorboard_log_dir, args.tensorboard_port)
            print(f"TensorBoard available at: {tensorboard_url}")
            print("TensorBoard will show training metrics in real-time")
        except Exception as e:
            print(f"Failed to start TensorBoard: {e}")
            print("Continuing training without TensorBoard...")

    logger.info("Starting training...")
    run.start()
    logger.info("Training completed!")
    if use_tensorboard:
        close_tensorboard_writer()
        logger.info("TensorBoard writer closed")


if __name__ == "__main__":
    main()
