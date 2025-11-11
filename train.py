"""Training entry point for the SAC + DrQ-v2 MetaWorld agent."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config.optim import OptimizerConfig
from config.rl import DrQSACConfig, OffPolicyTrainingConfig
from envs.metaworld import MetaworldConfig
from monitoring.utils import close_tensorboard_writer, set_tensorboard_writer
from run import Run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train SAC + DrQ-v2 on MetaWorld tasks"
    )
    parser.add_argument("--env-id", type=str, default="MT10")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=int(1e6))
    parser.add_argument("--warmstart-steps", type=int, default=5_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--evaluation-frequency", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--initial-temperature", type=float, default=0.1)
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--augmentation-pad", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=84)
    parser.add_argument(
        "--camera-names",
        type=str,
        default="corner,corner2,topview",
        help="Comma-separated list of camera names",
    )
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("./checkpoints"),
    )
    parser.add_argument(
        "--logs-dir",
        type=Path,
        default=Path("./logs"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    camera_names = tuple(name.strip() for name in args.camera_names.split(",") if name.strip())
    image_shape = (args.image_size, args.image_size, 3)

    env_config = MetaworldConfig(
        env_id=args.env_id,
        pixel_observations=True,
        camera_names=camera_names,
        image_shape=image_shape,
        terminate_on_success=False,
    )

    num_tasks = env_config.task_one_hot_dim or 1

    algorithm_config = DrQSACConfig(
        num_tasks=num_tasks,
        gamma=args.gamma,
        tau=args.tau,
        initial_temperature=args.initial_temperature,
        actor_optimizer=OptimizerConfig(lr=args.actor_lr),
        critic_optimizer=OptimizerConfig(lr=args.critic_lr),
        encoder_optimizer=OptimizerConfig(lr=args.encoder_lr),
        temperature_optimizer=OptimizerConfig(lr=args.alpha_lr, max_grad_norm=None),
        augmentation_pad=args.augmentation_pad,
        channels_last=True,
    )

    training_config = OffPolicyTrainingConfig(
        total_steps=args.total_steps,
        warmstart_steps=args.warmstart_steps,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        evaluation_frequency=args.evaluation_frequency,
    )

    run_name = args.run_name or f"{args.env_id.lower()}_drqsac"

    data_dir = args.data_dir.expanduser()
    data_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = None
    if not args.no_tensorboard:
        logs_dir = (args.logs_dir / run_name / str(args.seed)).expanduser()
        logs_dir.mkdir(parents=True, exist_ok=True)
        set_tensorboard_writer(str(logs_dir))

    run = Run(
        run_name=run_name,
        seed=args.seed,
        data_dir=data_dir,
        env=env_config,
        algorithm=algorithm_config,
        training_config=training_config,
        resume=args.resume,
    )

    try:
        run.start()
    finally:
        if logs_dir is not None:
            close_tensorboard_writer()


if __name__ == "__main__":
    main()
