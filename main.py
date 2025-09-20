#!/usr/bin/env python3
"""
Main entry point for the agent server and evaluation.

This script provides multiple commands:
- server: Creates an agent implementation and starts the RPC server
- eval: Runs local evaluation of the agent with visual rendering
"""

import argparse
import logging
import subprocess
import sys
import threading
import time
import webbrowser

from agent import RLAgent
from evaluation import AgentEvaluator


def setup_logging(level=logging.INFO):
    """Configure logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
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
                    log_dir,
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
            # TensorBoard failed to start, but don't crash the evaluation
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Agent server and evaluation tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py server --host localhost --port 8000
  python main.py eval --task reach-v3 --episodes 5
  python main.py eval --task push-v3 --episodes 10 --render-mode rgb_array
  python main.py eval --task reach-v3 --episodes 20 --no-tensorboard
  python main.py eval --task door-open-v3 --log-dir custom_logs/
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Server subcommand
    server_parser = subparsers.add_parser("server", help="Start the agent server")
    server_parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind the server to"
    )
    server_parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the server to"
    )
    server_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level",
    )

    # Evaluation subcommand
    eval_parser = subparsers.add_parser("eval", help="Run local agent evaluation")
    eval_parser.add_argument(
        "--task",
        type=str,
        default="reach-v3",
        help="MetaWorld task name (default: reach-v3)",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of episodes to run (default: 5)",
    )
    eval_parser.add_argument(
        "--steps",
        type=int,
        default=200,
        help="Maximum steps per episode (default: 200)",
    )
    eval_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    eval_parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array"],
        help="Rendering mode (default: human)",
    )
    eval_parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    eval_parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List available MetaWorld tasks and exit",
    )
    eval_parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging (default: True)",
    )
    eval_parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    eval_parser.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (auto-generated if not specified)",
    )

    args = parser.parse_args()

    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    log_level = getattr(logging, args.log_level)
    setup_logging(log_level)
    logger = logging.getLogger(__name__)

    if args.command == "server":
        run_server(args, logger)
    elif args.command == "eval":
        run_evaluation(args, logger)


def run_server(args, logger):
    """Run the agent server."""
    # Import server functionality only when needed to avoid capnp dependency for eval
    try:
        from agent_server import start_server
    except ImportError as e:
        logger.error(f"Failed to import server functionality: {e}")
        logger.error("Make sure capnp and other server dependencies are installed")
        sys.exit(1)

    logger.info(f"Starting agent server on {args.host}:{args.port}")

    # Create the RLAgent
    agent = RLAgent()

    # Start the server
    try:
        start_server(agent, args.host, args.port)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Error starting server: {e}", exc_info=True)
        sys.exit(1)


def run_evaluation(args, logger):
    """Run local agent evaluation."""
    logger.info("Running local evaluation")

    # Determine TensorBoard usage
    use_tensorboard = args.tensorboard and not args.no_tensorboard

    # Setup log directory if using TensorBoard
    log_dir = args.log_dir
    if use_tensorboard and not log_dir:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"runs/{args.task}_{timestamp}"

    # Create evaluator
    evaluator = AgentEvaluator(
        task_name=args.task,
        render_mode=args.render_mode,
        max_episodes=args.episodes,
        max_steps_per_episode=args.steps,
        seed=args.seed,
        use_tensorboard=use_tensorboard,
        log_dir=log_dir,
    )

    if args.list_tasks:
        evaluator.list_available_tasks()
        return

    # Launch TensorBoard if enabled
    tensorboard_url = None
    if use_tensorboard and log_dir:
        logger.info("Starting TensorBoard...")
        try:
            tensorboard_url = launch_tensorboard(log_dir)
            logger.info(f"TensorBoard available at: {tensorboard_url}")
            logger.info("TensorBoard will show metrics in real-time during evaluation")
        except Exception as e:
            logger.warning(f"Failed to start TensorBoard: {e}")
            logger.info("Continuing evaluation without TensorBoard...")

    try:
        evaluator.run_evaluation()
        logger.info("Evaluation completed successfully")

        if tensorboard_url:
            logger.info(f"View detailed metrics at: {tensorboard_url}")
            logger.info("TensorBoard will continue running in the background")

        # Optionally save results to file
        # import json
        # with open("evaluation_results.json", "w") as f:
        #     json.dump(results, f, indent=2)
        # logger.info("Results saved to evaluation_results.json")

    except KeyboardInterrupt:
        logger.info("Evaluation stopped by user")
        if tensorboard_url:
            logger.info(f"TensorBoard still available at: {tensorboard_url}")
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        if tensorboard_url:
            logger.info(f"TensorBoard still available at: {tensorboard_url}")
        sys.exit(1)


if __name__ == "__main__":
    main()
