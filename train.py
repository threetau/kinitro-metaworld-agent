"""Training entry-point placeholder during migration to SAC + DrQ-v2."""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the next-generation MetaWorld agent.")
    parser.add_argument(
        "--pixel-observations",
        action="store_true",
        help="Reserved for the upcoming SAC + DrQ-v2 pipeline.",
    )
    return parser.parse_args()


def main() -> None:
    _ = parse_args()
    raise NotImplementedError(
        "PPO training has been removed. The SAC + DrQ-v2 training pipeline will be added soon."
    )


if __name__ == "__main__":
    main()
*** End of File
