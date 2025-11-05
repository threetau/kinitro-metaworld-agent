from config.rl import AlgorithmConfig, DrQSACConfig

from .base import Algorithm, OffPolicyAlgorithm
from .drq import DrQSAC
from .mtsac import MTSAC, MTSACConfig
from .ppo import PPOConfig, PPO
from .sac import SAC, SACConfig
from .mamltrpo import MAMLTRPO, MAMLTRPOConfig
from .rl2 import RL2, RL2Config


def get_algorithm_for_config(config: AlgorithmConfig) -> type[Algorithm]:
    if type(config) is MTSACConfig:
        return MTSAC
    elif type(config) is DrQSACConfig:
        return DrQSAC
    elif type(config) is PPOConfig:
        return PPO
    elif type(config) is SACConfig:
        return SAC
    elif type(config) is MAMLTRPOConfig:
        return MAMLTRPO
    elif type(config) is RL2Config:
        return RL2
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


__all__ = [
    "Algorithm",
    "OffPolicyAlgorithm",
    "DrQSAC",
    "DrQSACConfig",
    "MTSAC",
    "PPO",
    "MTSACConfig",
    "PPOConfig",
    "SAC",
    "SACConfig",
    "MAMLTRPO",
    "MAMLTRPOConfig",
    "RL2",
    "RL2Config",
]
