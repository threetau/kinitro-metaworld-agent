import flax.linen as nn

import metaworld_algorithms.config.nn

from .base import  VanillaNetwork
from .care import CARENetwork
from .film import FiLMNetwork
from .moore import MOORENetwork
from .multi_head import MultiHeadNetwork
from .paco import PaCoNetwork
from .soft_modules import SoftModularizationNetwork


def get_nn_arch_for_config(
    config: metaworld_algorithms.config.nn.NeuralNetworkConfig,
) -> type[nn.Module]:
    if type(config) is metaworld_algorithms.config.nn.MultiHeadConfig:
        return MultiHeadNetwork
    elif type(config) is metaworld_algorithms.config.nn.SoftModulesConfig:
        return SoftModularizationNetwork
    elif type(config) is metaworld_algorithms.config.nn.PaCoConfig:
        return PaCoNetwork
    elif type(config) is metaworld_algorithms.config.nn.CAREConfig:
        return CARENetwork
    elif type(config) is metaworld_algorithms.config.nn.FiLMConfig:
        return FiLMNetwork
    elif type(config) is metaworld_algorithms.config.nn.MOOREConfig:
        return MOORENetwork
    elif type(config) is metaworld_algorithms.config.nn.VanillaNetworkConfig:
        return VanillaNetwork
    else:
        raise ValueError(
            f"Unknown config type: {type(config)}. (NeuralNetworkConfig by itself is not supported, use VanillaNeworkConfig)"
        )


__all__ = ["VanillaNetwork", "MultiHeadNetwork", "SoftModularizationNetwork"]
