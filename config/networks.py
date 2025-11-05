from dataclasses import dataclass

from typing import Sequence, Tuple

from config.utils import Activation, Initializer, StdType

from .nn import NeuralNetworkConfig, RecurrentNeuralNetworkConfig, VanillaNetworkConfig


@dataclass(frozen=True)
class ContinuousActionPolicyConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    squash_tanh: bool = True
    """Whether or not to squash the outputs with tanh."""

    log_std_min: float | None = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float | None = 2.0
    """The maximum possible log standard deviation for each action distribution."""

    std_type: StdType = StdType.MLP_HEAD
    """How to learn the standard deviation of the distribution.
    `MLP_HEAD` means it will be an output head from the last layer of the MLP torso and therefore state-dependent.
    `PARAM` means it will be a learned parameter per action dimension that will be state-independent."""

    head_kernel_init: Initializer | None = None
    """Override the initializer to use for the MLP head weights."""

    head_bias_init: Initializer | None = None
    """Override the initializer to use for the MLP head biases."""


@dataclass(frozen=True)
class RecurrentContinuousActionPolicyConfig:
    network_config: RecurrentNeuralNetworkConfig = RecurrentNeuralNetworkConfig()
    """The config for the neural network to use for function approximation."""

    encoder_config: NeuralNetworkConfig | None = VanillaNetworkConfig(
        width=400, depth=2
    )
    """The config for the neural network to use for encoding the observations. The optimizer config for this network is ignored."""

    squash_tanh: bool = True
    """Whether or not to squash the outputs with tanh."""

    log_std_min: float | None = -20.0
    """The minimum possible log standard deviation for each action distribution."""

    log_std_max: float | None = 2.0
    """The maximum possible log standard deviation for each action distribution."""

    std_type: StdType = StdType.MLP_HEAD
    """How to learn the standard deviation of the distribution.
    `MLP_HEAD` means it will be an output head from the last layer of the MLP torso and therefore state-dependent.
    `PARAM` means it will be a learned parameter per action dimension that will be state-independent."""

    head_kernel_init: Initializer | None = None
    """Override the initializer to use for the MLP head weights."""

    head_bias_init: Initializer | None = None
    """Override the initializer to use for the MLP head biases."""

    activate_head: bool = False
    """Whether or not to activate the MLP head after the RNN layer."""


@dataclass(frozen=True)
class QValueFunctionConfig:
    network_config: NeuralNetworkConfig = VanillaNetworkConfig(width=400, depth=3)
    """The config for the neural network to use for function approximation."""

    use_classification: bool = False
    """Whether or not to use classification instead of regression."""


@dataclass(frozen=True)
class ValueFunctionConfig(QValueFunctionConfig): ...


@dataclass(frozen=True)
class PixelEncoderConfig:
    """Configuration for the shared DrQ-style convolutional encoder."""

    feature_dim: int = 256
    num_layers: int = 4
    num_filters: int = 32
    kernel_size: int = 3
    strides: Sequence[int] = (2, 1, 1, 1)
    activation: Activation = Activation.ReLU
    use_bias: bool = True
    layer_norm: bool = False
    channels_last: bool = True
    output_activation: Activation = Activation.Identity


@dataclass(frozen=True)
class ProprioEncoderConfig:
    """Configuration for the proprioceptive MLP encoder."""

    network_config: VanillaNetworkConfig = VanillaNetworkConfig(width=256, depth=2)
    output_dim: int = 128
    activation: Activation = Activation.ReLU


@dataclass(frozen=True)
class TaskEmbeddingConfig:
    """Configuration for embedding task one-hot vectors."""

    embedding_dim: int = 64
    activation: Activation = Activation.ReLU


@dataclass(frozen=True)
class ObservationFusionConfig:
    """Configuration for combining image, proprio, and task features."""

    pixel_encoder: PixelEncoderConfig = PixelEncoderConfig()
    proprio_encoder: ProprioEncoderConfig = ProprioEncoderConfig()
    task_embedding: TaskEmbeddingConfig = TaskEmbeddingConfig()
    fusion_mlp: VanillaNetworkConfig = VanillaNetworkConfig(width=512, depth=2)
    latent_dim: int = 256
    view_names: Tuple[str, ...] = ("corner", "corner2", "topview")
