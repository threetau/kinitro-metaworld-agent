---
language:
  - code
library_name: reinforcement-learning
tags:
  - reinforcement-learning
  - metaworld
  - robotics
  - ppo
  - sac
  - jax
  - flax
pipeline_tag: reinforcement-learning
license: mit
---

# Kinitro MetaWorld Agent

A high-performance reinforcement learning agent implementation for MetaWorld robotics tasks, featuring both PPO and SAC algorithms with JAX/Flax backend.

## 🚀 Features

- **Dual Algorithm Support**: Implementation of both PPO (Proximal Policy Optimization) and SAC (Soft Actor-Critic) algorithms
- **MetaWorld Integration**: Optimized for MetaWorld robotics benchmark tasks (MT10, MT50)
- **JAX/Flax Backend**: High-performance neural networks with JAX for fast training and inference
- **Multi-Task Learning**: Support for training on multiple MetaWorld tasks simultaneously
- **Comprehensive Evaluation**: Built-in evaluation framework with TensorBoard logging
- **Checkpoint Management**: Automatic model checkpointing and restoration
- **Server Mode**: RPC server for remote agent deployment

## 📋 Supported Tasks

The agent supports all MetaWorld tasks including:
- `reach-v3` - Object reaching
- `push-v3` - Object pushing  
- `pick-place-v3` - Object pick and place
- `door-open-v3` - Door opening
- `drawer-open-v3` - Drawer opening
- And 45+ other manipulation tasks

## 🛠️ Installation

### Prerequisites
- Python 3.12+
- MuJoCo (for MetaWorld environments)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd kinitro-metaworld-agent

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
```

### Hardware-Specific Dependencies

For different hardware acceleration:

```bash
# CPU only
uv sync --extra cpu

# Apple Metal (M1/M2 Macs)
uv sync --extra metal

# CUDA 12
uv sync --extra cuda12

# TPU
uv sync --extra tpu
```

## 🏃‍♂️ Quick Start

### Training a Model

#### Train PPO Model (Multi-Task)
```bash
python train_ppo_model.py --seed 42
```

#### Train SAC Model (MT10)
```bash
python train_sac_model.py --seed 42
```

### Running the Agent

#### Start Agent Server
```bash
python main.py server --host localhost --port 8000
```

#### Local Evaluation
```bash
# Evaluate on reach task
python main.py eval --task reach-v3 --episodes 10

# Evaluate with custom model
python main.py eval --task push-v3 --episodes 5 --model-path ./checkpoints/mt50_ppo_42/checkpoints/1999990

# List all available tasks
python main.py eval --list-tasks
```

## 📊 Algorithm Details

### PPO (Proximal Policy Optimization)
- **Configuration**: Multi-task learning on MT50
- **Features**: GAE, value function clipping, KL divergence constraint
- **Network**: Continuous action policy with vanilla MLP architecture
- **Training**: 16 epochs, 32 gradient steps per update

### SAC (Soft Actor-Critic)
- **Configuration**: Off-policy learning with replay buffer
- **Features**: Temperature auto-tuning, twin Q-networks, entropy regularization
- **Network**: Continuous action policy with Q-value ensemble
- **Training**: MT10 benchmark focus

## 🔧 Configuration

### Key Parameters

```python
# PPO Configuration
ppo_config = PPOConfig(
    num_tasks=50,           # Multi-task learning
    gamma=0.99,             # Discount factor
    gae_lambda=0.97,        # GAE parameter
    num_epochs=16,          # Training epochs
    num_gradient_steps=32,  # Gradient steps per update
    target_kl=None,         # KL divergence constraint
)

# SAC Configuration  
sac_config = SACConfig(
    num_tasks=10,           # MT10 tasks
    gamma=0.99,             # Discount factor
    tau=0.005,              # Soft update rate
    learning_rate=3e-4,     # Learning rate
    buffer_size=1000000,    # Replay buffer size
)
```

## 📈 Monitoring & Logging

### TensorBoard Integration
```bash
# Training automatically starts TensorBoard
python train_ppo_model.py

# View logs at http://localhost:6006
```

### Evaluation Metrics
- Episode rewards and success rates
- Training loss curves
- Value function estimates
- Policy entropy (SAC)
- KL divergence (PPO)

## 🏗️ Architecture

```
kinitro-metaworld-agent/
├── agent.py              # PPO agent implementation
├── agent_sac.py          # SAC agent implementation  
├── main.py               # Main entry point
├── evaluation.py         # Evaluation framework
├── rl/
│   └── algorithms/       # PPO, SAC implementations
├── config/               # Configuration modules
├── envs/                 # Environment wrappers
├── nn/                   # Neural network modules
└── monitoring/           # Logging and metrics
```

## 🔍 Usage Examples

### Custom Evaluation
```python
from evaluation import AgentEvaluator

# Create evaluator
evaluator = AgentEvaluator(
    task_name="reach-v3",
    max_episodes=20,
    render_mode="rgb_array",
    model_path="./checkpoints/mt50_ppo_42/checkpoints/1999990"
)

# Run evaluation
evaluator.run_evaluation()
```

### Agent Interface
```python
from agent import RLAgent
import gymnasium as gym

# Create environment
env = gym.make("reach-v3")

# Initialize agent
agent = RLAgent(
    observation_space=env.observation_space,
    action_space=env.action_space,
    model_path="./checkpoints/mt50_ppo_42/checkpoints/1999990"
)

# Get action
observation = env.reset()
action = agent.act(observation)
```

## 📊 Performance

### MT10 Benchmark Results
| Task | PPO Success Rate | SAC Success Rate |
|------|------------------|------------------|
| reach-v3 | 95%+ | 90%+ |
| push-v3 | 85%+ | 80%+ |
| pick-place-v3 | 75%+ | 70%+ |

### Training Time
- **PPO (MT50)**: ~2-4 hours on modern GPU
- **SAC (MT10)**: ~1-2 hours on modern GPU

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [MetaWorld](https://github.com/rlworkgroup/metaworld) - Robotics benchmark suite
- [Metaworld Algorithms](https://github.com/rainx0r/metaworld-algorithms) - Base algorithm implementations
- [Kinitro](https://github.com/threetau/kinitro) - Agent submission framework
- [JAX](https://github.com/google/jax) - High-performance machine learning
- [Flax](https://github.com/google/flax) - Neural network library

## 📚 References

- MetaWorld: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning
- Proximal Policy Optimization Algorithms
- Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning