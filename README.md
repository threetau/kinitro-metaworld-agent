# MetaWorld Agent Migration (PPO → SAC + DrQ-v2)
## Installation

### Prerequisites
- Python 3.12+
- MuJoCo (for MetaWorld environments)

### Quick Start

```bash
# Install with uv (recommended)
uv venv --python python 3.12
source .venv/bin/activate
uv sync
```

### Training a Model

> ⚠️ The legacy PPO trainer has been removed. A new SAC + DrQ-v2 training
> pipeline is under construction and will land in an upcoming update.


#### Local Evaluation
```bash
# Evaluate on reach task
python evaluation.py --task <task_name> --episodes <num_episodes> --render-mode <human|rgb_array> --model-path <path_to_trained_model>
```
Available tasks:
```
reach-v3
push-v3
pick-place-v3
door-open-v3
door-close-v3
drawer-open-v3
button-press-topdown-v3
button-press-v3
peg-insert-side-v3
lever-pull-v3
```

## Roadmap
- ✅ Enable multi-view pixel observations (corner, corner2, topview) alongside
  proprioception and task one-hot inputs.
- 🚧 Replace the PPO baseline with a SAC + DrQ-v2 implementation tuned for
  MetaWorld MT10 multi-task learning.
- 🔜 Reintroduce end-to-end training and evaluation documentation once the new
  agent lands.

### TensorBoard Integration
```bash
# Training automatically starts TensorBoard
python train.py

# View logs at http://localhost:6006
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MetaWorld](https://github.com/rlworkgroup/metaworld) - Robotics benchmark suite
- [Metaworld Algorithms](https://github.com/rainx0r/metaworld-algorithms) - Base algorithm implementations
- [Kinitro](https://github.com/threetau/kinitro) - Agent submission framework
- [JAX](https://github.com/google/jax) - High-performance machine learning
- [Flax](https://github.com/google/flax) - Neural network library
