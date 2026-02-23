# Snake Deep Reinforcement Learning (DQN + PER + Curriculum)

A Deep Reinforcement Learning project: a Snake environment trained with an improved DQN agent (Double DQN + target network, Prioritized Experience Replay) and a simple curriculum (grid size + obstacles).

## Prerequisites

- Python 3.10+
- Git

Notes about PyTorch:

- Installing `torch` can vary by OS/CUDA.
- If `pip install torch` fails, follow the official PyTorch install instructions for your platform, then install the remaining dependencies.

## Clone

```bash
git clone https://github.com/justweb-s/snake-drl.git
cd snake-drl
```

## Quickstart

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS (bash/zsh):

```bash
source .venv/bin/activate
```

```bash
pip install -e .
python -m snake_drl.train --episodes 500
python -m snake_drl.play --checkpoint runs/<RUN_NAME>/checkpoints/best.pt --device cpu
```

## Demo

Add a short GIF/video here (e.g. `assets/demo.gif`).

Recommended workflow:

1. Train a checkpoint.
2. Run `play` with `--fps 30`.
3. Record a short clip (10–20s) using a screen recorder.
4. Convert to a GIF and place it under `assets/`.

## Highlights

- DQN (PyTorch) with dropout
- Double DQN + target network
- Prioritized Experience Replay (PER)
- Gradient clipping
- Curriculum learning (grid size + obstacles)
- Reproducible training (seed)
- Logging (TensorBoard + CSV)
- Checkpointing (`best.pt` / `last.pt`)

## Project structure

- `snake_drl/`
  - `env.py`: Snake environment
  - `model.py`: DQN network
  - `memory.py`: PER replay buffer
  - `agent.py`: DQN agent (Double DQN)
  - `curriculum.py`: curriculum schedule
  - `train.py`: training CLI
  - `play.py`: play/evaluate a trained checkpoint
- `scripts/plot_metrics.py`: plot `metrics.csv`
- `assets/`: put demo media here

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

Linux/macOS (bash/zsh):

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Editable install (recommended)

```bash
pip install -e .
```

This enables the console scripts:

```bash
snake-drl-train --help
snake-drl-play --help
```

## Train

Training is headless (no game window): progress, checkpoints and metrics are written to disk under `runs/`.

Headless training (default):

```bash
python -m snake_drl.train --episodes 2000 --seed 42
```

Print progress more frequently (every episode):

```bash
python -m snake_drl.train --episodes 500 --print-every 1
```

Disable console output:

```bash
python -m snake_drl.train --episodes 500 --print-every 0
```

Live training (render the agent while training):

```bash
python -m snake_drl.train --episodes 500 --render --render-fps 30 --cell-size 20
```

Controls (when `--render` is enabled):

- Close the window or press `ESC` to stop training gracefully.

To speed up rendering you can skip frames (render every N environment steps):

```bash
python -m snake_drl.train --episodes 500 --render --render-skip 5
```

If you installed the package in editable mode (`pip install -e .`), you can also use:

```bash
snake-drl-train --episodes 2000 --seed 42
```

Checkpoints are saved under:

```text
runs/<RUN_NAME>/checkpoints/best.pt
runs/<RUN_NAME>/checkpoints/last.pt
```

Metrics are saved under:

```text
runs/<RUN_NAME>/metrics.csv
runs/<RUN_NAME>/tensorboard/
```

With evaluation every N episodes:

```bash
python -m snake_drl.train --episodes 2000 --eval-every 200 --eval-episodes 10
```

Resume from a checkpoint:

```bash
python -m snake_drl.train --resume runs/<RUN_NAME>/checkpoints/last.pt
```

## TensorBoard

```bash
tensorboard --logdir runs
```

## Play (render a trained agent)

```bash
python -m snake_drl.play --checkpoint runs/<RUN_NAME>/checkpoints/best.pt
```

Or with the console script:

```bash
snake-drl-play --checkpoint runs/<RUN_NAME>/checkpoints/best.pt
```

Force CPU (recommended for portability):

```bash
python -m snake_drl.play --checkpoint runs/<RUN_NAME>/checkpoints/best.pt --device cpu
```

## Plot results

```bash
python scripts/plot_metrics.py --metrics runs/<RUN_NAME>/metrics.csv
```

You can commit the resulting PNG (e.g. `assets/training_curve.png`) and embed it here.

## Reproducibility

- Training uses `--seed` (NumPy/Python/Torch).
- Use `--deterministic` for more deterministic CUDA behavior (at a performance cost).
- Results can still vary due to GPU/driver differences and the stochastic nature of RL.

## Development

Install dev extras:

```bash
pip install -e .[dev,plots]
```

Enable pre-commit hooks:

```bash
pre-commit install
```

Run lint/format:

```bash
ruff check .
ruff format .
```

Run tests:

```bash
pytest -q
```

## Notes

- Action space is relative to the current direction:
  - `0`: forward
  - `1`: turn right
  - `2`: turn left
- Observation is a compact feature vector (danger in 8 directions, direction one-hot, food direction, normalized distances/position/length, nearest obstacle distance).

## License

MIT.
