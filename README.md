# Frogger RL 🐸

A reinforcement learning implementation of the classic Frogger game, featuring a trained RL agent and an interactive UI.

## Quick Start

Run the interactive UI to play yourself or watch the trained RL agent:

```bash
python frogger_ui.py
```

### Features

- **Two Play Modes:**
  - 🎮 Play yourself with real-time game updates (cars keep moving!)
  - 🤖 Watch the trained RL agent navigate through traffic

- **Two Rendering Modes:**
  - ASCII mode (classic text-based)
  - Emoji mode (🐸🚗🏆)

- **Three Speed Settings (Human Play):**
  - Fast (0.75s per step)
  - Medium (1s per step)
  - Slow (1.25s per step)

### Controls (Human Play)

The game runs continuously - cars keep moving even when you don't act! Time your moves carefully.

- `W`: Move up
- `S`: Move down
- `A`: Move left
- `D`: Move right
- `Space`: Stay in place (or just don't press anything)
- `Q`: Quit

## Project Structure

- `frogger_ui.py` - Interactive UI (recommended way to play)
- `frogger_env.py` - Frogger environment implementation
- `frogger_policy.py` - Neural network policy
- `train_agent.py` - Training script for the RL agent
- `simulate_frogger_agent.py` - Agent evaluation and visualization
- `human_play.py` - Legacy human play mode
- `checkpoints/frogger_policy.pt` - Trained policy weights

## Training Your Own Agent

```bash
python train_agent.py
```

## Requirements

- Python 3.x
- PyTorch
- NumPy

Install dependencies:
```bash
pip install torch numpy
```
