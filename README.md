# MARL-OHT-Optimization

Multi-Agent Reinforcement Learning project for OHT (Overhead Hoist Transport)
traffic optimization in semiconductor fab-style rail networks.

The current implementation focuses on:

- PettingZoo parallel multi-agent OHT environment
- Mega-fab directed rail graph generation with NetworkX
- Hot Lot priority delivery and yield behavior
- RLlib shared-policy PPO training and evaluation
- Dijkstra and random-policy baselines
- Density sweep evaluation for 20/30/40/50 OHT traffic
- Pygame visualization with zoom, pan, pause, replay, and recording

## Project Structure

```text
envs/
  grid_map.py                 Fab rail graph generator
  oht_env.py                  PettingZoo ParallelEnv OHT simulator

agents/
  train_ppo_rllib.py          PPO training, checkpointing, evaluation helpers
  evaluate_density_sweep.py   PPO vs Dijkstra density sweep and visualization
  train_rllib_comparison.py   PPO/APPO/IMPALA comparison runner
  run_curriculum.py           Curriculum training stages
  dijkstra_baseline.py        Shortest-path baseline policy

utils/
  scenario_scheduler.py       Deterministic task and Hot Lot batch scheduler
  visualization.py            Pygame OHT simulator viewer
  wandb_logger.py             W&B logging wrapper with console fallback

tests/
  test_*.py                   Environment, PettingZoo, metric, and RLlib smoke tests

checkpoints/
  best/ppo_50oht_collision_safe/
                              Best PPO checkpoint used by density sweep
```

## Setup

Python 3.10+ is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `wandb` is not configured, the project still runs in console logging mode.

## Quick Start

Run the main density sweep evaluation:

```bash
python3 agents/evaluate_density_sweep.py
```

By default this uses the best evaluation configuration:

```text
OHT_DENSITY_OHTS=20,30,40,50
OHT_DENSITY_EPISODES=5
OHT_DENSITY_MAX_STEPS=400
OHT_BATCH_HOT_LOT_RATIO=0.3
OHT_TERMINATE_ON_COLLISION=1
OHT_MAP_WIDTH=300
OHT_MAP_HEIGHT=200
OHT_BAY_INTERVAL=10
OHT_BAY_DEPTH=10
OHT_DENSITY_SEED=1234
```

The script evaluates PPO and Dijkstra on identical task batches for each OHT
density. Results are written to:

```text
results/density_sweep_eval.csv
```

## Visualization

Visualize one selected density episode without running the full sweep:

```bash
OHT_DENSITY_VISUALIZE_ONLY=1 \
OHT_DENSITY_VISUALIZE_POLICY=ppo \
OHT_DENSITY_VISUALIZE_OHTS=50 \
OHT_DENSITY_VISUALIZE_EPISODE=0 \
python3 agents/evaluate_density_sweep.py
```

Available policies:

```text
ppo
dijkstra
both
```

Viewer controls:

- Mouse wheel: zoom in/out
- Mouse drag: pan
- `+` / `-`: zoom in/out
- `F`: fit whole map
- `SPACE`: pause/resume
- `LEFT` / `RIGHT`: step through frames while paused
- `R`: start/stop frame recording
- `ESC`: quit

When an episode ends early, the final frame stays open so the terminal can show
why it ended, for example collision/stall truncation or Hot Lot batch completion.

## Training

Train or fine-tune shared-policy PPO:

```bash
python3 agents/train_ppo_rllib.py
```

Useful environment variables:

```text
OHT_NUM_OHTS                 Training OHT count
OHT_NUM_ITERATIONS           PPO training iterations
OHT_MAP_WIDTH                Map width
OHT_MAP_HEIGHT               Map height
OHT_BAY_INTERVAL             Bay spacing
OHT_BAY_DEPTH                Bay loop depth
OHT_HOT_LOT_PROBABILITY      Probability of generated Hot Lot tasks
OHT_CHECKPOINT_IN            Restore checkpoint path
OHT_CHECKPOINT_OUT           Save checkpoint path
OHT_BEST_CHECKPOINT_DIR      Save best periodic evaluation checkpoint
OHT_SKIP_EVAL=1              Train only, skip final evaluation
```

Example:

```bash
OHT_NUM_OHTS=50 \
OHT_NUM_ITERATIONS=100 \
OHT_CHECKPOINT_OUT=checkpoints/ppo_50oht_run \
python3 agents/train_ppo_rllib.py
```

## Algorithm Comparison

Compare RLlib PPO, APPO, and IMPALA:

```bash
python3 agents/train_rllib_comparison.py
```

Select algorithms:

```bash
OHT_ALGORITHMS=ppo,appo python3 agents/train_rllib_comparison.py
```

## Evaluation Meaning

The density sweep runs each OHT density for multiple deterministic task batches.
With the default configuration:

```text
20 OHT x 5 episodes x PPO/Dijkstra
30 OHT x 5 episodes x PPO/Dijkstra
40 OHT x 5 episodes x PPO/Dijkstra
50 OHT x 5 episodes x PPO/Dijkstra
```

For each episode, `OHT_BATCH_HOT_LOT_RATIO=0.3` marks 30% of the initially
assigned tasks as Hot Lots. The script reports how many Hot Lots are completed,
how long the Hot Lot batch takes, delivery throughput, collision rate, invalid
actions, and episode return.

Key metrics:

- `HotDone`: average completed Hot Lots
- `HotBCR`: Hot Lot batch completion rate
- `HotSteps`: steps needed to finish the Hot Lot batch, capped at max steps
- `Delivery`: average total deliveries
- `TP`: throughput, deliveries per step
- `Collision`: average collision count
- `SafeEp`: fraction of collision-free episodes
- `Return`: average total reward

## Tests

Run tests with pytest:

```bash
python3 -m pytest -q
```

Lightweight smoke checks:

```bash
python3 tests/test_pettingzoo_api.py
python3 tests/test_rllib_env.py
python3 tests/test_max_steps.py
```

## Notes

- `checkpoints/best/ppo_50oht_collision_safe` is intentionally tracked because
  it is the reference checkpoint for density sweep evaluation and visualization.
- Experiment outputs such as `results/`, `ray_results/`, `wandb/`, recordings,
  and cache folders are ignored by git.
- The environment default map is intentionally small for fast smoke tests. The
  density sweep explicitly overrides map settings to the 300x200 best-evaluation
  configuration.
