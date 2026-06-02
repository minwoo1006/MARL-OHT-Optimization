"""
Density sweep evaluation for PPO vs Dijkstra on identical Hot Lot batches.

This script is evaluation-only. It restores a PPO checkpoint and compares it
against Dijkstra over the same generated task batches at multiple OHT densities.
The main extra KPI is Hot Lot batch completion: how many steps are needed to
finish all initially assigned Hot Lots.
"""

import csv
import os
import random
import sys

os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.getcwd(), ".matplotlib_cache"),
)

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from agents.dijkstra_baseline import DijkstraBaselineAgent
from agents.train_ppo_rllib import (
    DEFAULT_MAP_CONFIG,
    configure_ray_storage,
    env_creator,
    parse_int_list,
    policy_mapping_fn,
    read_map_config,
    read_reward_config,
    safe_compute_action,
)
from envs.oht_env import OHTFabEnv
from utils.scenario_scheduler import Task


CSV_FIELDS = [
    "policy",
    "num_ohts",
    "episodes",
    "hot_lot_batch_size",
    "avg_hot_lot_completed",
    "avg_hot_lot_batch_completion_rate",
    "avg_steps_to_complete_hot_lots",
    "avg_delivery_count",
    "avg_throughput",
    "avg_cycle_time",
    "avg_hot_lot_cycle_time",
    "avg_collision_count",
    "avg_collision_free_rate",
    "avg_invalid_action_count",
    "avg_episode_return",
]

BEST_DENSITY_SWEEP_CONFIG = {
    "densities": [20, 30, 40, 50],
    "episodes": 5,
    "max_steps": 400,
    "hot_lot_ratio": 0.3,
    "terminate_on_collision": 1,
    "map_width": 300,
    "map_height": 200,
    "bay_interval": 10,
    "bay_depth": 10,
    "seed": 1234,
}

def env_default(name, default, best_default, use_best_config):
    if name in os.environ:
        return os.environ[name]
    return str(best_default if use_best_config else default)


def read_density_map_config(use_best_config):
    if not use_best_config:
        return read_map_config()
    return {
        "width": int(env_default("OHT_MAP_WIDTH", DEFAULT_MAP_CONFIG["width"], BEST_DENSITY_SWEEP_CONFIG["map_width"], True)),
        "height": int(env_default("OHT_MAP_HEIGHT", DEFAULT_MAP_CONFIG["height"], BEST_DENSITY_SWEEP_CONFIG["map_height"], True)),
        "bay_interval": int(env_default("OHT_BAY_INTERVAL", DEFAULT_MAP_CONFIG["bay_interval"], BEST_DENSITY_SWEEP_CONFIG["bay_interval"], True)),
        "bay_depth": int(env_default("OHT_BAY_DEPTH", DEFAULT_MAP_CONFIG["bay_depth"], BEST_DENSITY_SWEEP_CONFIG["bay_depth"], True)),
    }


def append_csv_row(path, row):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, "r", encoding="utf-8") as csv_file:
            existing_header = csv_file.readline().strip().split(",")
        if existing_header != CSV_FIELDS:
            file_exists = False

    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in CSV_FIELDS})


def build_ppo(checkpoint_path, env_config):
    register_env("oht_fab_env", env_creator)
    temp_env = OHTFabEnv(**env_config)
    obs_space = temp_env.observation_space("oht_0")
    act_space = temp_env.action_space("oht_0")

    config = (
        PPOConfig()
        .environment(env="oht_fab_env", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=float(os.environ.get("OHT_NUM_GPUS", "0")))
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=int(os.environ.get("OHT_TRAIN_BATCH_SIZE", "1000")),
            minibatch_size=int(os.environ.get("OHT_MINIBATCH_SIZE", "128")),
            num_epochs=int(os.environ.get("OHT_NUM_EPOCHS", "10")),
            lr=float(os.environ.get("OHT_LR", "3e-4")),
            gamma=float(os.environ.get("OHT_GAMMA", "0.99")),
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    algo = config.build_algo()
    algo.restore(os.path.abspath(checkpoint_path))
    return algo


def make_task_batch(ports, num_ohts, hot_lot_batch_size, rng, dist_fn):
    if num_ohts > len(ports):
        raise ValueError(
            f"num_ohts={num_ohts} exceeds available unique ports={len(ports)}. "
            "Use a larger map or lower OHT count."
        )

    starts = rng.sample(ports, num_ohts)
    tasks = []
    for idx, start in enumerate(starts):
        reachable_goals = [
            port
            for port in ports
            if port != start and dist_fn(start, port) < 100
        ]
        if not reachable_goals:
            raise ValueError(f"No reachable port goal found from start={start}")
        goal = rng.choice(reachable_goals)
        tasks.append(Task(start, goal, is_hot_lot=idx < hot_lot_batch_size))
    rng.shuffle(tasks)
    return tasks


def run_episode(
    policy,
    num_ohts,
    max_steps,
    map_config,
    reward_config,
    tasks,
    algo=None,
    visualize=False,
    visualize_fps=6,
    keep_visualization_open=False,
):
    env = OHTFabEnv(
        num_ohts=num_ohts,
        max_steps=max_steps,
        reward_config=reward_config,
        initial_tasks=tasks,
        **map_config,
    )
    obs, infos = env.reset()
    dijkstra_agent = DijkstraBaselineAgent(env.graph) if policy == "dijkstra" else None
    episode_return = 0.0
    steps_to_complete_hot_lots = None
    target_hot_lots = env.hot_lot_assigned_count
    viz = None
    end_reason = "max_steps"

    if visualize:
        from utils.visualization import OHTVisualizer

        viz = OHTVisualizer(env, fps=visualize_fps)
        viz.init()

    try:
        for _ in range(max_steps):
            if policy == "ppo":
                action_dict = {
                    agent_id: safe_compute_action(algo, obs[agent_id], env=env, agent_id=agent_id)
                    for agent_id in env.agents
                }
            elif policy == "dijkstra":
                action_dict = {
                    agent_id: dijkstra_agent.get_action(env, agent_id)
                    for agent_id in env.agents
                }
            else:
                raise ValueError(f"Unsupported policy: {policy}")

            prev_positions = env.agent_positions.copy()
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            episode_return += sum(rewards.values())

            if viz:
                collision_nodes = viz.detect_collisions(prev_positions, env)
                viz.push_snapshot(env.current_step, infos, collision_nodes)
                if not viz.render():
                    end_reason = "viewer_closed"
                    break

            if target_hot_lots > 0 and env.hot_lot_delivery_count >= target_hot_lots:
                steps_to_complete_hot_lots = env.current_step
                end_reason = "hot_lot_batch_completed"
                break
            if all(truncations.values()):
                end_reason = (
                    "collision_or_stall_truncation"
                    if env.collision_count > 0
                    else "max_steps_or_stall_truncation"
                )
                break
            if all(terminations.values()):
                end_reason = "terminated"
                break
            if not env.agents:
                end_reason = "no_active_agents"
                break
    finally:
        if viz:
            print(
                f"Visualization episode ended at step={env.current_step} "
                f"reason={end_reason} deliveries={env.delivery_count} "
                f"hot_lots={env.hot_lot_delivery_count}/{target_hot_lots} "
                f"collisions={env.collision_count} invalid={env.invalid_action_count}"
            )
            if keep_visualization_open and end_reason != "viewer_closed":
                viz.wait_until_closed()
            viz.close()

    metrics = env.get_episode_metrics()
    return {
        "hot_lot_completed": env.hot_lot_delivery_count,
        "hot_lot_batch_completion_rate": (
            env.hot_lot_delivery_count / target_hot_lots if target_hot_lots else 0.0
        ),
        "steps_to_complete_hot_lots": (
            steps_to_complete_hot_lots if steps_to_complete_hot_lots is not None else max_steps
        ),
        "delivery_count": env.delivery_count,
        "throughput": metrics["throughput"],
        "avg_cycle_time": metrics["avg_cycle_time"],
        "avg_hot_lot_cycle_time": metrics["avg_hot_lot_cycle_time"],
        "collision_count": env.collision_count,
        "collision_free": 1.0 if env.collision_count == 0 else 0.0,
        "invalid_action_count": env.invalid_action_count,
        "episode_return": episode_return,
    }


def summarize(policy, num_ohts, episodes, hot_lot_batch_size, rows):
    return {
        "policy": policy,
        "num_ohts": num_ohts,
        "episodes": episodes,
        "hot_lot_batch_size": hot_lot_batch_size,
        "avg_hot_lot_completed": np.mean([row["hot_lot_completed"] for row in rows]),
        "avg_hot_lot_batch_completion_rate": np.mean(
            [row["hot_lot_batch_completion_rate"] for row in rows]
        ),
        "avg_steps_to_complete_hot_lots": np.mean(
            [row["steps_to_complete_hot_lots"] for row in rows]
        ),
        "avg_delivery_count": np.mean([row["delivery_count"] for row in rows]),
        "avg_throughput": np.mean([row["throughput"] for row in rows]),
        "avg_cycle_time": np.mean([row["avg_cycle_time"] for row in rows]),
        "avg_hot_lot_cycle_time": np.mean([row["avg_hot_lot_cycle_time"] for row in rows]),
        "avg_collision_count": np.mean([row["collision_count"] for row in rows]),
        "avg_collision_free_rate": np.mean([row["collision_free"] for row in rows]),
        "avg_invalid_action_count": np.mean([row["invalid_action_count"] for row in rows]),
        "avg_episode_return": np.mean([row["episode_return"] for row in rows]),
    }


def print_table(rows):
    print("\n" + "=" * 110)
    print("Density Sweep: PPO vs Dijkstra on identical Hot Lot batches")
    print("=" * 110)
    print(
        f"{'Policy':<10} {'OHTs':<6} {'HotBatch':<9} {'HotDone':<8} "
        f"{'HotBCR':<8} {'HotSteps':<9} {'Delivery':<9} {'TP':<7} "
        f"{'Collision':<10} {'SafeEp':<7} {'Return':<10}"
    )
    print("-" * 110)
    for row in rows:
        print(
            f"{row['policy']:<10} {row['num_ohts']:<6} {row['hot_lot_batch_size']:<9} "
            f"{row['avg_hot_lot_completed']:<8.2f} "
            f"{row['avg_hot_lot_batch_completion_rate']:<8.2f} "
            f"{row['avg_steps_to_complete_hot_lots']:<9.2f} "
            f"{row['avg_delivery_count']:<9.2f} "
            f"{row['avg_throughput']:<7.3f} "
            f"{row['avg_collision_count']:<10.2f} "
            f"{row['avg_collision_free_rate']:<7.2f} "
            f"{row['avg_episode_return']:<10.2f}"
        )


def print_recommendation(rows):
    by_density = {}
    for row in rows:
        by_density.setdefault(row["num_ohts"], {})[row["policy"]] = row

    strict_candidates = []
    stress_candidates = []
    for num_ohts, policies in sorted(by_density.items()):
        ppo = policies.get("ppo")
        dijkstra = policies.get("dijkstra")
        if ppo is None or dijkstra is None:
            continue

        hot_lot_win = (
            ppo["avg_hot_lot_batch_completion_rate"]
            > dijkstra["avg_hot_lot_batch_completion_rate"]
        )
        delivery_win = ppo["avg_delivery_count"] > dijkstra["avg_delivery_count"]
        strict_safe = (
            ppo["avg_collision_count"] == 0
            and ppo["avg_collision_free_rate"] == 1.0
        )
        if hot_lot_win and delivery_win and strict_safe:
            strict_candidates.append(num_ohts)
        elif hot_lot_win and delivery_win:
            stress_candidates.append(num_ohts)

    print("\nRecommendation")
    if strict_candidates:
        print(
            "Strict safety demo density: "
            f"{max(strict_candidates)} OHTs "
            "(PPO beats Dijkstra on Hot Lot and delivery with zero collisions)."
        )
    else:
        print("Strict safety demo density: none in this sweep.")

    if stress_candidates:
        print(
            "Stress-test density: "
            f"{max(stress_candidates)} OHTs "
            "(PPO has KPI upside, but collision handling still needs work)."
        )


def main():
    configure_ray_storage()
    visualize_only = os.environ.get("OHT_DENSITY_VISUALIZE_ONLY", "0") == "1"
    use_best_config = (
        os.environ.get("OHT_DENSITY_BEST_CONFIG", "1") == "1"
        or visualize_only
    )
    map_config = read_density_map_config(use_best_config)
    reward_config = read_reward_config()
    if use_best_config and "OHT_TERMINATE_ON_COLLISION" not in os.environ:
        reward_config["terminate_on_collision"] = BEST_DENSITY_SWEEP_CONFIG["terminate_on_collision"]

    max_steps = int(env_default("OHT_DENSITY_MAX_STEPS", 400, BEST_DENSITY_SWEEP_CONFIG["max_steps"], use_best_config))
    episodes = int(env_default("OHT_DENSITY_EPISODES", 5, BEST_DENSITY_SWEEP_CONFIG["episodes"], use_best_config))
    densities = parse_int_list(
        os.environ.get("OHT_DENSITY_OHTS"),
        BEST_DENSITY_SWEEP_CONFIG["densities"] if use_best_config else [20, 30, 40, 50],
    )
    hot_lot_ratio = float(env_default("OHT_BATCH_HOT_LOT_RATIO", 0.3, BEST_DENSITY_SWEEP_CONFIG["hot_lot_ratio"], use_best_config))
    seed = int(env_default("OHT_DENSITY_SEED", 1234, BEST_DENSITY_SWEEP_CONFIG["seed"], use_best_config))
    visualize = os.environ.get("OHT_DENSITY_VISUALIZE", "0") == "1" or visualize_only
    default_visualize_policy = "ppo" if visualize_only else "dijkstra"
    visualize_policy = os.environ.get("OHT_DENSITY_VISUALIZE_POLICY", default_visualize_policy).lower()
    default_visualize_ohts = BEST_DENSITY_SWEEP_CONFIG["densities"][-1] if use_best_config else densities[0]
    visualize_ohts = int(os.environ.get("OHT_DENSITY_VISUALIZE_OHTS", str(default_visualize_ohts)))
    visualize_episode = int(os.environ.get("OHT_DENSITY_VISUALIZE_EPISODE", "0"))
    visualize_fps = int(os.environ.get("OHT_DENSITY_VISUALIZE_FPS", "6"))
    checkpoint_path = os.environ.get(
        "OHT_CHECKPOINT_IN",
        os.path.join("checkpoints", "best", "ppo_50oht_collision_safe"),
    )
    csv_path = os.path.abspath(
        os.environ.get(
            "OHT_DENSITY_CSV",
            os.path.join("results", "density_sweep_eval.csv"),
        )
    )

    base_env = OHTFabEnv(num_ohts=1, max_steps=max_steps, reward_config=reward_config, **map_config)
    print(f"Map ports={len(base_env.port_nodes)} | densities={densities} | max_steps={max_steps}")
    if use_best_config:
        print(
            "Best density config active: "
            f"OHTs={densities}, episodes={episodes}, hot_lot_ratio={hot_lot_ratio}, "
            f"terminate_on_collision={reward_config['terminate_on_collision']}, "
            f"map={map_config['width']}x{map_config['height']}, "
            f"bay_interval={map_config['bay_interval']}, bay_depth={map_config['bay_depth']}, "
            f"seed={seed}"
        )
    if visualize:
        print(
            "Visualization enabled: "
            f"policy={visualize_policy}, OHTs={visualize_ohts}, "
            f"episode={visualize_episode}, fps={visualize_fps}"
        )

    env_config = {
        "num_ohts": max(densities),
        "max_steps": max_steps,
        "reward_config": reward_config,
        **map_config,
    }

    needs_ppo = not visualize_only or visualize_policy in ("ppo", "both")
    algo = None
    if needs_ppo:
        ray.init(ignore_reinit_error=True, include_dashboard=False)
        algo = build_ppo(checkpoint_path, env_config)

    if visualize_only:
        if visualize_ohts not in densities:
            raise ValueError(
                f"OHT_DENSITY_VISUALIZE_OHTS={visualize_ohts} is not in OHT_DENSITY_OHTS={densities}"
            )
        if visualize_episode < 0 or visualize_episode >= episodes:
            raise ValueError(
                f"OHT_DENSITY_VISUALIZE_EPISODE={visualize_episode} must be between 0 and {episodes - 1}"
            )

        hot_lot_batch_size = max(1, int(round(visualize_ohts * hot_lot_ratio)))
        rng = random.Random(seed + visualize_ohts * 1000 + visualize_episode)
        tasks = make_task_batch(
            base_env.port_nodes,
            visualize_ohts,
            hot_lot_batch_size,
            rng,
            base_env._get_shortest_dist,
        )
        selected_policies = ["ppo", "dijkstra"] if visualize_policy == "both" else [visualize_policy]
        rows = []
        for policy in selected_policies:
            rows.append(
                {
                    **run_episode(
                        policy=policy,
                        num_ohts=visualize_ohts,
                        max_steps=max_steps,
                        map_config=map_config,
                        reward_config=reward_config,
                        tasks=tasks,
                        algo=algo,
                        visualize=True,
                        visualize_fps=visualize_fps,
                        keep_visualization_open=True,
                    ),
                    "policy": policy,
                    "num_ohts": visualize_ohts,
                    "episodes": 1,
                    "hot_lot_batch_size": hot_lot_batch_size,
                }
            )
        print_table([
            summarize(row["policy"], row["num_ohts"], 1, row["hot_lot_batch_size"], [row])
            for row in rows
        ])
        if algo:
            algo.stop()
            ray.shutdown()
        return

    summaries = []
    for num_ohts in densities:
        hot_lot_batch_size = max(1, int(round(num_ohts * hot_lot_ratio)))
        policy_rows = {"ppo": [], "dijkstra": []}
        for episode_idx in range(episodes):
            rng = random.Random(seed + num_ohts * 1000 + episode_idx)
            tasks = make_task_batch(
                base_env.port_nodes,
                num_ohts,
                hot_lot_batch_size,
                rng,
                base_env._get_shortest_dist,
            )
            for policy in policy_rows:
                should_visualize = (
                    visualize
                    and num_ohts == visualize_ohts
                    and episode_idx == visualize_episode
                    and (visualize_policy == "both" or visualize_policy == policy)
                )
                policy_rows[policy].append(
                    run_episode(
                        policy=policy,
                        num_ohts=num_ohts,
                        max_steps=max_steps,
                        map_config=map_config,
                        reward_config=reward_config,
                        tasks=tasks,
                        algo=algo,
                        visualize=should_visualize,
                        visualize_fps=visualize_fps,
                        keep_visualization_open=False,
                    )
                )

        for policy, rows in policy_rows.items():
            summary = summarize(policy, num_ohts, episodes, hot_lot_batch_size, rows)
            summaries.append(summary)
            append_csv_row(csv_path, summary)

    print_table(summaries)
    print_recommendation(summaries)
    print(f"Saved density sweep CSV: {csv_path}")
    if algo:
        algo.stop()
        ray.shutdown()


if __name__ == "__main__":
    main()
