'''
시각화 추가한 버전. 신욱이가 만든 파일은 유지하고 새롭게 만들었어요
'''

import os
import sys
import csv
import json
import numpy as np
import torch
from ray.rllib.core.columns import Columns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
import ray.tune.trainable.trainable as tune_trainable
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from envs.oht_env import DEFAULT_REWARD_CONFIG, OHTFabEnv
from utils.wandb_logger import WandBLogger  # ✅ [추가] W&B 로거 임포트

DEFAULT_MAP_CONFIG = {
    "width": 100,
    "height": 60,
    "bay_interval": 10,
    "bay_depth": 5,
}


def moving_average(values, window):
    if not values:
        return None
    return float(np.mean(values[-window:]))


def parse_int_list(value, default):
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def read_reward_config():
    reward_config = {}
    for key, default_value in DEFAULT_REWARD_CONFIG.items():
        env_key = f"OHT_{key.upper()}"
        raw_value = os.environ.get(env_key)
        if raw_value is None:
            reward_config[key] = default_value
        elif isinstance(default_value, int):
            reward_config[key] = int(raw_value)
        else:
            reward_config[key] = float(raw_value)
    return reward_config


def read_map_config():
    return {
        "width": int(os.environ.get("OHT_MAP_WIDTH", DEFAULT_MAP_CONFIG["width"])),
        "height": int(os.environ.get("OHT_MAP_HEIGHT", DEFAULT_MAP_CONFIG["height"])),
        "bay_interval": int(os.environ.get("OHT_BAY_INTERVAL", DEFAULT_MAP_CONFIG["bay_interval"])),
        "bay_depth": int(os.environ.get("OHT_BAY_DEPTH", DEFAULT_MAP_CONFIG["bay_depth"])),
    }


def summarize_episode_results(policy, num_ohts, num_episodes, episode_results):
    return {
        "policy": policy,
        "num_ohts": num_ohts,
        "num_episodes": num_episodes,
        "avg_delivery_count": np.mean([r["delivery_count"] for r in episode_results]),
        "avg_hot_lot_delivery_count": np.mean([r["hot_lot_delivery_count"] for r in episode_results]),
        "avg_hot_lot_assigned_count": np.mean([r["hot_lot_assigned_count"] for r in episode_results]),
        "avg_hot_lot_completion_rate": np.mean([r["hot_lot_completion_rate"] for r in episode_results]),
        "avg_throughput": np.mean([r["throughput"] for r in episode_results]),
        "avg_cycle_time": np.mean([r["avg_cycle_time"] for r in episode_results]),
        "avg_hot_lot_cycle_time": np.mean([r["avg_hot_lot_cycle_time"] for r in episode_results]),
        "avg_hot_lot_yield_success_rate": np.mean([r["hot_lot_yield_success_rate"] for r in episode_results]),
        "avg_hot_lot_yield_opportunities": np.mean([r["hot_lot_yield_opportunities"] for r in episode_results]),
        "avg_collision_count": np.mean([r["collision_count"] for r in episode_results]),
        "avg_invalid_action_count": np.mean([r["invalid_action_count"] for r in episode_results]),
        "avg_current_step": np.mean([r["current_step"] for r in episode_results]),
        "avg_episode_return": np.mean([r["episode_return"] for r in episode_results]),
    }


def compute_eval_score(summary):
    """Safety-first score for choosing checkpoints from periodic evaluation."""
    return (
        summary.get("avg_delivery_count", 0.0) * 100.0
        + summary.get("avg_hot_lot_delivery_count", 0.0) * 180.0
        + summary.get("avg_hot_lot_completion_rate", 0.0) * 500.0
        + summary.get("avg_throughput", 0.0) * 1000.0
        - summary.get("avg_collision_count", 0.0) * 500.0
        - summary.get("avg_invalid_action_count", 0.0) * 1000.0
        - summary.get("avg_cycle_time", 0.0) * 0.5
    )


def configure_ray_storage():
    storage_path = os.path.abspath(os.path.join(os.getcwd(), "ray_results"))
    os.makedirs(storage_path, exist_ok=True)
    tune_trainable.DEFAULT_STORAGE_PATH = storage_path
    return storage_path


def append_csv_row(path, row, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    if file_exists:
        with open(path, "r", encoding="utf-8") as csv_file:
            existing_header = csv_file.readline().strip().split(",")
        if existing_header != fieldnames:
            file_exists = False

    mode = "a" if file_exists else "w"
    with open(path, mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({field: row.get(field) for field in fieldnames})


EVAL_CSV_FIELDS = [
    "iteration",
    "stage",
    "policy",
    "num_ohts",
    "num_episodes",
    "avg_delivery_count",
    "avg_hot_lot_delivery_count",
    "avg_hot_lot_assigned_count",
    "avg_hot_lot_completion_rate",
    "avg_throughput",
    "avg_cycle_time",
    "avg_hot_lot_cycle_time",
    "avg_hot_lot_yield_success_rate",
    "avg_hot_lot_yield_opportunities",
    "avg_collision_count",
    "avg_invalid_action_count",
    "avg_current_step",
    "avg_episode_return",
]


def env_creator(config):
    """
    RLlib이 호출할 환경 생성 함수.
    OHTFabEnv를 PettingZoo ParallelEnv wrapper로 감싼다.
    """
    num_ohts = config.get("num_ohts", 5)
    max_steps = config.get("max_steps", 200)
    map_config = {key: config.get(key, value) for key, value in DEFAULT_MAP_CONFIG.items()}
    reward_config = config.get("reward_config", DEFAULT_REWARD_CONFIG)
    hot_lot_probability = config.get("hot_lot_probability", 0.1)
    raw_env = OHTFabEnv(
        num_ohts=num_ohts,
        max_steps=max_steps,
        reward_config=reward_config,
        hot_lot_probability=hot_lot_probability,
        **map_config,
    )
    return ParallelPettingZooEnv(raw_env)


def policy_mapping_fn(agent_id, *args, **kwargs):
    """
    모든 OHT가 같은 정책을 공유한다.
    즉, oht_0, oht_1, ... 모두 shared_policy 사용.
    """
    return "shared_policy"


def safe_compute_action(algo, obs, env=None, agent_id=None):
    """
    Ray RLlib new API stack 기준으로 학습된 RLModule에서 action을 계산한다.
    env와 agent_id가 주어지면 현재 노드에서 가능한 action만 선택하도록 masking한다.
    """
    valid_actions = None
    if env is not None and agent_id is not None:
        curr_node = env.agent_positions[agent_id]
        neighbors = env.successors_cache.get(
            curr_node,
            list(env.graph.successors(curr_node)),
        )
        valid_actions = [0]
        if len(neighbors) >= 1:
            valid_actions.append(1)
        if len(neighbors) >= 2:
            valid_actions.append(2)

    try:
        module = algo.get_module("shared_policy")
    except Exception:
        action = algo.compute_single_action(
            obs,
            policy_id="shared_policy",
            explore=False,
        )
        if isinstance(action, tuple):
            action = action[0]
        action = int(np.asarray(action).reshape(-1)[0])
        if valid_actions is not None and action not in valid_actions:
            return 0
        return action

    try:
        device = next(module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    obs_tensor = torch.as_tensor(
        np.asarray(obs, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    ).unsqueeze(0)

    with torch.no_grad():
        output = module.forward_inference({"obs": obs_tensor})

    if Columns.ACTION_DIST_INPUTS in output:
        logits = output[Columns.ACTION_DIST_INPUTS].clone()
    elif "action_dist_inputs" in output:
        logits = output["action_dist_inputs"].clone()
    elif Columns.ACTIONS in output:
        action_tensor = output[Columns.ACTIONS]
        return int(action_tensor.detach().cpu().numpy().reshape(-1)[0])
    elif "actions" in output:
        action_tensor = output["actions"]
        return int(action_tensor.detach().cpu().numpy().reshape(-1)[0])
    else:
        raise RuntimeError(
            f"Cannot find action output from RLModule. Output keys: {list(output.keys())}"
        )

    if valid_actions is not None:
        mask = torch.full_like(logits, -1e9)
        mask[:, valid_actions] = 0.0
        logits = logits + mask

    action_tensor = torch.argmax(logits, dim=-1)
    return int(action_tensor.detach().cpu().numpy().reshape(-1)[0])


def evaluate_ppo_policy(
    algo,
    num_ohts=5,
    max_steps=200,
    num_episodes=5,
    render=False,
    map_config=None,
    reward_config=None,
    hot_lot_probability=0.1,
):
    episode_results = []
    map_config = map_config or DEFAULT_MAP_CONFIG
    reward_config = reward_config or DEFAULT_REWARD_CONFIG

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(
            num_ohts=num_ohts,
            max_steps=max_steps,
            reward_config=reward_config,
            hot_lot_probability=hot_lot_probability,
            **map_config,
        )
        obs, infos = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            action_dict = {}
            for agent_id in env.agents:
                action_dict[agent_id] = safe_compute_action(
                    algo, obs[agent_id], env=env, agent_id=agent_id,
                )
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            episode_return += sum(rewards.values())
            if render:
                env.render(step=step + 1, action_dict=action_dict, rewards=rewards)
            if all(terminations.values()) or all(truncations.values()):
                break

        episode_results.append({
            "episode": episode_idx + 1,
            "policy": "ppo",
            "num_ohts": num_ohts,
            "delivery_count": env.delivery_count,
            "hot_lot_delivery_count": env.hot_lot_delivery_count,
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
            **env.get_episode_metrics(),
        })

    summary = summarize_episode_results("ppo", num_ohts, num_episodes, episode_results)
    return summary, episode_results


def evaluate_random_policy(
    num_ohts=5,
    max_steps=200,
    num_episodes=5,
    map_config=None,
    reward_config=None,
    hot_lot_probability=0.1,
):
    episode_results = []
    map_config = map_config or DEFAULT_MAP_CONFIG
    reward_config = reward_config or DEFAULT_REWARD_CONFIG

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(
            num_ohts=num_ohts,
            max_steps=max_steps,
            reward_config=reward_config,
            hot_lot_probability=hot_lot_probability,
            **map_config,
        )
        obs, infos = env.reset()
        episode_return = 0.0

        for step in range(max_steps):
            action_dict = {
                agent_id: env.action_space(agent_id).sample()
                for agent_id in env.agents
            }
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            episode_return += sum(rewards.values())
            if all(terminations.values()) or all(truncations.values()):
                break

        episode_results.append({
            "episode": episode_idx + 1,
            "policy": "random",
            "num_ohts": num_ohts,
            "delivery_count": env.delivery_count,
            "hot_lot_delivery_count": env.hot_lot_delivery_count,
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
            **env.get_episode_metrics(),
        })

    summary = summarize_episode_results("random", num_ohts, num_episodes, episode_results)
    return summary, episode_results


from agents.dijkstra_baseline import DijkstraBaselineAgent


def evaluate_dijkstra_policy(
    num_ohts=5,
    max_steps=200,
    num_episodes=5,
    map_config=None,
    reward_config=None,
    hot_lot_probability=0.1,
):
    episode_results = []
    map_config = map_config or DEFAULT_MAP_CONFIG
    reward_config = reward_config or DEFAULT_REWARD_CONFIG

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(
            num_ohts=num_ohts,
            max_steps=max_steps,
            reward_config=reward_config,
            hot_lot_probability=hot_lot_probability,
            **map_config,
        )
        obs, infos = env.reset()
        dijkstra_agent = DijkstraBaselineAgent(env.graph)
        episode_return = 0.0

        for step in range(max_steps):
            action_dict = {
                agent_id: dijkstra_agent.get_action(env, agent_id)
                for agent_id in env.agents
            }
            obs, rewards, terminations, truncations, infos = env.step(action_dict)
            episode_return += sum(rewards.values())
            if all(terminations.values()) or all(truncations.values()):
                break

        episode_results.append({
            "episode": episode_idx + 1,
            "policy": "dijkstra",
            "num_ohts": num_ohts,
            "delivery_count": env.delivery_count,
            "hot_lot_delivery_count": env.hot_lot_delivery_count,
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
            **env.get_episode_metrics(),
        })

    summary = summarize_episode_results("dijkstra", num_ohts, num_episodes, episode_results)
    return summary, episode_results


def print_comparison_table(summaries, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"{'Policy':<12} {'OHTs':<6} {'Delivery':<12} "
        f"{'HotDone':<10} {'HotAsg':<8} {'HotCR':<8} {'TP':<8} {'Cycle':<8} {'Yield':<8} "
        f"{'Collision':<12} {'Invalid':<12} {'Steps':<10} {'Return':<12}"
    )
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['policy']:<12} "
            f"{s['num_ohts']:<6} "
            f"{s['avg_delivery_count']:<12.2f} "
            f"{s.get('avg_hot_lot_delivery_count', 0.0):<10.2f} "
            f"{s.get('avg_hot_lot_assigned_count', 0.0):<8.2f} "
            f"{s.get('avg_hot_lot_completion_rate', 0.0):<8.2f} "
            f"{s.get('avg_throughput', 0.0):<8.3f} "
            f"{s.get('avg_cycle_time', 0.0):<8.2f} "
            f"{s.get('avg_hot_lot_yield_success_rate', 0.0):<8.2f} "
            f"{s['avg_collision_count']:<12.2f} "
            f"{s['avg_invalid_action_count']:<12.2f} "
            f"{s['avg_current_step']:<10.2f} "
            f"{s['avg_episode_return']:<12.2f}"
        )


def main():
    storage_path = configure_ray_storage()
    print(f"Ray storage path: {storage_path}")

    stage_name = os.environ.get("OHT_STAGE", "single")
    train_num_ohts = int(os.environ.get("OHT_NUM_OHTS", "5"))
    map_config = read_map_config()
    reward_config = read_reward_config()
    hot_lot_probability = float(os.environ.get("OHT_HOT_LOT_PROBABILITY", "0.1"))
    env_config = {
        "num_ohts": train_num_ohts,
        "max_steps": 200,
        "reward_config": reward_config,
        "hot_lot_probability": hot_lot_probability,
        **map_config,
    }
    results_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
    train_csv_path = os.path.join(results_dir, "ppo_training_log.csv")
    eval_csv_path = os.path.join(results_dir, "ppo_eval_log.csv")
    num_gpus = float(os.environ.get("OHT_NUM_GPUS", "1"))
    train_batch_size = int(os.environ.get("OHT_TRAIN_BATCH_SIZE", "1000"))
    minibatch_size = int(os.environ.get("OHT_MINIBATCH_SIZE", str(min(128, train_batch_size))))
    num_epochs = int(os.environ.get("OHT_NUM_EPOCHS", "10"))
    lr = float(os.environ.get("OHT_LR", "3e-4"))
    gamma = float(os.environ.get("OHT_GAMMA", "0.99"))
    num_iterations = int(os.environ.get("OHT_NUM_ITERATIONS", "100"))
    eval_interval = int(os.environ.get("OHT_EVAL_INTERVAL", "25"))
    eval_episodes = int(os.environ.get("OHT_EVAL_EPISODES", "5"))
    eval_ohts = parse_int_list(os.environ.get("OHT_EVAL_OHTS"), [10])
    final_eval_ohts = parse_int_list(os.environ.get("OHT_FINAL_EVAL_OHTS"), [2, 5, 10])
    final_eval_episodes = int(os.environ.get("OHT_FINAL_EVAL_EPISODES", "20"))
    checkpoint_in = os.environ.get("OHT_CHECKPOINT_IN")
    checkpoint_out = os.environ.get("OHT_CHECKPOINT_OUT")
    best_checkpoint_dir = os.environ.get("OHT_BEST_CHECKPOINT_DIR")
    if checkpoint_in:
        checkpoint_in = os.path.abspath(checkpoint_in)
    if checkpoint_out:
        checkpoint_out = os.path.abspath(checkpoint_out)
    if best_checkpoint_dir:
        best_checkpoint_dir = os.path.abspath(best_checkpoint_dir)
    skip_eval = os.environ.get("OHT_SKIP_EVAL", "0") == "1"
    best_eval_score = None
    best_eval_summary = None

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env("oht_fab_env", env_creator)

    temp_env = OHTFabEnv(**env_config)
    obs_space = temp_env.observation_space("oht_0")
    act_space = temp_env.action_space("oht_0")

    config = (
        PPOConfig()
        .environment(
            env="oht_fab_env",
            env_config=env_config,
        )
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=0)
        .training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
        )
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    algo = config.build_algo()
    if checkpoint_in:
        print(f"Restoring PPO checkpoint: {checkpoint_in}")
        algo.restore(checkpoint_in)

    # ✅ [추가] W&B 로거 초기화
    logger = WandBLogger(
        project  = "MARL-OHT-Optimization",
        run_name = f"ppo_{stage_name}_{train_num_ohts}ohts_lr{lr:g}_batch{train_batch_size}"
    )
    logger.init(config={
        "num_ohts":       train_num_ohts,
        "stage":          stage_name,
        "max_steps":      200,
        "map_config":     map_config,
        "reward_config":  reward_config,
        "hot_lot_probability": hot_lot_probability,
        "lr":             lr,
        "train_batch":    train_batch_size,
        "minibatch_size": minibatch_size,
        "num_epochs":     num_epochs,
        "gamma":          gamma,
        "num_iterations": num_iterations,
        "eval_interval":  eval_interval,
        "eval_episodes":  eval_episodes,
        "eval_ohts":      eval_ohts,
        "final_eval_ohts": final_eval_ohts,
        "final_eval_episodes": final_eval_episodes,
        "best_checkpoint_dir": best_checkpoint_dir,
    })

    # =========================
    # 1. PPO 학습
    # =========================
    log_interval = 10
    return_history = []

    for i in range(num_iterations):
        result = algo.train()

        # ✅ [추가] 매 iteration W&B에 기록
        logger.log_train(i, result)

        env_runners = result.get("env_runners", {})
        episode_return_mean = env_runners.get("episode_return_mean")
        episode_return_min  = env_runners.get("episode_return_min")
        episode_return_max  = env_runners.get("episode_return_max")
        episode_len_mean    = env_runners.get("episode_len_mean")
        num_episodes        = env_runners.get("num_episodes")
        num_env_steps       = env_runners.get("num_env_steps_sampled_lifetime")
        num_module_steps    = env_runners.get("num_module_steps_sampled_lifetime")
        if episode_return_mean is not None:
            return_history.append(episode_return_mean)

        episode_return_ma10 = moving_average(return_history, 10)
        episode_return_ma50 = moving_average(return_history, 50)
        train_row = {
            "iteration": i + 1,
            "stage": stage_name,
            "episode_return_mean": episode_return_mean,
            "episode_return_ma10": episode_return_ma10,
            "episode_return_ma50": episode_return_ma50,
            "episode_return_min": episode_return_min,
            "episode_return_max": episode_return_max,
            "episode_len_mean": episode_len_mean,
            "num_episodes": num_episodes,
            "env_steps_total": num_env_steps,
            "module_steps_total": json.dumps(num_module_steps, ensure_ascii=False),
        }
        append_csv_row(
            train_csv_path,
            train_row,
            [
                "iteration",
                "stage",
                "episode_return_mean",
                "episode_return_ma10",
                "episode_return_ma50",
                "episode_return_min",
                "episode_return_max",
                "episode_len_mean",
                "num_episodes",
                "env_steps_total",
                "module_steps_total",
            ],
        )
        logger.log_metrics(
            {
                "train/episode_return_ma10": episode_return_ma10,
                "train/episode_return_ma50": episode_return_ma50,
            },
            step=i + 1,
        )

        if (i + 1) % log_interval == 0 or i == 0:
            print("=" * 60)
            print(f"Iteration {i + 1}")
            print(f"episode_return_mean : {episode_return_mean}")
            print(f"episode_return_ma10 : {episode_return_ma10}")
            print(f"episode_return_ma50 : {episode_return_ma50}")
            print(f"episode_return_min  : {episode_return_min}")
            print(f"episode_return_max  : {episode_return_max}")
            print(f"episode_len_mean    : {episode_len_mean}")
            print(f"num_episodes        : {num_episodes}")
            print(f"env_steps_total     : {num_env_steps}")
            print(f"module_steps_total  : {num_module_steps}")

        if eval_interval > 0 and (i + 1) % eval_interval == 0:
            print(f"\nRunning periodic PPO evaluation at iteration {i + 1}...")
            for eval_num_ohts in eval_ohts:
                eval_summary, _ = evaluate_ppo_policy(
                    algo,
                    num_ohts=eval_num_ohts,
                    max_steps=env_config["max_steps"],
                    num_episodes=eval_episodes,
                    render=False,
                    map_config=map_config,
                    reward_config=reward_config,
                    hot_lot_probability=hot_lot_probability,
                )
                eval_summary["iteration"] = i + 1
                eval_summary["stage"] = stage_name
                append_csv_row(
                    eval_csv_path,
                    eval_summary,
                    EVAL_CSV_FIELDS,
                )
                print_comparison_table(
                    [eval_summary],
                    title=f"Periodic PPO Evaluation: {eval_num_ohts} OHTs @ Iter {i + 1}",
                )
                eval_score = compute_eval_score(eval_summary)
                eval_summary["eval_score"] = eval_score
                print(f"eval_score        : {eval_score:.2f}")
                if best_checkpoint_dir and (
                    best_eval_score is None or eval_score > best_eval_score
                ):
                    best_eval_score = eval_score
                    best_eval_summary = eval_summary.copy()
                    checkpoint_result = algo.save(best_checkpoint_dir)
                    checkpoint_path = getattr(getattr(checkpoint_result, "checkpoint", None), "path", checkpoint_result)
                    print(f"Saved best PPO checkpoint: {checkpoint_path}")
                logger.log_eval(eval_summary)

    if skip_eval:
        if checkpoint_out:
            checkpoint_result = algo.save(checkpoint_out)
            checkpoint_path = getattr(getattr(checkpoint_result, "checkpoint", None), "path", checkpoint_result)
            print(f"Saved PPO checkpoint: {checkpoint_path}")
        logger.finish()
        algo.stop()
        ray.shutdown()
        return

    # =========================
    # 2. Random / Dijkstra / PPO 비교 평가
    # =========================
    print("\nRunning policy evaluation...")

    final_summaries = []
    for final_eval_num_ohts in final_eval_ohts:
        random_summary,   _ = evaluate_random_policy(num_ohts=final_eval_num_ohts, max_steps=200, num_episodes=final_eval_episodes, map_config=map_config, reward_config=reward_config, hot_lot_probability=hot_lot_probability)
        dijkstra_summary, _ = evaluate_dijkstra_policy(num_ohts=final_eval_num_ohts, max_steps=200, num_episodes=final_eval_episodes, map_config=map_config, reward_config=reward_config, hot_lot_probability=hot_lot_probability)
        ppo_summary,      _ = evaluate_ppo_policy(algo, num_ohts=final_eval_num_ohts, max_steps=200, num_episodes=final_eval_episodes, render=False, map_config=map_config, reward_config=reward_config, hot_lot_probability=hot_lot_probability)
        group = [random_summary, dijkstra_summary, ppo_summary]
        final_summaries.extend(group)
        print_comparison_table(group, title=f"Policy Comparison: {final_eval_num_ohts} OHTs")

    for summary in final_summaries:
        summary["iteration"] = num_iterations
        summary["stage"] = stage_name
        append_csv_row(eval_csv_path, summary, EVAL_CSV_FIELDS)

    # ✅ [추가] 평가 결과 W&B에 기록 후 종료
    logger.log_eval(*final_summaries)
    if checkpoint_out:
        checkpoint_result = algo.save(checkpoint_out)
        checkpoint_path = getattr(getattr(checkpoint_result, "checkpoint", None), "path", checkpoint_result)
        print(f"Saved PPO checkpoint: {checkpoint_path}")
    if best_eval_summary:
        print("\nBest periodic PPO evaluation checkpoint:")
        print_comparison_table(
            [best_eval_summary],
            title=f"Best Periodic PPO Evaluation: score={best_eval_score:.2f}",
        )
    logger.finish()

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
