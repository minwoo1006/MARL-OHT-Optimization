'''
시각화 추가한 버전. 신욱이가 만든 파일은 유지하고 새롭게 만들었어요
'''

import os
import sys
import numpy as np
import torch
from ray.rllib.core.columns import Columns
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from envs.oht_env import OHTFabEnv
from utils.wandb_logger import WandBLogger  # ✅ [추가] W&B 로거 임포트


def env_creator(config):
    """
    RLlib이 호출할 환경 생성 함수.
    OHTFabEnv를 PettingZoo ParallelEnv wrapper로 감싼다.
    """
    num_ohts = config.get("num_ohts", 5)
    max_steps = config.get("max_steps", 200)
    raw_env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
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
    module = algo.get_module("shared_policy")

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

    if env is not None and agent_id is not None:
        curr_node = env.agent_positions[agent_id]
        neighbors = list(env.graph.successors(curr_node))
        valid_actions = [0]
        if len(neighbors) >= 1:
            valid_actions.append(1)
        if len(neighbors) >= 2:
            valid_actions.append(2)
        mask = torch.full_like(logits, -1e9)
        mask[:, valid_actions] = 0.0
        logits = logits + mask

    action_tensor = torch.argmax(logits, dim=-1)
    return int(action_tensor.detach().cpu().numpy().reshape(-1)[0])


def evaluate_ppo_policy(algo, num_ohts=5, max_steps=200, num_episodes=5, render=False):
    episode_results = []

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
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
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
        })

    summary = {
        "policy": "ppo",
        "num_ohts": num_ohts,
        "num_episodes": num_episodes,
        "avg_delivery_count": np.mean([r["delivery_count"] for r in episode_results]),
        "avg_collision_count": np.mean([r["collision_count"] for r in episode_results]),
        "avg_invalid_action_count": np.mean([r["invalid_action_count"] for r in episode_results]),
        "avg_current_step": np.mean([r["current_step"] for r in episode_results]),
        "avg_episode_return": np.mean([r["episode_return"] for r in episode_results]),
    }
    return summary, episode_results


def evaluate_random_policy(num_ohts=5, max_steps=200, num_episodes=5):
    episode_results = []

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
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
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
        })

    summary = {
        "policy": "random",
        "num_ohts": num_ohts,
        "num_episodes": num_episodes,
        "avg_delivery_count": np.mean([r["delivery_count"] for r in episode_results]),
        "avg_collision_count": np.mean([r["collision_count"] for r in episode_results]),
        "avg_invalid_action_count": np.mean([r["invalid_action_count"] for r in episode_results]),
        "avg_current_step": np.mean([r["current_step"] for r in episode_results]),
        "avg_episode_return": np.mean([r["episode_return"] for r in episode_results]),
    }
    return summary, episode_results


from agents.dijkstra_baseline import DijkstraBaselineAgent


def evaluate_dijkstra_policy(num_ohts=5, max_steps=200, num_episodes=5):
    episode_results = []

    for episode_idx in range(num_episodes):
        env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
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
            "collision_count": env.collision_count,
            "invalid_action_count": env.invalid_action_count,
            "current_step": env.current_step,
            "episode_return": episode_return,
        })

    summary = {
        "policy": "dijkstra",
        "num_ohts": num_ohts,
        "num_episodes": num_episodes,
        "avg_delivery_count": np.mean([r["delivery_count"] for r in episode_results]),
        "avg_collision_count": np.mean([r["collision_count"] for r in episode_results]),
        "avg_invalid_action_count": np.mean([r["invalid_action_count"] for r in episode_results]),
        "avg_current_step": np.mean([r["current_step"] for r in episode_results]),
        "avg_episode_return": np.mean([r["episode_return"] for r in episode_results]),
    }
    return summary, episode_results


def print_comparison_table(summaries, title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(
        f"{'Policy':<12} {'OHTs':<6} {'Delivery':<12} "
        f"{'Collision':<12} {'Invalid':<12} {'Steps':<10} {'Return':<12}"
    )
    print("-" * 80)
    for s in summaries:
        print(
            f"{s['policy']:<12} "
            f"{s['num_ohts']:<6} "
            f"{s['avg_delivery_count']:<12.2f} "
            f"{s['avg_collision_count']:<12.2f} "
            f"{s['avg_invalid_action_count']:<12.2f} "
            f"{s['avg_current_step']:<10.2f} "
            f"{s['avg_episode_return']:<12.2f}"
        )


def main():
    ray.init(ignore_reinit_error=True)
    register_env("oht_fab_env", env_creator)

    temp_env = OHTFabEnv(num_ohts=5, max_steps=200)
    obs_space = temp_env.observation_space("oht_0")
    act_space = temp_env.action_space("oht_0")

    config = (
        PPOConfig()
        .environment(
            env="oht_fab_env",
            env_config={"num_ohts": 5, "max_steps": 200},
        )
        .framework("torch")
        .resources(num_gpus=1)
        .env_runners(num_env_runners=0)
        .training(train_batch_size=1000, lr=3e-4, gamma=0.99)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    algo = config.build_algo()

    # ✅ [추가] W&B 로거 초기화
    logger = WandBLogger(
        project  = "MARL-OHT-Optimization",
        run_name = "ppo_5ohts_lr3e4_batch1000"
    )
    logger.init(config={
        "num_ohts":       5,
        "max_steps":      200,
        "lr":             3e-4,
        "train_batch":    1000,
        "gamma":          0.99,
        "num_iterations": 1000,
    })

    # =========================
    # 1. PPO 학습
    # =========================
    num_iterations = 1000
    log_interval = 10

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

        if (i + 1) % log_interval == 0 or i == 0:
            print("=" * 60)
            print(f"Iteration {i + 1}")
            print(f"episode_return_mean : {episode_return_mean}")
            print(f"episode_return_min  : {episode_return_min}")
            print(f"episode_return_max  : {episode_return_max}")
            print(f"episode_len_mean    : {episode_len_mean}")
            print(f"num_episodes        : {num_episodes}")
            print(f"env_steps_total     : {num_env_steps}")
            print(f"module_steps_total  : {num_module_steps}")

    # =========================
    # 2. Random / Dijkstra / PPO 비교 평가
    # =========================
    print("\nRunning policy evaluation...")

    random_2_summary,   _ = evaluate_random_policy(num_ohts=2,  max_steps=200, num_episodes=20)
    dijkstra_2_summary, _ = evaluate_dijkstra_policy(num_ohts=2, max_steps=200, num_episodes=20)
    ppo_2_summary,      _ = evaluate_ppo_policy(algo, num_ohts=2, max_steps=200, num_episodes=20, render=False)
    print_comparison_table([random_2_summary, dijkstra_2_summary, ppo_2_summary], title="Policy Comparison: 2 OHTs")

    random_5_summary,   _ = evaluate_random_policy(num_ohts=5,  max_steps=200, num_episodes=20)
    dijkstra_5_summary, _ = evaluate_dijkstra_policy(num_ohts=5, max_steps=200, num_episodes=20)
    ppo_5_summary,      _ = evaluate_ppo_policy(algo, num_ohts=5, max_steps=200, num_episodes=5,  render=False)
    print_comparison_table([random_5_summary, dijkstra_5_summary, ppo_5_summary], title="Policy Comparison: 5 OHTs")

    random_10_summary,   _ = evaluate_random_policy(num_ohts=10, max_steps=200, num_episodes=20)
    dijkstra_10_summary, _ = evaluate_dijkstra_policy(num_ohts=10, max_steps=200, num_episodes=20)
    ppo_10_summary,      _ = evaluate_ppo_policy(algo, num_ohts=10, max_steps=200, num_episodes=20, render=False)
    print_comparison_table([random_10_summary, dijkstra_10_summary, ppo_10_summary], title="Policy Comparison: 10 OHTs")

    # ✅ [추가] 평가 결과 W&B에 기록 후 종료
    logger.log_eval(
        random_2_summary,  dijkstra_2_summary,  ppo_2_summary,
        random_5_summary,  dijkstra_5_summary,  ppo_5_summary,
        random_10_summary, dijkstra_10_summary, ppo_10_summary,
    )
    logger.finish()

    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    main()
