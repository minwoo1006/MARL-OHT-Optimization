"""
Train and compare a small set of RLlib algorithms on the OHT environment.

Default comparison:
- PPO: policy-gradient actor-critic baseline already used by the project.
- APPO: PPO-family asynchronous policy-gradient baseline.
- IMPALA: off-policy actor-critic with V-trace correction.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ray
from ray.rllib.algorithms.appo import APPOConfig
from ray.rllib.algorithms.impala import IMPALAConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

from agents.train_ppo_rllib import (
    DEFAULT_MAP_CONFIG,
    EVAL_CSV_FIELDS,
    append_csv_row,
    configure_ray_storage,
    env_creator,
    evaluate_dijkstra_policy,
    evaluate_ppo_policy,
    evaluate_random_policy,
    parse_int_list,
    policy_mapping_fn,
    print_comparison_table,
    read_map_config,
    read_reward_config,
)
from envs.oht_env import OHTFabEnv


ALGORITHM_CONFIGS = {
    "ppo": PPOConfig,
    "appo": APPOConfig,
    "impala": IMPALAConfig,
}


def parse_algorithms(raw_value):
    if not raw_value:
        return ["ppo", "appo", "impala"]
    algorithms = [item.strip().lower() for item in raw_value.split(",") if item.strip()]
    unknown = [algo for algo in algorithms if algo not in ALGORITHM_CONFIGS]
    if unknown:
        raise ValueError(f"Unknown algorithms: {unknown}. Available: {sorted(ALGORITHM_CONFIGS)}")
    return algorithms


def build_algorithm(algo_name, env_config, obs_space, act_space, num_gpus, train_batch_size, minibatch_size, num_epochs, lr, gamma):
    config = (
        ALGORITHM_CONFIGS[algo_name]()
        .environment(env="oht_fab_env", env_config=env_config)
        .framework("torch")
        .resources(num_gpus=num_gpus)
        .env_runners(num_env_runners=0)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
    )

    if algo_name == "ppo":
        config = config.training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            num_epochs=num_epochs,
            lr=lr,
            gamma=gamma,
        )
    elif algo_name == "appo":
        config = config.training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            lr=lr,
            gamma=gamma,
        )
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    elif algo_name == "impala":
        config = config.training(
            train_batch_size=train_batch_size,
            minibatch_size=minibatch_size,
            lr=lr,
            gamma=gamma,
        )
        config = config.api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
    return config.build_algo()


def stop_algorithm(algo, algo_name):
    try:
        algo.stop()
    except Exception as exc:
        print(f"Warning: {algo_name.upper()} cleanup failed after training/eval: {exc}")


def main():
    storage_path = configure_ray_storage()
    print(f"Ray storage path: {storage_path}")

    algorithms = parse_algorithms(os.environ.get("OHT_ALGORITHMS"))
    stage_name = os.environ.get("OHT_STAGE", "rl_comparison")
    train_num_ohts = int(os.environ.get("OHT_NUM_OHTS", "10"))
    map_config = read_map_config()
    reward_config = read_reward_config()
    hot_lot_probability = float(os.environ.get("OHT_HOT_LOT_PROBABILITY", "0.1"))
    env_config = {
        "num_ohts": train_num_ohts,
        "max_steps": int(os.environ.get("OHT_MAX_STEPS", "200")),
        "reward_config": reward_config,
        "hot_lot_probability": hot_lot_probability,
        **map_config,
    }

    num_gpus = float(os.environ.get("OHT_NUM_GPUS", "0"))
    train_batch_size = int(os.environ.get("OHT_TRAIN_BATCH_SIZE", "800"))
    minibatch_size = int(os.environ.get("OHT_MINIBATCH_SIZE", str(min(128, train_batch_size))))
    num_epochs = int(os.environ.get("OHT_NUM_EPOCHS", "5"))
    lr = float(os.environ.get("OHT_LR", "3e-4"))
    gamma = float(os.environ.get("OHT_GAMMA", "0.99"))
    num_iterations = int(os.environ.get("OHT_NUM_ITERATIONS", "10"))
    final_eval_ohts = parse_int_list(os.environ.get("OHT_FINAL_EVAL_OHTS"), [train_num_ohts])
    final_eval_episodes = int(os.environ.get("OHT_FINAL_EVAL_EPISODES", "5"))

    results_dir = os.path.abspath(os.path.join(os.getcwd(), "results"))
    eval_csv_path = os.path.join(results_dir, "rllib_comparison_eval_log.csv")

    ray.init(ignore_reinit_error=True, include_dashboard=False)
    register_env("oht_fab_env", env_creator)

    temp_env = OHTFabEnv(**env_config)
    obs_space = temp_env.observation_space("oht_0")
    act_space = temp_env.action_space("oht_0")

    all_eval_summaries = []
    for algo_name in algorithms:
        print("=" * 80)
        print(f"Training {algo_name.upper()} | OHTs={train_num_ohts} | iterations={num_iterations}")
        print("=" * 80)

        algo = build_algorithm(
            algo_name,
            env_config,
            obs_space,
            act_space,
            num_gpus,
            train_batch_size,
            minibatch_size,
            num_epochs,
            lr,
            gamma,
        )

        for iteration in range(1, num_iterations + 1):
            result = algo.train()
            env_runners = result.get("env_runners", {})
            episode_return_mean = env_runners.get("episode_return_mean")
            if iteration == 1 or iteration == num_iterations or iteration % 10 == 0:
                print(
                    f"{algo_name.upper()} iter={iteration} | "
                    f"return_mean={episode_return_mean} | "
                    f"episodes={env_runners.get('num_episodes')}"
                )

        for eval_num_ohts in final_eval_ohts:
            summary, _ = evaluate_ppo_policy(
                algo,
                num_ohts=eval_num_ohts,
                max_steps=env_config["max_steps"],
                num_episodes=final_eval_episodes,
                render=False,
                map_config=map_config,
                reward_config=reward_config,
                hot_lot_probability=hot_lot_probability,
            )
            summary["policy"] = algo_name
            summary["iteration"] = num_iterations
            summary["stage"] = stage_name
            append_csv_row(eval_csv_path, summary, EVAL_CSV_FIELDS)
            all_eval_summaries.append(summary)

        stop_algorithm(algo, algo_name)

    baseline_summaries = []
    for eval_num_ohts in final_eval_ohts:
        random_summary, _ = evaluate_random_policy(
            num_ohts=eval_num_ohts,
            max_steps=env_config["max_steps"],
            num_episodes=final_eval_episodes,
            map_config=map_config,
            reward_config=reward_config,
            hot_lot_probability=hot_lot_probability,
        )
        dijkstra_summary, _ = evaluate_dijkstra_policy(
            num_ohts=eval_num_ohts,
            max_steps=env_config["max_steps"],
            num_episodes=final_eval_episodes,
            map_config=map_config,
            reward_config=reward_config,
            hot_lot_probability=hot_lot_probability,
        )
        for summary in (random_summary, dijkstra_summary):
            summary["iteration"] = num_iterations
            summary["stage"] = stage_name
            append_csv_row(eval_csv_path, summary, EVAL_CSV_FIELDS)
        baseline_summaries.extend([random_summary, dijkstra_summary])

    print_comparison_table(
        baseline_summaries + all_eval_summaries,
        title=f"RLlib Algorithm Comparison: {', '.join(algo.upper() for algo in algorithms)}",
    )
    print(f"Saved comparison CSV: {eval_csv_path}")

    ray.shutdown()


if __name__ == "__main__":
    main()
