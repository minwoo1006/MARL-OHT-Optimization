import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv
from agents.dijkstra_baseline import DijkstraBaselineAgent


def run_random_policy(num_ohts=10, max_steps=500):
    env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
    obs, infos = env.reset()

    for step in range(max_steps):
        action_dict = {
            agent: np.random.randint(0, 3)
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        if all(terminations.values()) or all(truncations.values()):
            break

    return {
        "policy": "random",
        "num_ohts": num_ohts,
        "delivery_count": env.delivery_count,
        "collision_count": env.collision_count,
        "invalid_action_count": env.invalid_action_count,
        "current_step": env.current_step,
    }


def run_dijkstra_policy(num_ohts=10, max_steps=500):
    env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
    obs, infos = env.reset()
    dijkstra_agent = DijkstraBaselineAgent(env.graph)

    for step in range(max_steps):
        action_dict = {
            agent_id: dijkstra_agent.get_action(env, agent_id)
            for agent_id in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        if all(terminations.values()) or all(truncations.values()):
            break

    return {
        "policy": "dijkstra",
        "num_ohts": num_ohts,
        "delivery_count": env.delivery_count,
        "collision_count": env.collision_count,
        "invalid_action_count": env.invalid_action_count,
        "current_step": env.current_step,
    }


if __name__ == "__main__":
    print("\n=== Policy Comparison: 5 OHTs ===")
    print(run_random_policy(num_ohts=5, max_steps=300))
    print(run_dijkstra_policy(num_ohts=5, max_steps=300))

    print("\n=== Policy Comparison: 10 OHTs ===")
    print(run_random_policy(num_ohts=10, max_steps=500))
    print(run_dijkstra_policy(num_ohts=10, max_steps=500))