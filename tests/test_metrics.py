import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv
from utils.scenario_scheduler import Task


def test_kpi_metrics_are_reported():
    env = OHTFabEnv(num_ohts=2, max_steps=5)
    obs, info = env.reset()

    action_dict = {agent: 0 for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(action_dict)
    metrics = env.get_episode_metrics()

    assert "throughput" in metrics
    assert "avg_cycle_time" in metrics
    assert "avg_hot_lot_cycle_time" in metrics
    assert "hot_lot_yield_success_rate" in metrics

    for agent_info in infos.values():
        assert "throughput" in agent_info
        assert "avg_cycle_time" in agent_info
        assert "hot_lot_yield_success_rate" in agent_info


def test_initial_oht_positions_are_unique_when_ports_are_available():
    env = OHTFabEnv(num_ohts=10, max_steps=5, width=100, height=60, bay_interval=10, bay_depth=5)
    obs, info = env.reset()

    positions = list(env.agent_positions.values())
    assert len(positions) == len(set(positions))


def test_idle_agent_does_not_generate_repeated_deliveries():
    probe_env = OHTFabEnv(num_ohts=1, max_steps=5)
    start, goal = next(
        (source, target)
        for source in probe_env.port_nodes
        for target in probe_env.port_nodes
        if source != target and probe_env._get_shortest_dist(source, target) < 100
    )
    env = OHTFabEnv(
        num_ohts=1,
        max_steps=200,
        initial_tasks=[Task(start, goal, is_hot_lot=False)],
    )
    obs, info = env.reset()

    delivered = False
    for _ in range(200):
        curr = env.agent_positions["oht_0"]
        target = env.agent_targets["oht_0"]
        next_hop = env._get_next_hop(curr, target)
        neighbors = env.successors_cache[curr]
        if next_hop in neighbors:
            action = neighbors.index(next_hop) + 1
        else:
            action = 0
        obs, rewards, terminations, truncations, infos = env.step({"oht_0": action})
        if env.delivery_count == 1:
            delivered = True
        if delivered and env.agent_states["oht_0"] == 0 and "oht_0" not in env.task_start_steps:
            for _ in range(5):
                obs, rewards, terminations, truncations, infos = env.step({"oht_0": 0})
            break

    assert env.delivery_count == 1


def test_metrics():
    env = OHTFabEnv(num_ohts=2, max_steps=30)
    obs, info = env.reset()

    for step in range(1, 31):
        action_dict = {
            agent: np.random.randint(0, 3)
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        print(f"\nstep={step}")
        print("actions:", action_dict)
        print("rewards:", rewards)
        print("delivery_count:", env.delivery_count)
        print("collision_count:", env.collision_count)
        print("invalid_action_count:", env.invalid_action_count)
        print("infos:", infos)

        if all(truncations.values()) or all(terminations.values()):
            print("\nEpisode finished.")
            break


if __name__ == "__main__":
    test_metrics()
