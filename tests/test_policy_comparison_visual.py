import sys
import os
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv
from agents.dijkstra_baseline import DijkstraBaselineAgent
from utils.visualization import OHTVisualizer  # [추가] 시각화 임포트


def run_random_policy(num_ohts=10, max_steps=500, visualize=False):  # [추가] visualize 파라미터
    env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
    obs, infos = env.reset()

    # [추가] 시각화 초기화
    viz = None
    if visualize:
        viz = OHTVisualizer(env, fps=6)
        viz.init()

    for step in range(max_steps):
        action_dict = {
            agent: np.random.randint(0, 3)
            for agent in env.agents
        }

        prev_positions = env.agent_positions.copy()  # [추가] 충돌 감지용 이전 위치 저장
        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        # [추가] 스냅샷 저장 및 렌더링
        if viz:
            collision_nodes = viz.detect_collisions(prev_positions, env)
            viz.push_snapshot(step, infos, collision_nodes)
            if not viz.render():
                break

        if all(terminations.values()) or all(truncations.values()):
            break

    # [추가] 시각화 종료
    if viz:
        viz.close()

    return {
        "policy": "random",
        "num_ohts": num_ohts,
        "delivery_count": env.delivery_count,
        "collision_count": env.collision_count,
        "invalid_action_count": env.invalid_action_count,
        "current_step": env.current_step,
    }


def run_dijkstra_policy(num_ohts=10, max_steps=500, visualize=False):  # ✅ [추가] visualize 파라미터
    env = OHTFabEnv(num_ohts=num_ohts, max_steps=max_steps)
    obs, infos = env.reset()
    dijkstra_agent = DijkstraBaselineAgent(env.graph)

    # [추가] 시각화 초기화
    viz = None
    if visualize:
        viz = OHTVisualizer(env, fps=6)
        viz.init()

    for step in range(max_steps):
        action_dict = {
            agent_id: dijkstra_agent.get_action(env, agent_id)
            for agent_id in env.agents
        }

        prev_positions = env.agent_positions.copy()  # [추가] 충돌 감지용 이전 위치 저장
        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        # [추가] 스냅샷 저장 및 렌더링
        if viz:
            collision_nodes = viz.detect_collisions(prev_positions, env)
            viz.push_snapshot(step, infos, collision_nodes)
            if not viz.render():
                break

        if all(terminations.values()) or all(truncations.values()):
            break

    # [추가] 시각화 종료
    if viz:
        viz.close()

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
    print(run_random_policy(num_ohts=5, max_steps=300, visualize=False))
    print(run_dijkstra_policy(num_ohts=5, max_steps=300, visualize=True))  # [수정] 시각화 켜기

    print("\n=== Policy Comparison: 10 OHTs ===")
    print(run_random_policy(num_ohts=10, max_steps=500, visualize=False))
    print(run_dijkstra_policy(num_ohts=10, max_steps=500, visualize=False))
