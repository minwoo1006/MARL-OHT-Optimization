import networkx as nx
from collections import defaultdict
import time
import random
import numpy as np


# ──────────────────────────────────────────────
# 핵심 베이스라인 에이전트 클래스
# ──────────────────────────────────────────────

class DijkstraBaselineAgent:
    def __init__(self, graph: nx.DiGraph):
        """graph: grid_map.py 가 생성한 NetworkX 방향 그래프 (레일 네트워크)"""
        self.graph = graph
        # 에이전트별 경로 캐시: {agent_id: [node, node, ...]}
        self._path_cache: dict[str, list] = {}

    def compute_path(self, agent_id: str, start: int, goal: int) -> list:
        """start → goal 까지 최단 경로를 계산하고 저장"""
        try:
            path = nx.dijkstra_path(self.graph, start, goal, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = [start]

        self._path_cache[agent_id] = path
        return path

    def get_action(self, env, agent_id: str) -> int:
        """
        현재 환경 상태를 기반으로 다음 Action(0, 1, 2)을 반환
        """
        current_node = env.agent_positions[agent_id]
        goal_node = env.agent_targets[agent_id]

        # LOADING 상태이면 무조건 정지(0)
        if env.agent_states[agent_id] == 1:
            return 0

        # 목적지 도착 시 정지
        if current_node == goal_node:
            return 0

        path = self._path_cache.get(agent_id, [])

        # 경로가 없거나 현재 위치가 경로에서 벗어난 경우 재계산
        if not path or current_node not in path or path[-1] != goal_node:
            path = self.compute_path(agent_id, current_node, goal_node)

        # 현재 노드 이후의 다음 노드 찾기
        try:
            idx = path.index(current_node)
            if idx + 1 < len(path):
                next_node = path[idx + 1]
                
                # 다음 노드로 가기 위한 Action 번호 찾기
                neighbors = list(self.graph.successors(current_node))
                if next_node in neighbors:
                    return neighbors.index(next_node) + 1
        except (ValueError, IndexError):
            pass

        return 0  # 이동 불가 또는 경로 끝 → 정지

    def reset_agent(self, agent_id: str):
        """특정 에이전트의 경로 캐시를 초기화합니다."""
        self._path_cache.pop(agent_id, None)

    def reset_all(self):
        """모든 에이전트의 경로 캐시를 초기화합니다."""
        self._path_cache.clear()


# ──────────────────────────────────────────────
# 베이스라인 실행 함수 (환경과 연결)
# ──────────────────────────────────────────────

def run_dijkstra_baseline(env, max_steps: int = 500, verbose: bool = True) -> dict:
    """
    Dijkstra 베이스라인으로 한 에피소드를 실행하고 성능 지표를 반환
    """
    env.reset()
    agent = DijkstraBaselineAgent(env.graph)

    metrics = {
        "total_deliveries": 0,
        "total_steps": 0,
        "total_collisions": 0,
        "deadlock_count": 0,
        "avg_cycle_time": 0.0,
    }

    # 데드락 감지용: 연속으로 같은 노드에 머문 횟수 (LOADING 상태 제외)
    stall_counter = defaultdict(int)
    STALL_THRESHOLD = 10 

    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}
        for agent_id in env.agents:
            actions[agent_id] = agent.get_action(env, agent_id)

        prev_positions = env.agent_positions.copy()
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # ── 지표 수집 ──
        for agent_id in env.agents:
            # 데드락 감지 (MOVING 상태인데 제자리 유지)
            if env.agent_states[agent_id] == 0: # MOVING
                if env.agent_positions[agent_id] == prev_positions[agent_id]:
                    stall_counter[agent_id] += 1
                    if stall_counter[agent_id] == STALL_THRESHOLD:
                        metrics["deadlock_count"] += 1
                        if verbose:
                            print(f"  ⚠️  [Step {step}] {agent_id} 데드락 의심 (노드 {env.agent_positions[agent_id]})")
                else:
                    stall_counter[agent_id] = 0
            else:
                stall_counter[agent_id] = 0

        metrics["total_steps"] += 1
        
        # 마지막 스텝의 info에서 누적 지표 가져오기
        last_info = next(iter(infos.values()))
        metrics["total_deliveries"] = last_info["delivery_count"]
        metrics["total_collisions"] = last_info["collision_count"]

        if verbose and step % 50 == 0:
            print(f"  [Step {step:4d}] 배송: {metrics['total_deliveries']} | 충돌: {metrics['total_collisions']} | 데드락: {metrics['deadlock_count']}")

        if all(terminations.values()) or all(truncations.values()):
            break

    if metrics["total_deliveries"] > 0:
        metrics["avg_cycle_time"] = metrics["total_steps"] / metrics["total_deliveries"]
    else:
        metrics["avg_cycle_time"] = float("inf")

    return metrics

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from envs.oht_env import OHTFabEnv
    
    print("🚀 Dijkstra 베이스라인 성능 측정을 시작합니다...")
    env = OHTFabEnv(num_ohts=5, max_steps=500) # 에이전트 5대로 테스트
    results = run_dijkstra_baseline(env, verbose=True)
    
    print("\n" + "="*40)
    print("📊 최종 성능 리포트 (Dijkstra)")
    print("="*40)
    print(f"총 배송 횟수: {results['total_deliveries']}건")
    print(f"총 충돌 횟수: {results['total_collisions']}회")
    print(f"데드락 발생: {results['deadlock_count']}회")
    print(f"평균 사이클 타임: {results['avg_cycle_time']:.2f} steps/job")
    print("="*40)