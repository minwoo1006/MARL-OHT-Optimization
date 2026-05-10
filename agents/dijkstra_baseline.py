import networkx as nx
from collections import defaultdict
import time


# ──────────────────────────────────────────────
# 핵심 베이스라인 에이전트 클래스
# ──────────────────────────────────────────────

class DijkstraBaselineAgent:
    def __init__(self, graph: nx.DiGraph): #graph: grid_map.py 가 생성한 NetworkX 방향 그래프 (레일 네트워크)
     
        self.graph = graph
        # 에이전트별 경로 캐시: {agent_id: [node, node, ...]}
        self._path_cache: dict[str, list] = {}

    def compute_path(self, agent_id: str, start: int, goal: int) -> list:
        """
        # start → goal 까지 최단 경로를 계산하고  저장
        """
        try:
            path = nx.dijkstra_path(self.graph, start, goal, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            # 경로가 없으면 제자리 유지
            path = [start]

        self._path_cache[agent_id] = path
        return path

    def get_action(self, agent_id: str, current_node: int, goal: int) -> int:
        """
        현재 위치와 목적지를 받아 다음 이동할 노드를 반환

        경로가 캐시에 없거나 현재 위치가 바뀐 경우 재계산
        """
        # 목적지 도착
        if current_node == goal:
            return current_node

        path = self._path_cache.get(agent_id, [])

        # 경로가 없거나 현재 위치가 경로에서 벗어난 경우 재계산
        if not path or current_node not in path:
            path = self.compute_path(agent_id, current_node, goal)

        # 현재 노드 이후의 다음 노드 반환
        try:
            idx = path.index(current_node)
            if idx + 1 < len(path):
                return path[idx + 1]
        except ValueError:
            pass

        return current_node  # 이동 불가 → 제자리

    def reset_agent(self, agent_id: str):
        """특정 에이전트의 경로 캐시를 초기화합니다 (새 임무 시작 시 호출)."""
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
    observations, infos = env.reset()

    # 환경에서 그래프 가져오기 (OHTFabEnv가 self.graph 를 노출한다고 가정)
    graph = env.graph
    agent = DijkstraBaselineAgent(graph)

    metrics = {
        "total_deliveries": 0,
        "total_steps": 0,
        "deadlock_count": 0,
        "avg_cycle_time": 0.0,
        "elapsed_sec": 0.0,
    }

    # 데드락 감지용: 연속으로 같은 노드에 머문 횟수
    stall_counter = defaultdict(int)
    STALL_THRESHOLD = 5  # 5스텝 연속 제자리 → 데드락으로 간주

    start_time = time.time()

    for step in range(max_steps):
        if not env.agents:
            break

        actions = {}
        for agent_id in env.agents:
            obs = observations[agent_id]

            # obs 구조: (current_node, goal_node, ...) 첫 두 값 사용
            # ※ OHTFabEnv의 실제 observation 구조에 맞게 수정 필요
            current_node = int(obs[0])
            goal_node    = int(obs[1])

            next_node = agent.get_action(agent_id, current_node, goal_node)
            actions[agent_id] = next_node

        observations, rewards, terminations, truncations, infos = env.step(actions)

        # ── 지표 수집 ──
        for agent_id in env.agents:
            obs = observations[agent_id]
            current_node = int(obs[0])
            goal_node    = int(obs[1])

            # 목적지 도달 감지
            if current_node == goal_node:
                metrics["total_deliveries"] += 1
                agent.reset_agent(agent_id)  # 다음 임무를 위해 경로 초기화
                stall_counter[agent_id] = 0

            # 데드락 감지 (제자리 연속)
            prev_node = getattr(env, "_prev_positions", {}).get(agent_id, -1)
            if current_node == prev_node:
                stall_counter[agent_id] += 1
                if stall_counter[agent_id] == STALL_THRESHOLD:
                    metrics["deadlock_count"] += 1
                    if verbose:
                        print(f"  ⚠️  [Step {step}] {agent_id} 데드락 감지 (노드 {current_node})")
            else:
                stall_counter[agent_id] = 0

        metrics["total_steps"] += 1

        if verbose and step % 50 == 0:
            print(f"  [Step {step:4d}] 배송완료: {metrics['total_deliveries']}건 | "
                  f"데드락: {metrics['deadlock_count']}회")

    # ── 최종 지표 계산 ──
    metrics["elapsed_sec"] = time.time() - start_time
    if metrics["total_deliveries"] > 0:
        metrics["avg_cycle_time"] = metrics["total_steps"] / metrics["total_deliveries"]
    else:
        metrics["avg_cycle_time"] = float("inf")

    return metrics