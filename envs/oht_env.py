import random
import numpy as np
import networkx as nx
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
from envs.grid_map import create_fab_graph
from utils.scenario_scheduler import ScenarioScheduler, create_default_scenario

class OHTFabEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "oht_fab_v1"}

    def __init__(self, num_ohts=2, max_steps=200, scheduler=None, **map_kwargs):
        # 수백 x 수백 스케일 지원 (기본 300x200)
        if not map_kwargs:
            map_kwargs = {"width": 300, "height": 200, "bay_interval": 30, "bay_depth": 20}
            
        self.graph = create_fab_graph(layout_type="mega", **map_kwargs)
        self.num_ohts = num_ohts
        self.max_steps = max_steps
        self.current_step = 0
        
        # [MES 연동] 시나리오 스케줄러 설정
        self.scheduler = scheduler
        if self.scheduler is None:
            ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
            tasks = create_default_scenario(ports, num_tasks=max_steps * num_ohts)
            self.scheduler = ScenarioScheduler(tasks)

        # [PettingZoo 규격] 에이전트 이름 리스트
        self.possible_agents = [f"oht_{i}" for i in range(num_ohts)]
        self.agents = self.possible_agents[:]

        # [PettingZoo 규격] 행동 및 관측 공간 정의
        # 행동: 0(정지), 1(1번 분기), 2(2번 분기) -> Discrete(3)
        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        # 관측: 9차원 1D 벡터 (0~1 사이 값)
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.agent_positions = {}
        self.agent_targets = {}
        self.agent_priorities = {} # Hot Lot 여부 (0: 일반, 1: Hot Lot)
        self.cumulative_rewards = {}
        self.stall_counters = {} # 데드락 감지용 카운터
        self.delivery_count = 0
        self.collision_count = 0
        self.invalid_action_count = 0
        
        # [State Transition] 상태 및 타이머 변수 추가
        self.agent_states = {}   # 0: 이동 중(MOVING), 1: 상하차 중(LOADING)
        self.loading_timers = {} # 상하차 남은 스텝 수
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self.current_step = 0
        self.delivery_count = 0
        self.collision_count = 0
        self.invalid_action_count = 0
        
        # 스케줄러 초기화
        ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
        tasks = create_default_scenario(ports, num_tasks=self.max_steps * self.num_ohts)
        self.scheduler.reset(tasks)

        for i, agent in enumerate(self.agents):
            task = self.scheduler.get_next_task(agent)
            if task:
                self.agent_positions[agent] = task.start_node
                self.agent_targets[agent] = task.goal_node
                self.agent_priorities[agent] = 1 if task.is_hot_lot else 0
            else:
                self.agent_positions[agent] = random.choice(ports)
                self.agent_targets[agent] = self.agent_positions[agent]
                self.agent_priorities[agent] = 0

            self.cumulative_rewards[agent] = 0.0
            self.stall_counters[agent] = 0
            
            self.agent_states[agent] = 0 # 모두 '이동 중' 상태로 시작
            self.loading_timers[agent] = 0
            
        # PettingZoo reset 반환 규격: (obs, infos)
        return self._get_obs(), {agent: {} for agent in self.agents}

    def _get_obs(self):
        return {agent: self._compute_single_obs(agent) for agent in self.agents}

    def _compute_single_obs(self, agent):
        """9차원 관측 벡터 추출 (Hot Lot 인지 기능 추가)"""
        curr_node = self.agent_positions[agent]
        target_node = self.agent_targets[agent]
        
        # 1. 목적지 거리 정규화
        try:
            dist = nx.shortest_path_length(self.graph, curr_node, target_node)
        except nx.NetworkXNoPath:
            dist = 100
        norm_dist = min(dist / 100.0, 1.0) # 고정값으로 정규화 (맵 확장 대비)
        
        # 2. 나의 로딩 상태
        my_state = float(self.agent_states[agent])
        
        # 3 & 4. 전방 거리 및 앞차 상태
        forward_dist, forward_state = 1.0, 0.0
        try:
            path = nx.shortest_path(self.graph, curr_node, target_node)
            if len(path) > 1:
                next_node = path[1]
                for other_agent in self.agents:
                    if other_agent != agent and self.agent_positions[other_agent] == next_node:
                        forward_dist = 0.0
                        forward_state = float(self.agent_states[other_agent])
                        break
        except nx.NetworkXNoPath:
            pass

        # 5 & 6. 갈림길 정체도 (Spine-and-Bay 구조 기반)
        neighbors = list(self.graph.successors(curr_node))
        branch_a_cong, branch_b_cong = 0.0, 0.0
        if len(neighbors) > 0:
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[0]: branch_a_cong = 1.0
        if len(neighbors) > 1:
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[1]: branch_b_cong = 1.0
                    
        # 7. 나의 우선순위 (Hot Lot 여부)
        my_priority = float(self.agent_priorities[agent])
        
        # 8 & 9. 후방 레이더 (내 뒤에 Hot Lot이 오고 있는가?)
        rear_hot_lot_dist = 1.0
        rear_hot_lot_present = 0.0
        
        # 나를 향해 오는 노드들(Predecessors) 탐색
        preds = list(self.graph.predecessors(curr_node))
        for p in preds:
            for other in self.agents:
                if other != agent and self.agent_positions[other] == p:
                    if self.agent_priorities[other] == 1:
                        rear_hot_lot_dist = 0.0
                        rear_hot_lot_present = 1.0
                        break
        
        return np.array([norm_dist, my_state, forward_dist, forward_state, 
                         branch_a_cong, branch_b_cong, my_priority, 
                         rear_hot_lot_dist, rear_hot_lot_present], dtype=np.float32)

    def step(self, action_dict):
        # PettingZoo 최신 step 반환 규격: obs, rewards, terminations, truncations, infos
        self.current_step += 1
        obs, rewards, terminations, truncations, infos = {}, {}, {}, {}, {}
        intended_positions = {}
        prev_positions = self.agent_positions.copy()
        
        # 1. 상태 전이(State Transition) 및 행동 계산
        for agent in self.agents:
            rewards[agent], terminations[agent], truncations[agent], infos[agent] = 0.0, False, False, {}
            
            # [State 1: LOADING 중일 때]
            if self.agent_states[agent] == 1:
                self.loading_timers[agent] -= 1
                intended_positions[agent] = self.agent_positions[agent] # 강제 대기
                
                # 로딩이 끝났다면?
                if self.loading_timers[agent] <= 0:
                    self.agent_states[agent] = 0 # 이동 상태로 복귀
                    
                    # [MES 연동] 스케줄러에서 다음 작업 할당
                    task = self.scheduler.get_next_task(agent)
                    if task:
                        self.agent_positions[agent] = task.start_node
                        self.agent_targets[agent] = task.goal_node
                        self.agent_priorities[agent] = 1 if task.is_hot_lot else 0
                    else:
                        # 작업이 없으면 제자리 대기 (목적지를 현재 위치로)
                        self.agent_targets[agent] = self.agent_positions[agent]
                        self.agent_priorities[agent] = 0
                continue # 로딩 중이면 아래의 이동(Action) 로직 생략
                
            # [State 0: MOVING 중일 때] - 들어온 Action 처리
            action = action_dict.get(agent, 0)
            curr_node = self.agent_positions[agent]
            neighbors = list(self.graph.successors(curr_node))
            
            if action == 0:
                intended_positions[agent] = curr_node
                
                # [Yielding Reward] 주변에 Hot Lot이 있는데 양보했다면 큰 보너스
                is_yielding_for_hot_lot = False
                for other in self.agents:
                    if other != agent and self.agent_priorities[other] == 1:
                        # 내 다음 갈 수 있는 노드에 Hot Lot이 있다면
                        if self.agent_positions[other] in neighbors:
                            is_yielding_for_hot_lot = True
                            break
                
                if is_yielding_for_hot_lot:
                    rewards[agent] = 2.0 # 적극적 양보 보상
                else:
                    # 일반 양보 보상 체크
                    others_nearby = False
                    for other in self.agents:
                        if other != agent and self.agent_positions[other] in neighbors:
                            others_nearby = True
                            break
                    rewards[agent] = 0.05 if others_nearby else -0.1
            elif action > 0 and action <= len(neighbors):
                intended_positions[agent] = neighbors[action - 1]
                rewards[agent] = -0.1
            else:
                intended_positions[agent] = curr_node
                rewards[agent] = -1.0 # 에러 페널티
                self.invalid_action_count += 1
                
        # 2. 충돌 감지 (모션 체크)
        pos_counts = {}
        for pos in intended_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        # 3. 이동 승인 및 보상 업데이트
        for agent in self.agents:
            if self.agent_states[agent] == 0: # MOVING 상태에서만 체크
                next_pos = intended_positions[agent]
                if pos_counts[next_pos] == 1:
                    old_pos = prev_positions[agent]
                    old_target = self.agent_targets[agent]

                    # 이동 전 거리
                    try:
                        old_dist = nx.shortest_path_length(self.graph, old_pos, old_target)
                    except nx.NetworkXNoPath:
                        old_dist = 100

                    self.agent_positions[agent] = next_pos

                    # 이동 후 거리
                    try:
                        new_dist = nx.shortest_path_length(self.graph, next_pos, old_target)
                    except nx.NetworkXNoPath:
                        new_dist = 100

                    # 목적지에 가까워졌으면 작은 보상, 멀어졌으면 작은 페널티
                    if new_dist < old_dist:
                        rewards[agent] += 0.2
                    elif new_dist > old_dist:
                        rewards[agent] -= 0.2

                    # 목적지 도착 이벤트 발생!
                    if next_pos == self.agent_targets[agent]:
                        # Hot Lot이면 더 큰 보상
                        delivery_bonus = 50.0 if self.agent_priorities[agent] == 1 else 20.0
                        rewards[agent] += delivery_bonus
                        self.delivery_count += 1
                        self.agent_states[agent] = 1
                        self.loading_timers[agent] = 5
                else:
                    rewards[agent] -= 15.0 # 충돌 페널티 강화
                    self.collision_count += 1

                # 데드락(Stall) 카운터 업데이트
                if self.agent_positions[agent] == prev_positions[agent]:
                    self.stall_counters[agent] += 1
                else:
                    self.stall_counters[agent] = 0

                # 조기 종료(Truncation) 처리: 15스텝 이상 정체 시
                if self.stall_counters[agent] >= 15:
                    truncations[agent] = True
                    rewards[agent] -= 50.0 # 데드락 페널티
                    # 한 에이전트라도 데드락이면 episode 전체를 종료
                    for a in self.agents:
                        truncations[a] = True
            
            self.cumulative_rewards[agent] += rewards[agent]
            obs[agent] = self._compute_single_obs(agent)
            infos[agent].update({
                "delivery_count": self.delivery_count,
                "collision_count": self.collision_count,
                "invalid_action_count": self.invalid_action_count,
                "current_step": self.current_step,
                "stall_count": self.stall_counters[agent],
                "is_hot_lot": self.agent_priorities[agent] # 인터페이스 정의
            })

        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
                
        return obs, rewards, terminations, truncations, infos

    def render(self, step, action_dict=None, rewards=None):
        print(f"\n=== [Step {step}] OHT 팹 모니터링 (Mega-Fab Scale) ===")
        # 맵 크기에 맞게 그리드 동적 계산
        nodes = list(self.graph.nodes())
        max_x = max(n[0] for n in nodes) + 1
        max_y = max(n[1] for n in nodes) + 1

        # 너무 크면 텍스트 렌더링 축소 (중앙 Spine 위주)
        render_width = min(max_x, 40) 
        render_height = min(max_y, 20)

        grid = [[' . ' for _ in range(render_width)] for _ in range(render_height)]
        for (x, y), data in self.graph.nodes(data=True):
            if x < render_width and y < render_height:
                if data.get('is_port'): grid[y][x] = '[P]'
                elif data.get('is_stocker'): grid[y][x] = '(S)'
                else: grid[y][x] = ' + '

        for agent in self.agents:
            x, y = self.agent_positions[agent]
            if x < render_width and y < render_height:
                marker = ' H ' if self.agent_priorities[agent] == 1 else f' {agent[-1]} '
                grid[y][x] = marker

        for row in grid:
            print("".join(row))
        print("-" * 60)

        # 상태 출력
        for agent in self.agents:
            pos = self.agent_positions[agent]
            target = self.agent_targets[agent]
            cum_reward = round(self.cumulative_rewards[agent], 2)
            stall = self.stall_counters[agent]
            priority = "🔥 HOT" if self.agent_priorities[agent] == 1 else "Normal"

            # 로딩 상태 표시
            state_str = "🚀 주행중"
            if self.agent_states[agent] == 1:
                state_str = f"📦 로딩중 ({self.loading_timers[agent]}/5)"
            elif stall > 0:
                state_str = f"⚠️ 정체 ({stall}/15)"

            act_str = f"| 행동: {action_dict[agent]}" if action_dict and agent in action_dict else ""
            rew_str = f"| 획득: {rewards[agent]}" if rewards and agent in rewards else ""

            print(f" 🚛 {agent} [{priority}]: {state_str} | 위치 {pos} ➡️ 목적지 {target} {act_str} {rew_str} | 총점: {cum_reward}")
        print("=" * 60 + "\n")