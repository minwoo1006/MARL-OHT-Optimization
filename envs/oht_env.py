import random
import numpy as np
import networkx as nx
from pettingzoo.utils.env import ParallelEnv
from gymnasium.spaces import Discrete, Box
from envs.grid_map import create_fab_graph

class OHTFabEnv(ParallelEnv):
    metadata = {'render_modes': ['human'], "name": "oht_fab_v1"}

    def __init__(self, num_ohts=2, max_steps=200):
        self.graph = create_fab_graph()
        self.num_ohts = num_ohts
        self.max_steps = max_steps
        self.current_step = 0
        
        # [PettingZoo 규격] 에이전트 이름 리스트
        self.possible_agents = [f"oht_{i}" for i in range(num_ohts)]
        self.agents = self.possible_agents[:]

        # [PettingZoo 규격] 행동 및 관측 공간 정의
        # 행동: 0(정지), 1(1번 분기), 2(2번 분기) -> Discrete(3)
        self.action_spaces = {agent: Discrete(3) for agent in self.possible_agents}
        # 관측: 7차원 1D 벡터 (0~1 사이 값) -> Box(7)
        self.observation_spaces = {
            agent: Box(low=0.0, high=1.0, shape=(7,), dtype=np.float32) 
            for agent in self.possible_agents
        }

        self.agent_positions = {}
        self.agent_targets = {}
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
        ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
        start_nodes = random.sample(ports, self.num_ohts)
        
        for i, agent in enumerate(self.agents):
            self.agent_positions[agent] = start_nodes[i]
            self.agent_targets[agent] = random.choice([p for p in ports if p != start_nodes[i]])
            self.cumulative_rewards[agent] = 0.0
            self.stall_counters[agent] = 0 # 초기화
            
            self.agent_states[agent] = 0 # 모두 '이동 중' 상태로 시작
            self.loading_timers[agent] = 0
            
        # PettingZoo reset 반환 규격: (obs, infos)
        return self._get_obs(), {agent: {} for agent in self.agents}

    def _get_obs(self):
        return {agent: self._compute_single_obs(agent) for agent in self.agents}

    def _compute_single_obs(self, agent):
        """7차원 관측 벡터 추출 (이전 설계 반영)"""
        curr_node = self.agent_positions[agent]
        target_node = self.agent_targets[agent]
        
        # 1. 목적지 거리 정규화
        try:
            dist = nx.shortest_path_length(self.graph, curr_node, target_node)
        except nx.NetworkXNoPath:
            dist = 100
        norm_dist = min(dist / len(self.graph.nodes), 1.0)
        
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

        # 5 & 6. 갈림길 정체도
        neighbors = list(self.graph.successors(curr_node))
        branch_a_cong, branch_b_cong = 0.0, 0.0
        if len(neighbors) > 0:
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[0]: branch_a_cong = 1.0
        if len(neighbors) > 1:
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[1]: branch_b_cong = 1.0
                    
        # 7. 후방 긴급 물량 유무 (예비)
        rear_priority = 0.0
        
        return np.array([norm_dist, my_state, forward_dist, forward_state, 
                         branch_a_cong, branch_b_cong, rear_priority], dtype=np.float32)

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
                    ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
                    curr_pos = self.agent_positions[agent]
                    # 새로운 목적지 발급
                    self.agent_targets[agent] = random.choice([p for p in ports if p != curr_pos])
                continue # 로딩 중이면 아래의 이동(Action) 로직 생략
                
            # [State 0: MOVING 중일 때] - 들어온 Action 처리
            action = action_dict.get(agent, 0)
            curr_node = self.agent_positions[agent]
            neighbors = list(self.graph.successors(curr_node))
            
            if action == 0:
                intended_positions[agent] = curr_node
                # [Yielding Reward] 주변에 다른 차가 있는데 멈춰줬다면 보너스
                neighbors = list(self.graph.successors(curr_node))
                others_nearby = False
                for other in self.agents:
                    if other != agent and self.agent_positions[other] in neighbors:
                        others_nearby = True
                        break

                if others_nearby:
                    rewards[agent] = 0.05 # 양보 보상
                else:
                    rewards[agent] = -0.1 # 그냥 대기 페널티
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
                    self.agent_positions[agent] = next_pos
                    
                    # 목적지 도착 이벤트 발생!
                    if next_pos == self.agent_targets[agent]:
                        rewards[agent] += 20.0 # 배송 보상 강화
                        self.delivery_count += 1
                        self.agent_states[agent] = 1 # 즉시 LOADING 상태로 전환
                        self.loading_timers[agent] = 5 # 5스텝 딜레이 시작
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
            
            self.cumulative_rewards[agent] += rewards[agent]
            obs[agent] = self._compute_single_obs(agent)
            infos[agent].update({
                "delivery_count": self.delivery_count,
                "collision_count": self.collision_count,
                "invalid_action_count": self.invalid_action_count,
                "current_step": self.current_step,
                "stall_count": self.stall_counters[agent]
            })

        if self.current_step >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
                
        return obs, rewards, terminations, truncations, infos

    def render(self, step, action_dict=None, rewards=None):
        print(f"\n=== [Step {step}] OHT 팹 모니터링 (Spine-and-Bay) ===")
        # 신규 맵 크기에 맞게 그리드 동적 생성 (10x6)
        grid = [[' . ' for _ in range(10)] for _ in range(6)]
        for (x, y), data in self.graph.nodes(data=True):
            grid[y][x] = '[P]' if data.get('is_port') else ' + '
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            grid[y][x] = f' {agent[-1]} ' # "oht_0"에서 숫자만 표기
        for row in grid:
            print("".join(row))
        print("-" * 55)
        
        # 상태 출력
        for agent in self.agents:
            pos = self.agent_positions[agent]
            target = self.agent_targets[agent]
            cum_reward = round(self.cumulative_rewards[agent], 2)
            stall = self.stall_counters[agent]
            
            # 로딩 상태 표시
            state_str = "🚀 주행중"
            if self.agent_states[agent] == 1:
                state_str = f"📦 로딩중 ({self.loading_timers[agent]}/5)"
            elif stall > 0:
                state_str = f"⚠️ 정체 ({stall}/15)"
                
            act_str = f"| 행동: {action_dict[agent]}" if action_dict and agent in action_dict else ""
            rew_str = f"| 획득: {rewards[agent]}" if rewards and agent in rewards else ""
            
            print(f" 🚛 {agent}: {state_str} | 위치 {pos} ➡️ 목적지 {target} {act_str} {rew_str} | 총점: {cum_reward}")
        print("=" * 55 + "\n")