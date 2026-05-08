import random
from envs.grid_map import create_fab_graph
import numpy as np
import networkx as nx

class OHTFabEnv:
    def __init__(self, num_ohts=2):
        self.graph = create_fab_graph()
        self.num_ohts = num_ohts
        self.agents = [i for i in range(num_ohts)]
        
        self.agent_positions = {}
        self.agent_targets = {}
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents} # 누적 보상 추가
    
    def reset(self):
        ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
        start_nodes = random.sample(ports, self.num_ohts)
        
        self.cumulative_rewards = {agent: 0.0 for agent in self.agents} # 초기화
        
        for agent in self.agents:
            self.agent_positions[agent] = start_nodes[agent]
            target = random.choice([p for p in ports if p != start_nodes[agent]])
            self.agent_targets[agent] = target
            
        return self._get_obs(), {}


    def _get_obs(self):
        """모든 OHT의 관측값을 딕셔너리로 반환합니다."""
        obs = {}
        for agent in self.agents:
            obs[agent] = self._compute_single_obs(agent)
        return obs

    def _compute_single_obs(self, agent):
        """
        단일 OHT 에이전트의 시야(Observation)를 7차원 벡터로 계산하여 반환합니다.
        NetworkX 그래프를 활용해 자신, 전방, 후방, 갈림길 상태를 수치화합니다.
        """
        curr_node = self.agent_positions[agent]
        target_node = self.agent_targets[agent]
        
        # 1. Self: 목적지까지의 거리 (정규화)
        try:
            # NetworkX의 다익스트라(Dijkstra) 기반 최단 경로 길이 계산
            dist = nx.shortest_path_length(self.graph, curr_node, target_node)
        except nx.NetworkXNoPath:
            dist = 100 # 길이 끊긴 예외 상황 대비
            
        max_dist = len(self.graph.nodes) # 맵의 전체 노드 수를 최대 거리로 간주
        norm_dist = dist / max_dist # 0.0 ~ 1.0 사이로 정규화
        
        # 2. Self: 내 상태 (현재는 모두 이동 중이므로 0.0)
        # 추후 목적지 도착 시 '상하차(Loading) 상태' 구현을 위한 자리입니다.
        my_state = 0.0
        
        # 3 & 4. Forward: 전방 상황 파악
        forward_dist = 1.0  # 앞이 뻥 뚫려있음 (기본값)
        forward_state = 0.0 # 앞차의 상태
        
        try:
            path = nx.shortest_path(self.graph, curr_node, target_node)
            if len(path) > 1:
                next_node = path[1] # 내가 가야 할 바로 다음 칸
                
                # 다음 칸에 다른 OHT가 존재하는지 스캔
                for other_agent in self.agents:
                    if other_agent != agent and self.agent_positions[other_agent] == next_node:
                        forward_dist = 0.0 # 바로 앞에 장애물(OHT) 있음
                        # 추후 앞차의 my_state(로딩 중인지)를 가져와서 forward_state에 넣을 수 있습니다.
                        break
        except nx.NetworkXNoPath:
            pass

        # 5 & 6. Routing: 갈림길(분기점) 정체도 파악
        neighbors = list(self.graph.successors(curr_node))
        branch_a_cong = 0.0
        branch_b_cong = 0.0
        
        if len(neighbors) > 0: # 1번 길 (직진 또는 분기점 A)
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[0]:
                    branch_a_cong = 1.0 # 해당 길에 OHT 있음
                    
        if len(neighbors) > 1: # 2번 길 (분기점 B)
            for other_agent in self.agents:
                if self.agent_positions[other_agent] == neighbors[1]:
                    branch_b_cong = 1.0 # 해당 길에 OHT 있음
                    
        # 7. Rear: 후방 Hot Lot(긴급 물량) 추격 여부 (3주차 구현을 위한 자리)
        rear_priority = 0.0
        
        # 최종 관측치 배열 생성 (인공지능 모델의 '눈')
        obs_vector = np.array([
            norm_dist,       # 목적지까지의 상대적 거리
            my_state,        # 나의 로딩 상태
            forward_dist,    # 전방 OHT와의 거리
            forward_state,   # 전방 OHT의 상태
            branch_a_cong,   # 1번 경로 정체도
            branch_b_cong,   # 2번 경로 정체도
            rear_priority    # 후방 긴급 물량 유무
        ], dtype=np.float32)
        
        return obs_vector

    def render(self, step, action_dict=None, rewards=None): # 파라미터 추가
        print(f"\n=== [Step {step}] OHT 팹 모니터링 ===")
        
        # 1. 맵 렌더링 (기존과 동일)
        grid = [[' . ' for _ in range(5)] for _ in range(5)]
        for (x, y), data in self.graph.nodes(data=True):
            grid[y][x] = '[P]' if data.get('is_port') else ' + '
        for agent in self.agents:
            x, y = self.agent_positions[agent]
            grid[y][x] = f' {agent} '
            
        for row in grid:
            print("".join(row))
        print("-" * 40)
        
        # 2. 상태, 행동, 보상, 누적 보상 종합 출력
        for agent in self.agents:
            pos = self.agent_positions[agent]
            target = self.agent_targets[agent]
            cum_reward = round(self.cumulative_rewards[agent], 2)
            
            # 이전 스텝의 행동과 보상이 있다면 출력
            act_str = f" | 선택 행동: {action_dict[agent]}" if action_dict else ""
            rew_str = f" | 획득 점수: {rewards[agent]}" if rewards else ""
            
            print(f" 🚛 OHT {agent}: 위치 {pos} ➡️ 목적지 {target}{act_str}{rew_str} | 총점: {cum_reward}")
        print("========================================\n")

    def step(self, action_dict):
        obs, rewards, dones, info = {}, {}, {'__all__': True}, {}
        intended_positions = {}
        
        for agent, action in action_dict.items():
            curr_node = self.agent_positions[agent]
            neighbors = list(self.graph.successors(curr_node))
            
            if action == 0:
                intended_positions[agent] = curr_node
                rewards[agent] = -0.1 
            elif action > 0 and action <= len(neighbors):
                intended_positions[agent] = neighbors[action - 1]
                rewards[agent] = -0.1
            else:
                intended_positions[agent] = curr_node
                rewards[agent] = -1.0
                
        pos_counts = {}
        for pos in intended_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        for agent in self.agents:
            next_pos = intended_positions[agent]
            
            if pos_counts[next_pos] == 1:
                self.agent_positions[agent] = next_pos
                if next_pos == self.agent_targets[agent]:
                    rewards[agent] += 10.0
            else:
                rewards[agent] = -10.0
                
            # 누적 보상 업데이트
            self.cumulative_rewards[agent] += rewards[agent]
            
            obs[agent] = self.agent_positions[agent]
            if self.agent_positions[agent] != self.agent_targets[agent]:
                dones['__all__'] = False
            
        return obs, rewards, dones, info