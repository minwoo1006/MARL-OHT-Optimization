import random
from envs.grid_map import create_fab_graph

class OHTFabEnv:
    def __init__(self, num_ohts=2):
        self.graph = create_fab_graph()
        self.num_ohts = num_ohts
        self.agents = [i for i in range(num_ohts)]
        
        self.agent_positions = {}
        self.agent_targets = {} # 각 OHT의 목적지
    
    def reset(self):
        """OHT들을 무작위 포트에 배치하고 목적지를 할당합니다."""
        ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
        start_nodes = random.sample(ports, self.num_ohts)
        
        for agent in self.agents:
            self.agent_positions[agent] = start_nodes[agent]
            # 출발지와 다른 무작위 포트를 목적지로 설정
            target = random.choice([p for p in ports if p != start_nodes[agent]])
            self.agent_targets[agent] = target
            
        return self._get_obs(), {}

    def _get_obs(self):
        return {agent: self.agent_positions[agent] for agent in self.agents}

    def render(self, step):
        """간단한 텍스트(Console) 시각화"""
        print(f"--- [Step {step}] OHT 현황 ---")
        for agent in self.agents:
            pos = self.agent_positions[agent]
            target = self.agent_targets[agent]
            print(f" 🚛 OHT {agent}: 현재 {pos} ➡️ 목적지 {target}")
        print("-" * 25)

    def step(self, action_dict):
        obs, rewards, dones, info = {}, {}, {'__all__': True}, {}
        intended_positions = {}
        
        # 1. Action Space: 행동 해석 및 희망 위치 계산
        for agent, action in action_dict.items():
            curr_node = self.agent_positions[agent]
            neighbors = list(self.graph.successors(curr_node)) # 연결된 다음 노드들
            
            if action == 0: # 정지
                intended_positions[agent] = curr_node
                rewards[agent] = -0.1 
            elif action > 0 and action <= len(neighbors): # 1번 또는 2번 경로 선택
                intended_positions[agent] = neighbors[action - 1] # 인덱스는 0부터 시작
                rewards[agent] = -0.1
            else: # 없는 길(예: 갈림길이 아닌데 2번 선택)
                intended_positions[agent] = curr_node
                rewards[agent] = -1.0 # 잘못된 명령 페널티
                
        # 2. 충돌 감지 (모션 체크)
        pos_counts = {}
        for pos in intended_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        # 3. 이동 승인 및 보상 함수(Reward Function) 적용
        for agent in self.agents:
            next_pos = intended_positions[agent]
            
            if pos_counts[next_pos] == 1:
                self.agent_positions[agent] = next_pos
                
                # 목적지 도착 체크!
                if next_pos == self.agent_targets[agent]:
                    rewards[agent] += 10.0 # 도착 보상
                    # 실제 프로젝트에서는 여기서 새 목적지를 주거나 대기 상태로 변경
            else:
                # 충돌 발생!
                rewards[agent] = -10.0
                
            obs[agent] = self.agent_positions[agent]
            # 도착하지 않은 OHT가 하나라도 있으면 에피소드는 끝나지 않음
            if self.agent_positions[agent] != self.agent_targets[agent]:
                dones['__all__'] = False
            
        return obs, rewards, dones, info