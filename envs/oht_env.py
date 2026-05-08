import random
from envs.grid_map import create_fab_graph

class OHTFabEnv:
    def __init__(self, num_ohts=2):
        self.graph = create_fab_graph()
        self.num_ohts = num_ohts
        self.agents = [i for i in range(num_ohts)]
        self.agent_positions = {}
    
    def reset(self):
        """에피소드 시작 시 OHT들을 팹의 포트(Port)에 무작위로 배치합니다."""
        ports = [n for n, d in self.graph.nodes(data=True) if d.get('is_port')]
        # 중복 없이 시작 위치 선정
        start_nodes = random.sample(ports, self.num_ohts)
        
        for agent in self.agents:
            self.agent_positions[agent] = start_nodes[agent]
            
        return self._get_obs(), {}

    def _get_obs(self):
        """각 OHT의 현재 위치(x, y)를 관측값으로 반환합니다."""
        return {agent: self.agent_positions[agent] for agent in self.agents}

    def step(self, action_dict):
        obs, rewards, dones, info = {}, {}, {'__all__': False}, {}
        
        # [Flatland 철학 1단계] 독립 이동 계산 (희망 다음 위치 찾기)
        intended_positions = {}
        for agent, action in action_dict.items():
            curr_node = self.agent_positions[agent]
            neighbors = list(self.graph.successors(curr_node)) # 갈 수 있는 다음 노드들
            
            # Action 1(전진)이고 갈 길이 있다면 첫 번째 경로로 이동한다고 가정
            if action == 1 and len(neighbors) > 0:
                intended_positions[agent] = neighbors[0]
            else:
                # Action 0(정지) 또는 막다른 길이면 제자리
                intended_positions[agent] = curr_node
                
        # [Flatland 철학 2단계] 모션 체크 및 데드락 감지
        # 두 대 이상의 OHT가 같은 칸(Node)으로 가려고 하는지 카운트
        pos_counts = {}
        for pos in intended_positions.values():
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
            
        # [Flatland 철학 3단계] 실제 이동 승인 및 보상(Reward) 부여
        for agent in self.agents:
            next_pos = intended_positions[agent]
            
            # 가려는 곳에 경쟁자가 없으면 이동 승인!
            if pos_counts[next_pos] == 1:
                self.agent_positions[agent] = next_pos
                rewards[agent] = 1 if action == 1 else -0.1 # 전진하면 +1점, 무의미하게 멈추면 -0.1점 페널티
            else:
                # 충돌 발생! 이동을 취소하고 제자리에 강제 대기시킴
                rewards[agent] = -10 # 충돌 페널티 크게 부여
                
            obs[agent] = self.agent_positions[agent]
            dones[agent] = False
            info[agent] = {}
            
        return obs, rewards, dones, info