import random
from envs.grid_map import create_fab_graph

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
        return {agent: self.agent_positions[agent] for agent in self.agents}

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