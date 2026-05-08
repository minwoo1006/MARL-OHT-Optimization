import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv

def test_random_run():
    env = OHTFabEnv(num_ohts=2) 
    obs, info = env.reset()
    env.render(step=0)

    # 상하차 지연(5초)을 확인하기 위해 스텝을 15번으로 늘려서 테스트
    for step in range(1, 16):
        action_dict = {}
        for agent in env.agents: # 'oht_0', 'oht_1'
            action_dict[agent] = np.random.randint(0, 3) 

        # PettingZoo 최신 규격 반환값 (5개)
        next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
        env.render(step=step, action_dict=action_dict, rewards=rewards)

if __name__ == "__main__":
    test_random_run()