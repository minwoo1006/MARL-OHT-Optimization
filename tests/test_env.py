import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv

def test_random_run():
    env = OHTFabEnv(num_ohts=2) 
    obs, info = env.reset()
    
    # 시작 상태 렌더링
    env.render(step=0)

    for step in range(1, 600):
        action_dict = {}
        for i in range(2):
            action_dict[i] = np.random.randint(0, 3) 

        next_obs, rewards, dones, info = env.step(action_dict)
        
        # 행동이 적용된 후의 결과 렌더링
        env.render(step=step, action_dict=action_dict, rewards=rewards)

if __name__ == "__main__":
    test_random_run()