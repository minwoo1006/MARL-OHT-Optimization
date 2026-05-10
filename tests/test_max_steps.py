import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv


def test_max_steps():
    env = OHTFabEnv(num_ohts=2, max_steps=5)
    obs, info = env.reset()

    for step in range(1, 10):
        action_dict = {
            agent: np.random.randint(0, 3)
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        print(f"step={step}")
        print("truncations:", truncations)

        if all(truncations.values()):
            print("max_steps reached. Episode truncated.")
            break


if __name__ == "__main__":
    test_max_steps()