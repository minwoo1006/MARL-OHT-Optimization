import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv


def test_metrics():
    env = OHTFabEnv(num_ohts=2, max_steps=30)
    obs, info = env.reset()

    for step in range(1, 31):
        action_dict = {
            agent: np.random.randint(0, 3)
            for agent in env.agents
        }

        obs, rewards, terminations, truncations, infos = env.step(action_dict)

        print(f"\nstep={step}")
        print("actions:", action_dict)
        print("rewards:", rewards)
        print("delivery_count:", env.delivery_count)
        print("collision_count:", env.collision_count)
        print("invalid_action_count:", env.invalid_action_count)
        print("infos:", infos)

        if all(truncations.values()) or all(terminations.values()):
            print("\nEpisode finished.")
            break


if __name__ == "__main__":
    test_metrics()