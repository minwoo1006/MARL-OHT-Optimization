import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pettingzoo.test import parallel_api_test
from envs.oht_env import OHTFabEnv


def test_parallel_api():
    env = OHTFabEnv(num_ohts=3)
    parallel_api_test(env, num_cycles=100)


if __name__ == "__main__":
    test_parallel_api()
    print("PettingZoo Parallel API test passed.")