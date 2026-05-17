import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv


def test_rllib_env_wrapper():
    raw_env = OHTFabEnv(num_ohts=5, max_steps=100)
    env = ParallelPettingZooEnv(raw_env)

    obs, info = env.reset()

    print("RLlib wrapper reset 성공")
    print("obs type:", type(obs))
    print("agents:", list(obs.keys()))

    action_dict = {}

    for agent_id in obs.keys():
        action_dict[agent_id] = raw_env.action_space(agent_id).sample()

    next_obs, rewards, terminateds, truncateds, infos = env.step(action_dict)

    print("\nRLlib wrapper step 성공")
    print("next_obs keys:", list(next_obs.keys()))
    print("rewards:", rewards)
    print("terminateds:", terminateds)
    print("truncateds:", truncateds)
    print("infos sample:", next(iter(infos.values())))


if __name__ == "__main__":
    test_rllib_env_wrapper()