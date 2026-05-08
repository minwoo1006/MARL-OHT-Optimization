import numpy as np
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv

class OHTFabEnv:
    def __init__(self, num_ohts=2):
        # 1. 관측 방식 설정 (전방 분기점을 트리 형태로 예측)
        obs_builder = TreeObsForRailEnv(max_depth=2)

        # 2. Flatland 환경(엔진) 인스턴스화
        # 맵 생성기(rail_generator)를 생략하면 내부적으로 가장 안정적인 기본 맵(sparse_rail)을 알아서 깔아줍니다.
        self.env = RailEnv(
            width=50,   # 기본 맵 생성기가 맵을 그릴 수 있게 20x20으로 공간을 조금 키워줍니다.
            height=50,
            number_of_agents=num_ohts,
            obs_builder_object=obs_builder,
            remove_agents_at_target=False # OHT는 도착해도 사라지면 안 됨
        )

    def reset(self):
        # 팹 환경 초기화
        obs, info = self.env.reset()
        return obs, info

    def step(self, action_dict):
        # 에이전트들의 행동을 환경에 적용
        obs, rewards, dones, info = self.env.step(action_dict)
        return obs, rewards, dones, info