import numpy as np
import sys
import os
# 상위 폴더의 모듈을 불러오기 위한 경로 설정
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.oht_env import OHTFabEnv

def test_random_run():
    print("=== 🏭 OHT Fab Simulator 초기화 중 ===")
    # OHT 2대로 환경 세팅
    env = OHTFabEnv(num_ohts=2) 
    obs, info = env.reset()
    
    print(f"✅ 초기화 완료! 현재 활성화된 OHT 대수: {len(obs)}")
    print("-" * 40)

    # 5 스텝(시간) 동안 아무렇게나(Random) 주행시켜보기
    print("=== 🚦 랜덤 주행 테스트 시작 ===")
    for step in range(500):
        action_dict = {}
        # 2대의 OHT에게 0~4 사이의 무작위 행동 지시
        # (0: 기존 행동 유지, 1: 좌회전, 2: 직진, 3: 우회전, 4: 정지)
        for i in range(2):
            action_dict[i] = np.random.randint(0, 5) 

        # 환경에 행동을 적용하고 결과 받아오기
        next_obs, rewards, dones, info = env.step(action_dict)

        print(f"[Step {step+1}]")
        print(f" └ 선택한 행동: {action_dict}")
        print(f" └ 받은 보상: {rewards}")
        print(f" └ 에피소드 종료 여부: {dones['__all__']}\n")

if __name__ == "__main__":
    test_random_run()