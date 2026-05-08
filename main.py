from envs.oht_env import OHTFabEnv
# from agents.mappo import train_mappo # 추후 팀원 2가 작성

def main():
    print("=== MARL OHT 최적화 프로젝트 시작 ===")
    env = OHTFabEnv(num_ohts=3)
    obs, info = env.reset()
    # train_mappo(env) # 학습 실행

if __name__ == "__main__":
    main()