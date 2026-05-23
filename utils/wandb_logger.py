"""
utils/wandb_logger.py
W&B(Weights & Biases) 학습 곡선 모니터링

기록 지표:
  [학습 중 매 iteration]
  - episode_return_mean / min / max  : 학습 수렴 여부
  - episode_len_mean                 : 에피소드 평균 길이
  - num_episodes                     : 총 에피소드 수
  - env_steps_total                  : 누적 환경 스텝 수

  [평가 완료 후]
  - avg_delivery_count  : 평균 배송 횟수 (Throughput)
  - avg_collision_count : 평균 충돌 횟수
  - avg_stall_count     : 평균 정체 카운트 (정체 구간 해소 여부)
  - avg_episode_return  : 평균 에피소드 리턴
  - avg_current_step    : 평균 에피소드 종료 스텝
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class WandBLogger:

    def __init__(self, project: str = "MARL-OHT-Optimization", run_name: str = None):
        self.project  = project
        self.run_name = run_name
        self._wandb   = None
        self._enabled = False

    def init(self, config: dict = None):
        """
        W&B 실행을 초기화
        """
        try:
            import wandb
            self._wandb = wandb
            wandb.init(
                project = self.project,
                name    = self.run_name,
                config  = config or {},
            )
            self._enabled = True
            print(f"✅ W&B 연동 성공 | Project: {self.project} | Run: {self.run_name}")
            print(f"   대시보드: {wandb.run.get_url()}")
        except ImportError:
            print("⚠️  wandb가 설치되지 않았습니다. 콘솔 출력 모드로 실행합니다.")
            print("   설치: pip install wandb")
            self._enabled = False
        except Exception as e:
            print(f"⚠️  W&B 초기화 실패: {e}")
            print("   콘솔 출력 모드로 실행합니다.")
            self._enabled = False

    def log_train(self, iteration: int, result: dict):
        """
        매 학습 iteration 결과를 W&B에 기록
        """
        env_runners = result.get("env_runners", {})

        log_dict = {
            "train/iteration":           iteration + 1,
            "train/episode_return_mean": env_runners.get("episode_return_mean"),
            "train/episode_return_min":  env_runners.get("episode_return_min"),
            "train/episode_return_max":  env_runners.get("episode_return_max"),
            "train/episode_len_mean":    env_runners.get("episode_len_mean"),
            "train/num_episodes":        env_runners.get("num_episodes"),
            "train/env_steps_total":     env_runners.get("num_env_steps_sampled_lifetime"),
            "train/module_steps_total":  env_runners.get("num_module_steps_sampled_lifetime"),
        }

        # None 값 제거
        log_dict = {k: v for k, v in log_dict.items() if v is not None}

        if self._enabled:
            self._wandb.log(log_dict, step=iteration + 1)
        else:
            # W&B 없을 때 콘솔 출력
            mean = log_dict.get("train/episode_return_mean", "N/A")
            steps = log_dict.get("train/env_steps_total", "N/A")
            print(f"  [W&B-console] iter={iteration+1} | return_mean={mean} | env_steps={steps}")

    def log_eval(self, *summaries):
        """
        평가 완료 후 정책별 비교 지표를 W&B에 기록
        """
        for summary in summaries:
            policy   = summary.get("policy", "unknown")
            num_ohts = summary.get("num_ohts", 0)
            prefix   = f"eval/{policy}_{num_ohts}ohts"

            log_dict = {
                f"{prefix}/avg_delivery_count":     summary.get("avg_delivery_count"),
                f"{prefix}/avg_collision_count":    summary.get("avg_collision_count"),
                f"{prefix}/avg_invalid_action":     summary.get("avg_invalid_action_count"),
                f"{prefix}/avg_episode_return":     summary.get("avg_episode_return"),
                f"{prefix}/avg_current_step":       summary.get("avg_current_step"),
            }

            log_dict = {k: v for k, v in log_dict.items() if v is not None}

            if self._enabled:
                self._wandb.log(log_dict)
            else:
                print(f"  [W&B-console] EVAL | {policy} {num_ohts}ohts | "
                      f"delivery={summary.get('avg_delivery_count', 'N/A'):.2f} | "
                      f"collision={summary.get('avg_collision_count', 'N/A'):.2f} | "
                      f"return={summary.get('avg_episode_return', 'N/A'):.2f}")

    def log_stall(self, iteration: int, stall_data: dict):
        """
        정체 구간 해소 데이터를 기록
        """
        log_dict = {
            "stall/avg_stall_count":  stall_data.get("avg_stall"),
            "stall/max_stall_count":  stall_data.get("max_stall"),
            "stall/deadlock_count":   stall_data.get("deadlock_count"),
        }
        log_dict = {k: v for k, v in log_dict.items() if v is not None}

        if self._enabled:
            self._wandb.log(log_dict, step=iteration + 1)
        else:
            print(f"  [W&B-console] STALL | iter={iteration+1} | {stall_data}")

    def finish(self):
        """W&B 실행을 종료합니다."""
        if self._enabled:
            self._wandb.finish()
            print("✅ W&B 기록 완료")



if __name__ == "__main__":
    # 단독 실행 테스트 (W&B 연결 없이 콘솔 출력 확인)
    print("=== WandBLogger 단독 테스트 ===")

    logger = WandBLogger(project="MARL-OHT-Optimization", run_name="test_run")
    logger.init(config={"num_ohts": 5, "lr": 3e-4})

    # 가짜 학습 결과로 테스트
    fake_result = {
        "env_runners": {
            "episode_return_mean": -120.5,
            "episode_return_min":  -300.0,
            "episode_return_max":   50.0,
            "episode_len_mean":     85.3,
            "num_episodes":         12,
            "num_env_steps_sampled_lifetime": 1000,
        }
    }
    logger.log_train(0, fake_result)

    # 가짜 평가 결과로 테스트
    fake_summary = {
        "policy": "ppo", "num_ohts": 5,
        "avg_delivery_count": 19.6,
        "avg_collision_count": 30.0,
        "avg_invalid_action_count": 0.0,
        "avg_episode_return": -130.78,
        "avg_current_step": 91.0,
    }
    logger.log_eval(fake_summary)
    logger.finish()
