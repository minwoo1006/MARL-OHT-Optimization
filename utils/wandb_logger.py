import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _enable_utf8_console():
    """Keep emoji/status output intact on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


class WandBLogger:
    def __init__(self, project: str = "MARL-OHT-Optimization", run_name: str = None):
        self.project = project
        self.run_name = run_name
        self._wandb = None
        self._enabled = False

    def init(self, config: dict = None):
        _enable_utf8_console()

        try:
            wandb_dir = os.path.abspath(os.getcwd())
            wandb_cache_dir = os.path.abspath(os.path.join(os.getcwd(), ".wandb_cache"))
            os.makedirs(wandb_dir, exist_ok=True)
            os.makedirs(wandb_cache_dir, exist_ok=True)
            os.environ.setdefault("WANDB_DIR", wandb_dir)
            os.environ.setdefault("WANDB_CACHE_DIR", wandb_cache_dir)
            os.environ.setdefault("WANDB_CONFIG_DIR", wandb_cache_dir)

            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.project,
                name=self.run_name,
                config=config or {},
            )
            self._enabled = True
            print(f"✅ W&B 연동 성공 | Project: {self.project} | Run: {self.run_name}")
            print(f"   📊 대시보드: {wandb.run.get_url()}")
        except ImportError:
            print("⚠️  wandb가 설치되지 않았습니다. 콘솔 출력 모드로 실행합니다.")
            print("   설치: pip install wandb")
            self._enabled = False
        except Exception as exc:
            print(f"⚠️  W&B 초기화 실패: {exc}")
            print("   콘솔 출력 모드로 실행합니다.")
            self._enabled = False

    def log_train(self, iteration: int, result: dict):
        env_runners = result.get("env_runners", {})

        log_dict = {
            "train/iteration": iteration + 1,
            "train/episode_return_mean": env_runners.get("episode_return_mean"),
            "train/episode_return_min": env_runners.get("episode_return_min"),
            "train/episode_return_max": env_runners.get("episode_return_max"),
            "train/episode_len_mean": env_runners.get("episode_len_mean"),
            "train/num_episodes": env_runners.get("num_episodes"),
            "train/env_steps_total": env_runners.get("num_env_steps_sampled_lifetime"),
            "train/module_steps_total": env_runners.get("num_module_steps_sampled_lifetime"),
        }
        log_dict = {key: value for key, value in log_dict.items() if value is not None}

        if self._enabled:
            self._wandb.log(log_dict, step=iteration + 1)
        else:
            mean = log_dict.get("train/episode_return_mean", "N/A")
            steps = log_dict.get("train/env_steps_total", "N/A")
            print(f"  [W&B-console] iter={iteration + 1} | return_mean={mean} | env_steps={steps}")

    def log_metrics(self, metrics: dict, step: int = None):
        metrics = {key: value for key, value in metrics.items() if value is not None}
        if not metrics:
            return

        if self._enabled:
            self._wandb.log(metrics, step=step)
        else:
            print(f"  [W&B-console] metrics | {metrics}")

    def log_eval(self, *summaries):
        for summary in summaries:
            policy = summary.get("policy", "unknown")
            num_ohts = summary.get("num_ohts", 0)
            prefix = f"eval/{policy}_{num_ohts}ohts"

            log_dict = {
                f"{prefix}/avg_delivery_count": summary.get("avg_delivery_count"),
                f"{prefix}/avg_hot_lot_delivery_count": summary.get("avg_hot_lot_delivery_count"),
                f"{prefix}/avg_collision_count": summary.get("avg_collision_count"),
                f"{prefix}/avg_invalid_action": summary.get("avg_invalid_action_count"),
                f"{prefix}/avg_episode_return": summary.get("avg_episode_return"),
                f"{prefix}/avg_current_step": summary.get("avg_current_step"),
            }
            log_dict = {key: value for key, value in log_dict.items() if value is not None}

            if self._enabled:
                self._wandb.log(log_dict)
            else:
                print(
                    f"  [W&B-console] EVAL | {policy} {num_ohts}ohts | "
                    f"delivery={summary.get('avg_delivery_count', 0.0):.2f} | "
                    f"collision={summary.get('avg_collision_count', 0.0):.2f} | "
                    f"return={summary.get('avg_episode_return', 0.0):.2f}"
                )

    def log_stall(self, iteration: int, stall_data: dict):
        log_dict = {
            "stall/avg_stall_count": stall_data.get("avg_stall"),
            "stall/max_stall_count": stall_data.get("max_stall"),
            "stall/deadlock_count": stall_data.get("deadlock_count"),
        }
        log_dict = {key: value for key, value in log_dict.items() if value is not None}

        if self._enabled:
            self._wandb.log(log_dict, step=iteration + 1)
        else:
            print(f"  [W&B-console] STALL | iter={iteration + 1} | {stall_data}")

    def finish(self):
        if self._enabled:
            self._wandb.finish()
            print("✅ W&B 기록 완료")


if __name__ == "__main__":
    logger = WandBLogger(project="MARL-OHT-Optimization", run_name="test_run")
    logger.init(config={"num_ohts": 5, "lr": 3e-4})
    logger.log_train(
        0,
        {
            "env_runners": {
                "episode_return_mean": -120.5,
                "episode_return_min": -300.0,
                "episode_return_max": 50.0,
                "episode_len_mean": 85.3,
                "num_episodes": 12,
                "num_env_steps_sampled_lifetime": 1000,
            }
        },
    )
    logger.log_eval(
        {
            "policy": "ppo",
            "num_ohts": 5,
            "avg_delivery_count": 19.6,
            "avg_collision_count": 30.0,
            "avg_invalid_action_count": 0.0,
            "avg_episode_return": -130.78,
            "avg_current_step": 91.0,
        }
    )
    logger.finish()
