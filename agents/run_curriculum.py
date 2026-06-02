import os
import subprocess
import sys


DEFAULT_STAGES = [
    ("navigation_2oht", 2, 50),
    ("congestion_5oht", 5, 100),
    ("density_10oht", 10, 100),
]


def parse_stages(raw_value):
    if not raw_value:
        return DEFAULT_STAGES

    stages = []
    for item in raw_value.split(","):
        name, num_ohts, iterations = item.split(":")
        stages.append((name.strip(), int(num_ohts), int(iterations)))
    return stages


def main():
    stages = parse_stages(os.environ.get("OHT_CURRICULUM"))
    checkpoint_dir = os.path.abspath(os.path.join(os.getcwd(), "checkpoints", "curriculum"))
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_in = None
    for stage_name, num_ohts, iterations in stages:
        checkpoint_out = os.path.join(checkpoint_dir, stage_name)
        env = os.environ.copy()
        env.update(
            {
                "OHT_STAGE": stage_name,
                "OHT_NUM_OHTS": str(num_ohts),
                "OHT_NUM_ITERATIONS": str(iterations),
                "OHT_CHECKPOINT_OUT": checkpoint_out,
            }
        )
        if checkpoint_in:
            env["OHT_CHECKPOINT_IN"] = checkpoint_in

        print("=" * 80)
        print(f"Starting curriculum stage: {stage_name} | OHTs={num_ohts} | iterations={iterations}")
        print("=" * 80)

        subprocess.run(
            [sys.executable, "-u", "agents/train_ppo_rllib.py"],
            check=True,
            env=env,
        )
        checkpoint_in = checkpoint_out


if __name__ == "__main__":
    main()
