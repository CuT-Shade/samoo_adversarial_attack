# run_batch.py
# 批量运行实验脚本
# 可以同时运行多个配置文件或参数组合
# 可以选取./experiment_configs路径下的yaml

import subprocess
import sys
from pathlib import Path
import time

def run_single_experiment(config_file: str, log_file: str = None):
    """
    运行单个实验

    Args:
        config_file: 配置文件名
        log_file: 日志文件路径（可选）
    """
    cmd = [sys.executable, "run_experiment.py", config_file]

    print(f"Running experiment: {config_file}")
    start_time = time.time()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)

        end_time = time.time()
        duration = end_time - start_time

        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"Experiment: {config_file}\n")
                f.write(f"Duration: {duration:.2f} seconds\n")
                f.write(f"Return code: {result.returncode}\n")
                if result.stdout:
                    f.write("STDOUT:\n")
                    f.write(result.stdout)
                if result.stderr:
                    f.write("STDERR:\n")
                    f.write(result.stderr)
                f.write(f"{'='*50}\n")

        if result.returncode == 0:
            print(f"✓ Success ({duration:.2f} seconds)")
        else:
            print(f"✗ Failed (exit code: {result.returncode})")
            if result.stderr:
                print(f"Error: {result.stderr.strip()}")

    except Exception as e:
        print(f"✗ Error running {config_file}: {e}")
        if log_file:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Exception in {config_file}: {e}\n")

def main():
    """主函数：运行批量实验"""
    experiments = [
        "rgb_sim_config.yaml",
        "v_channel_config.yaml",
        "targeted_attack_config.yaml",
        # 添加更多实验...
    ]

    # 创建日志文件
    log_file = f"batch_run_{time.strftime('%Y%m%d_%H%M%S')}.log"

    print(f"Starting batch run with {len(experiments)} experiments")
    print(f"Results will be logged to: {log_file}")

    for i, config in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] ", end="")
        run_single_experiment(config, log_file)

    print("Batch run completed!")
    print(f"Check {log_file} for detailed results")

if __name__ == "__main__":
    main()