# main.py
# 主执行模块：SA-MOO对抗攻击算法的主要流程
# 这个模块整合所有子模块，执行完整的攻击过程
# 包括初始化、进化循环、结果分析和保存

import os
import sys
import warnings
import random
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
import torch

# 导入配置
from .config.config import *

# 导入各模块
from .utils.logger import Logger
from .data.data_loader import load_target_image_and_model
from .core.attack_runner import run_samoo_attack

def main() -> None:
    """
    SA-MOO对抗攻击算法的主函数。

    执行完整的攻击流程：
    1. 加载数据和模型
    2. 初始化种群
    3. 进化循环（初始化->交叉->变异->选择）
    4. 结果分析和保存
    """
    # 过滤警告信息
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models._utils")

    # 创建运行目录和日志
    time_str = datetime.now().strftime("%m%d%H%M")
    attack_type = "targeted" if IS_TARGETED_ATTACK else "non_targeted"
    run_name = f"img{TARGET_IMAGE_ID}_{attack_type}_k{FIXED_K}_{PERTURBATION_MODE}_gen{NUM_GENERATIONS}_{time_str}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "report.txt")
    sys.stdout = logger  # 重定向输出到日志文件

    seed_env = os.getenv("SA_MOO_SEED")
    if seed_env is not None:
        try:
            seed_value = int(seed_env)
            np.random.seed(seed_value)
            random.seed(seed_value)
            torch.manual_seed(seed_value)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed_value)
            print(f"[Seed] Global random seed set to {seed_value}")
        except ValueError:
            print(f"[Seed] Invalid SA_MOO_SEED value: {seed_env}")

    try:
        # 获取计算设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def persist_metrics(payload: dict) -> None:
            """Save metrics to run directory and optional external path."""

            metrics_env_path = os.getenv("SA_MOO_METRICS_JSON")

            def sanitize(value):
                if isinstance(value, (np.generic,)):
                    return sanitize(value.item())
                if isinstance(value, (float, int)):
                    if isinstance(value, float) and not math.isfinite(value):
                        return None
                    return float(value)
                if isinstance(value, Path):
                    return str(value)
                if isinstance(value, dict):
                    return {k: sanitize(v) for k, v in value.items()}
                if isinstance(value, (list, tuple)):
                    return [sanitize(v) for v in value]
                return value

            payload_serializable = sanitize(payload)

            try:
                with (run_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                    json.dump(payload_serializable, fh, indent=2, ensure_ascii=False)
            except Exception as exc:
                print(f"[Metrics] Failed to write metrics.json: {exc}")

            if metrics_env_path:
                try:
                    env_path = Path(metrics_env_path)
                    env_path.parent.mkdir(parents=True, exist_ok=True)
                    with env_path.open("w", encoding="utf-8") as fh:
                        json.dump(payload_serializable, fh, indent=2, ensure_ascii=False)
                except Exception as exc:
                    print(f"[Metrics] Failed to write metrics to {metrics_env_path}: {exc}")

        print("[Init] Loading target image, dataset, and model weights...", flush=True)
        original_rgb, original_v, true_label, model, class_names = load_target_image_and_model(
            TARGET_IMAGE_ID, MODEL_WEIGHTS_PATH, DATA_ROOT_DIR, PERTURBATION_MODE
        )
        print("[Init] Data and model ready.", flush=True)

        result = run_samoo_attack(
            original_rgb=original_rgb,
            original_v=original_v,
            true_label=true_label,
            model=model,
            class_names=class_names,
            device=device,
            run_dir=run_dir,
            time_str=time_str,
            persist_metrics=persist_metrics,
            verbose=True,
        )

        print("\n--- Attack Completed ---")
        if not result.success:
            print(f"Total Queries: {result.queries}")
            print("Failed to find an adversarial example.")
            return

        best_obj = result.best_objective
        print("Final Best Solution Stats:")
        print(f"  Adversarial: {best_obj[0]}")
        print(f"  L2 Norm: {best_obj[2]:.4f}")
        print(f"  L0 Norm: {best_obj[3]}")
        if result.metrics.get("psnr") is not None:
            print(f"  PSNR: {result.metrics['psnr']:.2f} dB")
        if result.metrics.get("lpips") is not None:
            print(f"  LPIPS: {result.metrics['lpips']:.4f}")

        true_label_name = class_names[true_label]
        predicted_name = result.predicted_class_name or str(result.predicted_class)
        print("\n--- Prediction Verification ---")
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            resize_info = "enabled" if ENABLE_RESIZE_PREPROCESSING else "disabled"
            print(
                "Note: Verification applied real-world preprocessing "
                f"(JPEG quality: {JPEG_QUALITY}, resize: {resize_info})"
            )
        else:
            print("Note: Verification without preprocessing")

        if IS_TARGETED_ATTACK:
            target_name = class_names[TARGET_CLASS_ID]
            print(f"Original class: {true_label_name} ({true_label})")
            print(f"Target class: {target_name} ({TARGET_CLASS_ID})")
            print(f"Predicted class after attack: {predicted_name} ({result.predicted_class})")
            success_flag = result.predicted_class == TARGET_CLASS_ID
            print(f"Attack Success: {success_flag}")
        else:
            print(f"Original class: {true_label_name} ({true_label})")
            print(f"Predicted class after attack: {predicted_name} ({result.predicted_class})")
            success_flag = result.predicted_class != true_label
            print(f"Attack Success: {success_flag}")

        print(f"Original class probability dropped from {result.metrics['true_prob_before']:.2f}%"
              f" to {result.metrics['true_prob_after']:.2f}%")
        print(
            f"Model now confidently predicts '{predicted_name}' with "
            f"{result.metrics['predicted_prob']:.2f}% confidence."
        )

        print(f"Total Queries: {result.queries}")
        if run_dir is not None:
            print(f"\nResults saved in: {run_dir.absolute()}")

    finally:
        # 恢复原始stdout并关闭日志
        sys.stdout = logger.terminal
        logger.close()

if __name__ == "__main__":
    main()