"""Command-line tool to evaluate SA-MOO on RobustBench models."""


from __future__ import annotations

import argparse
import importlib
import os
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from matplotlib.colors import rgb_to_hsv

from dotenv import load_dotenv
from .config.config import apply_config, load_config
from .core.attack_runner import run_samoo_attack

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _parse_args() -> argparse.Namespace:
    # 先加载 .env
    load_dotenv(override=False)
    # 读取 .env 配置
    env_data_dir = os.getenv("DATA_ROOT_DIR")
    env_output_dir = os.getenv("OUTPUT_DIR")
    env_torch_home = os.getenv("TORCH_HOME")
    env_rb_dir = os.getenv("ROBUST_BENCH_DIR")

    parser = argparse.ArgumentParser(description="Evaluate SA-MOO attack on RobustBench models.")
    parser.add_argument("--model-name", required=True, help="RobustBench model name, e.g., 'Standard'.")
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet"], help="Dataset key.")
    parser.add_argument("--threat-model", default="Linf", help="RobustBench threat model key.")
    parser.add_argument("--n-examples", type=int, default=100, help="Number of test examples to attack.")
    parser.add_argument("--start-index", type=int, default=0, help="Starting index within the RobustBench evaluation subset.")
    parser.add_argument("--indices", type=str, help="Comma-separated list of explicit sample indices (overrides --start-index/--n-examples).")
    parser.add_argument("--random-sample", action="store_true", help="Randomly choose n-examples distinct indices (dataset subset order).")
    parser.add_argument("--random-seed", type=int, help="Seed for --random-sample to ensure reproducibility.")
    parser.add_argument("--population-size", type=int, help="Override SA-MOO population size.")
    parser.add_argument("--num-generations", type=int, help="Override SA-MOO generation count.")
    parser.add_argument("--fixed-k", type=int, help="Override SA-MOO fixed K budget.")
    parser.add_argument("--crossover-prob", type=float, help="Override crossover probability.")
    parser.add_argument("--zero-sample-prob", type=float, help="Override zero sample probability.")
    parser.add_argument("--perturbation-mode", type=str, choices=["rgb_sim", "channel", "v_channel"], help="Override perturbation mode.")
    parser.add_argument("--targeted", action="store_true", help="Evaluate a targeted attack.")
    parser.add_argument("--target-class-id", type=int, help="Target class id when --targeted is set.")
    parser.add_argument("--config", type=str, help="Optional path to base SA-MOO config file.")
    parser.add_argument("--output-dir", type=str, default=env_output_dir or "robustbench_runs", help="Directory for evaluation outputs.")
    parser.add_argument("--data-dir", type=str, default=env_data_dir or "./data", help="Directory to cache RobustBench datasets.")
    parser.add_argument("--device", type=str, default=None, help="Computation device, e.g., 'cuda:0' or 'cpu'.")
    parser.add_argument("--save-artifacts", action="store_true", help="Persist per-sample artifacts (images, logs).")
    parser.add_argument("--results-json", type=str, help="Optional path to store aggregated metrics as JSON.")
    parser.add_argument("--disable-edge-guidance", action="store_true", help="Disable edge guidance for evaluation.")
    parser.add_argument("--disable-visualization", action="store_true", help="Disable interactive visualization during evaluation.")
    parser.add_argument("--verbose", action="store_true", help="Print per-example progress details.")
    parser.add_argument("--debug-paths", action="store_true", help="Print internal cache/model paths before loading model.")
    parser.add_argument("--no-download", action="store_true", help="Do not attempt to download model; require cached file to exist.")
    # 额外：将 TORCH_HOME、ROBUST_BENCH_DIR 注入环境变量（优先 .env）
    if env_torch_home:
        os.environ["TORCH_HOME"] = env_torch_home
    if env_rb_dir:
        os.environ["ROBUST_BENCH_DIR"] = env_rb_dir
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace, output_dir: Path, placeholder_weights: Path) -> Dict[str, object]:
    overrides: Dict[str, object] = {
        "output_dir": str(output_dir),
        "data_root_dir": str(Path(args.data_dir).resolve()),
        "model_weights_path": str(placeholder_weights),
        "target_image_id": int(args.start_index),
    }

    if args.population_size is not None:
        overrides["population_size"] = args.population_size
    if args.num_generations is not None:
        overrides["num_generations"] = args.num_generations
    if args.fixed_k is not None:
        overrides["fixed_k"] = args.fixed_k
    if args.crossover_prob is not None:
        overrides["crossover_prob"] = args.crossover_prob
    if args.zero_sample_prob is not None:
        overrides["zero_sample_prob"] = args.zero_sample_prob
    if args.perturbation_mode is not None:
        overrides["perturbation_mode"] = args.perturbation_mode
    if args.targeted:
        overrides["is_targeted_attack"] = True
        if args.target_class_id is None:
            raise ValueError("--targeted requires --target-class-id to be specified.")
        overrides["target_class_id"] = args.target_class_id
    else:
        overrides["is_targeted_attack"] = False

    if args.disable_edge_guidance:
        overrides["edge_guidance"] = {"enabled": False}

    if args.disable_visualization:
        overrides["enable_visualization"] = False

    return overrides


def _load_dataset(dataset: str, n_examples: int, data_dir: Path) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    try:
        data_module = importlib.import_module("robustbench.data")
    except ModuleNotFoundError as exc:  # pragma: no cover
        if exc.name in {"robustbench", "robustbench.data"}:
            raise ImportError(
                "robustbench is required for this script. Install it with 'pip install robustbench'."
            ) from exc
        if exc.name == "autoattack":
            raise ImportError(
                "robustbench requires the 'autoattack' package. Install it with 'pip install autoattack'."
            ) from exc
        raise

    load_cifar10 = getattr(data_module, "load_cifar10")
    load_cifar100 = getattr(data_module, "load_cifar100")
    load_imagenet = getattr(data_module, "load_imagenet")

    if dataset == "cifar10":
        x, y = load_cifar10(n_examples=n_examples, data_dir=data_dir)
        class_names = CIFAR10_CLASS_NAMES
    elif dataset == "cifar100":
        x, y = load_cifar100(n_examples=n_examples, data_dir=data_dir)
        class_names = [str(idx) for idx in range(100)]
    elif dataset == "imagenet":
        x, y = load_imagenet(n_examples=n_examples, data_dir=data_dir)
        class_names = [str(idx) for idx in range(1000)]
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return x, y, class_names


def main() -> None:
    args = _parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    placeholder_weights = output_dir / "robustbench_placeholder.pt"
    placeholder_weights.touch(exist_ok=True)

    overrides = _build_overrides(args, output_dir, placeholder_weights)
    cfg = load_config(config_file=args.config, overrides=overrides)
    apply_config(cfg)

    # Allow overriding the RobustBench cache directory via environment or .env.
    raw_rb_dir = os.getenv("ROBUST_BENCH_DIR")
    if not raw_rb_dir:
        raw_rb_dir = overrides.get("robust_bench_dir") if overrides else None
    robustbench_root = Path(raw_rb_dir) if raw_rb_dir else Path(cfg.TORCH_HOME) / "robustbench"
    # If user pointed to parent directory but actually stored under a nested 'robustbench' folder, auto-adjust.
    nested_candidate = robustbench_root / "robustbench" / "models"
    if not (robustbench_root / "models").exists() and nested_candidate.exists():
        # Use the outer robustbench_root/robustbench as real root.
        robustbench_root = robustbench_root / "robustbench"
    os.environ["ROBUST_BENCH_DIR"] = str(robustbench_root)
    robustbench_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        utils_module = importlib.import_module("robustbench.utils")
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency missing at runtime.
        if exc.name in {"robustbench", "robustbench.utils"}:
            raise ImportError(
                "robustbench is required for this script. Install it with 'pip install robustbench'."
            ) from exc
        if exc.name == "autoattack":
            raise ImportError(
                "robustbench requires the 'autoattack' package. Install it with 'pip install autoattack'."
            ) from exc
        raise

    load_model = getattr(utils_module, "load_model")

    # Compute expected model path (RobustBench default logic constructs path under ROBUST_BENCH_DIR/models/...)
    expected_model_path = robustbench_root / "models" / args.dataset / args.threat_model / f"{args.model_name}.pt"
    # 某些新版 robustbench.utils 可能直接使用相对路径 'models/...'; 我们检测当前工作目录下该相对文件是否存在
    relative_model_path = Path("models") / args.dataset / args.threat_model / f"{args.model_name}.pt"
    if args.debug_paths:
        print("[Debug] ROBUST_BENCH_DIR:", robustbench_root)
        print("[Debug] Expected model file:", expected_model_path)
        print("[Debug] Exists:", expected_model_path.exists())
        print("[Debug] Relative model file (CWD):", relative_model_path.resolve())
        print("[Debug] Relative exists:", relative_model_path.exists())

    # 如果缓存存在但相对路径不存在，且我们推测 load_model 会用相对路径，则复制/同步。
    # 触发条件：expected 存在 + relative 不存在。
    if expected_model_path.exists() and not relative_model_path.exists():
        try:
            relative_model_path.parent.mkdir(parents=True, exist_ok=True)
            # 复制文件
            import shutil
            shutil.copy2(expected_model_path, relative_model_path)
            if args.debug_paths:
                print(f"[Debug] Copied cached model to working directory: {relative_model_path}")
        except Exception as copy_e:  # pragma: no cover
            if args.debug_paths:
                print("[Debug] Failed to copy model to relative path:", repr(copy_e))

    # 如果文件已经存在，或用户指定 --no-download，则我们尝试全面阻断下载相关函数。
    if expected_model_path.exists() or relative_model_path.exists() or args.no_download:
        import inspect
        try:  # 屏蔽真正下载
            import robustbench.utils as _rb_utils  # type: ignore
            blocked_names = []
            def _skip_download(*a, **k):  # noqa: ANN001, ANN002
                print(f"[Info] Skip download, using cached model: {expected_model_path}")
                # 不返回任何内容；某些函数可能期望返回路径，若需要再适配
                return
            for attr_name in dir(_rb_utils):
                if "download" in attr_name.lower() or "get_and_save" in attr_name.lower():
                    obj = getattr(_rb_utils, attr_name)
                    if callable(obj):
                        try:
                            setattr(_rb_utils, attr_name, _skip_download)
                            blocked_names.append(attr_name)
                        except Exception:
                            pass
            if args.no_download and not expected_model_path.exists():
                # 缓存不存在，直接报错（在 patch 之后）
                raise FileNotFoundError(f"--no-download 已启用，但缓存模型不存在: {expected_model_path}")
            if args.debug_paths:
                print(f"[Debug] Patched download-related functions: {blocked_names}")
        except Exception as _e:  # pragma: no cover
            if args.debug_paths:
                print("[Debug] Monkey patch failed:", repr(_e))

    if args.no_download and not expected_model_path.exists():  # 双重保护
        raise FileNotFoundError(f"--no-download: 需要缓存文件 {expected_model_path} 但未找到")
    model = load_model(model_name=args.model_name, dataset=args.dataset, threat_model=args.threat_model)
    model.to(device).eval()
    model.requires_grad_(False)

    data_dir = Path(args.data_dir).resolve()
    # 加载至少覆盖可能用到的最大索引数量的数据。
    # 对于 CIFAR10/100 固定顺序，若使用自定义 indices 直接加载完整评估子集（RobustBench 默认顺序）。
    dataset_default_sizes = {
        "cifar10": 10000,
        "cifar100": 10000,
        "imagenet": 50000,
    }

    max_needed = args.start_index + args.n_examples
    if args.indices:
        # 解析 indices 字符串
        raw_list = [s.strip() for s in args.indices.split(',') if s.strip()]
        explicit_indices = sorted({int(v) for v in raw_list})
        max_needed = max(explicit_indices) + 1
    else:
        explicit_indices = None

    if args.random_sample:
        max_needed = max(max_needed, dataset_default_sizes.get(args.dataset, max_needed))

    x_test, y_test, class_names = _load_dataset(args.dataset, max_needed, data_dir)

    # 如果 random-sample 模式，采样 n_exmaples 个不重复索引
    if args.random_sample and explicit_indices is not None:
        raise ValueError("--random-sample 与 --indices 不能同时使用")
    if args.random_sample:
        rng = np.random.default_rng(args.random_seed)
        explicit_indices = sorted(rng.choice(range(len(y_test)), size=args.n_examples, replace=False).tolist())

    if explicit_indices is None:
        selected_indices = list(range(args.start_index, args.start_index + args.n_examples))
    else:
        selected_indices = explicit_indices

    if len(selected_indices) == 0:
        raise ValueError("没有可评估的样本索引")

    if args.debug_paths:
        print(f"[Debug] Using sample indices: {selected_indices[:10]}{' ...' if len(selected_indices)>10 else ''}")

    total = 0
    success = 0
    query_sum = 0
    records: List[Dict[str, object]] = []

    for local_idx, sample_idx in enumerate(selected_indices):
        image = x_test[sample_idx].permute(1, 2, 0).cpu().numpy().astype(np.float32)
        label = int(y_test[sample_idx])

        if cfg.PERTURBATION_MODE == "v_channel":
            original_v = rgb_to_hsv(image)[:, :, 2]
        else:
            original_v = None

        run_dir: Optional[Path] = None
        if args.save_artifacts:
            run_dir = output_dir / f"sample_{sample_idx:05d}"
            run_dir.mkdir(parents=True, exist_ok=True)

        attack_result = run_samoo_attack(
            original_rgb=image,
            original_v=original_v,
            true_label=label,
            model=model,
            class_names=class_names,
            device=device,
            run_dir=run_dir,
            time_str=None,
            persist_metrics=None,
            verbose=args.verbose,
        )

        is_success = attack_result.success
        predicted_label = attack_result.predicted_class
        if not args.targeted and predicted_label == label:
            is_success = False
        if args.targeted and predicted_label != cfg.TARGET_CLASS_ID:
            is_success = False

        total += 1
        success += int(is_success)
        query_sum += attack_result.queries

        if args.verbose:
            status = "SUCCESS" if is_success else "FAIL"
            print(f"[Eval] idx={sample_idx} label={label} pred={predicted_label} status={status} queries={attack_result.queries}")

        record = {
            "sample_index": sample_idx,
            "true_label": label,
            "predicted_label": predicted_label,
            "success": is_success,
            "queries": attack_result.queries,
            "metrics": attack_result.metrics,
        }
        records.append(record)

    robust_accuracy = 1.0 - (success / max(total, 1))
    avg_queries = query_sum / max(total, 1)

    summary = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "threat_model": args.threat_model,
        "n_examples": total,
        "success_count": success,
        "robust_accuracy": robust_accuracy,
        "average_queries": avg_queries,
        "indices": selected_indices,
    }

    print("\n=== RobustBench Evaluation Summary ===")
    print(json.dumps(summary, indent=2))

    if args.results_json:
        results_path = Path(args.results_json).resolve()
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("w", encoding="utf-8") as fh:
            json.dump({"summary": summary, "records": records}, fh, indent=2)
    

if __name__ == "__main__":
    main()
