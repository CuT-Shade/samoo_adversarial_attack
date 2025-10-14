"""Batch evaluation helper to run SA-MOO (robustbench_eval.py) over multiple
perturbation modes on the SAME fixed sample index set and aggregate stats.

Usage example (random 100 samples, three modes):
  python -m samoo_adversarial_unified_attack_div.src.robustbench_batch_modes \
    --model-name Standard --dataset cifar10 --threat-model Linf \
    --n-examples 100 --random-sample --random-seed 42 \
    --modes rgb_sim,channel,v_channel --base-output-dir batch_runs_cifar10

It will:
  1. Generate (or load) a shared indices list.
  2. For each mode invoke robustbench_eval.py with --indices=<same list>.
  3. Collect each per-mode results.json and compute aggregated metrics.
  4. Write aggregate JSON + CSV into base output dir.

Design notes:
  - We call robustbench_eval.py via subprocess to avoid refactoring its CLI code.
  - Mean of metrics (if present in records[i]['metrics']) is computed over
    successful attacks only; if no successes, means are omitted.
  - Dataset size defaults (test splits) are assumed: CIFAR10/100=10000, ImageNet=50000.
    Override with --dataset-size if needed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from dotenv import load_dotenv


def parse_args() -> argparse.Namespace:
    load_dotenv(override=False)
    env_data_dir = os.getenv("DATA_ROOT_DIR")
    env_output_dir = os.getenv("OUTPUT_DIR")

    p = argparse.ArgumentParser(description="Batch multi-mode RobustBench evaluation for SA-MOO.")
    p.add_argument("--model-name", required=True)
    p.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagenet"])
    p.add_argument("--threat-model", default="Linf")
    p.add_argument("--modes", default="rgb_sim,channel,v_channel", help="Comma separated perturbation modes to evaluate.")
    p.add_argument("--n-examples", type=int, default=100, help="Number of samples to evaluate (used with --random-sample unless --indices provided).")
    p.add_argument("--indices", type=str, help="Explicit comma separated indices list (overrides random sampling).")
    p.add_argument("--indices-file", type=str, help="Path to JSON or txt file containing indices list (overrides --indices / random).")
    p.add_argument("--random-sample", action="store_true", help="Randomly sample n-examples indices (ignored if --indices or --indices-file).")
    p.add_argument("--random-seed", type=int, default=0, help="Seed for random sampling.")
    p.add_argument("--dataset-size", type=int, help="Override assumed test dataset size for sampling.")
    p.add_argument("--config", type=str, help="Path to base SA-MOO config file passed to evaluator.")
    # Pass-through overrides used by robustbench_eval
    p.add_argument("--population-size", type=int)
    p.add_argument("--num-generations", type=int)
    p.add_argument("--fixed-k", type=int)
    p.add_argument("--crossover-prob", type=float)
    p.add_argument("--zero-sample-prob", type=float)
    p.add_argument("--targeted", action="store_true")
    p.add_argument("--target-class-id", type=int)
    p.add_argument("--disable-edge-guidance", action="store_true")
    p.add_argument("--disable-visualization", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--save-artifacts", action="store_true", help="Store per-sample artifacts inside each mode subfolder.")
    p.add_argument("--no-download", action="store_true", help="Disallow model download attempts.")
    p.add_argument("--debug-paths", action="store_true")
    p.add_argument("--device", type=str)
    p.add_argument("--data-dir", type=str, default=env_data_dir or "./data")
    p.add_argument("--base-output-dir", type=str, default=env_output_dir or "robustbench_batch_runs")
    p.add_argument("--results-prefix", type=str, default="results", help="Filename prefix for per-mode JSON (default: results).")
    p.add_argument("--shared-indices-json", type=str, help="Path to save shared indices list (default base_output_dir/indices.json).")
    p.add_argument("--aggregate-json", type=str, help="Path for aggregated JSON (default base_output_dir/aggregate.json).")
    p.add_argument("--aggregate-csv", type=str, help="Path for aggregated CSV (default base_output_dir/aggregate.csv).")
    return p.parse_args()


def _load_indices(args: argparse.Namespace, dataset_size: int) -> List[int]:
    if args.indices_file:
        path = Path(args.indices_file)
        if not path.exists():
            raise FileNotFoundError(f"--indices-file not found: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            data = json.loads(text)
            if not isinstance(data, list):
                raise ValueError("JSON indices file must contain a list")
            indices = [int(v) for v in data]
        except json.JSONDecodeError:
            # Fallback: comma/whitespace separated numbers
            raw = [tok for tok in text.replace("\n", ",").split(',') if tok.strip()]
            indices = [int(v) for v in raw]
        return sorted(set(indices))
    if args.indices:
        return sorted({int(s) for s in args.indices.split(',') if s.strip()})
    if args.random_sample:
        rng = np.random.default_rng(args.random_seed)
        if args.n_examples > dataset_size:
            raise ValueError(f"n-examples ({args.n_examples}) > dataset-size ({dataset_size})")
        return sorted(rng.choice(range(dataset_size), size=args.n_examples, replace=False).tolist())
    # Default: first n indices
    return list(range(args.n_examples))


def _build_common_pass_through(args: argparse.Namespace) -> Dict[str, str | None]:
    return {
        "--population-size": str(args.population_size) if args.population_size is not None else None,
        "--num-generations": str(args.num_generations) if args.num_generations is not None else None,
        "--fixed-k": str(args.fixed_k) if args.fixed_k is not None else None,
        "--crossover-prob": str(args.crossover_prob) if args.crossover_prob is not None else None,
        "--zero-sample-prob": str(args.zero_sample_prob) if args.zero_sample_prob is not None else None,
        "--config": args.config,
        "--target-class-id": str(args.target_class_id) if args.target_class_id is not None else None,
        "--device": args.device,
        "--data-dir": args.data_dir,
    }


def _run_mode(mode: str, indices: Sequence[int], args: argparse.Namespace, pass_map: Dict[str, str | None], base_dir: Path) -> Path:
    """Invoke robustbench_eval.py for a single perturbation mode.

    Returns path to the results JSON file.
    """
    mode_dir = base_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    results_path = mode_dir / f"{args.results_prefix}.json"

    # Build command list
    eval_module = "src.robustbench_eval"
    if __package__:
        # When installed as a package, prefer the fully qualified package path.
        candidate = f"{__package__}.robustbench_eval"
        try:
            __import__(candidate)
            eval_module = candidate
        except ModuleNotFoundError:
            pass

    cmd: List[str] = [sys.executable, "-m", eval_module,
                      "--model-name", args.model_name,
                      "--dataset", args.dataset,
                      "--threat-model", args.threat_model,
                      "--n-examples", str(len(indices)),
                      "--indices", ",".join(map(str, indices)),
                      "--perturbation-mode", mode,
                      "--output-dir", str(mode_dir),
                      "--results-json", str(results_path)]

    if args.save_artifacts:
        cmd.append("--save-artifacts")
    if args.no_download:
        cmd.append("--no-download")
    if args.debug_paths:
        cmd.append("--debug-paths")
    if args.verbose:
        cmd.append("--verbose")
    if args.targeted:
        cmd.append("--targeted")
    if args.disable_edge_guidance:
        cmd.append("--disable-edge-guidance")
    if args.disable_visualization:
        cmd.append("--disable-visualization")

    for k, v in pass_map.items():
        if v is not None:
            cmd.extend([k, v])

    print(f"[Batch] Running mode={mode} ...")
    # Execute
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (mode_dir / "run_stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        print(proc.stdout)
        raise RuntimeError(f"Mode {mode} failed with return code {proc.returncode}")
    if not results_path.exists():
        raise FileNotFoundError(f"Expected results JSON not found for mode {mode}: {results_path}")
    return results_path


def _aggregate(per_mode_json: Dict[str, Path]) -> Dict[str, object]:
    summary_table = []
    per_mode_details = {}
    metric_union = set()
    # First pass: collect basic
    for mode, path in per_mode_json.items():
        data = json.loads(path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        records = data.get("records", [])
        # Compute metric means across successful attacks if metrics present
        success_metrics: Dict[str, List[float]] = {}
        for rec in records:
            if rec.get("success") and isinstance(rec.get("metrics"), dict):
                for mk, mv in rec["metrics"].items():
                    if isinstance(mv, (int, float)):
                        success_metrics.setdefault(mk, []).append(float(mv))
                        metric_union.add(mk)
        metric_means = {mk: (float(np.mean(vals)) if len(vals) > 0 else None) for mk, vals in success_metrics.items()}
        per_mode_details[mode] = {
            "summary": summary,
            "success_metric_means": metric_means,
            "n_success_metrics": {k: len(v) for k, v in success_metrics.items()},
        }
        summary_table.append({
            "mode": mode,
            "success_count": summary.get("success_count"),
            "n_examples": summary.get("n_examples"),
            "robust_accuracy": summary.get("robust_accuracy"),
            "average_queries": summary.get("average_queries"),
            **{f"mean_{k}": metric_means.get(k) for k in metric_means}
        })
    return {
        "modes": list(per_mode_json.keys()),
        "per_mode": per_mode_details,
        "comparison": summary_table,
        "all_metrics": sorted(metric_union),
    }


def _write_csv(aggregate: Dict[str, object], path: Path) -> None:
    rows = aggregate.get("comparison", [])
    if not rows:
        return
    # Collect all column names
    cols = set()
    for r in rows:
        cols.update(r.keys())
    ordered = ["mode", "n_examples", "success_count", "robust_accuracy", "average_queries"] + sorted(c for c in cols if c not in {"mode", "n_examples", "success_count", "robust_accuracy", "average_queries"})
    lines = [",".join(ordered)]
    for r in rows:
        line = []
        for c in ordered:
            v = r.get(c)
            if isinstance(v, float):
                line.append(f"{v:.6f}")
            else:
                line.append("" if v is None else str(v))
        lines.append(",".join(line))
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    if not modes:
        raise ValueError("No modes provided via --modes")

    dataset_default_sizes = {"cifar10": 10000, "cifar100": 10000, "imagenet": 50000}
    dataset_size = args.dataset_size or dataset_default_sizes[args.dataset]

    indices = _load_indices(args, dataset_size)
    if len(indices) != len(set(indices)):
        raise ValueError("Duplicate indices detected")
    if args.n_examples and not args.indices and not args.indices_file and not args.random_sample:
        # indices list already derived from first n; ensure length matches request
        indices = indices[: args.n_examples]

    base_dir = Path(args.base_output_dir).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    # Save shared indices
    shared_indices_path = Path(args.shared_indices_json) if args.shared_indices_json else base_dir / "indices.json"
    shared_indices_path.write_text(json.dumps(indices, indent=2), encoding="utf-8")
    print(f"[Batch] Shared indices saved: {shared_indices_path} (count={len(indices)})")

    pass_map = _build_common_pass_through(args)
    per_mode_json: Dict[str, Path] = {}
    for mode in modes:
        per_mode_json[mode] = _run_mode(mode, indices, args, pass_map, base_dir)

    aggregate = _aggregate(per_mode_json)
    aggregate_path = Path(args.aggregate_json) if args.aggregate_json else base_dir / "aggregate.json"
    aggregate_path.write_text(json.dumps({"indices": indices, **aggregate}, indent=2), encoding="utf-8")
    print(f"[Batch] Aggregate JSON written: {aggregate_path}")

    csv_path = Path(args.aggregate_csv) if args.aggregate_csv else base_dir / "aggregate.csv"
    _write_csv(aggregate, csv_path)
    print(f"[Batch] Aggregate CSV written: {csv_path}")

    print("\n=== Batch Comparison (key columns) ===")
    for row in aggregate["comparison"]:
        print(f"{row['mode']:10s} success={row['success_count']}/{row['n_examples']} robust_acc={row['robust_accuracy']:.4f} avg_queries={row['average_queries']:.2f}")


if __name__ == "__main__":  # pragma: no cover
    main()
