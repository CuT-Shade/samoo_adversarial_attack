"""Parameter sweep runner for SA-MOO adversarial attack experiments.

This utility automates large batches of experiments by:
  * expanding a Cartesian product of parameter overrides defined in a YAML spec,
  * generating per-run configuration files derived from an optional base config,
  * launching `run_experiment.py` for every combination (with basic parallelism),
  * parsing the resulting `report.txt` artefacts to harvest key metrics, and
  * compiling an aggregate report (CSV + JSON) sorted by configurable priorities.

Spec template: see `sweep_configs/sweep_template.yaml` for a heavily commented guide.

Quick start::

    python run_parameter_sweep.py sweep_configs/my_sweep.yaml

The script creates two major outputs:

1. Experiment run folders placed under `<output_root>/sweep_<timestamp>/`, each
   containing the usual attack artefacts.
2. A sweep workspace under `sweep_runs/sweep_<timestamp>/` with:
   - `summary.csv` & `summary.json`: tabular results for every combination
   - `stdout/` & `stderr/`: raw console logs per run for debugging
   - `configs/`: generated YAML configs (unless `keep_configs` is false)
   - optional `harvest/`: collected artefacts (e.g., frequency heatmaps)

Key metrics extracted from `report.txt`:
    success, l2, l0, psnr, lpips, mean_delta_e00, max_delta_e00,
    orig_confidence, adv_confidence, confidence_drop

Run with `-h/--help` for CLI overrides (e.g., limiting runs, dry-run mode).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import shutil
import sys
import time
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

REPO_ROOT = Path(__file__).resolve().parent
PYTHON_EXECUTABLE = sys.executable
RUN_EXPERIMENT_SCRIPT = REPO_ROOT / "run_experiment.py"
SWEEP_ROOT = REPO_ROOT / "sweep_runs"

METRIC_KEYS = (
    "success",
    "l2",
    "l0",
    "psnr",
    "lpips",
    "mean_delta_e00",
    "max_delta_e00",
    "orig_confidence",
    "adv_confidence",
    "confidence_drop",
    "runtime_sec",
)

HEARTBEAT_INTERVAL_SEC = 60

REPORT_PATTERNS = {
    "success": re.compile(r"Attack Success:\s*(True|False)", re.IGNORECASE),
    "l2": re.compile(r"L2 Norm:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "l0": re.compile(r"L0 Norm:\s*(\d+)", re.IGNORECASE),
    "psnr": re.compile(r"PSNR:\s*([0-9]*\.?[0-9]+)\s*dB", re.IGNORECASE),
    "lpips": re.compile(r"LPIPS:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "mean_delta_e00": re.compile(r"Mean ΔE00:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "max_delta_e00": re.compile(r"Max ΔE00:\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    "confidence_pair": re.compile(
        r"Original class probability dropped from\s*([0-9]*\.?[0-9]+)%\s*to\s*([0-9]*\.?[0-9]+)%",
        re.IGNORECASE,
    ),
    "result_dir": re.compile(r"Results saved in:\s*(.+)")
}

BOOLEAN_TRUE = {"true", "1", "yes", "on", True}


class SweepError(RuntimeError):
    """Custom exception for sweep-specific failures."""


@dataclass
class RankingRule:
    key: str
    order: str = "desc"
    weight: float = 1.0

    def validate(self) -> None:
        if self.key not in METRIC_KEYS:
            raise SweepError(f"Unknown ranking metric: {self.key}")
        if self.order not in {"asc", "desc"}:
            raise SweepError(f"Ranking order must be 'asc' or 'desc', got {self.order}")
        if not isinstance(self.weight, (float, int)):
            raise SweepError(f"Ranking weight must be numeric, got {type(self.weight)}")


@dataclass
class ParameterChoice:
    value: Any
    extras: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None


@dataclass
class SweepSpec:
    spec_path: Path
    base_config: Optional[Path]
    overrides: Dict[str, Any]
    output_root: Path
    keep_configs: bool
    parameter_grid: Dict[str, List[ParameterChoice]]
    ranking: List[RankingRule]
    max_parallel: int
    resume: bool
    environment: Dict[str, str]
    harvest: Sequence[str]

    @staticmethod
    def from_yaml(path: Path) -> "SweepSpec":
        with open(path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        base_config = raw.get("base_config")
        base_config_path = (path.parent / base_config).resolve() if base_config else None
        if base_config_path and not base_config_path.exists():
            raise SweepError(f"Base config '{base_config_path}' does not exist")

        overrides = raw.get("overrides") or {}
        output_root = Path(raw.get("output_root", "attack_results")).resolve()
        keep_configs = bool(raw.get("keep_configs", True))

        raw_grid = raw.get("parameters") or {}
        if not raw_grid:
            raise SweepError("Parameter grid is empty; define at least one varying parameter")
        parameter_grid: Dict[str, List[ParameterChoice]] = {}
        for raw_key, raw_values in raw_grid.items():
            key = str(raw_key)
            if not isinstance(raw_values, Sequence) or isinstance(raw_values, (str, bytes)):
                raise SweepError(f"Parameter '{key}' must map to a list of choices")
            choices: List[ParameterChoice] = []
            for raw_choice in raw_values:
                choice = parse_parameter_choice(raw_choice)
                choices.append(choice)
            if not choices:
                raise SweepError(f"Parameter '{key}' must have at least one choice")
            parameter_grid[key] = choices

        ranking_rules = []
        ranking_section = raw.get("ranking") or {}
        for item in ranking_section.get("priority", []) or []:
            rule = RankingRule(
                key=str(item.get("key")),
                order=str(item.get("order", "desc")).lower(),
                weight=float(item.get("weight", 1.0)),
            )
            rule.validate()
            ranking_rules.append(rule)

        max_parallel = int(raw.get("max_parallel", 1))
        if max_parallel < 1:
            raise SweepError("max_parallel must be >= 1")

        resume_flag = bool(raw.get("resume", False))
        env_section = raw.get("environment") or {}
        env_updates = {str(k): str(v) for k, v in env_section.items()}

        harvest_patterns = raw.get("harvest") or []

        return SweepSpec(
            spec_path=path.resolve(),
            base_config=base_config_path,
            overrides=overrides,
            output_root=output_root,
            keep_configs=keep_configs,
            parameter_grid=parameter_grid,
            ranking=ranking_rules,
            max_parallel=max_parallel,
            resume=resume_flag,
            environment=env_updates,
            harvest=tuple(str(p) for p in harvest_patterns),
        )


@dataclass
class Combination:
    index: int
    selections: Dict[str, ParameterChoice]
    slug: str
    slug_safe: str


@dataclass
class RunResult:
    combination: Combination
    config_path: Path
    stdout_path: Path
    stderr_path: Path
    status: str
    metrics: Dict[str, Optional[float]]
    run_dir: Optional[Path]
    error_message: Optional[str] = None
    raw_stdout: str = ""
    raw_stderr: str = ""
    composite_score: Optional[float] = None

    def metric_row(self) -> Dict[str, Any]:
        row = {
            "index": self.combination.index,
            "slug": self.combination.slug,
            "status": self.status,
            "run_dir": str(self.run_dir) if self.run_dir else None,
        }
        row.update(self.metrics)
        if self.composite_score is not None:
            row["composite_score"] = self.composite_score
        if self.error_message:
            row["error"] = self.error_message
        return row


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def dump_yaml(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, allow_unicode=True, sort_keys=False)


def deep_copy(data: Dict[str, Any]) -> Dict[str, Any]:
    return json.loads(json.dumps(data))


def set_nested(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split('.')
    cursor = config
    for part in parts[:-1]:
        if part not in cursor or not isinstance(cursor[part], dict):
            cursor[part] = {}
        cursor = cursor[part]
    cursor[parts[-1]] = value


def parse_parameter_choice(raw_choice: Any) -> ParameterChoice:
    if isinstance(raw_choice, ParameterChoice):
        return raw_choice

    if isinstance(raw_choice, dict):
        label = raw_choice.get("label")
        value_present = "value" in raw_choice
        value = raw_choice.get("value") if value_present else None

        extras = None
        for candidate_key in ("overrides", "extras", "set", "assign"):
            if candidate_key in raw_choice:
                extras = raw_choice.get(candidate_key)
                break

        if extras is not None and not isinstance(extras, dict):
            raise SweepError("Parameter overrides must be a mapping of key -> value")

        if extras is None:
            # Allow shorthand: {"value": true, "target_class_id": 5}
            inferred = {
                str(k): v for k, v in raw_choice.items()
                if k not in {"value", "label", "overrides", "extras", "set", "assign"}
            }
            extras = inferred if inferred else {}

        extras_str_keys = {str(k): v for k, v in extras.items()} if extras else {}

        if not value_present and not extras_str_keys:
            raise SweepError("Dictionary parameter choice must specify 'value' or additional overrides")

        return ParameterChoice(value=value, extras=extras_str_keys, label=label)

    return ParameterChoice(value=raw_choice)


def build_combinations(grid: Dict[str, List[ParameterChoice]]) -> List[Combination]:
    keys = list(grid.keys())
    values_product = list(itertools.product(*(grid[k] for k in keys)))
    combos: List[Combination] = []
    for idx, choices in enumerate(values_product, start=1):
        selection_map = {keys[i]: choices[i] for i in range(len(keys))}
        slug = _slugify(selection_map)
        slug_safe = _slug_to_filename(slug)
        combos.append(Combination(idx, selection_map, slug, slug_safe))
    return combos


def _slugify(selections: Dict[str, ParameterChoice]) -> str:
    segments = []
    for key, choice in selections.items():
        key_fragment = key.replace('.', '_')
        value_fragment = _format_value_for_slug(choice.label if choice.label is not None else choice.value)
        segments.append(f"{key_fragment}={value_fragment}")
    return "__".join(segments)


def _format_value_for_slug(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Remove trailing zeros for floats
        formatted = f"{value}"
        if isinstance(value, float):
            formatted = f"{value:.6g}"  # compact form
        return formatted.replace('.', 'p').replace('-', 'm')
    fragment = str(value).strip().replace(' ', '_')
    return re.sub(r"[^A-Za-z0-9_]+", "-", fragment)


def _slug_to_filename(slug: str, max_length: int = 120) -> str:
    if len(slug) <= max_length:
        return slug
    digest = hashlib.sha1(slug.encode("utf-8")).hexdigest()[:10]
    cutoff = max_length - len(digest) - 2
    truncated = slug[:cutoff]
    return f"{truncated}__{digest}"


def merge_base_config(base: Optional[Path]) -> Dict[str, Any]:
    if not base:
        return {}
    return deep_copy(load_yaml(base))


def apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    for key, value in overrides.items():
        if isinstance(value, dict):
            existing = config.get(key)
            if not isinstance(existing, dict):
                existing = {}
            config[key] = existing
            apply_overrides(existing, value)
        else:
            config[key] = value


def generate_config(base_config: Optional[Path], overrides: Dict[str, Any], combo: Combination,
                    sweep_output_root: Path) -> Dict[str, Any]:
    config = merge_base_config(base_config)
    apply_overrides(config, overrides)
    # Ensure our sweep output root is used for all runs
    config["output_dir"] = str(sweep_output_root)
    for dotted_key, choice in combo.selections.items():
        if choice.value is not None:
            set_nested(config, dotted_key, choice.value)
        for extra_key, extra_value in choice.extras.items():
            set_nested(config, str(extra_key), extra_value)
    return config


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def launch_experiment(
    config_path: Path,
    env_updates: Dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> Tuple[Any, float, Any, Any]:
    env = os.environ.copy()
    env.update(env_updates)
    command = [PYTHON_EXECUTABLE, str(RUN_EXPERIMENT_SCRIPT), str(config_path)]
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    stdout_handle = stdout_path.open("w", encoding="utf-8")
    stderr_handle = stderr_path.open("w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    except Exception:
        stdout_handle.close()
        stderr_handle.close()
        raise
    start_ts = time.time()
    return proc, start_ts, stdout_handle, stderr_handle


@dataclass
class ActiveProcess:
    process: Any
    combination: Combination
    config_path: Path
    stdout_path: Path
    stderr_path: Path
    start_ts: float
    config_data: Dict[str, Any]
    stdout_handle: Any
    stderr_handle: Any
    last_heartbeat: float


def wait_for_process(active: ActiveProcess) -> Tuple[int, str, str, float]:
    retcode = active.process.wait()
    duration = time.time() - active.start_ts
    try:
        active.stdout_handle.flush()
    finally:
        active.stdout_handle.close()
    try:
        active.stderr_handle.flush()
    finally:
        active.stderr_handle.close()
    stdout = active.stdout_path.read_text(encoding="utf-8", errors="ignore") if active.stdout_path.exists() else ""
    stderr = active.stderr_path.read_text(encoding="utf-8", errors="ignore") if active.stderr_path.exists() else ""
    return retcode, stdout, stderr, duration


def parse_report(report_path: Path) -> Dict[str, Optional[float]]:
    metrics: Dict[str, Optional[float]] = {key: None for key in METRIC_KEYS}
    if not report_path.exists():
        return metrics

    content = report_path.read_text(encoding="utf-8", errors="ignore")
    for key, pattern in REPORT_PATTERNS.items():
        if key == "confidence_pair" or key == "result_dir":
            continue
        match = pattern.search(content)
        if match:
            raw = match.group(1)
            if key == "success":
                metrics[key] = 1.0 if raw.lower() == "true" else 0.0
            elif key == "l0":
                metrics[key] = float(int(raw))
            else:
                metrics[key] = float(raw)

    conf_match = REPORT_PATTERNS["confidence_pair"].search(content)
    if conf_match:
        orig = float(conf_match.group(1)) / 100.0
        adv = float(conf_match.group(2)) / 100.0
        metrics["orig_confidence"] = orig
        metrics["adv_confidence"] = adv
        metrics["confidence_drop"] = orig - adv
    else:
        metrics["orig_confidence"] = None
        metrics["adv_confidence"] = None
        metrics["confidence_drop"] = None

    return metrics


def find_run_dir(stdout: str, sweep_output_root: Path, config: Dict[str, Any], start_ts: float) -> Optional[Path]:
    result_match = REPORT_PATTERNS["result_dir"].search(stdout)
    if result_match:
        candidate = Path(result_match.group(1).strip())
        if candidate.exists():
            return candidate

    # Fallback: search by prefix + recent modification time within sweep folder
    sweep_output_root.mkdir(parents=True, exist_ok=True)
    target_id = config.get("target_image_id")
    fixed_k = config.get("fixed_k")
    perturbation_mode = config.get("perturbation_mode")
    generations = config.get("num_generations")
    is_targeted = config.get("is_targeted_attack", False)
    prefix = f"img{target_id}_" + ("targeted" if is_targeted else "non_targeted")
    prefix += f"_k{fixed_k}_{perturbation_mode}_gen{generations}_"
    candidates = []
    for path in sweep_output_root.glob(f"{prefix}*"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, path))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    threshold = start_ts - 60  # 1 minute before launch start
    for mtime, path in candidates:
        if mtime >= threshold:
            return path
    return candidates[0][1]


def compute_scores(results: List[RunResult], rules: List[RankingRule]) -> None:
    if not rules:
        return

    # Initialise composite score
    for result in results:
        result.composite_score = 0.0

    for rule in rules:
        values = [res.metrics.get(rule.key) for res in results if res.metrics.get(rule.key) is not None]
        if not values:
            continue
        min_val, max_val = min(values), max(values)
        span = max_val - min_val
        for res in results:
            raw_value = res.metrics.get(rule.key)
            if raw_value is None:
                continue
            if span <= 1e-12:
                score = 1.0
            else:
                if rule.order == "desc":
                    score = (raw_value - min_val) / span
                else:
                    score = (max_val - raw_value) / span
            res.composite_score = (res.composite_score or 0.0) + float(rule.weight) * score

    # Normalise composite score by total weight
    total_weight = sum(rule.weight for rule in rules)
    if total_weight <= 0:
        return
    for res in results:
        if res.composite_score is not None:
            res.composite_score /= total_weight


def sort_results(results: List[RunResult], rules: List[RankingRule]) -> List[RunResult]:
    if not rules:
        # Default: sort by success desc, confidence_drop desc, l2 asc
        def fallback_key(res: RunResult) -> Tuple:
            return (
                -(res.metrics.get("success") or 0.0),
                -(res.metrics.get("confidence_drop") or 0.0),
                res.metrics.get("l2") or float("inf"),
            )

        return sorted(results, key=fallback_key)

    def sort_key(res: RunResult) -> Tuple:
        keys: List[Any] = []
        for rule in rules:
            value = res.metrics.get(rule.key)
            if value is None:
                # Missing values sorted last
                keys.append(float("inf") if rule.order == "asc" else -float("inf"))
            else:
                keys.append(value if rule.order == "desc" else -value)
        score = res.composite_score if res.composite_score is not None else -float("inf")
        keys.append(score)
        return tuple(keys)

    return sorted(results, key=sort_key, reverse=True)


def save_summary(results: List[RunResult], sweep_workspace: Path) -> None:
    records = [res.metric_row() for res in results]
    csv_path = sweep_workspace / "summary.csv"
    json_path = sweep_workspace / "summary.json"

    fieldnames = ["index", "slug", "status", "run_dir", "composite_score"] + list(METRIC_KEYS) + ["error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)

    with open(json_path, "w", encoding="utf-8") as jsonfile:
        json.dump(records, jsonfile, ensure_ascii=False, indent=2)


def harvest_artifacts(run_dir: Path, harvest_patterns: Sequence[str], harvest_root: Path, slug: str) -> None:
    if not harvest_patterns or not run_dir.exists():
        return
    destination = harvest_root / slug
    destination.mkdir(parents=True, exist_ok=True)
    for pattern in harvest_patterns:
        for file in run_dir.glob(pattern):
            target = destination / file.name
            try:
                shutil.copy2(file, target)
            except OSError as exc:
                print(f"[Harvest] Failed to copy {file} -> {target}: {exc}")


def run_sweep(spec: SweepSpec, args: argparse.Namespace) -> List[RunResult]:
    sweep_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_workspace = SWEEP_ROOT / f"sweep_{sweep_id}"
    configs_dir = sweep_workspace / "configs"
    stdout_dir = sweep_workspace / "stdout"
    stderr_dir = sweep_workspace / "stderr"
    harvest_dir = sweep_workspace / "harvest"
    ensure_dirs(sweep_workspace, stdout_dir, stderr_dir)
    if spec.keep_configs:
        ensure_dirs(configs_dir)

    sweep_output_root = spec.output_root / f"sweep_{sweep_id}"
    sweep_output_root.mkdir(parents=True, exist_ok=True)

    combinations = build_combinations(spec.parameter_grid)
    rng = random.Random(args.random_seed)

    if args.max_random_runs is not None:
        if args.max_runs is not None:
            raise SweepError("--max-runs and --max-random-runs cannot be used together")
        if args.max_random_runs < 1:
            raise SweepError("--max-random-runs must be >= 1")
        if args.max_random_runs >= len(combinations):
            combinations = list(combinations)
            rng.shuffle(combinations)
        else:
            combinations = rng.sample(combinations, args.max_random_runs)
    elif args.max_runs is not None:
        if args.max_runs < 1:
            raise SweepError("--max-runs must be >= 1")
        combinations = combinations[: args.max_runs]

    if args.dry_run:
        print("[Dry run] The following combinations would be executed:")
        for combo in combinations:
            print(f"#{combo.index:03d} {combo.slug}")
        return []

    print(f"Launching sweep with {len(combinations)} combinations (max_parallel={spec.max_parallel})")

    results: List[RunResult] = []
    active_processes: List[ActiveProcess] = []

    pending: List[Combination] = list(combinations)
    combo_iter = iter(pending)
    completed_count = 0
    total_count = len(combinations)

    def print_progress(note: str = "") -> None:
        message = f"(正在进行:{len(active_processes)} 已完成:{completed_count} 总计划数:{total_count})"
        if note:
            message = f"{note} {message}"
        print(message)

    while True:
        # Launch new processes if capacity allows
        while len(active_processes) < spec.max_parallel:
            try:
                combo = next(combo_iter)
            except StopIteration:
                break

            generated_config = generate_config(spec.base_config, spec.overrides, combo, sweep_output_root)
            config_filename = f"{combo.index:03d}_{combo.slug_safe}.yaml"
            config_path = (configs_dir / config_filename) if spec.keep_configs else (sweep_workspace / config_filename)
            dump_yaml(generated_config, config_path)

            stdout_path = stdout_dir / f"{combo.index:03d}_{combo.slug_safe}.log"
            stderr_path = stderr_dir / f"{combo.index:03d}_{combo.slug_safe}.log"

            if spec.resume:
                existing_run_dir = find_run_dir_from_previous(sweep_output_root, generated_config)
                if existing_run_dir and (existing_run_dir / "report.txt").exists():
                    metrics = parse_report(existing_run_dir / "report.txt")
                    metrics["runtime_sec"] = None
                    result = RunResult(
                        combination=combo,
                        config_path=config_path,
                        stdout_path=stdout_path,
                        stderr_path=stderr_path,
                        status="reused",
                        metrics=metrics,
                        run_dir=existing_run_dir,
                    )
                    harvest_artifacts(existing_run_dir, spec.harvest, harvest_dir, combo.slug_safe)
                    results.append(result)
                    completed_count += 1
                    print(f"[{combo.index:03d}] Reused existing run at {existing_run_dir}")
                    continue

            proc, start_ts, stdout_handle, stderr_handle = launch_experiment(
                config_path, spec.environment, stdout_path, stderr_path
            )
            active_processes.append(ActiveProcess(
                process=proc,
                combination=combo,
                config_path=config_path,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
                start_ts=start_ts,
                config_data=deep_copy(generated_config),
                stdout_handle=stdout_handle,
                stderr_handle=stderr_handle,
                last_heartbeat=start_ts,
            ))
            print_progress(f"[{combo.index:03d}] Launched (PID={proc.pid}) -> {combo.slug}")

        if not active_processes:
            break

        time.sleep(0.5)
        now = time.time()
        for active in list(active_processes):
            if now - active.last_heartbeat >= HEARTBEAT_INTERVAL_SEC:
                elapsed = now - active.start_ts
                print(
                    f"[{active.combination.index:03d}] Still running ({elapsed:.0f}s elapsed). "
                    f"Logs: {active.stdout_path}"
                )
                active.last_heartbeat = now
        finished: List[ActiveProcess] = []
        for active in active_processes:
            if active.process.poll() is not None:
                finished.append(active)

        for active in finished:
            retcode, stdout, stderr, runtime = wait_for_process(active)
            metrics = {key: None for key in METRIC_KEYS}
            metrics["runtime_sec"] = runtime

            run_dir = find_run_dir(stdout, sweep_output_root, active.config_data, active.start_ts)

            if retcode == 0 and run_dir:
                report_metrics = parse_report(run_dir / "report.txt")
                report_metrics.pop("runtime_sec", None)
                metrics.update(report_metrics)
                status = "ok"
                error_message = None
                harvest_artifacts(run_dir, spec.harvest, harvest_dir, active.combination.slug_safe)
            else:
                status = "failed"
                error_message = stderr.strip() or stdout.strip() or f"Exit code {retcode}"
                run_dir = run_dir if run_dir and run_dir.exists() else None

            result = RunResult(
                combination=active.combination,
                config_path=active.config_path,
                stdout_path=active.stdout_path,
                stderr_path=active.stderr_path,
                status=status,
                metrics=metrics,
                run_dir=run_dir,
                error_message=error_message,
                raw_stdout=stdout,
                raw_stderr=stderr,
            )
            results.append(result)
            active_processes.remove(active)
            completed_count += 1
            print_progress(f"[{active.combination.index:03d}] Completed (status={status}, runtime={runtime:.1f}s)")

    compute_scores(results, spec.ranking)
    ordered = sort_results(results, spec.ranking)
    save_summary(ordered, sweep_workspace)

    # Persist spec snapshot for reproducibility
    spec_snapshot_path = sweep_workspace / "sweep_spec.yaml"
    shutil.copy2(spec.spec_path, spec_snapshot_path)

    print(f"Sweep finished: results saved under {sweep_workspace}")
    return ordered


def find_run_dir_from_previous(sweep_output_root: Path, config: Dict[str, Any]) -> Optional[Path]:
    sweep_output_root.mkdir(parents=True, exist_ok=True)
    target_id = config.get("target_image_id")
    fixed_k = config.get("fixed_k")
    perturbation_mode = config.get("perturbation_mode")
    generations = config.get("num_generations")
    is_targeted = config.get("is_targeted_attack", False)
    prefix = f"img{target_id}_" + ("targeted" if is_targeted else "non_targeted")
    prefix += f"_k{fixed_k}_{perturbation_mode}_gen{generations}_"
    candidates = list(sweep_output_root.glob(f"{prefix}*"))
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch parameter sweep for SA-MOO attacks")
    parser.add_argument("spec", type=Path, help="Path to sweep specification YAML")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit number of combinations")
    parser.add_argument(
        "--max-random-runs",
        type=int,
        default=None,
        help="Randomly sample this many combinations (mutually exclusive with --max-runs)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Seed for random sampling when using --max-random-runs",
    )
    parser.add_argument("--dry-run", action="store_true", help="List combinations without executing")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    spec = SweepSpec.from_yaml(args.spec)
    try:
        run_sweep(spec, args)
    except SweepError as exc:
        print(f"[SweepError] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nSweep cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
