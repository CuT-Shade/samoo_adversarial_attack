"""Multi-objective hyperparameter search driver for SA-MOO attacks using Optuna.

This script orchestrates large-scale searches over the SA-MOO configuration space while
retaining the existing `run_experiment.py` execution pipeline. It supports:

* multi-objective optimisation of success rate, L0, LPIPS, and query count;
* persistent Optuna storage (SQLite by default) with resume-on-start;
* configurable search space covering evolutionary and semantic guidance hyper-parameters;
* repeat runs per trial to estimate success rate under stochastic operators;
* graceful interruption with immediate Pareto front reporting and artefact harvesting;
* automatic report generation (CSV + HTML/PNG plots) and solution gallery exports;
* hard constraints for query budget and L0 sparsity enforced per repeat.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import optuna
import yaml

try:  # Optional visualisation dependencies
    import plotly.express as px
except Exception:  # pragma: no cover - handled at runtime
    px = None  # type: ignore

try:  # Pandas is already part of project requirements; guard for clarity
    import pandas as pd
except Exception:  # pragma: no cover - handled at runtime
    pd = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parent
RUN_EXPERIMENT_SCRIPT = REPO_ROOT / "run_experiment.py"
DEFAULT_STORAGE = "sqlite:///samoo-attack-opt.db"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "optuna_results"
DEFAULT_BASE_CONFIG = REPO_ROOT / "complete_config.yaml"


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def deep_update(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``overrides`` into ``base`` and return a new dict."""

    result = json.loads(json.dumps(base)) if base else {}
    stack: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [(result, overrides or {})]
    while stack:
        dst, src = stack.pop()
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                stack.append((dst[key], value))
            else:
                dst[key] = value
    return result


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sanitize_number(value: Any, *, fallback: float) -> float:
    try:
        if value is None:
            raise ValueError
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            raise ValueError
        return numeric
    except Exception:
        return float(fallback)


@dataclass
class RepeatRecord:
    index: int
    config_path: Path
    metrics_path: Path
    stdout_path: Path
    stderr_path: Path
    duration_sec: float
    returncode: int
    metrics: Dict[str, Any]

@dataclass
class TrialSummary:
    trial_number: int
    overrides: Dict[str, Any]
    success_rate: float
    average_l0: float
    average_lpips: float
    average_queries: float
    repeats: List[RepeatRecord]
    trial_dir: Path
    constraints_ok: bool
    raw_average_l0: float
    raw_average_queries: float

    @property
    def objective_vector(self) -> Tuple[float, float, float, float]:
        return (
            float(self.success_rate),
            float(self.average_l0),
            float(self.average_lpips),
            float(self.average_queries),
        )


# ---------------------------------------------------------------------------
# Optuna objective wrapper
# ---------------------------------------------------------------------------


class AttackObjective:
    def __init__(
        self,
        base_config_path: Path,
        study_name: str,
        repeats: int,
        keep_runs: bool,
        results_root: Path,
        perturbation_mode: Optional[str],
        output_dir_override: Optional[Path],
        global_seed: Optional[int],
        max_queries: int,
        max_l0: int,
    ) -> None:
        self.base_config_path = base_config_path
        with base_config_path.open("r", encoding="utf-8") as fh:
            self.base_config: Dict[str, Any] = yaml.safe_load(fh) or {}
        self.study_name = study_name
        self.repeats = max(1, repeats)
        self.keep_runs = keep_runs
        self.results_root = ensure_dir(results_root)
        self.trials_root = ensure_dir(self.results_root / "trials")
        self.perturbation_mode = perturbation_mode
        self.output_dir_override = output_dir_override
        self.global_seed = global_seed
        self.max_queries = int(max_queries)
        self.max_l0 = int(max_l0)

    # ------------------------------------------------------------------
    # Primary Optuna hook
    # ------------------------------------------------------------------
    def __call__(self, trial: optuna.trial.Trial) -> Tuple[float, float, float, float]:
        overrides = self._sample_overrides(trial)
        summary = self._execute_trial(trial, overrides)
        trial.set_user_attr("overrides", overrides)
        trial.set_user_attr("aggregated_metrics", {
            "success_rate": summary.success_rate,
            "average_l0": summary.average_l0,
            "average_lpips": summary.average_lpips,
            "average_queries": summary.average_queries,
            "constraints_ok": summary.constraints_ok,
            "raw_average_l0": summary.raw_average_l0,
            "raw_average_queries": summary.raw_average_queries,
        })
        trial.set_user_attr("trial_dir", str(summary.trial_dir))
        trial.set_user_attr(
            "repeat_details",
            [
                {
                    "index": record.index,
                    "metrics": record.metrics,
                    "metrics_path": str(record.metrics_path),
                    "stdout_path": str(record.stdout_path),
                    "stderr_path": str(record.stderr_path),
                    "config_path": str(record.config_path),
                    "duration_sec": record.duration_sec,
                }
                for record in summary.repeats
            ],
        )
        return summary.objective_vector

    # ------------------------------------------------------------------
    # Configuration sampling helpers
    # ------------------------------------------------------------------
    def _sample_overrides(self, trial: optuna.trial.Trial) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}

        population = trial.suggest_int("population_size", 2, 12)
        generations = trial.suggest_int("num_generations", 200, 1200, step=50)
        fixed_k = trial.suggest_int("fixed_k", 4, 48, step=2)
        crossover_prob = trial.suggest_float("crossover_prob", 0.05, 0.6, step=0.01, log=False)
        zero_sample_prob = trial.suggest_float("zero_sample_prob", 0.05, 0.6, step=0.01, log=False)

        overrides.update(
            {
                "population_size": population,
                "num_generations": generations,
                "fixed_k": fixed_k,
                "crossover_prob": round(crossover_prob, 3),
                "zero_sample_prob": round(zero_sample_prob, 3),
            }
        )

        continuous_enabled = trial.suggest_categorical("continuous/enabled", [False, True])
        overrides["enable_continuous_perturbation"] = continuous_enabled
        if continuous_enabled:
            lower = trial.suggest_float("continuous/lower", -1.0, -0.05)
            upper = trial.suggest_float("continuous/upper", 0.05, 1.0)
            if upper - lower < 0.05:
                upper = min(1.0, lower + 0.05)
            decimals = trial.suggest_int("continuous/decimals", 1, 4)
            overrides.update(
                {
                    "continuous_lower_bound": round(lower, 3),
                    "continuous_upper_bound": round(upper, 3),
                    "continuous_decimal_places": decimals,
                }
            )
        else:
            overrides.update(
                {
                    "continuous_lower_bound": -1.0,
                    "continuous_upper_bound": 1.0,
                    "continuous_decimal_places": 2,
                }
            )

        # Edge guidance & semantic guidance parameters
        edge_enabled = trial.suggest_categorical("edge/enabled", [False, True])
        edge_overrides: Dict[str, Any] = {"enabled": edge_enabled}
        if edge_enabled:
            edge_overrides["method"] = trial.suggest_categorical(
                "edge/method", ["sobel", "scharr", "prewitt", "canny"]
            )
            edge_overrides["gaussian_sigma"] = round(
                trial.suggest_float("edge/gaussian_sigma", 0.0, 2.0), 3
            )
            edge_overrides["exponent"] = round(trial.suggest_float("edge/exponent", 1.0, 3.5), 3)
            edge_overrides["uniform_mix"] = round(trial.suggest_float("edge/uniform_mix", 0.0, 0.4), 3)
            edge_overrides["min_value"] = 1e-4

            semantic_enabled = trial.suggest_categorical("edge/semantic/enabled", [False, True])
            semantic_cfg: Dict[str, Any] = {
                "enabled": semantic_enabled,
                "fail_on_missing": False,
                "path": None,
            }
            if semantic_enabled:
                semantic_cfg.update(
                    {
                        "weight": round(trial.suggest_float("edge/semantic/weight", 0.1, 1.0), 3),
                        "exponent": round(trial.suggest_float("edge/semantic/exponent", 0.5, 2.5), 3),
                        "normalize": trial.suggest_categorical(
                            "edge/semantic/normalize", [True, False]
                        ),
                        "invert": trial.suggest_categorical("edge/semantic/invert", [False, True]),
                        "blur_sigma": round(
                            trial.suggest_float("edge/semantic/blur_sigma", 0.0, 1.5), 3
                        ),
                    }
                )
            edge_overrides["semantic"] = semantic_cfg
        overrides["edge_guidance"] = deep_update(self._edge_defaults(), edge_overrides)

        if self.perturbation_mode:
            overrides["perturbation_mode"] = self.perturbation_mode
        if self.output_dir_override:
            overrides["output_dir"] = str(self.output_dir_override)

        return overrides

    @staticmethod
    def _edge_defaults() -> Dict[str, Any]:
        return {
            "enabled": False,
            "method": "sobel",
            "gaussian_sigma": 0.8,
            "exponent": 1.5,
            "uniform_mix": 0.15,
            "min_value": 1e-4,
            "semantic": {
                "enabled": False,
                "fail_on_missing": False,
                "path": None,
                "weight": 0.5,
                "normalize": True,
                "invert": False,
                "blur_sigma": 0.0,
                "exponent": 1.0,
            },
        }

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------
    def _execute_trial(
        self,
        trial: optuna.trial.Trial,
        overrides: Dict[str, Any],
    ) -> TrialSummary:
        config_data = deep_update(self.base_config, overrides)
        trial_dir = self.trials_root / f"trial_{trial.number:05d}"
        if trial_dir.exists():
            shutil.rmtree(trial_dir, ignore_errors=True)
        ensure_dir(trial_dir)

        with (trial_dir / "overrides.yaml").open("w", encoding="utf-8") as fh:
            yaml.safe_dump(overrides, fh, allow_unicode=True, sort_keys=False)

        repeats: List[RepeatRecord] = []
        fallback_queries = (
            int(config_data.get("num_generations", 1000))
            * int(config_data.get("population_size", 2))
            * 2
        )

        for rep in range(self.repeats):
            repeat_dir = ensure_dir(trial_dir / f"repeat_{rep:02d}")
            repeat_config_path = repeat_dir / "config.yaml"
            with repeat_config_path.open("w", encoding="utf-8") as fh:
                yaml.safe_dump(config_data, fh, allow_unicode=True, sort_keys=False)

            metrics_path = repeat_dir / "metrics.json"
            seed = self._compute_seed(trial.number, rep)
            env = os.environ.copy()
            env["SA_MOO_METRICS_JSON"] = str(metrics_path)
            env["SA_MOO_SEED"] = str(seed)

            start = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, str(RUN_EXPERIMENT_SCRIPT), str(repeat_config_path)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            duration = time.perf_counter() - start

            stdout_path = repeat_dir / "stdout.log"
            stderr_path = repeat_dir / "stderr.log"
            stdout_path.write_text(proc.stdout or "", encoding="utf-8")
            stderr_path.write_text(proc.stderr or "", encoding="utf-8")

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Trial {trial.number} repeat {rep} failed with return code {proc.returncode}."
                )

            if not metrics_path.exists():
                raise FileNotFoundError(
                    f"Metrics file missing for trial {trial.number} repeat {rep}: {metrics_path}"
                )

            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics.setdefault("queries", fallback_queries)

            repeats.append(
                RepeatRecord(
                    index=rep,
                    config_path=repeat_config_path,
                    metrics_path=metrics_path,
                    stdout_path=stdout_path,
                    stderr_path=stderr_path,
                    duration_sec=duration,
                    returncode=proc.returncode,
                    metrics=metrics,
                )
            )

            if not self.keep_runs:
                run_dir = metrics.get("output_dir") or metrics.get("run_dir")
                if run_dir:
                    shutil.rmtree(Path(run_dir), ignore_errors=True)

        summary = self._aggregate(trial.number, overrides, repeats, trial_dir, fallback_queries)
        with (trial_dir / "aggregate.json").open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "trial_number": summary.trial_number,
                    "overrides": summary.overrides,
                    "success_rate": summary.success_rate,
                    "average_l0": summary.average_l0,
                    "average_lpips": summary.average_lpips,
                    "average_queries": summary.average_queries,
                    "constraints_ok": summary.constraints_ok,
                    "raw_average_l0": summary.raw_average_l0,
                    "raw_average_queries": summary.raw_average_queries,
                    "repeats": [
                        {
                            "index": record.index,
                            "metrics": record.metrics,
                            "metrics_path": str(record.metrics_path),
                            "stdout_path": str(record.stdout_path),
                            "stderr_path": str(record.stderr_path),
                            "config_path": str(record.config_path),
                            "duration_sec": record.duration_sec,
                        }
                        for record in summary.repeats
                    ],
                },
                fh,
                indent=2,
                ensure_ascii=False,
            )
        return summary

    def _compute_seed(self, trial_number: int, repeat_index: int) -> int:
        if self.global_seed is None:
            base = int(time.time())
        else:
            base = int(self.global_seed)
        return base + trial_number * 997 + repeat_index * 31

    def _aggregate(
        self,
        trial_number: int,
        overrides: Dict[str, Any],
        repeats: Sequence[RepeatRecord],
        trial_dir: Path,
        fallback_queries: int,
    ) -> TrialSummary:
        constraint_hits = 0
        for rec in repeats:
            metrics = dict(rec.metrics)
            queries_val = sanitize_number(metrics.get("queries"), fallback=float(fallback_queries))
            l0_val = sanitize_number(metrics.get("l0"), fallback=float(self.max_l0 + 1))
            violates = queries_val > float(self.max_queries) or l0_val > float(self.max_l0)
            metrics["queries"] = queries_val
            metrics["l0"] = l0_val
            if violates:
                constraint_hits += 1
                if metrics.get("success"):
                    metrics["success"] = False
                notes = metrics.get("notes")
                if not isinstance(notes, list):
                    notes = [] if notes is None else [notes]
                notes.append(
                    {
                        "type": "constraint",
                        "max_queries": self.max_queries,
                        "max_l0": self.max_l0,
                        "observed_queries": queries_val,
                        "observed_l0": l0_val,
                    }
                )
                metrics["notes"] = notes
            rec.metrics = metrics

        if constraint_hits:
            total = len(repeats)
            print(
                f"[Constraint] Trial {trial_number}: {constraint_hits}/{total} repeats exceeded "
                f"queries≤{self.max_queries} or L0≤{self.max_l0}; marked unsuccessful."
            )

        success_flags = [1 if rec.metrics.get("success") else 0 for rec in repeats]
        success_rate = sum(success_flags) / max(1, len(success_flags))

        def collect(key: str, *, default: float) -> List[float]:
            values: List[float] = []
            for rec in repeats:
                raw = rec.metrics.get(key)
                numeric = sanitize_number(raw, fallback=default)
                if rec.metrics.get("success") is False and key in {"l0", "lpips"}:
                    numeric = max(numeric, default)
                values.append(numeric)
            return values

        raw_avg_l0 = statistics.fmean(collect("l0", default=float(self.max_l0 + 1)))
        avg_lpips = statistics.fmean(collect("lpips", default=1.0))
        penalty_queries_default = float(max(self.max_queries * 2, fallback_queries))
        raw_avg_queries = statistics.fmean(collect("queries", default=penalty_queries_default))

        constraints_ok = raw_avg_l0 <= float(self.max_l0) and raw_avg_queries <= float(self.max_queries)

        penalised_success = success_rate if constraints_ok else 0.0
        penalised_l0 = raw_avg_l0 if constraints_ok else max(raw_avg_l0, float(self.max_l0) + 1.0)
        penalised_queries = raw_avg_queries if constraints_ok else max(
            raw_avg_queries, float(self.max_queries) + 1.0
        )

        return TrialSummary(
            trial_number=trial_number,
            overrides=overrides,
            success_rate=penalised_success,
            average_l0=penalised_l0,
            average_lpips=avg_lpips,
            average_queries=penalised_queries,
            repeats=list(repeats),
            trial_dir=trial_dir,
            constraints_ok=constraints_ok,
            raw_average_l0=raw_avg_l0,
            raw_average_queries=raw_avg_queries,
        )


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------


def _trial_rows(study: optuna.Study) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for trial in study.get_trials(deepcopy=False):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        aggregated = trial.user_attrs.get("aggregated_metrics")
        overrides = trial.user_attrs.get("overrides", {})
        if not aggregated:
            continue
        row = {
            "trial": trial.number,
            "success_rate": aggregated.get("success_rate"),
            "average_l0": aggregated.get("average_l0"),
            "average_lpips": aggregated.get("average_lpips"),
            "average_queries": aggregated.get("average_queries"),
            "raw_average_l0": aggregated.get("raw_average_l0"),
            "raw_average_queries": aggregated.get("raw_average_queries"),
            "constraints_ok": aggregated.get("constraints_ok", True),
            "overrides": overrides,
            "trial_dir": trial.user_attrs.get("trial_dir"),
        }
        rows.append(row)
    return rows


def generate_reports(study: optuna.Study, results_root: Path) -> None:
    rows = _trial_rows(study)
    if not rows:
        print("[Report] No completed trials to summarise yet.")
        return

    ensure_dir(results_root)
    csv_path = results_root / "trial_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "trial",
                "success_rate",
                "average_l0",
                "average_lpips",
                "average_queries",
                "raw_average_l0",
                "raw_average_queries",
                "constraints_ok",
                "overrides",
                "trial_dir",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    json_path = results_root / "trial_metrics.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    pareto_trials = study.best_trials
    pareto_rows = [row for row in rows if row["trial"] in {t.number for t in pareto_trials}]
    (results_root / "pareto_front.json").write_text(
        json.dumps(pareto_rows, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if px and pd and pareto_rows:
        try:
            df_fig = pd.DataFrame(rows)
            if not df_fig.empty:
                chrome_available = any(
                    shutil.which(candidate)
                    for candidate in [
                        "chrome",
                        "chrome.exe",
                        "google-chrome",
                        "chromium",
                        "chromium-browser",
                    ]
                )
                scatter3d = px.scatter_3d(
                    df_fig,
                    x="average_l0",
                    y="average_queries",
                    z="average_lpips",
                    color="success_rate",
                    hover_name="trial",
                    title="Trial distribution (L0 / Queries / LPIPS)",
                )
                scatter3d.write_html(str(results_root / "pareto_front_3d.html"))
                if chrome_available:
                    try:
                        scatter3d.write_image(str(results_root / "pareto_front_3d.png"))
                    except Exception as exc:  # pragma: no cover - optional dependency
                        print(f"[Report] Failed to export 3D PNG: {exc}")
                else:
                    print("[Report] Skipping 3D PNG export (Chrome executable not found).")

                scatter2d = px.scatter(
                    df_fig,
                    x="average_queries",
                    y="average_lpips",
                    color="success_rate",
                    hover_name="trial",
                    title="Queries vs LPIPS",
                )
                scatter2d.write_html(str(results_root / "pareto_front_2d.html"))
                if chrome_available:
                    try:
                        scatter2d.write_image(str(results_root / "pareto_front_2d.png"))
                    except Exception as exc:  # pragma: no cover
                        print(f"[Report] Failed to export 2D PNG: {exc}")
                else:
                    print("[Report] Skipping 2D PNG export (Chrome executable not found).")
        except Exception as exc:  # pragma: no cover
            print(f"[Report] Plotly visualisation failed: {exc}")

    solutions_dir = ensure_dir(results_root / "pareto_solutions")
    for trial in pareto_trials:
        details = trial.user_attrs.get("repeat_details", [])
        for detail in details:
            metrics = detail.get("metrics", {})
            run_dir = metrics.get("output_dir") or metrics.get("run_dir")
            if not run_dir:
                continue
            run_path = Path(run_dir)
            if not run_path.exists():
                continue
            for artifact in [
                "adversarial_result.png",
                "final_perturbed.png",
                "noise_RGB_L2.png",
                "noise_V.png",
            ]:
                src = run_path / artifact
                if not src.exists():
                    continue
                dest = solutions_dir / f"trial{trial.number:05d}_rep{detail['index']:02d}_{artifact}"
                shutil.copy2(src, dest)

    print(f"[Report] Trial summary saved to {csv_path}")
    if pareto_rows:
        print(
            "[Report] Pareto front updated with"
            f" {len(pareto_rows)} solutions (see pareto_front.json)."
        )


def print_current_best(study: optuna.Study, limit: int = 5) -> None:
    rows = _trial_rows(study)
    if not rows:
        print("[Summary] No completed trials yet.")
        return

    rows.sort(
        key=lambda row: (
            -(row["success_rate"] or 0.0),
            row["average_lpips"] or float("inf"),
            row["average_queries"] or float("inf"),
            row["average_l0"] or float("inf"),
        )
    )

    print("\n[Summary] Top current solutions:")
    for idx, row in enumerate(rows[:limit], start=1):
        print(
            f"  #{idx}: Trial {row['trial']} | success={row['success_rate']:.2f} "
            f"| L0={row['average_l0']:.2f} | LPIPS={row['average_lpips']:.4f} "
            f"| queries={row['average_queries']:.1f} | constraints_ok={row.get('constraints_ok')}"
        )


# ---------------------------------------------------------------------------
# CLI handling
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-objective hyperparameter optimisation for SA-MOO attacks",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=DEFAULT_BASE_CONFIG,
        help="Path to the base YAML configuration (default: complete_config.yaml)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="samoo-attack",
        help="Optuna study name (used for storage and result folders)",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=DEFAULT_STORAGE,
        help="Optuna storage URL (default: sqlite:///samoo-attack-opt.db)",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory to store optimisation artefacts and reports",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials to run",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Optional time budget in seconds",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Number of stochastic repeats per trial to estimate success rate",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel Optuna workers (subprocess runs remain serial per trial)",
    )
    parser.add_argument(
        "--perturbation-mode",
        type=str,
        default=None,
        choices=["rgb_sim", "channel", "v_channel", None],
        help="Override perturbation mode for all trials",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Override output directory for attack artefacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Base seed for reproducibility across trials",
    )
    parser.add_argument(
        "--cleanup-runs",
        action="store_true",
        help="Remove per-trial attack output directories after harvesting metrics",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=10000,
        help="Upper bound on query count; repeats exceeding此会被标记为失败",
    )
    parser.add_argument(
        "--max-l0",
        type=int,
        default=30,
        help="Upper bound on L0 sparsity; repeats超过此阈值将被视为失败",
    )
    return parser.parse_args(argv)


def create_study(args: argparse.Namespace) -> optuna.Study:
    return optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        directions=["maximize", "minimize", "minimize", "minimize"],
        load_if_exists=True,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    study = create_study(args)
    objective = AttackObjective(
        base_config_path=args.base_config,
        study_name=args.study_name,
        repeats=args.repeats,
        keep_runs=not args.cleanup_runs,
        results_root=args.results_root / args.study_name,
        perturbation_mode=args.perturbation_mode,
        output_dir_override=args.output_dir,
        global_seed=args.seed,
        max_queries=args.max_queries,
        max_l0=args.max_l0,
    )

    print(
        f"[Optuna] Study '{args.study_name}' with storage {args.storage}"
        f" | repeats={args.repeats} | n_trials={args.n_trials}"
    )

    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout,
            n_jobs=args.n_jobs,
            catch=(RuntimeError, FileNotFoundError),
        )
    except KeyboardInterrupt:
        print("\n[Optuna] Interrupted by user. Generating current reports...")
    finally:
        generate_reports(study, objective.results_root)
        print_current_best(study)


if __name__ == "__main__":
    main()
