"""Reusable execution pipeline for the SA-MOO adversarial attack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import math
import numpy as np
import torch
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from PIL import Image
from skimage.color import deltaE_ciede2000, rgb2lab
from skimage.metrics import peak_signal_noise_ratio as psnr

import lpips

from ..config import config as cfg
from .dynamic_parameters import DynamicParameterScheduler
from .evolutionary_operators import crossover, initialize_population, mutate, selection
from .objectives import (
    apply_real_world_preprocessing,
    dominates,
    evaluate_objectives_batch,
    get_query_counter,
    non_dominated_sort,
    reset_query_counter,
)
from ..utils.dct_low_frequency import decompose_low_high_frequency
from ..utils.edge_guidance import EdgeGuidanceWeights, compute_edge_guidance_weights
from ..visualization.visualization import (
    init_visualization,
    save_adversarial_result,
    save_convergence_plot,
    save_delta_L_heatmap,
    save_edge_guidance_artifacts,
    save_noise_heatmap,
    update_visualization,
)


@dataclass
class AttackRunResult:
    """Container for attack outputs that other entrypoints can reuse."""

    success: bool
    best_objective: Tuple[bool, float, float, int]
    best_solution: Optional[Tuple[np.ndarray, np.ndarray]]
    final_rgb: Optional[np.ndarray]
    metrics: Dict[str, float]
    history: Dict[str, List[Optional[float]]]
    queries: int
    predicted_class: Optional[int]
    predicted_class_name: Optional[str]


def _maybe_compute_edge_guidance(
    original_rgb: np.ndarray,
    config: Dict[str, object],
    run_dir: Optional[Path],
    verbose: bool,
) -> Tuple[Optional[EdgeGuidanceWeights], Optional[np.ndarray]]:
    if not config.get("enabled", False):
        return None, None

    try:
        weights = compute_edge_guidance_weights(original_rgb, config)
    except Exception as exc:  # pragma: no cover - defensive guard
        if verbose:
            print(f"[Edge Guidance] Initialization failed: {exc}. Falling back to uniform sampling.")
        return None, None

    priority_map: Optional[np.ndarray] = None
    if weights and weights.pixel_weights is not None:
        try:
            priority_map = weights.pixel_weights.reshape(original_rgb.shape[0], original_rgb.shape[1])
        except Exception:
            priority_map = None

    semantic_cfg = config.get("semantic")
    artifacts = getattr(weights, "artifacts", None)
    if (
        isinstance(semantic_cfg, dict)
        and semantic_cfg.get("enabled", False)
        and artifacts
        and run_dir is not None
    ):
        try:
            save_edge_guidance_artifacts(artifacts, run_dir, prefix="semantic_guidance")
            if verbose:
                print("[Edge Guidance] Semantic guidance diagnostics saved.")
        except Exception as viz_exc:  # pragma: no cover - defensive guard
            if verbose:
                print(f"[Edge Guidance] Failed to save semantic diagnostics: {viz_exc}")

    return weights, priority_map


def run_samoo_attack(
    original_rgb: np.ndarray,
    original_v: Optional[np.ndarray],
    true_label: int,
    model: torch.nn.Module,
    class_names: Sequence[str],
    *,
    device: Optional[torch.device] = None,
    run_dir: Optional[Path] = None,
    time_str: Optional[str] = None,
    persist_metrics: Optional[Callable[[Dict[str, object]], None]] = None,
    verbose: bool = True,
    compute_visualization: Optional[bool] = None,
) -> AttackRunResult:
    """Execute the SA-MOO attack pipeline and return aggregated metrics."""

    reset_query_counter()

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_guidance_weights, edge_priority_map = _maybe_compute_edge_guidance(
        original_rgb, cfg.EDGE_GUIDANCE_CONFIG, run_dir, verbose
    )

    do_visualization = compute_visualization
    if do_visualization is None:
        do_visualization = cfg.ENABLE_VISUALIZATION
    if do_visualization:
        init_visualization(original_rgb)

    scheduler = DynamicParameterScheduler(
        cfg.DYNAMIC_PARAMETER_CONFIG,
        total_generations=cfg.NUM_GENERATIONS,
        is_targeted_attack=cfg.IS_TARGETED_ATTACK,
    )

    initial_params = scheduler.get_params(0)
    current_zero_sample_prob = float(initial_params.get("zero_sample_prob", cfg.ZERO_SAMPLE_PROB))
    current_crossover_prob = float(initial_params.get("crossover_prob", cfg.CROSSOVER_PROB))
    runtime_fixed_k = max(1, int(round(initial_params.get("fixed_k", cfg.FIXED_K))))

    fallback_start_pm = 0.2 if cfg.IS_TARGETED_ATTACK else 0.4
    mutation_initial = float(
        scheduler.get_value("mutation_probability", 0, fallback_start_pm) or fallback_start_pm
    )

    population = initialize_population(
        cfg.POPULATION_SIZE,
        runtime_fixed_k,
        cfg.PERTURBATION_MODE,
        zero_sample_prob=current_zero_sample_prob,
        edge_guidance=edge_guidance_weights,
    )

    best_solution: Optional[Tuple[np.ndarray, np.ndarray]] = None
    best_objective: Tuple[bool, float, float, int] = (False, float("inf"), float("inf"), float("inf"))
    history: Dict[str, List[Optional[float]]] = {"l2": [], "l0": []}

    if verbose:
        attack_type = "Targeted" if cfg.IS_TARGETED_ATTACK else "Non-Targeted"
        print("--- Starting Unified SA-MOO Attack ---")
        print(f"Perturbation Mode: {cfg.PERTURBATION_MODE}")
        print(f"Attack Type: {attack_type}")
        if cfg.IS_TARGETED_ATTACK:
            target_name = class_names[cfg.TARGET_CLASS_ID] if class_names else str(cfg.TARGET_CLASS_ID)
            print(f"Target Class: {target_name} ({cfg.TARGET_CLASS_ID})")
        print(f"Fixed K (L0 budget): {runtime_fixed_k}")
        print(f"Generations: {cfg.NUM_GENERATIONS}, Population Size: {cfg.POPULATION_SIZE}")
        print(f"Crossover Prob: {current_crossover_prob:.3f}")
        print(f"Zero-sample Prob: {current_zero_sample_prob:.3f}")
        print(f"Mutation Prob (initial): {mutation_initial:.3f}")
        if cfg.ENABLE_REAL_WORLD_ROBUSTNESS:
            resize_flag = "ENABLED" if cfg.ENABLE_RESIZE_PREPROCESSING else "DISABLED"
            print(
                "Real World Robustness: ENABLED "
                f"(JPEG Quality: {cfg.JPEG_QUALITY}, Resize: {resize_flag}, Scale: {cfg.RESIZE_SCALE})"
            )
        else:
            print("Real World Robustness: DISABLED")

    for generation in range(cfg.NUM_GENERATIONS):
        gen_params = scheduler.get_params(generation)

        mutation_prob = float(gen_params.get("mutation_probability", mutation_initial))
        current_crossover_prob = float(gen_params.get("crossover_prob", current_crossover_prob))
        current_zero_sample_prob = float(gen_params.get("zero_sample_prob", current_zero_sample_prob))
        runtime_fixed_k = max(1, int(round(gen_params.get("fixed_k", runtime_fixed_k))))

        offspring: List[Tuple[np.ndarray, np.ndarray]] = []
        while len(offspring) < cfg.POPULATION_SIZE:
            i1, i2 = np.random.choice(cfg.POPULATION_SIZE, 2, replace=False)
            parent1, parent2 = population[i1], population[i2]

            if np.random.rand() < current_crossover_prob:
                child1, child2 = crossover(
                    parent1,
                    parent2,
                    runtime_fixed_k,
                    cfg.PERTURBATION_MODE,
                    current_crossover_prob,
                )
                offspring.extend([child1, child2])
            else:
                offspring.extend([parent1, parent2])

        offspring = offspring[: cfg.POPULATION_SIZE]

        mutated_offspring = [
            mutate(
                individual,
                runtime_fixed_k,
                mutation_prob,
                cfg.PERTURBATION_MODE,
                current_zero_sample_prob,
                edge_guidance=edge_guidance_weights,
            )
            for individual in offspring
        ]

        combined = population + mutated_offspring

        dct_context: Optional[Dict[str, object]] = None
        if cfg.DCT_LOW_FREQ_CONFIG.get("enabled", False):
            max_generations = int(cfg.DCT_LOW_FREQ_CONFIG.get("max_generations", 0) or 0)
            dct_active = generation < max_generations if max_generations > 0 else True
            dct_context = {
                "config": cfg.DCT_LOW_FREQ_CONFIG,
                "active": dct_active,
            }
            if edge_priority_map is not None and cfg.EDGE_GUIDANCE_CONFIG.get("enabled", False):
                dct_context["edge_map"] = edge_priority_map

        objective_values = evaluate_objectives_batch(
            combined,
            original_rgb,
            original_v,
            true_label,
            model,
            cfg.IS_TARGETED_ATTACK,
            cfg.TARGET_CLASS_ID if cfg.IS_TARGETED_ATTACK else None,
            cfg.PERTURBATION_MODE,
            dct_context=dct_context,
        )

        population = selection(combined, objective_values, cfg.POPULATION_SIZE)

        current_values = evaluate_objectives_batch(
            population,
            original_rgb,
            original_v,
            true_label,
            model,
            cfg.IS_TARGETED_ATTACK,
            cfg.TARGET_CLASS_ID if cfg.IS_TARGETED_ATTACK else None,
            cfg.PERTURBATION_MODE,
            dct_context=dct_context,
        )

        fronts = non_dominated_sort(current_values)
        best_idx = fronts[0][0] if fronts and fronts[0] else 0

        if dominates(current_values[best_idx], best_objective):
            best_solution = population[best_idx]
            best_objective = current_values[best_idx]

        if best_objective[0]:
            history["l2"].append(best_objective[2])
            history["l0"].append(best_objective[3])
        else:
            history["l2"].append(None)
            history["l0"].append(None)

        if verbose:
            print(
                f"Gen {generation + 1}/{cfg.NUM_GENERATIONS} | PM: {mutation_prob:.3f} | "
                f"XO: {current_crossover_prob:.3f} | ZP: {current_zero_sample_prob:.3f} | "
                f"Best L2: {best_objective[2]:.2f}, L0: {best_objective[3]}, Success: {best_objective[0]}"
            )

        if do_visualization and (generation + 1) % max(1, cfg.VISUALIZE_INTERVAL) == 0:
            update_visualization(
                original_rgb,
                best_solution,
                best_objective,
                generation + 1,
                cfg.PERTURBATION_MODE,
                original_v,
            )

    if do_visualization:
        import matplotlib.pyplot as plt

        plt.ioff()
        plt.show()

    if best_solution is None:
        queries = get_query_counter()
        metrics_payload = {
            "success": False,
            "loss": float(best_objective[1]) if isinstance(best_objective[1], (int, float)) else math.inf,
            "l2": float(best_objective[2]) if isinstance(best_objective[2], (int, float)) else math.inf,
            "l0": int(best_objective[3]) if isinstance(best_objective[3], (int, float)) else math.inf,
            "lpips": None,
            "psnr": None,
            "mean_delta_e00": None,
            "max_delta_e00": None,
            "queries": queries,
            "timestamp": time_str,
            "perturbation_mode": cfg.PERTURBATION_MODE,
            "num_generations": cfg.NUM_GENERATIONS,
            "population_size": cfg.POPULATION_SIZE,
        }
        if persist_metrics:
            persist_metrics(metrics_payload)
        return AttackRunResult(
            success=False,
            best_objective=best_objective,
            best_solution=None,
            final_rgb=None,
            metrics=metrics_payload,
            history=history,
            queries=queries,
            predicted_class=None,
            predicted_class_name=None,
        )

    indices, perturbations = best_solution
    original_hsv = rgb_to_hsv(original_rgb)
    if cfg.PERTURBATION_MODE == "v_channel":
        noise_v = np.zeros_like(original_v, dtype=np.float32)
        noise_v.flat[indices] = perturbations
        final_v = np.clip(original_hsv[:, :, 2] + noise_v, 0.0, 1.0)
        final_hsv = np.stack([
            original_hsv[:, :, 0],
            original_hsv[:, :, 1],
            final_v,
        ], axis=-1)
        final_rgb = hsv_to_rgb(final_hsv)
        noise_for_display = noise_v
    else:
        noise_rgb = np.zeros_like(original_rgb, dtype=np.float32)
        noise_rgb.flat[indices] = perturbations
        final_rgb = np.clip(original_rgb + noise_rgb, 0.0, 1.0)
        noise_for_display = noise_rgb

    if cfg.DCT_LOW_FREQ_CONFIG.get("enabled", False) and run_dir is not None:
        try:
            low_delta, high_delta = decompose_low_high_frequency(
                original_rgb,
                final_rgb,
                cfg.DCT_LOW_FREQ_CONFIG,
            )
            save_noise_heatmap(np.linalg.norm(low_delta, axis=2), "Low Frequency L2", run_dir / "freq_low_heatmap.png")
            save_noise_heatmap(np.linalg.norm(high_delta, axis=2), "High Frequency L2", run_dir / "freq_high_heatmap.png")
            if verbose:
                print("[DCT] Saved low/high frequency heatmaps.")
        except Exception as freq_exc:  # pragma: no cover - defensive guard
            if verbose:
                print(f"[DCT] Failed to save frequency decomposition: {freq_exc}")

    psnr_value = psnr(original_rgb, final_rgb, data_range=1.0)

    loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
    orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
    pert_tensor = torch.from_numpy(final_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
    with torch.no_grad():
        lpips_val = float(loss_fn_vgg(orig_tensor, pert_tensor).item())

    original_lab = rgb2lab(original_rgb)
    perturbed_lab = rgb2lab(final_rgb)
    delta_e00 = deltaE_ciede2000(original_lab, perturbed_lab)
    mean_delta_e00 = float(np.mean(delta_e00))
    max_delta_e00 = float(np.max(delta_e00))

    delta_L = perturbed_lab[:, :, 0] - original_lab[:, :, 0]
    if run_dir is not None:
        save_delta_L_heatmap(delta_L, run_dir / "deltaL_heatmap.png")

    verification_rgb = final_rgb
    if cfg.ENABLE_REAL_WORLD_ROBUSTNESS:
        verification_rgb = apply_real_world_preprocessing(final_rgb)

    final_tensor = torch.from_numpy(verification_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        output_logits = model(final_tensor)
        predicted_idx = int(torch.argmax(output_logits, dim=1).item())

    predicted_name = class_names[predicted_idx] if class_names else str(predicted_idx)

    orig_tensor_for_probs = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        orig_logits = model(orig_tensor_for_probs)
        orig_probs = torch.softmax(orig_logits, dim=1).cpu().squeeze()
        pert_probs = torch.softmax(output_logits, dim=1).cpu().squeeze()

    true_class_name = class_names[true_label] if class_names else str(true_label)
    orig_true_prob = float(orig_probs[true_label].item() * 100)
    pert_true_prob = float(pert_probs[true_label].item() * 100)
    pert_max_prob = float(pert_probs[predicted_idx].item() * 100)

    if run_dir is not None:
        if cfg.PERTURBATION_MODE == "v_channel":
            save_noise_heatmap(noise_for_display, "Value", run_dir / "noise_V.png")
            noise_display = noise_for_display.copy()
        else:
            noise_l2 = np.linalg.norm(noise_for_display, axis=2)
            save_noise_heatmap(noise_l2, "RGB L2", run_dir / "noise_RGB_L2.png")
            noise_display = noise_l2.copy()

        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())

        save_adversarial_result(
            original_rgb,
            noise_display,
            final_rgb,
            true_class_name,
            predicted_name,
            best_objective[3],
            cfg.PERTURBATION_MODE,
            run_dir / "adversarial_result.png",
        )

        final_img_only_path = run_dir / "final_perturbed.png"
        Image.fromarray((final_rgb * 255).astype(np.uint8)).save(final_img_only_path, format="PNG", compress_level=0)

        if cfg.ENABLE_REAL_WORLD_ROBUSTNESS:
            preprocessed_rgb = apply_real_world_preprocessing(final_rgb)
            Image.fromarray((preprocessed_rgb * 255).astype(np.uint8)).save(
                run_dir / "final_perturbed_preprocessed.png",
                format="PNG",
                compress_level=0,
            )

        save_convergence_plot(history["l2"], history["l0"], run_dir / "convergence.png")

    metrics_payload = {
        "success": bool(best_objective[0]),
        "loss": float(best_objective[1]),
        "l2": float(best_objective[2]),
        "l0": int(best_objective[3]),
        "lpips": lpips_val,
        "psnr": psnr_value,
        "mean_delta_e00": mean_delta_e00,
        "max_delta_e00": max_delta_e00,
        "queries": get_query_counter(),
        "timestamp": time_str,
        "perturbation_mode": cfg.PERTURBATION_MODE,
        "num_generations": cfg.NUM_GENERATIONS,
        "population_size": cfg.POPULATION_SIZE,
        "true_prob_before": orig_true_prob,
        "true_prob_after": pert_true_prob,
        "predicted_prob": pert_max_prob,
        "true_label": true_label,
        "predicted_label": predicted_idx,
    }

    if persist_metrics:
        persist_metrics(metrics_payload)

    return AttackRunResult(
        success=bool(best_objective[0]),
        best_objective=best_objective,
        best_solution=best_solution,
        final_rgb=final_rgb,
        metrics=metrics_payload,
        history=history,
        queries=get_query_counter(),
        predicted_class=predicted_idx,
        predicted_class_name=predicted_name,
    )