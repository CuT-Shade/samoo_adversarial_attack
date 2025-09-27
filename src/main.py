import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import lpips
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from config.config import *

from utils.logger import Logger
from utils.edge_guidance import compute_edge_guidance_weights
from data.data_loader import load_target_image_and_model
from core.objectives import evaluate_objectives_batch, dominates, non_dominated_sort
from core.evolutionary_operators import (
    initialize_population, crossover, mutate, selection
)
from core.dynamic_parameters import DynamicParameterScheduler
from visualization.visualization import (
    init_visualization, update_visualization, save_noise_heatmap,
    save_delta_L_heatmap, save_adversarial_result, save_convergence_plot,
    save_edge_guidance_artifacts,
)

def main() -> None:

    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")
    warnings.filterwarnings("ignore", category=FutureWarning, module="torchvision.models._utils")

    time_str = datetime.now().strftime("%m%d%H%M")
    attack_type = "targeted" if IS_TARGETED_ATTACK else "non_targeted"
    run_name = f"img{TARGET_IMAGE_ID}_{attack_type}_k{FIXED_K}_{PERTURBATION_MODE}_gen{NUM_GENERATIONS}_{time_str}"
    run_dir = OUTPUT_DIR / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = Logger(run_dir / "report.txt")
    sys.stdout = logger 

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        original_rgb, original_v, true_label, model, class_names = load_target_image_and_model(
            TARGET_IMAGE_ID, MODEL_WEIGHTS_PATH, DATA_ROOT_DIR, PERTURBATION_MODE
        )
        edge_guidance_weights = None
        edge_guidance_status = "DISABLED"
        if EDGE_GUIDANCE_CONFIG.get("enabled", False):
            try:
                edge_guidance_weights = compute_edge_guidance_weights(original_rgb, EDGE_GUIDANCE_CONFIG)
                method = EDGE_GUIDANCE_CONFIG.get("method", "sobel")
                edge_guidance_status = f"ENABLED (method: {method})"
                semantic_cfg = EDGE_GUIDANCE_CONFIG.get("semantic")
                artifacts = getattr(edge_guidance_weights, "artifacts", None)
                if (
                    isinstance(semantic_cfg, dict)
                    and semantic_cfg.get("enabled", False)
                    and artifacts
                ):
                    try:
                        save_edge_guidance_artifacts(
                            artifacts,
                            run_dir,
                            prefix="semantic_guidance",
                        )
                        print("[Edge Guidance] Semantic guidance diagnostics saved.")
                    except Exception as viz_exc:
                        print(f"[Edge Guidance] Failed to save semantic diagnostics: {viz_exc}")
            except Exception as exc:
                print(f"[Edge Guidance] Initialization failed: {exc}. Falling back to uniform sampling.")
                edge_guidance_weights = None

        if ENABLE_VISUALIZATION:
            init_visualization(original_rgb)

        scheduler = DynamicParameterScheduler(
            DYNAMIC_PARAMETER_CONFIG,
            total_generations=NUM_GENERATIONS,
            is_targeted_attack=IS_TARGETED_ATTACK
        )

        initial_params = scheduler.get_params(0)
        zero_sample_initial = initial_params.get("zero_sample_prob")
        crossover_initial = initial_params.get("crossover_prob")
        fixed_k_initial = initial_params.get("fixed_k")

        current_zero_sample_prob = float(zero_sample_initial if zero_sample_initial is not None else ZERO_SAMPLE_PROB)
        current_crossover_prob = float(crossover_initial if crossover_initial is not None else CROSSOVER_PROB)
        runtime_fixed_k = int(round(fixed_k_initial)) if fixed_k_initial is not None else FIXED_K
        runtime_fixed_k = max(1, runtime_fixed_k)

        fallback_start_pm = 0.4 if not IS_TARGETED_ATTACK else 0.2
        initial_mutation_prob_raw = scheduler.get_value("mutation_probability", 0, fallback_start_pm)
        initial_mutation_prob = float(initial_mutation_prob_raw if initial_mutation_prob_raw is not None else fallback_start_pm)

        mutation_final_raw = scheduler.get_value("mutation_probability", NUM_GENERATIONS - 1, initial_mutation_prob)
        mutation_final = float(mutation_final_raw if mutation_final_raw is not None else initial_mutation_prob)
        crossover_final_raw = scheduler.get_value("crossover_prob", NUM_GENERATIONS - 1, current_crossover_prob)
        crossover_final = float(crossover_final_raw if crossover_final_raw is not None else current_crossover_prob)
        zero_sample_final_raw = scheduler.get_value("zero_sample_prob", NUM_GENERATIONS - 1, current_zero_sample_prob)
        zero_sample_final = float(zero_sample_final_raw if zero_sample_final_raw is not None else current_zero_sample_prob)
        fixed_k_final_raw = scheduler.get_value("fixed_k", NUM_GENERATIONS - 1, runtime_fixed_k)
        fixed_k_final = int(round(fixed_k_final_raw)) if fixed_k_final_raw is not None else runtime_fixed_k
        fixed_k_final = max(1, fixed_k_final)

        def format_range_float(start: float, end: float, precision: int = 3) -> str:
            if abs(end - start) < 1e-9:
                return f"{start:.{precision}f}"
            return f"{start:.{precision}f} -> {end:.{precision}f}"

        def format_range_int(start: int, end: int) -> str:
            if start == end:
                return str(start)
            return f"{start} -> {end}"

        population = initialize_population(
            POPULATION_SIZE,
            runtime_fixed_k,
            PERTURBATION_MODE,
            zero_sample_prob=current_zero_sample_prob,
            edge_guidance=edge_guidance_weights
        )

        best_sol = None
        best_obj = (False, float("inf"), float("inf"), float("inf"))

        history = {"l2": [], "l0": []}

        print("--- Starting Unified SA-MOO Attack ---")
        print(f"Perturbation Mode: {PERTURBATION_MODE}")
        print(f"Attack Type: {'Targeted' if IS_TARGETED_ATTACK else 'Non-Targeted'}")
        if IS_TARGETED_ATTACK:
            print(f"Target Class: {class_names[TARGET_CLASS_ID]} ({TARGET_CLASS_ID})")
        print(f"Fixed K (L0): {format_range_int(runtime_fixed_k, fixed_k_final)}")
        print(f"Generations: {NUM_GENERATIONS}, Population Size: {POPULATION_SIZE}")
        print(f"Crossover Prob: {format_range_float(current_crossover_prob, crossover_final)}")
        print(f"Zero-sample Prob: {format_range_float(current_zero_sample_prob, zero_sample_final)}")
        print(f"Mutation Prob: {format_range_float(initial_mutation_prob, mutation_final)}")
        print(f"Visualization: {'ENABLED' if ENABLE_VISUALIZATION else 'DISABLED'}")
        print(f"Real World Robustness: {'ENABLED' if ENABLE_REAL_WORLD_ROBUSTNESS else 'DISABLED'}")
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            print(f"  JPEG Quality: {JPEG_QUALITY}, Resize: {'ENABLED' if ENABLE_RESIZE_PREPROCESSING else 'DISABLED'}")
            if ENABLE_RESIZE_PREPROCESSING:
                print(f"  Resize Scale: {RESIZE_SCALE}")
        print(f"Edge Guidance: {edge_guidance_status}")

        for gen in range(NUM_GENERATIONS):
            gen_params = scheduler.get_params(gen)

            pm_override = gen_params.get("mutation_probability")
            if pm_override is not None:
                current_pm = float(pm_override)
            else:
                progress = gen / max(NUM_GENERATIONS - 1, 1)
                current_pm = max(0.001, fallback_start_pm * (1 - progress))

            crossover_override = gen_params.get("crossover_prob")
            if crossover_override is not None:
                current_crossover_prob = float(crossover_override)

            zero_sample_override = gen_params.get("zero_sample_prob")
            if zero_sample_override is not None:
                current_zero_sample_prob = float(zero_sample_override)

            fixed_k_override = gen_params.get("fixed_k")
            if fixed_k_override is not None:
                runtime_fixed_k = max(1, int(round(fixed_k_override)))

            offspring = []
            while len(offspring) < POPULATION_SIZE:

                i1, i2 = np.random.choice(POPULATION_SIZE, 2, replace=False)
                p1, p2 = population[i1], population[i2]

                if np.random.rand() < current_crossover_prob:
                    c1, c2 = crossover(p1, p2, runtime_fixed_k, PERTURBATION_MODE, current_crossover_prob)
                    offspring.extend([c1, c2])
                else:
                    offspring.extend([p1, p2])

            offspring = offspring[:POPULATION_SIZE]  # 确保子代数量正确

            mutated_offspring = [
                mutate(
                    ind,
                    runtime_fixed_k,
                    current_pm,
                    PERTURBATION_MODE,
                    current_zero_sample_prob,
                    edge_guidance=edge_guidance_weights,
                )
                for ind in offspring
            ]

            combined = population + mutated_offspring
            obj_vals = evaluate_objectives_batch(
                combined, original_rgb, original_v, true_label, model,
                IS_TARGETED_ATTACK, TARGET_CLASS_ID if IS_TARGETED_ATTACK else None,
                PERTURBATION_MODE
            )

            population = selection(combined, obj_vals, POPULATION_SIZE)

            current_obj = evaluate_objectives_batch(
                population, original_rgb, original_v, true_label, model,
                IS_TARGETED_ATTACK, TARGET_CLASS_ID if IS_TARGETED_ATTACK else None,
                PERTURBATION_MODE
            )

            fronts = non_dominated_sort(current_obj)
            best_idx = fronts[0][0] if fronts[0] else 0

            if dominates(current_obj[best_idx], best_obj):
                best_sol = population[best_idx]
                best_obj = current_obj[best_idx]

            if best_obj[0]:
                history["l2"].append(best_obj[2])
                history["l0"].append(best_obj[3])
            else:
                history["l2"].append(None)
                history["l0"].append(None)

            print(
                f"Gen {gen+1}/{NUM_GENERATIONS} | PM: {current_pm:.3f} | XO: {current_crossover_prob:.3f} | "
                f"ZP: {current_zero_sample_prob:.3f} | Best L2: {best_obj[2]:.2f}, "
                f"L0: {best_obj[3]}, Success: {best_obj[0]}"
            )

            if ENABLE_VISUALIZATION and (gen + 1) % VISUALIZE_INTERVAL == 0:
                update_visualization(
                    original_rgb, best_sol, best_obj, gen + 1,
                    PERTURBATION_MODE, original_v
                )

        if ENABLE_VISUALIZATION:
            plt.ioff()
            plt.show()

        print("\n--- Attack Completed ---")

        if best_sol is None:
            print("Failed to find an adversarial example.")
            return

        indices, perturbations = best_sol
        if PERTURBATION_MODE == "v_channel":
            noise_full = np.zeros_like(original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            original_hsv = rgb_to_hsv(original_rgb)
            final_v = np.clip(original_hsv[:, :, 2] + noise_full, 0.0, 1.0)
            final_hsv = np.stack([
                original_hsv[:, :, 0],
                original_hsv[:, :, 1],
                final_v
            ], axis=-1)
            final_rgb = hsv_to_rgb(final_hsv)
        else:
            noise_full = np.zeros_like(original_rgb, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            final_rgb = np.clip(original_rgb + noise_full, 0.0, 1.0)

        psnr_value = psnr(original_rgb, final_rgb, data_range=1.0)

        print("Final Best Solution Stats:")
        print(f"  Adversarial: {best_obj[0]}")
        print(f"  L2 Norm: {best_obj[2]:.4f}")
        print(f"  L0 Norm: {best_obj[3]}")
        print(f"  PSNR: {psnr_value:.2f} dB")

        loss_fn_vgg = lpips.LPIPS(net="vgg").to(device)
        orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        pert_tensor = torch.from_numpy(final_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device) * 2.0 - 1.0
        with torch.no_grad():
            lpips_val = float(loss_fn_vgg(orig_tensor, pert_tensor).item())
        print(f"  LPIPS: {lpips_val:.4f}")

        original_lab = rgb2lab(original_rgb)
        perturbed_lab = rgb2lab(final_rgb)
        delta_E00 = deltaE_ciede2000(original_lab, perturbed_lab)
        mean_delta_E00 = float(np.mean(delta_E00))
        max_delta_E00 = float(np.max(delta_E00))
        print("\n--- CIELAB Analysis ---")
        print(f"  Mean ΔE00: {mean_delta_E00:.3f}")
        print(f"  Max ΔE00: {max_delta_E00:.3f}")

        delta_L = perturbed_lab[:, :, 0] - original_lab[:, :, 0]
        save_delta_L_heatmap(delta_L, run_dir / "deltaL_heatmap.png")

        with torch.no_grad():
            verification_rgb = final_rgb
            if ENABLE_REAL_WORLD_ROBUSTNESS:
                from core.objectives import apply_real_world_preprocessing
                verification_rgb = apply_real_world_preprocessing(final_rgb)
            
            final_tensor = torch.from_numpy(verification_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
            output_logits = model(final_tensor)
            predicted_class_idx = int(torch.argmax(output_logits, dim=1).item())
            predicted_class_name = class_names[predicted_class_idx]

        print("\n--- Prediction Verification ---")
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            print(f"Note: Verification applied real-world preprocessing (JPEG quality: {JPEG_QUALITY}, resize: {'enabled' if ENABLE_RESIZE_PREPROCESSING else 'disabled'})")
        else:
            print("Note: Verification without preprocessing")

        orig_tensor = torch.from_numpy(original_rgb).permute(2, 0, 1).unsqueeze(0).to(device)
        with torch.no_grad():
            orig_logits = model(orig_tensor)
            orig_probs = torch.softmax(orig_logits, dim=1).cpu().squeeze()

        pert_probs = torch.softmax(output_logits, dim=1).cpu().squeeze()
        true_class_name = class_names[true_label]
        orig_true_prob = float(orig_probs[true_label].item() * 100)
        pert_true_prob = float(pert_probs[true_label].item() * 100)
        pert_max_prob = float(pert_probs[predicted_class_idx].item() * 100)

        if IS_TARGETED_ATTACK:
            target_class_name = class_names[TARGET_CLASS_ID]
            orig_target_prob = float(orig_probs[TARGET_CLASS_ID].item() * 100)
            pert_target_prob = float(pert_probs[TARGET_CLASS_ID].item() * 100)
            print(f"Original class: {true_class_name} ({true_label})")
            print(f"Target class: {target_class_name} ({TARGET_CLASS_ID})")
            print(f"Predicted class after attack: {predicted_class_name} ({predicted_class_idx})")
            print(f"Attack Success: {predicted_class_idx == TARGET_CLASS_ID}")
            print(f"Original class probability dropped from {orig_true_prob:.2f}% to {pert_true_prob:.2f}%")
            print(f"Target class probability increased from {orig_target_prob:.2f}% to {pert_target_prob:.2f}%")
            if predicted_class_idx == TARGET_CLASS_ID:
                print(f"Model now confidently predicts the target '{target_class_name}' with {pert_max_prob:.2f}% confidence.")
            else:
                print(f"Model failed to predict the target. It predicts '{predicted_class_name}' with {pert_max_prob:.2f}% confidence.")
        else:
            print(f"Original class: {true_class_name} ({true_label})")
            print(f"Predicted class after attack: {predicted_class_name} ({predicted_class_idx})")
            print(f"Attack Success: {predicted_class_idx != true_label}")
            print(f"Original class probability dropped from {orig_true_prob:.2f}% to {pert_true_prob:.2f}%")
            print(f"Model now confidently predicts '{predicted_class_name}' with {pert_max_prob:.2f}% confidence.")

        if PERTURBATION_MODE == "v_channel":
            save_noise_heatmap(noise_full, "Value", run_dir / "noise_V.png")
        else:
            noise_l2 = np.linalg.norm(noise_full, axis=2)
            save_noise_heatmap(noise_l2, "RGB L2", run_dir / "noise_RGB_L2.png")

        if PERTURBATION_MODE == "v_channel":
            noise_display = noise_full.copy()
        else:
            noise_display = noise_l2.copy()

        if noise_display.max() > noise_display.min():
            noise_display = (noise_display - noise_display.min()) / (noise_display.max() - noise_display.min())

        save_adversarial_result(
            original_rgb, noise_display, final_rgb,
            true_class_name, predicted_class_name, best_obj[3],
            PERTURBATION_MODE, run_dir / "adversarial_result.png"
        )

        final_img_only_path = run_dir / "final_perturbed.png"
        final_perturbed_uint8 = (final_rgb * 255).astype(np.uint8)
        img_pil = Image.fromarray(final_perturbed_uint8)
        img_pil.save(final_img_only_path, format="PNG", compress_level=0)

        if ENABLE_REAL_WORLD_ROBUSTNESS:
            from core.objectives import apply_real_world_preprocessing
            preprocessed_rgb = apply_real_world_preprocessing(final_rgb)
            preprocessed_uint8 = (preprocessed_rgb * 255).astype(np.uint8)
            preprocessed_path = run_dir / "final_perturbed_preprocessed.png"
            preprocessed_pil = Image.fromarray(preprocessed_uint8)
            preprocessed_pil.save(preprocessed_path, format="PNG", compress_level=0)
            print(f"Preprocessed image saved as: final_perturbed_preprocessed.png")

        save_convergence_plot(
            history["l2"], history["l0"], run_dir / "convergence.png"
        )

        print(f"\nResults saved in: {run_dir.absolute()}")
        print(f"Noise heatmap saved as: {'noise_V.png' if PERTURBATION_MODE == 'v_channel' else 'noise_RGB_L2.png'}")
        print(f"Final 32x32 perturbed image saved as: {final_img_only_path.name}")

    finally:
        sys.stdout = logger.terminal
        logger.close()

if __name__ == "__main__":
    main()