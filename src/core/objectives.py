import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from io import BytesIO
from config.config import ENABLE_REAL_WORLD_ROBUSTNESS, JPEG_QUALITY, ENABLE_RESIZE_PREPROCESSING, RESIZE_SCALE, DOMINANCE_CONFIG

def evaluate_objectives_batch(
    population: List[Tuple[np.ndarray, np.ndarray]],
    original_rgb: np.ndarray,
    original_v: Optional[np.ndarray],
    true_label: int,
    model: nn.Module,
    targeted: bool,
    target_label: Optional[int],
    mode: str
) -> List[Tuple[bool, float, float, int]]:
    pop_size = len(population)
    device = next(model.parameters()).device
    batch_tensors = []
    for indices, perturbations in population:
        if mode == "v_channel":
            noise_v = np.zeros_like(original_v, dtype=np.float32)
            noise_v.flat[indices] = perturbations
            perturbed_v = np.clip(original_v + noise_v, 0.0, 1.0)
            original_hsv = rgb_to_hsv(original_rgb)
            perturbed_hsv = np.stack([
                original_hsv[:, :, 0],
                original_hsv[:, :, 1],
                perturbed_v
            ], axis=-1)
            perturbed_rgb = hsv_to_rgb(perturbed_hsv)
            perturbed_rgb = np.clip(perturbed_rgb, 0.0, 1.0).astype(np.float32)
        else:
            noise_rgb = np.zeros_like(original_rgb, dtype=np.float32)
            noise_rgb.flat[indices] = perturbations
            perturbed_rgb = np.clip(original_rgb + noise_rgb, 0.0, 1.0).astype(np.float32)
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            perturbed_rgb = apply_real_world_preprocessing(perturbed_rgb)
        tensor = torch.from_numpy(perturbed_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        batch_tensors.append(tensor)
    batch_input = torch.cat(batch_tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch_input)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
    objective_values = []
    for i in range(pop_size):
        pred_label = int(np.argmax(probs[i]))
        is_adversarial = False
        loss = 0.0
        if targeted:
            if target_label is None:
                raise ValueError("Target label must be provided for targeted attack.")
            is_adversarial = (pred_label == target_label)
            log_probs = torch.log_softmax(outputs[i:i+1], dim=1).cpu().squeeze()
            loss = -float(log_probs[target_label].item())
        else:
            is_adversarial = (pred_label != true_label)
            p_correct = float(probs[i, true_label])
            p_others = probs[i].copy()
            p_others[true_label] = -np.inf
            p_max_other = float(np.max(p_others))
            loss = p_correct - p_max_other
        if mode == "v_channel":
            noise_full = np.zeros_like(original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            l2_norm = float(np.linalg.norm(noise_full))
            l0_norm = int(np.count_nonzero(perturbations))
        else:
            noise_full = np.zeros_like(original_rgb, dtype=np.float32)
            noise_rgb.flat[indices] = perturbations
            l2_norm = float(np.linalg.norm(noise_rgb))
            if mode == "rgb_sim":
                pert_reshaped = perturbations.reshape(-1, 3)
                l0_norm = int(np.sum(np.any(pert_reshaped != 0, axis=1)))
            else:
                l0_norm = int(np.count_nonzero(perturbations))
        objective_values.append((is_adversarial, loss, l2_norm, l0_norm))
    return objective_values

def apply_real_world_preprocessing(image: np.ndarray) -> np.ndarray:
    pil_image = Image.fromarray((image * 255).astype(np.uint8))
    if JPEG_QUALITY < 100:
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        buffer.seek(0)
        pil_image = Image.open(buffer)
    if ENABLE_RESIZE_PREPROCESSING:
        original_size = pil_image.size
        new_size = (int(original_size[0] / RESIZE_SCALE), int(original_size[1] / RESIZE_SCALE))
        if new_size[0] <= 0 or new_size[1] <= 0:
            print(f"Warning: Resize scale {RESIZE_SCALE} too large for {original_size} image, skipping resize preprocessing")
        else:
            resized_image = pil_image.resize(new_size, Image.BILINEAR)
            pil_image = resized_image.resize(original_size, Image.BILINEAR)
    processed_array = np.array(pil_image, dtype=np.float32) / 255.0
    return processed_array

def _objective_to_dict(obj: Tuple[bool, float, float, int]) -> Dict[str, Any]:
    l0_value = obj[3]
    if isinstance(l0_value, (int, np.integer)):
        l0_norm = int(l0_value)
    elif isinstance(l0_value, float) and math.isfinite(l0_value):
        l0_norm = int(round(l0_value))
    else:
        l0_norm = float(l0_value)
    return {
        "is_adversarial": bool(obj[0]),
        "loss": float(obj[1]),
        "l2_norm": float(obj[2]),
        "l0_norm": l0_norm
    }

def _match_condition(condition: Dict[str, Any], self_metrics: Dict[str, Any], other_metrics: Dict[str, Any]) -> bool:
    if not condition:
        return True
    for key, expected in condition.items():
        if key.startswith("other_"):
            metric = key[len("other_"):]
            if other_metrics.get(metric) != expected:
                return False
        else:
            if self_metrics.get(key) != expected:
                return False
    return True

def _normalize_scalar(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)

def _compare_metric(rule: Dict[str, Any], metrics_a: Dict[str, Any], metrics_b: Dict[str, Any]) -> int:
    name = rule.get("name")
    if not name:
        return 0
    if name not in metrics_a or name not in metrics_b:
        return 0
    goal = str(rule.get("goal", "min")).lower()
    tolerance = float(rule.get("tolerance", 0.0) or 0.0)
    value_a = _normalize_scalar(metrics_a[name])
    value_b = _normalize_scalar(metrics_b[name])
    diff = value_a - value_b
    if goal == "min":
        if diff < -tolerance:
            return 1
        if diff > tolerance:
            return -1
    elif goal == "max":
        if diff > tolerance:
            return 1
        if diff < -tolerance:
            return -1
    return 0

def _legacy_compare(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int]) -> int:
    is_adv1, loss1, l2_1, l0_1 = obj1
    is_adv2, loss2, l2_2, l0_2 = obj2
    if is_adv1 and not is_adv2:
        return 1
    if not is_adv1 and is_adv2:
        return -1
    if is_adv1 and is_adv2:
        if l2_1 < l2_2:
            return 1
        if l2_1 > l2_2:
            return -1
        if l0_1 < l0_2:
            return 1
        if l0_1 > l0_2:
            return -1
    if not is_adv1 and not is_adv2:
        if loss1 < loss2:
            return 1
        if loss1 > loss2:
            return -1
    return 0

def _compare_with_config(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int], config: Dict[str, Any]) -> int:
    metrics_a = _objective_to_dict(obj1)
    metrics_b = _objective_to_dict(obj2)
    for rule in config.get("rules", []):
        condition = rule.get("when", {})
        if _match_condition(condition, metrics_a, metrics_b):
            prefer = rule.get("prefer")
            if prefer == "self":
                return 1
            if prefer == "other":
                return -1
            for metric_rule in rule.get("metrics", []):
                verdict = _compare_metric(metric_rule, metrics_a, metrics_b)
                if verdict != 0:
                    return verdict
    for metric_rule in config.get("tie_breakers", []):
        verdict = _compare_metric(metric_rule, metrics_a, metrics_b)
        if verdict != 0:
            return verdict
    return _legacy_compare(obj1, obj2)

def dominates(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int]) -> bool:
    verdict = _compare_with_config(obj1, obj2, DOMINANCE_CONFIG)
    if verdict > 0:
        return True
    if verdict < 0:
        return False
    reverse_verdict = _compare_with_config(obj2, obj1, DOMINANCE_CONFIG)
    return reverse_verdict < 0

def non_dominated_sort(objective_values: List[Tuple[bool, float, float, int]]) -> List[List[int]]:
    pop_size = len(objective_values)
    fronts: List[List[int]] = [[]]
    domination_info = [{"n": 0, "S": []} for _ in range(pop_size)]
    for p in range(pop_size):
        for q in range(p + 1, pop_size):
            if dominates(objective_values[p], objective_values[q]):
                domination_info[p]["S"].append(q)
                domination_info[q]["n"] += 1
            elif dominates(objective_values[q], objective_values[p]):
                domination_info[q]["S"].append(p)
                domination_info[p]["n"] += 1
        if domination_info[p]["n"] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in domination_info[p]["S"]:
                domination_info[q]["n"] -= 1
                if domination_info[q]["n"] == 0:
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)
        else:
            break
    return fronts