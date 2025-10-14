# core/objectives.py
# 目标函数与支配关系模块：定义多目标优化函数和支配关系
# 这个模块实现了SA-MOO算法的核心目标函数，包括对抗成功率、L2范数和L0范数
# 还包含了非支配排序算法，用于选择最优解

import math
import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Optional, Dict, Any
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from PIL import Image
from io import BytesIO
from ..config.config import (
    ENABLE_REAL_WORLD_ROBUSTNESS,
    JPEG_QUALITY,
    ENABLE_RESIZE_PREPROCESSING,
    RESIZE_SCALE,
    DOMINANCE_CONFIG,
    DCT_LOW_FREQ_CONFIG,
)
from ..utils.dct_low_frequency import apply_dct_low_frequency_interference

_QUERY_COUNTER: Dict[str, int] = {"total": 0}


def reset_query_counter() -> None:
    """Reset the global query counter before a new attack run."""

    _QUERY_COUNTER["total"] = 0


def increment_query_counter(count: int) -> None:
    if count <= 0:
        return
    _QUERY_COUNTER["total"] = _QUERY_COUNTER.get("total", 0) + int(count)


def get_query_counter() -> int:
    """Return the total number of model queries issued in the current run."""

    return int(_QUERY_COUNTER.get("total", 0))

def evaluate_objectives_batch(
    population: List[Tuple[np.ndarray, np.ndarray]],
    original_rgb: np.ndarray,
    original_v: Optional[np.ndarray],
    true_label: int,
    model: nn.Module,
    targeted: bool,
    target_label: Optional[int],
    mode: str,
    dct_context: Optional[Dict[str, Any]] = None,
) -> List[Tuple[bool, float, float, int]]:
    """
    批量评估种群中每个个体的多目标函数值。

    该函数对整个种群进行批量推理，计算每个个体的对抗性、损失值、L2范数和L0范数。
    支持目标攻击和非目标攻击两种模式。

    Args:
        population: 种群列表，每个个体为(indices, perturbations)的元组
        original_rgb: 原始RGB图像，形状[32, 32, 3]，值范围[0, 1]
        original_v: 原始V通道（仅v_channel模式），形状[32, 32]，值范围[0, 1]
        true_label: 原始图像的真实类别索引
        model: 预训练的分类模型
        targeted: 是否为目标攻击
        target_label: 目标类别索引（仅targeted=True时使用）
        mode: 扰动模式 ("rgb_sim", "channel", "v_channel")

    Returns:
        目标值列表，每个元素为(is_adversarial, loss, l2_norm, l0_norm)的元组
    """
    pop_size = len(population)
    device = next(model.parameters()).device
    batch_tensors = []
    metric_images: List[np.ndarray] = []

    dct_cfg: Optional[Dict[str, Any]] = None
    dct_active = False
    edge_map = None

    if dct_context:
        dct_cfg = dct_context.get("config")
        edge_map = dct_context.get("edge_map")
        dct_active = bool(dct_context.get("active", False) and dct_cfg and dct_cfg.get("enabled", False))

    # 构建批量输入张量
    for indices, perturbations in population:
        if mode == "v_channel":
            # V通道扰动：修改HSV的V通道
            noise_v = np.zeros_like(original_v, dtype=np.float32)
            noise_v.flat[indices] = perturbations
            perturbed_v = np.clip(original_v + noise_v, 0.0, 1.0)
            original_hsv = rgb_to_hsv(original_rgb)
            perturbed_hsv = np.stack([
                original_hsv[:, :, 0],  # H通道保持不变
                original_hsv[:, :, 1],  # S通道保持不变
                perturbed_v              # V通道应用扰动
            ], axis=-1)
            perturbed_rgb = hsv_to_rgb(perturbed_hsv)
            perturbed_rgb = np.clip(perturbed_rgb, 0.0, 1.0).astype(np.float32)
        else:
            # RGB或通道扰动：直接在RGB空间应用扰动
            noise_rgb = np.zeros_like(original_rgb, dtype=np.float32)
            noise_rgb.flat[indices] = perturbations
            perturbed_rgb = np.clip(original_rgb + noise_rgb, 0.0, 1.0).astype(np.float32)

        if dct_active:
            perturbed_rgb = apply_dct_low_frequency_interference(
                original_rgb,
                perturbed_rgb,
                dct_cfg,
                edge_map=edge_map,
            )

        metrics_rgb = perturbed_rgb

        # 应用真实世界鲁棒性预处理（如果启用）
        model_rgb = metrics_rgb
        if ENABLE_REAL_WORLD_ROBUSTNESS:
            model_rgb = apply_real_world_preprocessing(metrics_rgb)

        # 转换为PyTorch张量格式 [C, H, W]
        tensor = torch.from_numpy(model_rgb).permute(2, 0, 1).unsqueeze(0).float().to(device)
        batch_tensors.append(tensor)
        metric_images.append(metrics_rgb)

    # 批量推理
    batch_input = torch.cat(batch_tensors, dim=0)
    with torch.no_grad():
        outputs = model(batch_input)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()

    objective_values = []
    for i in range(pop_size):
        pred_label = int(np.argmax(probs[i]))
        metrics_rgb = metric_images[i]
        delta_rgb = metrics_rgb - original_rgb
        indices, perturbations = population[i]
        is_adversarial = False
        loss = 0.0

        if targeted:
            # 目标攻击：检查是否预测为目标类别
            if target_label is None:
                raise ValueError("Target label must be provided for targeted attack.")
            is_adversarial = (pred_label == target_label)
            # 损失：负对数概率（最小化以最大化目标类别概率）
            log_probs = torch.log_softmax(outputs[i:i+1], dim=1).cpu().squeeze()
            loss = -float(log_probs[target_label].item())
        else:
            # 非目标攻击：检查是否预测错误
            is_adversarial = (pred_label != true_label)
            # 损失：正确类别的概率与最高错误类别概率的差（margin loss）
            p_correct = float(probs[i, true_label])
            p_others = probs[i].copy()
            p_others[true_label] = -np.inf  # 排除正确类别
            p_max_other = float(np.max(p_others))
            loss = p_correct - p_max_other

        # 计算L2和L0范数
        if mode == "v_channel":
            # V通道模式：扰动只在V通道上
            noise_full = np.zeros_like(original_v, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            if dct_active:
                l2_norm = float(np.linalg.norm(delta_rgb))
            else:
                l2_norm = float(np.linalg.norm(noise_full))  # L2范数
            l0_norm = int(np.count_nonzero(perturbations))  # 非零扰动数量
        else:
            # RGB模式：扰动在RGB空间
            noise_full = np.zeros_like(original_rgb, dtype=np.float32)
            noise_full.flat[indices] = perturbations
            if dct_active:
                l2_norm = float(np.linalg.norm(delta_rgb))
            else:
                l2_norm = float(np.linalg.norm(noise_full))  # L2范数
            if mode == "rgb_sim":
                # RGB同步模式：L0为被扰动的像素数量
                pert_reshaped = perturbations.reshape(-1, 3)  # (k, 3)
                l0_norm = int(np.sum(np.any(pert_reshaped != 0, axis=1)))
            else:  # "channel"
                # 通道模式：L0为被扰动的通道数量
                l0_norm = int(np.count_nonzero(perturbations))

        objective_values.append((is_adversarial, loss, l2_norm, l0_norm))

    increment_query_counter(pop_size)
    return objective_values

def apply_real_world_preprocessing(image: np.ndarray) -> np.ndarray:
    """
    对图像应用真实世界预处理：可选的JPEG压缩和Resize + Interpolation。

    Args:
        image: 输入图像，形状[H, W, C]，值范围[0, 1]

    Returns:
        预处理后的图像，形状[H, W, C]，值范围[0, 1]
    """
    # 转换为PIL Image
    pil_image = Image.fromarray((image * 255).astype(np.uint8))

    # JPEG压缩（如果质量<100）
    if JPEG_QUALITY < 100:
        buffer = BytesIO()
        pil_image.save(buffer, format='JPEG', quality=JPEG_QUALITY)
        buffer.seek(0)
        pil_image = Image.open(buffer)

    # Resize + Interpolation（如果启用）
    if ENABLE_RESIZE_PREPROCESSING:
        original_size = pil_image.size  # (W, H)
        # 计算缩小后的尺寸
        new_size = (int(original_size[0] / RESIZE_SCALE), int(original_size[1] / RESIZE_SCALE))
        
        # 检查新尺寸是否有效
        if new_size[0] <= 0 or new_size[1] <= 0:
            print(f"Warning: Resize scale {RESIZE_SCALE} too large for {original_size} image, skipping resize preprocessing")
        else:
            # 缩小
            resized_image = pil_image.resize(new_size, Image.BILINEAR)
            # 放大回原尺寸
            pil_image = resized_image.resize(original_size, Image.BILINEAR)

    # 转换回numpy数组
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
            # 条件匹配但无明确偏好，继续评估下一条规则

    # 若规则未决，再尝试平局处理或兜底策略
    for metric_rule in config.get("tie_breakers", []):
        verdict = _compare_metric(metric_rule, metrics_a, metrics_b)
        if verdict != 0:
            return verdict

    return _legacy_compare(obj1, obj2)

def dominates(obj1: Tuple[bool, float, float, int], obj2: Tuple[bool, float, float, int]) -> bool:
    """
    判断obj1是否支配obj2（论文3.1节的支配关系定义）。

    支配关系定义：
    - 如果obj1是对抗性的而obj2不是，则obj1支配obj2
    - 如果两者都是对抗性的，则比较L2范数，较小的支配
    - 如果两者都不是对抗性的，则比较损失值，较小的支配

    Args:
        obj1: 第一个目标值元组 (is_adversarial, loss, l2_norm, l0_norm)
        obj2: 第二个目标值元组 (is_adversarial, loss, l2_norm, l0_norm)

    Returns:
        True如果obj1支配obj2，否则False
    """
    verdict = _compare_with_config(obj1, obj2, DOMINANCE_CONFIG)
    if verdict > 0:
        return True
    if verdict < 0:
        return False
    # 平局情况下按照反向比较确认，确保一致性
    reverse_verdict = _compare_with_config(obj2, obj1, DOMINANCE_CONFIG)
    return reverse_verdict < 0

def non_dominated_sort(objective_values: List[Tuple[bool, float, float, int]]) -> List[List[int]]:
    """
    执行非支配排序，返回各前沿的索引列表。

    非支配排序将种群分为多个前沿：
    - 第0前沿：非支配解（没有其他解支配它们）
    - 第1前沿：被第0前沿支配的解中非支配的，依此类推

    Args:
        objective_values: 所有个体的目标值列表

    Returns:
        前沿列表，每个前沿包含该前沿中个体的索引
    """
    pop_size = len(objective_values)
    fronts: List[List[int]] = [[]]  # 初始化第0前沿

    # domination_info[i] = {"n": 被多少个个体支配, "S": 支配的个体索引列表}
    domination_info = [{"n": 0, "S": []} for _ in range(pop_size)]

    # 计算支配关系
    for p in range(pop_size):
        for q in range(p + 1, pop_size):
            if dominates(objective_values[p], objective_values[q]):
                domination_info[p]["S"].append(q)
                domination_info[q]["n"] += 1
            elif dominates(objective_values[q], objective_values[p]):
                domination_info[q]["S"].append(p)
                domination_info[p]["n"] += 1

        # 如果没有被任何个体支配，加入第0前沿
        if domination_info[p]["n"] == 0:
            fronts[0].append(p)

    # 构建后续前沿
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