# core/evolutionary_operators.py
# 进化算子模块：实现遗传算法的初始化、交叉和变异操作
# 这个模块包含了SA-MOO算法的进化操作，支持不同的扰动模式
# 包括种群初始化、交叉算子和变异算子

import numpy as np
from typing import List, Tuple, Optional
from ..config.config import (
    ENABLE_CONTINUOUS_PERTURBATION,
    CONTINUOUS_LOWER_BOUND,
    CONTINUOUS_UPPER_BOUND,
    CONTINUOUS_DECIMAL_PLACES,
    ZERO_SAMPLE_PROB as DEFAULT_ZERO_SAMPLE_PROB,
    CROSSOVER_PROB as DEFAULT_CROSSOVER_PROB,
)
from .objectives import non_dominated_sort
from ..utils.edge_guidance import EdgeGuidanceWeights


def _sample_full_domain(
    mode: str,
    domain_size: int,
    sample_size: int,
    edge_guidance: Optional[EdgeGuidanceWeights],
) -> np.ndarray:
    """Sample indices from the full domain, optionally using edge guidance."""
    sample_size = min(sample_size, domain_size)
    if edge_guidance is not None and sample_size > 0:
        candidates = np.arange(domain_size, dtype=int)
        return edge_guidance.sample_without_replacement(mode, candidates, sample_size)
    return np.random.choice(domain_size, size=sample_size, replace=False)


def _sample_from_candidates(
    mode: str,
    candidates: np.ndarray,
    sample_size: int,
    edge_guidance: Optional[EdgeGuidanceWeights],
) -> np.ndarray:
    """Sample indices from a candidate set with optional guidance."""
    if candidates.size == 0 or sample_size <= 0:
        return np.empty(0, dtype=int)

    sample_size = min(sample_size, candidates.size)
    if edge_guidance is not None:
        return edge_guidance.sample_without_replacement(mode, candidates, sample_size)
    return np.random.choice(candidates, size=sample_size, replace=False)

def generate_perturbation_values(size: int, zero_sample_prob: float = DEFAULT_ZERO_SAMPLE_PROB) -> np.ndarray:
    """
    根据配置生成扰动值。

    如果启用连续扰动，从[lower_bound, upper_bound]中随机选择n位小数。
    否则，从{-1, 0, 1}中选择离散值。

    Args:
        size: 要生成的扰动值数量

    Returns:
        扰动值数组
    """
    if ENABLE_CONTINUOUS_PERTURBATION:
        # 连续扰动：从指定范围内随机选择
        values = np.random.uniform(CONTINUOUS_LOWER_BOUND, CONTINUOUS_UPPER_BOUND, size)
        # 四舍五入到指定小数位数
        values = np.round(values, CONTINUOUS_DECIMAL_PLACES)
        return values
    else:
        # 离散扰动：从{-1, 0, 1}中选择
        stay_prob = np.clip(zero_sample_prob, 0.0, 1.0)
        side_prob = (1 - stay_prob) / 2
        return np.random.choice([-1, 0, 1], size=size,
                               p=[side_prob, stay_prob, side_prob])

def initialize_population(
    pop_size: int,
    k: int,
    mode: str,
    zero_sample_prob: float = DEFAULT_ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    根据扰动模式初始化种群。

    为进化算法生成初始种群，每个个体包含扰动位置索引和扰动值。
    支持三种扰动模式，每种模式有不同的索引和值初始化策略。

    Args:
        pop_size: 种群大小（个体数量）
        k: 扰动预算（允许修改的最大数量）
        mode: 扰动模式
             - "v_channel": 扰动HSV的V通道
             - "rgb_sim": RGB同步扰动（像素级）
             - "channel": 独立通道扰动

    Returns:
        初始化后的种群列表，每个个体为(indices, perturbations)的元组
    """
    population = []
    total_pixels = 32 * 32  # CIFAR-10图像尺寸
    total_channels = total_pixels * 3  # RGB三通道

    for _ in range(pop_size):
        if mode == "v_channel":
            # V通道模式：选择k个像素位置，扰动值为-1,0,1
            indices = _sample_full_domain("v_channel", total_pixels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((indices, values))

        elif mode == "rgb_sim":
            # RGB同步模式：选择k个像素，每个像素的RGB三通道同时扰动
            pixel_indices = _sample_full_domain("rgb_sim", total_pixels, k, edge_guidance)
            # 为每个像素生成3个扰动值（RGB）
            values = generate_perturbation_values(k * 3, zero_sample_prob)
            # 将像素索引扩展为RGB索引：pixel_idx -> [pixel_idx*3, pixel_idx*3+1, pixel_idx*3+2]
            rgb_indices = np.repeat(pixel_indices * 3, 3) + np.tile([0, 1, 2], k)
            population.append((rgb_indices, values))

        elif mode == "channel":
            # 通道模式：选择k个独立通道位置进行扰动
            channel_indices = _sample_full_domain("channel", total_channels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((channel_indices, values))
        else:
            raise ValueError(f"Unknown PERTURBATION_MODE: {mode}")

    return population

def crossover_v_channel(ind1, ind2, k, crossover_prob: float = DEFAULT_CROSSOVER_PROB):
    """
    V通道专用交叉算子。

    实现论文中定义的交叉操作：从两个父代个体中交换部分扰动位置。
    只交换ind1有但ind2没有的位置对应的扰动值。

    Args:
        ind1: 父代个体1 (indices1, perturbations1)
        ind2: 父代个体2 (indices2, perturbations2)
        k: 扰动预算

    Returns:
        交叉后的两个子代个体 (child1, child2)
    """
    ma, da = ind1  # indices和perturbations of parent1
    mb, db = ind2

    # 安全检查：确保个体结构正确
    assert len(ma) == len(da), f"Individual p1 corrupted: len(ma)={len(ma)} != len(da)={len(da)}"
    assert len(mb) == len(db), f"Individual p2 corrupted: len(mb)={len(mb)} != len(db)={len(db)}"

    # 找到ind2有但ind1没有的像素位置
    set_ma = set(ma)
    set_mb = set(mb)
    u = list(set_mb - set_ma)  # ind2独有的位置

    if len(u) == 0:
        return ind1, ind2  # 无可交换位置，返回原个体

    # 计算最大交换数量：交叉概率 * k
    max_exchange = int(crossover_prob * k)
    exchange_count = min(max_exchange, len(u))

    # 随机选择要交换的位置
    a_indices = np.random.choice(len(ma), size=exchange_count, replace=False)
    b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

    # 构建子代1：移除选中的位置，添加来自ind2的位置
    new_ma = np.delete(ma, a_indices)
    new_b = np.array(u)[b_indices]
    new_ma = np.concatenate([new_ma, new_b])

    # 创建mb到db的索引映射，用于获取对应位置的扰动值
    mb_to_idx = {pixel: i for i, pixel in enumerate(mb)}
    indices_in_mb = [mb_to_idx[pixel] for pixel in new_b]  # 整数索引列表
    new_db_from_p2 = db[indices_in_mb]

    new_da = np.delete(da, a_indices)
    new_da = np.concatenate([new_da, new_db_from_p2])

    # 类似地构建子代2
    new_mb = np.delete(mb, b_indices)
    new_a = ma[a_indices]
    new_mb = np.concatenate([new_mb, new_a])

    ma_to_idx = {pixel: i for i, pixel in enumerate(ma)}
    indices_in_ma = [ma_to_idx[pixel] for pixel in new_a]
    new_da_from_p1 = da[indices_in_ma]

    new_db = np.delete(db, b_indices)
    new_db = np.concatenate([new_db, new_da_from_p1])

    return (new_ma, new_da), (new_mb, new_db)

def crossover(ind1, ind2, k, mode, crossover_prob: float = DEFAULT_CROSSOVER_PROB):
    """
    根据扰动模式选择合适的交叉算子。

    Args:
        ind1: 父代个体1
        ind2: 父代个体2
        k: 扰动预算
        mode: 扰动模式

    Returns:
        交叉后的两个子代个体
    """
    if mode == "v_channel":
        return crossover_v_channel(ind1, ind2, k, crossover_prob)
    elif mode == "rgb_sim":
        # RGB同步模式的交叉算子
        ma, da = ind1
        mb, db = ind2

        # 将RGB索引转换回像素索引
        pixel_ma = ma[::3] // 3  # 每3个连续索引对应一个像素
        pixel_mb = mb[::3] // 3

        set_ma = set(pixel_ma)
        set_mb = set(pixel_mb)
        u = list(set_mb - set_ma)  # ind2独有的像素

        if len(u) == 0:
            return ind1, ind2

        max_exchange = int(crossover_prob * k)
        exchange_count = min(max_exchange, len(u))

        a_indices = np.random.choice(len(pixel_ma), size=exchange_count, replace=False)
        b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

        # 构建新的像素索引
        new_pixel_ma = np.delete(pixel_ma, a_indices)
        new_b = np.array(u)[b_indices]
        new_pixel_ma = np.concatenate([new_pixel_ma, new_b])

        # 转换回RGB索引
        new_ma_rgb = np.repeat(new_pixel_ma * 3, 3) + np.tile([0, 1, 2], len(new_pixel_ma))

        # 处理扰动值
        da_pixels = da.reshape(-1, 3)  # (k, 3)
        new_da_pixels = np.delete(da_pixels, a_indices, axis=0)

        mb_to_idx = {p: i for i, p in enumerate(pixel_mb)}
        indices_in_mb = [mb_to_idx[p] for p in new_b]
        b_da_from_p2 = db.reshape(-1, 3)[indices_in_mb]

        new_da = np.concatenate([new_da_pixels, b_da_from_p2]).flatten()

        # 类似地处理子代2
        new_pixel_mb = np.delete(pixel_mb, b_indices)
        new_a = pixel_ma[a_indices]
        new_pixel_mb = np.concatenate([new_pixel_mb, new_a])
        new_mb_rgb = np.repeat(new_pixel_mb * 3, 3) + np.tile([0, 1, 2], len(new_pixel_mb))

        db_pixels = db.reshape(-1, 3)
        new_db_pixels = np.delete(db_pixels, b_indices, axis=0)

        indices_in_ma = [np.where(pixel_ma == p)[0][0] for p in new_a]
        a_db_from_p1 = da.reshape(-1, 3)[indices_in_ma]

        new_db = np.concatenate([new_db_pixels, a_db_from_p1]).flatten()

        return (new_ma_rgb, new_da), (new_mb_rgb, new_db)

    else:  # "channel" 模式
        ma, da = ind1
        mb, db = ind2

        set_ma = set(ma)
        set_mb = set(mb)
        u = list(set_mb - set_ma)

        if len(u) == 0:
            return ind1, ind2

        max_exchange = int(crossover_prob * k)
        exchange_count = min(max_exchange, len(u))

        a_indices = np.random.choice(len(ma), size=exchange_count, replace=False)
        b_indices = np.random.choice(len(u), size=exchange_count, replace=False)

        new_ma = np.delete(ma, a_indices)
        new_b = np.array(u)[b_indices]
        new_ma = np.concatenate([new_ma, new_b])

        new_da = np.delete(da, a_indices)

        mb_to_idx = {ch: i for i, ch in enumerate(mb)}
        indices_in_mb = [mb_to_idx[ch] for ch in new_b]
        b_da_from_p2 = db[indices_in_mb]

        new_da = np.concatenate([new_da, b_da_from_p2])

        new_mb = np.delete(mb, b_indices)
        new_a = ma[a_indices]
        new_mb = np.concatenate([new_mb, new_a])

        new_db = np.delete(db, b_indices)

        indices_in_ma = [np.where(ma == ch)[0][0] for ch in new_a]
        a_db_from_p1 = da[indices_in_ma]

        new_db = np.concatenate([new_db, a_db_from_p1])

        return (new_ma, new_da), (new_mb, new_db)

def mutate_v_channel(
    individual,
    k,
    pm,
    zero_sample_prob: float = DEFAULT_ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
):
    """
    V通道专用变异算子。

    通过添加新位置和移除现有位置来改变个体的扰动模式。

    Args:
        individual: 要变异的个体 (indices, perturbations)
        k: 扰动预算
        pm: 变异概率

    Returns:
        变异后的个体
    """
    ma, da = individual
    assert len(ma) == len(da), f"Mutate input corrupted: len(ma)={len(ma)} != len(da)={len(da)}"

    total_pixels = 32 * 32

    # 如果随机数大于变异概率，直接返回原个体
    if np.random.rand() >= pm:
        return individual

    # 找到未被扰动的像素位置
    set_ma = set(ma)
    t = list(set(range(total_pixels)) - set_ma)
    available_pixels = np.array(t, dtype=int)
    if available_pixels.size == 0:
        return individual

    # 计算要变异的数量：变异概率 * k，但至少1个，最多不超过可用位置
    remove_count = max(1, min(int(pm * k), len(ma), available_pixels.size))
    add_count = remove_count

    # 随机选择要移除和添加的位置
    a_indices = np.random.choice(len(ma), size=remove_count, replace=False)
    new_pixels = _sample_from_candidates("v_channel", available_pixels, add_count, edge_guidance)

    # 构建新的indices和perturbations
    new_ma = np.delete(ma, a_indices)
    new_ma = np.concatenate([new_ma, new_pixels])

    new_da = np.delete(da, a_indices)
    new_da_b = generate_perturbation_values(add_count, zero_sample_prob)
    new_da = np.concatenate([new_da, new_da_b])

    return (new_ma, new_da)

def mutate(
    individual,
    k,
    pm,
    mode,
    zero_sample_prob: float = DEFAULT_ZERO_SAMPLE_PROB,
    edge_guidance: Optional[EdgeGuidanceWeights] = None,
):
    """
    根据扰动模式选择合适的变异算子。

    Args:
        individual: 要变异的个体
        k: 扰动预算
        pm: 变异概率
        mode: 扰动模式

    Returns:
        变异后的个体
    """
    if mode == "v_channel":
        return mutate_v_channel(individual, k, pm, zero_sample_prob, edge_guidance)
    elif mode == "rgb_sim":
        ma, da = individual
        pixel_ma = ma[::3] // 3  # 像素索引
        total_pixels = 32 * 32
        set_ma = set(pixel_ma)
        t = list(set(range(total_pixels)) - set_ma)
        available_pixels = np.array(t, dtype=int)
        if available_pixels.size == 0:
            return individual

        set_size = max(1, min(int(pm * k), len(pixel_ma), available_pixels.size))

        a_indices = np.random.choice(len(pixel_ma), size=set_size, replace=False)
        new_pixels = _sample_from_candidates("rgb_sim", available_pixels, set_size, edge_guidance)

        new_pixel_ma = np.delete(pixel_ma, a_indices)
        new_pixel_ma = np.concatenate([new_pixel_ma, new_pixels])

        new_ma_rgb = np.repeat(new_pixel_ma * 3, 3) + np.tile([0, 1, 2], len(new_pixel_ma))

        da_pixels = da.reshape(-1, 3)
        keep_mask = np.ones(len(pixel_ma), dtype=bool)
        keep_mask[a_indices] = False
        new_da_kept = da_pixels[keep_mask].flatten()

        new_da_b = generate_perturbation_values(set_size * 3, zero_sample_prob)
        new_da = np.concatenate([new_da_kept, new_da_b])

        return (new_ma_rgb, new_da)

    else:  # "channel"
        ma, da = individual
        total_channels = 32 * 32 * 3
        set_ma = set(ma)
        t = list(set(range(total_channels)) - set_ma)
        available_channels = np.array(t, dtype=int)
        if available_channels.size == 0:
            return individual

        set_size = max(1, min(int(pm * k), len(ma), available_channels.size))

        a_indices = np.random.choice(len(ma), size=set_size, replace=False)
        new_channels = _sample_from_candidates("channel", available_channels, set_size, edge_guidance)

        new_ma = np.delete(ma, a_indices)
        new_ma = np.concatenate([new_ma, new_channels])

        new_da = np.delete(da, a_indices)
        new_da_b = generate_perturbation_values(set_size, zero_sample_prob)
        new_da = np.concatenate([new_da, new_da_b])

        return (new_ma, new_da)

def selection(combined_population, objective_values, pop_size):
    """
    基于非支配排序的选择操作。

    从父代和子代组合种群中选择下一代种群。

    Args:
        combined_population: 父代+子代种群
        objective_values: 对应的目标值
        pop_size: 选择后的种群大小

    Returns:
        选择的种群
    """
    fronts = non_dominated_sort(objective_values)
    selected = []

    for front in fronts:
        if len(selected) + len(front) <= pop_size:
            selected.extend([combined_population[i] for i in front])
        else:
            remaining = pop_size - len(selected)
            chosen = np.random.choice(front, remaining, replace=False)
            selected.extend([combined_population[i] for i in chosen])
            break

    return selected