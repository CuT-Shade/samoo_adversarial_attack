import numpy as np
from typing import List, Tuple, Optional
from config.config import (
    ENABLE_CONTINUOUS_PERTURBATION,
    CONTINUOUS_LOWER_BOUND,
    CONTINUOUS_UPPER_BOUND,
    CONTINUOUS_DECIMAL_PLACES,
    ZERO_SAMPLE_PROB as DEFAULT_ZERO_SAMPLE_PROB,
    CROSSOVER_PROB as DEFAULT_CROSSOVER_PROB,
)
from .objectives import non_dominated_sort
from utils.edge_guidance import EdgeGuidanceWeights

def _sample_full_domain(
    mode: str,
    domain_size: int,
    sample_size: int,
    edge_guidance: Optional[EdgeGuidanceWeights],
) -> np.ndarray:
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
    if candidates.size == 0 or sample_size <= 0:
        return np.empty(0, dtype=int)
    sample_size = min(sample_size, candidates.size)
    if edge_guidance is not None:
        return edge_guidance.sample_without_replacement(mode, candidates, sample_size)
    return np.random.choice(candidates, size=sample_size, replace=False)

def generate_perturbation_values(size: int, zero_sample_prob: float = DEFAULT_ZERO_SAMPLE_PROB) -> np.ndarray:
    if ENABLE_CONTINUOUS_PERTURBATION:
        values = np.random.uniform(CONTINUOUS_LOWER_BOUND, CONTINUOUS_UPPER_BOUND, size)
        values = np.round(values, CONTINUOUS_DECIMAL_PLACES)
        return values
    else:
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
    population = []
    total_pixels = 32 * 32  # CIFAR-10图像尺寸
    total_channels = total_pixels * 3  # RGB三通道
    for _ in range(pop_size):
        if mode == "v_channel":
            indices = _sample_full_domain("v_channel", total_pixels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((indices, values))
        elif mode == "rgb_sim":
            pixel_indices = _sample_full_domain("rgb_sim", total_pixels, k, edge_guidance)
            values = generate_perturbation_values(k * 3, zero_sample_prob)
            rgb_indices = np.repeat(pixel_indices * 3, 3) + np.tile([0, 1, 2], k)
            population.append((rgb_indices, values))
        elif mode == "channel":
            channel_indices = _sample_full_domain("channel", total_channels, k, edge_guidance)
            values = generate_perturbation_values(k, zero_sample_prob)
            population.append((channel_indices, values))
        else:
            raise ValueError(f"Unknown PERTURBATION_MODE: {mode}")
    return population

def crossover_v_channel(ind1, ind2, k, crossover_prob: float = DEFAULT_CROSSOVER_PROB):
    ma, da = ind1  # indices和perturbations of parent1
    mb, db = ind2
    assert len(ma) == len(da), f"Individual p1 corrupted: len(ma)={len(ma)} != len(da)={len(da)}"
    assert len(mb) == len(db), f"Individual p2 corrupted: len(mb)={len(mb)} != len(db)={len(db)}"
    set_ma = set(ma)
    set_mb = set(mb)
    u = list(set_mb - set_ma)  # ind2独有的位置
    if len(u) == 0:
        return ind1, ind2  # 无可交换位置，返回原个体
    max_exchange = int(crossover_prob * k)
    exchange_count = min(max_exchange, len(u))
    a_indices = np.random.choice(len(ma), size=exchange_count, replace=False)
    b_indices = np.random.choice(len(u), size=exchange_count, replace=False)
    new_ma = np.delete(ma, a_indices)
    new_b = np.array(u)[b_indices]
    new_ma = np.concatenate([new_ma, new_b])
    mb_to_idx = {pixel: i for i, pixel in enumerate(mb)}
    indices_in_mb = [mb_to_idx[pixel] for pixel in new_b]  # 整数索引列表
    new_db_from_p2 = db[indices_in_mb]
    new_da = np.delete(da, a_indices)
    new_da = np.concatenate([new_da, new_db_from_p2])
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
    if mode == "v_channel":
        return crossover_v_channel(ind1, ind2, k, crossover_prob)
    elif mode == "rgb_sim":
        ma, da = ind1
        mb, db = ind2
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
        new_pixel_ma = np.delete(pixel_ma, a_indices)
        new_b = np.array(u)[b_indices]
        new_pixel_ma = np.concatenate([new_pixel_ma, new_b])
        new_ma_rgb = np.repeat(new_pixel_ma * 3, 3) + np.tile([0, 1, 2], len(new_pixel_ma))
        da_pixels = da.reshape(-1, 3)  # (k, 3)
        new_da_pixels = np.delete(da_pixels, a_indices, axis=0)
        mb_to_idx = {p: i for i, p in enumerate(pixel_mb)}
        indices_in_mb = [mb_to_idx[p] for p in new_b]
        b_da_from_p2 = db.reshape(-1, 3)[indices_in_mb]
        new_da = np.concatenate([new_da_pixels, b_da_from_p2]).flatten()
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
    ma, da = individual
    assert len(ma) == len(da), f"Mutate input corrupted: len(ma)={len(ma)} != len(da)={len(da)}"
    total_pixels = 32 * 32
    if np.random.rand() >= pm:
        return individual
    set_ma = set(ma)
    t = list(set(range(total_pixels)) - set_ma)
    available_pixels = np.array(t, dtype=int)
    if available_pixels.size == 0:
        return individual
    remove_count = max(1, min(int(pm * k), len(ma), available_pixels.size))
    add_count = remove_count
    a_indices = np.random.choice(len(ma), size=remove_count, replace=False)
    new_pixels = _sample_from_candidates("v_channel", available_pixels, add_count, edge_guidance)
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