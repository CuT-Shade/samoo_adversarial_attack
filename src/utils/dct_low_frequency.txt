"""Utility functions for DCT-based low-frequency perturbation control."""

from __future__ import annotations

import numpy as np
from functools import lru_cache
from typing import Dict, Any, Optional

Array = np.ndarray


@lru_cache(maxsize=None)
def _dct_matrix(n: int) -> Array:
    """Return an orthonormal DCT-II transform matrix of size n×n."""
    if n <= 0:
        raise ValueError("DCT dimension must be positive.")

    k = np.arange(n, dtype=np.float32).reshape(-1, 1)
    i = np.arange(n, dtype=np.float32).reshape(1, -1)
    factor = np.pi / (2.0 * n)
    mat = np.cos((2.0 * i + 1.0) * k * factor)
    mat *= np.sqrt(2.0 / n)
    mat[0, :] /= np.sqrt(2.0)
    return mat.astype(np.float32)


def _dct2(data: Array) -> Array:
    """Compute the 2-D DCT-II with orthonormal scaling."""
    h, w = data.shape
    m_h = _dct_matrix(h)
    m_w = _dct_matrix(w)
    return m_h @ data @ m_w.T


def _idct2(coeffs: Array) -> Array:
    """Compute the inverse (2-D DCT-III) for orthonormal DCT-II."""
    h, w = coeffs.shape
    m_h = _dct_matrix(h)
    m_w = _dct_matrix(w)
    return m_h.T @ coeffs @ m_w


def _build_frequency_mask(shape: tuple[int, int], config: Dict[str, Any]) -> Array:
    """Create a low-frequency mask with optional smooth roll-off."""
    h, w = shape
    keep_ratio = float(config.get("keep_ratio", 0.25))
    keep_ratio = float(np.clip(keep_ratio, 0.0, 1.0))
    keep_rows = config.get("keep_rows")
    keep_cols = config.get("keep_cols")

    rows = int(config.get("min_rows", 1))
    cols = int(config.get("min_cols", 1))

    if keep_rows is not None:
        rows = max(rows, min(h, int(keep_rows)))
    else:
        rows = max(rows, int(np.ceil(h * keep_ratio)))

    if keep_cols is not None:
        cols = max(cols, min(w, int(keep_cols)))
    else:
        cols = max(cols, int(np.ceil(w * keep_ratio)))

    mask = np.zeros((h, w), dtype=np.float32)
    mask[:rows, :cols] = 1.0

    decay = config.get("decay", None)
    if decay is not None and decay > 0:
        # Apply a cosine decay from the retained corner outward to soften edges
        y = np.linspace(0.0, 1.0, rows, endpoint=True, dtype=np.float32)
        x = np.linspace(0.0, 1.0, cols, endpoint=True, dtype=np.float32)
        window_y = np.power(np.cos(0.5 * np.pi * y), decay)
        window_x = np.power(np.cos(0.5 * np.pi * x), decay)
        mask_block = np.outer(window_y, window_x)
        mask[:rows, :cols] = mask_block

    return mask


def apply_dct_low_frequency_interference(
    original: Array,
    perturbed: Array,
    config: Dict[str, Any],
    edge_map: Optional[Array] = None,
) -> Array:
    """Constrain perturbations to the low-frequency DCT band.

    The function projects the perturbation (difference between ``perturbed`` and
    ``original``) to the DCT domain, keeps the low-frequency coefficients per the
    configuration, optionally boosts them, and reconstructs a filtered
    perturbation. The final image is clipped into ``[0, 1]``.

    Args:
        original: Reference RGB image in ``[0, 1]`` with shape ``[H, W, 3]``.
        perturbed: Candidate RGB image in ``[0, 1]`` with shape ``[H, W, 3]``.
        config: Configuration dictionary. Supported keys:
            - ``enabled`` (bool): Guard; should be ``True`` to activate.
            - ``keep_ratio`` (float): Fraction of low frequencies (per axis) to
              preserve if explicit counts are not provided.
            - ``keep_rows`` / ``keep_cols`` (int): Explicit number of rows/cols
              to retain from the DCT corner.
            - ``min_rows`` / ``min_cols`` (int): Minimum rows/cols to retain.
            - ``decay`` (float): Optional cosine decay exponent for a smooth mask.
            - ``blend`` (float): In ``[0, 1]``. ``1`` replaces perturbation with
              the filtered version, ``0`` keeps the original perturbation.
            - ``boost`` (float): Multiplier applied to retained coefficients.
            - ``channel_weights`` (list[float]): Optional per-channel blend
              weights in ``[0, 1]``.
            - ``epsilon`` (float): If total perturbation energy is below this
              threshold, the original image is returned early.

    Returns:
        Filtered RGB image with values in ``[0, 1]``.
    """
    if not config.get("enabled", False):
        return np.clip(perturbed, 0.0, 1.0).astype(np.float32)

    if original.shape != perturbed.shape:
        raise ValueError("Original and perturbed images must share the same shape.")
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Inputs must be RGB images with shape [H, W, 3].")

    eps = float(config.get("epsilon", 0.0) or 0.0)
    delta = perturbed.astype(np.float32) - original.astype(np.float32)
    energy = float(np.linalg.norm(delta))
    if eps > 0.0 and energy < eps:
        return np.clip(perturbed, 0.0, 1.0).astype(np.float32)

    blend = float(config.get("blend", 1.0))
    blend = float(np.clip(blend, 0.0, 1.0))
    boost = float(config.get("boost", 1.0))
    channel_weights = config.get("channel_weights")

    mask = _build_frequency_mask(delta.shape[:2], config)
    has_channel_weights = isinstance(channel_weights, (list, tuple)) and len(channel_weights) == 3

    filtered = np.zeros_like(delta, dtype=np.float32)

    for c in range(delta.shape[2]):
        channel_delta = delta[:, :, c]
        coeffs = _dct2(channel_delta)
        low_coeffs = coeffs * mask
        if boost != 1.0:
            low_coeffs[: mask.shape[0], : mask.shape[1]] *= boost
        reconstructed = _idct2(low_coeffs)

        if has_channel_weights:
            channel_blend = float(np.clip(channel_weights[c], 0.0, 1.0))
        else:
            channel_blend = 1.0

        effective_blend = blend * channel_blend
        filtered[:, :, c] = (1.0 - effective_blend) * channel_delta + effective_blend * reconstructed

    if edge_map is not None:
        edge_arr = np.asarray(edge_map, dtype=np.float32)
        h, w = delta.shape[:2]
        if edge_arr.ndim == 1 and edge_arr.size == h * w:
            edge_arr = edge_arr.reshape((h, w))
        elif edge_arr.shape != (h, w):
            raise ValueError("Edge map must broadcast to (H, W) shape for DCT interference.")

        edge_arr = np.clip(edge_arr, 0.0, None)
        edge_min = float(edge_arr.min(initial=0.0))
        edge_max = float(edge_arr.max(initial=0.0))
        if edge_max > edge_min:
            edge_norm = (edge_arr - edge_min) / (edge_max - edge_min)
        else:
            edge_norm = np.ones_like(edge_arr, dtype=np.float32)

        exponent = float(config.get("edge_exponent", 1.0) or 1.0)
        if exponent != 1.0:
            edge_norm = np.power(np.clip(edge_norm, 0.0, 1.0), exponent)

        edge_blend = float(np.clip(config.get("edge_blend", 1.0), 0.0, 1.0))
        if edge_blend < 1.0:
            edge_norm = (1.0 - edge_blend) + edge_blend * edge_norm

        filtered *= edge_norm[:, :, None]

        if config.get("preserve_energy", False):
            original_norm = float(np.linalg.norm(delta))
            filtered_norm = float(np.linalg.norm(filtered))
            if filtered_norm > 0.0 and original_norm > 0.0:
                filtered *= (original_norm / filtered_norm)

    result = original + filtered
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def decompose_low_high_frequency(
    original: Array,
    perturbed: Array,
    config: Dict[str, Any],
) -> tuple[Array, Array]:
    """Return low- and high-frequency components of the perturbation.

    The function mirrors :func:`apply_dct_low_frequency_interference` but keeps
    both the filtered (low-frequency) and residual (high-frequency) parts.

    Args:
        original: Reference RGB image in ``[0, 1]`` with shape ``[H, W, 3]``.
        perturbed: Perturbed RGB image in ``[0, 1]`` with shape ``[H, W, 3]``.
        config: DCT配置，用于构建低频掩模。

    Returns:
        (low_delta, high_delta)：两个与输入形状相同的数组，分别表示低频扰动
        与高频残差（两者之和等于总扰动）。
    """
    if original.shape != perturbed.shape:
        raise ValueError("Original and perturbed images must share the same shape.")
    if original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Inputs must be RGB images with shape [H, W, 3].")

    delta = perturbed.astype(np.float32) - original.astype(np.float32)
    mask = _build_frequency_mask(delta.shape[:2], config)

    low_delta = np.zeros_like(delta, dtype=np.float32)
    high_delta = np.zeros_like(delta, dtype=np.float32)

    for c in range(delta.shape[2]):
        channel_delta = delta[:, :, c]
        coeffs = _dct2(channel_delta)
        low_coeffs = coeffs * mask
        reconstructed_low = _idct2(low_coeffs)
        reconstructed_high = channel_delta - reconstructed_low
        low_delta[:, :, c] = reconstructed_low
        high_delta[:, :, c] = reconstructed_high

    return low_delta, high_delta