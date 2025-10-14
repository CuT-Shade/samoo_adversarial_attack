"""Edge-guided sampling utilities.

This module computes edge-aware probability distributions that bias perturbation
indices toward semantic contours. The weights can be reused for both population
initialization and mutation operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import warnings

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, sobel, scharr, prewitt
from skimage.feature import canny
from skimage.io import imread
from skimage.transform import resize


_TORCHVISION_SEGMENTATION_CACHE: Dict[Tuple[str, str], Tuple[Any, Any]] = {}


@dataclass
class EdgeGuidanceWeights:
    """Container for per-domain sampling probabilities.

    Attributes:
        pixel_weights: Flattened probability distribution over image pixels.
            The length should be H * W.
        channel_weights: Flattened probability distribution over individual
            RGB channels. The length should be H * W * 3 and must sum to 1.
        artifacts: Optional debug artifacts (e.g., edge maps, semantic maps)
            captured during weight construction for visualization purposes.
    """

    pixel_weights: np.ndarray
    channel_weights: np.ndarray
    artifacts: Optional[Dict[str, np.ndarray]] = None

    def weights_for_mode(self, mode: str) -> np.ndarray:
        """Return the appropriate weight vector for the given perturbation mode."""
        if mode == "channel":
            return self.channel_weights
        return self.pixel_weights

    def sample_without_replacement(
        self,
        mode: str,
        candidate_indices: np.ndarray,
        sample_size: int,
    ) -> np.ndarray:
        """Sample indices with guidance, falling back to uniform if needed."""
        if sample_size <= 0 or candidate_indices.size == 0:
            return np.empty(0, dtype=int)

        weights = self.weights_for_mode(mode)
        subset_weights = weights[candidate_indices]
        total = float(subset_weights.sum())

        sample_size = min(sample_size, candidate_indices.size)
        if not np.isfinite(total) or total <= 0.0:
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        probabilities = subset_weights / total
        if np.any(~np.isfinite(probabilities)):
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        # Guard against numerical precision issues that might create slight negatives
        probabilities = np.clip(probabilities, 0.0, None)
        prob_sum = probabilities.sum()
        if prob_sum <= 0.0:
            return np.random.choice(candidate_indices, size=sample_size, replace=False)

        probabilities /= prob_sum
        return np.random.choice(candidate_indices, size=sample_size, replace=False, p=probabilities)


def _normalize_weights(flat_weights: np.ndarray, min_value: float, uniform_mix: float) -> np.ndarray:
    """Normalize and smooth a flattened weight map."""
    if flat_weights.size == 0:
        return np.array([], dtype=np.float64)

    flat = np.clip(flat_weights, 0.0, None)
    if not np.isfinite(flat).all():
        if flat.size == 0:
            return np.array([], dtype=np.float64)
        return np.full(flat.size, 1.0 / flat.size, dtype=np.float64)

    flat_min = float(flat.min(initial=0.0))
    flat -= flat_min
    flat_max = float(flat.max(initial=0.0))
    if flat_max > 0.0:
        flat /= flat_max

    flat += max(min_value, 0.0)
    total = float(flat.sum())
    if total <= 0.0 or not np.isfinite(total):
        flat = np.full(flat.size, 1.0 / max(flat.size, 1))
    else:
        flat /= total

    uniform_mix = float(np.clip(uniform_mix, 0.0, 1.0))
    if uniform_mix > 0.0 and flat.size > 0:
        uniform = np.full(flat.size, 1.0 / flat.size)
        flat = (1.0 - uniform_mix) * flat + uniform_mix * uniform
        flat /= float(flat.sum())

    return flat.astype(np.float64)


def _compute_core_edges(gray: np.ndarray, method: str, canny_sigma: float) -> np.ndarray:
    """Compute edge magnitude using the requested detector."""
    if method == "sobel":
        edges = sobel(gray)
    elif method == "scharr":
        edges = scharr(gray)
    elif method == "prewitt":
        edges = prewitt(gray)
    elif method == "canny":
        edges = canny(gray, sigma=canny_sigma).astype(np.float32)
    else:
        raise ValueError(f"Unsupported edge guidance method: {method}")

    return np.abs(edges.astype(np.float64))


def _compute_multi_scale_edges(
    gray: np.ndarray,
    method: str,
    base_gaussian_sigma: float,
    canny_sigma: float,
    multi_cfg: Dict[str, object],
) -> np.ndarray:
    """Aggregate edges computed over multiple spatial scales."""

    scales = multi_cfg.get("scales", [1.0, 0.5, 0.25])
    if not isinstance(scales, Sequence) or isinstance(scales, (str, bytes)):
        scales = [1.0]
    scales = list(scales)
    if all(abs(float(scale) - 1.0) > 1e-6 for scale in scales):
        scales.insert(0, 1.0)

    weights = multi_cfg.get("weights")
    gaussian_sigmas = multi_cfg.get("gaussian_sigmas")
    combine_mode = str(multi_cfg.get("combine", "mean")).lower()
    anti_aliasing = bool(multi_cfg.get("anti_aliasing", True))

    height, width = gray.shape
    aggregated = None
    weight_sum = 0.0

    for idx, scale_value in enumerate(scales):
        try:
            scale = float(scale_value)
        except (TypeError, ValueError):
            continue
        if scale <= 0.0:
            continue

        if abs(scale - 1.0) < 1e-6:
            scaled_gray = gray
        else:
            new_shape = (
                max(1, int(round(height * scale))),
                max(1, int(round(width * scale))),
            )
            scaled_gray = resize(
                gray,
                new_shape,
                mode="reflect",
                anti_aliasing=anti_aliasing,
                preserve_range=True,
            )

        sigma = base_gaussian_sigma
        if isinstance(gaussian_sigmas, Sequence) and not isinstance(gaussian_sigmas, (str, bytes)):
            if idx < len(gaussian_sigmas):
                try:
                    sigma = float(gaussian_sigmas[idx])
                except (TypeError, ValueError):
                    sigma = base_gaussian_sigma

        if sigma > 0.0:
            scaled_gray = gaussian(scaled_gray, sigma=sigma, mode="reflect")

        scaled_edges = _compute_core_edges(scaled_gray, method, canny_sigma)

        if scaled_edges.shape != (height, width):
            scaled_edges = resize(
                scaled_edges,
                (height, width),
                mode="reflect",
                anti_aliasing=True,
                preserve_range=True,
            )

        if aggregated is None:
            aggregated = np.zeros_like(scaled_edges)

        if combine_mode == "max":
            aggregated = np.maximum(aggregated, scaled_edges)
            continue

        weight = 1.0
        if isinstance(weights, Sequence) and not isinstance(weights, (str, bytes)):
            if idx < len(weights):
                try:
                    weight = float(weights[idx])
                except (TypeError, ValueError):
                    weight = 1.0

        aggregated += weight * scaled_edges
        weight_sum += weight

    if aggregated is None:
        if base_gaussian_sigma > 0.0:
            blurred = gaussian(gray, sigma=base_gaussian_sigma, mode="reflect")
        else:
            blurred = gray
        return _compute_core_edges(blurred, method, canny_sigma)

    if combine_mode == "max":
        return aggregated

    if weight_sum > 0.0:
        aggregated = aggregated / weight_sum

    return aggregated


def _load_semantic_map(
    semantic_cfg: Dict[str, object],
    target_shape: Sequence[int],
) -> Optional[np.ndarray]:
    """Load and preprocess a semantic attention map if configured."""

    path_value = semantic_cfg.get("path")
    semantic_array = semantic_cfg.get("array")
    if path_value is None and semantic_array is None:
        return None

    data: Optional[np.ndarray] = None

    if path_value is not None:
        path = Path(str(path_value))
        if not path.exists():
            if semantic_cfg.get("fail_on_missing", False):
                raise FileNotFoundError(f"Semantic map not found at: {path}")
            return None

        suffix = path.suffix.lower()
        if suffix in {".npy", ".npz"}:
            loaded = np.load(path)
            if isinstance(loaded, np.ndarray):
                data = loaded
            else:
                # npz archive: take the first array
                first_key = next(iter(loaded.files))
                data = loaded[first_key]
        else:
            data = imread(path)

    if data is None and semantic_array is not None:
        data = np.asarray(semantic_array)

    if data is None:
        return None

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 3:
        if data.shape[2] == 3:
            data = rgb2gray(data)
        else:
            data = np.mean(data, axis=2)
    elif data.ndim == 1 and data.size == int(target_shape[0]) * int(target_shape[1]):
        data = data.reshape(tuple(target_shape[:2]))
    elif data.ndim != 2:
        raise ValueError("Semantic map must be 2D or broadcastable to the image shape")

    target_hw = (int(target_shape[0]), int(target_shape[1]))
    if data.shape != target_hw:
        data = resize(
            data,
            target_hw,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

    if semantic_cfg.get("normalize", True):
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if np.isfinite(data_max - data_min) and data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

    if semantic_cfg.get("invert", False):
        data = 1.0 - data

    exponent = float(semantic_cfg.get("exponent", 1.0))
    if exponent != 1.0:
        data = np.power(np.clip(data, 0.0, None), exponent)

    blur_sigma = float(semantic_cfg.get("blur_sigma", 0.0))
    if blur_sigma > 0.0:
        data = gaussian(data, sigma=blur_sigma, mode="reflect")

    clip_low = semantic_cfg.get("clip_low")
    clip_high = semantic_cfg.get("clip_high")
    if clip_low is not None or clip_high is not None:
        lo = float(clip_low) if clip_low is not None else 0.0
        hi = float(clip_high) if clip_high is not None else np.inf
        data = np.clip(data, lo, hi)

    return np.clip(data, 0.0, None)


def _load_external_map(
    path_value: Optional[object],
    target_shape: Sequence[int],
    *,
    normalize: bool = False,
    clip_low: Optional[float] = None,
    clip_high: Optional[float] = None,
    fail_on_missing: bool = False,
) -> Optional[np.ndarray]:
    """Load a generic map produced by an external professional model."""

    if path_value is None:
        return None

    path = Path(str(path_value))
    if not path.exists():
        if fail_on_missing:
            raise FileNotFoundError(f"Professional preprocessing map not found at: {path}")
        return None

    suffix = path.suffix.lower()
    if suffix in {".npy", ".npz"}:
        loaded = np.load(path)
        if isinstance(loaded, np.ndarray):
            data = loaded
        else:
            first_key = next(iter(loaded.files))
            data = loaded[first_key]
    else:
        data = imread(path)

    data = np.asarray(data, dtype=np.float64)

    if data.ndim == 3:
        if data.shape[2] == 3:
            data = rgb2gray(data)
        else:
            data = np.mean(data, axis=2)
    elif data.ndim == 1 and data.size == int(target_shape[0]) * int(target_shape[1]):
        data = data.reshape(tuple(target_shape[:2]))
    elif data.ndim != 2:
        raise ValueError("Professional preprocessing map must be 2D or broadcastable to the image shape")

    target_hw = (int(target_shape[0]), int(target_shape[1]))
    if data.shape != target_hw:
        data = resize(
            data,
            target_hw,
            mode="reflect",
            anti_aliasing=True,
            preserve_range=True,
        )

    if normalize:
        data_min = float(np.min(data))
        data_max = float(np.max(data))
        if np.isfinite(data_max - data_min) and data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        else:
            data = np.zeros_like(data)

    if clip_low is not None or clip_high is not None:
        lo = float(clip_low) if clip_low is not None else -np.inf
        hi = float(clip_high) if clip_high is not None else np.inf
        data = np.clip(data, lo, hi)

    return np.asarray(data, dtype=np.float64)


def _normalize_map(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return data
    data = np.clip(data, 0.0, None)
    minimum = float(np.min(data))
    maximum = float(np.max(data))
    if maximum > minimum:
        data = (data - minimum) / (maximum - minimum)
    else:
        data = np.zeros_like(data)
    return data


def _resolve_device(device_pref: str) -> str:
    desired = str(device_pref or "cpu").lower()
    if desired in {"auto", "best"}:
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return desired


def _get_torchvision_segmentation_model(name: str, device: str) -> Tuple[Any, Any]:
    cache_key = (name, device)
    if cache_key in _TORCHVISION_SEGMENTATION_CACHE:
        return _TORCHVISION_SEGMENTATION_CACHE[cache_key]

    try:
        import torch  # type: ignore
        from torchvision.models.segmentation import (  # type: ignore
            deeplabv3_resnet50,
            DeepLabV3_ResNet50_Weights,
            fcn_resnet50,
            FCN_ResNet50_Weights,
            lraspp_mobilenet_v3_large,
            LRASPP_MobileNet_V3_Large_Weights,
        )
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Torchvision (with segmentation models) is required for professional preprocessing."
        ) from exc

    device_obj = torch.device(device)

    if name == "deeplabv3_resnet50":
        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        model = deeplabv3_resnet50(weights=weights)
    elif name == "fcn_resnet50":
        weights = FCN_ResNet50_Weights.DEFAULT
        model = fcn_resnet50(weights=weights)
    elif name == "lraspp_mobilenet_v3_large":
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT
        model = lraspp_mobilenet_v3_large(weights=weights)
    else:
        raise ValueError(f"Unsupported torchvision segmentation model: {name}")

    preprocess = weights.transforms()
    model.eval()
    model.to(device_obj)

    _TORCHVISION_SEGMENTATION_CACHE[cache_key] = (model, preprocess)
    return model, preprocess


def _run_semantic_model_from_cfg(
    rgb_image: np.ndarray,
    semantic_cfg: Dict[str, Any],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    try:
        import torch  # type: ignore
        import torch.nn.functional as F  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Running built-in semantic models requires torch and torchvision to be installed."
        ) from exc

    from PIL import Image  # Local import to avoid mandatory dependency when unused

    model_name = str(semantic_cfg.get("name", "deeplabv3_resnet50"))
    device = _resolve_device(str(semantic_cfg.get("device", "auto")))
    model, preprocess = _get_torchvision_segmentation_model(model_name, device)

    pil_image = Image.fromarray(
        np.clip(rgb_image * 255.0, 0, 255).astype(np.uint8)
    )
    processed = preprocess(pil_image)
    if processed.dim() == 3:
        processed = processed.unsqueeze(0)

    try:
        input_tensor = processed.to(device)
    except AttributeError:
        # Older torchvision transforms may return numpy arrays
        input_tensor = torch.from_numpy(processed).to(device)

    with torch.no_grad():
        output = model(input_tensor)["out"]
        probabilities = torch.softmax(output, dim=1)
        probabilities = F.interpolate(
            probabilities,
            size=rgb_image.shape[:2],
            mode="bilinear",
            align_corners=False,
        )

    output_mode = str(semantic_cfg.get("output", "max_prob")).lower()
    if output_mode == "max_prob":
        semantic_tensor = probabilities.max(dim=1).values.squeeze(0)
    elif output_mode == "entropy":
        eps = float(semantic_cfg.get("entropy_eps", 1e-6))
        semantic_tensor = -torch.sum(
            probabilities * torch.log(probabilities + eps), dim=1
        ).squeeze(0)
    elif output_mode == "target_class":
        target_idx = semantic_cfg.get("target_class")
        if target_idx is None:
            raise ValueError("semantic_model.target_class must be set when output='target_class'")
        target_idx = int(target_idx)
        if target_idx < 0 or target_idx >= probabilities.shape[1]:
            raise ValueError("semantic_model.target_class is out of range for the selected model")
        semantic_tensor = probabilities[:, target_idx, :, :].squeeze(0)
    elif output_mode == "top2_gap":
        top2 = torch.topk(probabilities, k=2, dim=1).values
        semantic_tensor = (top2[:, 0, :, :] - top2[:, 1, :, :]).squeeze(0)
    else:
        raise ValueError(f"Unsupported semantic_model.output value: {output_mode}")

    semantic_map = semantic_tensor.cpu().numpy()
    sigma = float(semantic_cfg.get("smooth_sigma", 0.0))
    if sigma > 0.0:
        semantic_map = gaussian(semantic_map, sigma=sigma, mode="reflect")

    return np.clip(semantic_map, 0.0, None), probabilities.squeeze(0).cpu().numpy()


def _generate_edge_map_from_cfg(
    rgb_image: np.ndarray,
    edge_cfg: Dict[str, Any],
    semantic_map: Optional[np.ndarray],
    semantic_probabilities: Optional[np.ndarray],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    method = str(edge_cfg.get("type", "semantic_gradient")).lower()

    if method == "semantic_gradient":
        if semantic_map is None:
            raise ValueError("semantic_gradient edge model requires a semantic map.")
        grad_y, grad_x = np.gradient(semantic_map)
        edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif method == "semantic_entropy":
        if semantic_probabilities is None:
            raise ValueError("semantic_entropy edge model requires semantic probabilities.")
        eps = float(edge_cfg.get("entropy_eps", 1e-6))
        edges = -np.sum(
            semantic_probabilities * np.log(semantic_probabilities + eps), axis=0
        )
    elif method.startswith("torchvision_"):
        # Allow direct edge mapping via a torchvision segmentation model
        model_name = method.replace("torchvision_", "")
        semantic_model_cfg = {
            "name": model_name,
            "device": edge_cfg.get("device", "auto"),
            "output": edge_cfg.get("semantic_output", "max_prob"),
            "smooth_sigma": edge_cfg.get("semantic_smooth_sigma", 0.0),
            "entropy_eps": edge_cfg.get("entropy_eps", 1e-6),
            "target_class": edge_cfg.get("target_class"),
        }
        semantic_map, semantic_probabilities = _run_semantic_model_from_cfg(
            rgb_image, semantic_model_cfg
        )
        grad_y, grad_x = np.gradient(semantic_map)
        edges = np.sqrt(grad_x ** 2 + grad_y ** 2)
    elif method == "canny_rgb":
        sigma = float(edge_cfg.get("canny_sigma", 1.0))
        edges = canny(rgb2gray(rgb_image), sigma=sigma).astype(np.float64)
    else:
        raise ValueError(f"Unsupported edge_model.type value: {method}")

    if edge_cfg.get("normalize", True):
        edges = _normalize_map(edges)

    smooth_sigma = float(edge_cfg.get("smooth_sigma", 0.0))
    if smooth_sigma > 0.0:
        edges = gaussian(edges, sigma=smooth_sigma, mode="reflect")

    return np.clip(edges, 0.0, None), semantic_map, semantic_probabilities


def _blend_with_semantic(edges: np.ndarray, semantic_map: np.ndarray, semantic_cfg: Dict[str, object]) -> np.ndarray:
    """Blend semantic attention into the edge magnitude map."""

    mode = str(semantic_cfg.get("blend_mode", "multiply")).lower()
    weight = float(semantic_cfg.get("weight", 0.5))
    weight = max(0.0, weight)

    if mode == "multiply":
        factor = (1.0 - weight) + weight * semantic_map
        return edges * factor
    if mode == "replace":
        return (1.0 - weight) * edges + weight * semantic_map
    if mode == "max":
        multiplier = weight if weight > 0 else 1.0
        return np.maximum(edges, semantic_map * multiplier)
    # Default to additive blend
    return edges + weight * semantic_map


def compute_edge_guidance_weights(
    rgb_image: np.ndarray,
    config: Dict[str, object],
) -> EdgeGuidanceWeights:
    """Compute edge-aware sampling weights from an RGB image.

    Args:
        rgb_image: Float image in [0, 1] with shape (H, W, 3).
        config: Configuration dictionary with optional keys:
            - enabled (bool): Whether guidance is enabled (caller usually checks
              this before calling).
            - method (str): Edge detector ('sobel', 'scharr', 'prewitt', 'canny').
            - gaussian_sigma (float): Sigma for Gaussian smoothing before edges.
            - canny_sigma (float): Sigma for Canny if method == 'canny'.
            - exponent (float): Exponent applied to edge magnitude to sharpen contrast.
            - uniform_mix (float): Blend ratio with uniform distribution.
            - min_value (float): Floor added before normalization.
            - multi_scale (dict): Optional multi-scale configuration with keys such as
              'enabled', 'scales', 'weights', 'gaussian_sigmas', 'combine'.
            - semantic (dict): Optional semantic guidance settings ('enabled', 'path',
              'weight', 'blend_mode', etc.).

    Returns:
        EdgeGuidanceWeights containing per-domain probability vectors.
    """
    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected RGB image with shape (H, W, 3)")

    method = str(config.get("method", "sobel")).lower()
    gaussian_sigma = float(config.get("gaussian_sigma", 0.8))
    exponent = float(config.get("exponent", 1.5))
    uniform_mix = float(config.get("uniform_mix", 0.15))
    min_value = float(config.get("min_value", 1e-4))
    canny_sigma = float(config.get("canny_sigma", 1.0))

    gray = rgb2gray(rgb_image)
    artifacts: Dict[str, np.ndarray] = {
        "gray": np.clip(np.asarray(gray, dtype=np.float64), 0.0, 1.0)
    }

    professional_cfg = config.get("professional_preprocessing") if isinstance(config.get("professional_preprocessing"), dict) else None
    edge_override: Optional[np.ndarray] = None
    semantic_override: Optional[np.ndarray] = None
    semantic_probabilities: Optional[np.ndarray] = None

    if isinstance(professional_cfg, dict) and professional_cfg.get("enabled", False):
        strict_errors = bool(professional_cfg.get("fail_on_missing", False))

        try:
            edge_override = _load_external_map(
                professional_cfg.get("edge_map_path"),
                gray.shape,
                normalize=bool(professional_cfg.get("edge_normalize", False)),
                clip_low=professional_cfg.get("edge_clip_low"),
                clip_high=professional_cfg.get("edge_clip_high"),
                fail_on_missing=strict_errors,
            )
            if edge_override is not None:
                artifacts["edge_magnitude_professional_file"] = np.clip(edge_override, 0.0, None)
        except Exception as exc:
            if strict_errors:
                raise
            warnings.warn(f"Failed to load professional edge map: {exc}")

        try:
            semantic_override = _load_external_map(
                professional_cfg.get("semantic_map_path"),
                gray.shape,
                normalize=bool(professional_cfg.get("semantic_normalize", True)),
                clip_low=professional_cfg.get("semantic_clip_low"),
                clip_high=professional_cfg.get("semantic_clip_high"),
                fail_on_missing=strict_errors,
            )
            if semantic_override is not None:
                artifacts["semantic_map_professional_file"] = np.clip(semantic_override, 0.0, None)
        except Exception as exc:
            if strict_errors:
                raise
            warnings.warn(f"Failed to load professional semantic map: {exc}")

        generated_semantic: Optional[np.ndarray] = None
        generated_edge: Optional[np.ndarray] = None

        semantic_model_cfg = professional_cfg.get("semantic_model")
        if isinstance(semantic_model_cfg, dict) and semantic_model_cfg.get("enabled", False):
            try:
                generated_semantic, semantic_probabilities = _run_semantic_model_from_cfg(rgb_image, semantic_model_cfg)
                artifacts["semantic_map_professional_model"] = np.clip(generated_semantic, 0.0, None)
            except Exception as exc:
                semantic_probabilities = None
                if strict_errors:
                    raise
                warnings.warn(f"Semantic model preprocessing failed: {exc}")

        if semantic_override is None and generated_semantic is not None:
            semantic_override = generated_semantic

        edge_model_cfg = professional_cfg.get("edge_model")
        if isinstance(edge_model_cfg, dict) and edge_model_cfg.get("enabled", False):
            try:
                generated_edge, generated_semantic_from_edge, generated_probs_from_edge = _generate_edge_map_from_cfg(
                    rgb_image,
                    edge_model_cfg,
                    semantic_override,
                    semantic_probabilities,
                )
                if generated_edge is not None:
                    artifacts["edge_magnitude_professional_model"] = np.clip(generated_edge, 0.0, None)
                if generated_semantic_from_edge is not None and semantic_override is None:
                    semantic_override = generated_semantic_from_edge
                if generated_probs_from_edge is not None and semantic_probabilities is None:
                    semantic_probabilities = generated_probs_from_edge
            except Exception as exc:
                if strict_errors:
                    raise
                warnings.warn(f"Edge model preprocessing failed: {exc}")

        if edge_override is None and generated_edge is not None:
            edge_override = generated_edge

    if edge_override is not None:
        edges = np.clip(edge_override, 0.0, None)
        artifacts["edge_magnitude_base"] = edges.copy()
    else:
        multi_cfg = config.get("multi_scale") if isinstance(config.get("multi_scale"), dict) else None
        if isinstance(multi_cfg, dict) and multi_cfg.get("enabled", False):
            base_edges = _compute_multi_scale_edges(gray, method, gaussian_sigma, canny_sigma, multi_cfg)
            edges = np.clip(base_edges, 0.0, None)
            artifacts["edge_magnitude_base"] = edges.copy()
        else:
            working_gray = gray
            if gaussian_sigma > 0.0:
                working_gray = gaussian(working_gray, sigma=gaussian_sigma, mode="reflect")
                artifacts["gray_smoothed"] = np.clip(working_gray, 0.0, None)
            base_edges = _compute_core_edges(working_gray, method, canny_sigma)
            edges = np.clip(base_edges, 0.0, None)
            artifacts["edge_magnitude_base"] = edges.copy()

    if exponent != 1.0:
        edges = np.power(edges, exponent)

    edges = np.clip(edges, 0.0, None)
    artifacts["edge_magnitude_post_exponent"] = edges.copy()

    semantic_cfg = config.get("semantic") if isinstance(config.get("semantic"), dict) else None
    if semantic_override is not None:
        base_cfg = semantic_cfg if isinstance(semantic_cfg, dict) else {}
        semantic_cfg = dict(base_cfg)
        semantic_cfg["array"] = semantic_override
        semantic_cfg.pop("path", None)
        semantic_cfg["enabled"] = True
        if isinstance(professional_cfg, dict) and "semantic_normalize" in professional_cfg:
            semantic_cfg["normalize"] = professional_cfg["semantic_normalize"]

    if isinstance(semantic_cfg, dict) and semantic_cfg.get("enabled", False):
        semantic_map = _load_semantic_map(semantic_cfg, gray.shape)
        if semantic_map is not None:
            artifacts["semantic_map"] = np.clip(semantic_map, 0.0, None)
            edges = _blend_with_semantic(edges, semantic_map, semantic_cfg)
            edges = np.clip(edges, 0.0, None)
            artifacts["edge_magnitude_post_semantic"] = edges.copy()
        elif semantic_cfg.get("fail_on_missing", False):
            raise FileNotFoundError("Semantic guidance enabled but no semantic map could be loaded.")

    edges = np.clip(edges, 0.0, None)

    pixel_weights = _normalize_weights(edges.reshape(-1), min_value, uniform_mix)
    channel_weights = np.repeat(pixel_weights / 3.0, 3)

    try:
        artifacts["normalized_weights"] = pixel_weights.reshape(gray.shape)
    except Exception:
        pass

    return EdgeGuidanceWeights(
        pixel_weights=pixel_weights,
        channel_weights=channel_weights,
        artifacts=artifacts,
    )