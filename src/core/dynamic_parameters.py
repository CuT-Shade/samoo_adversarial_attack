"""Dynamic parameter scheduling utilities for SA-MOO attack."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any


def _clamp(value: float, min_value: float | None, max_value: float | None) -> float:
    if min_value is not None and value < min_value:
        value = min_value
    if max_value is not None and value > max_value:
        value = max_value
    return value


@dataclass
class DynamicParameterScheduler:
    """Scheduler that produces per-generation parameter values."""

    config: Dict[str, Any]
    total_generations: int
    is_targeted_attack: bool

    def __post_init__(self) -> None:
        self.total_generations = max(int(self.total_generations), 1)
        if not isinstance(self.config, dict):
            self.config = {}

    def get_params(self, generation: int) -> Dict[str, float]:
        """Return all dynamic parameter values for the given generation index."""
        progress = self._compute_progress(generation)
        result: Dict[str, float] = {}
        for name, cfg in self.config.items():
            result[name] = self._compute_value(cfg or {}, progress)
        return result

    def get_value(self, name: str, generation: int, default: float | None = None) -> float | None:
        """Return a single dynamic parameter value, or the provided default if missing."""
        cfg = self.config.get(name)
        if cfg is None:
            return default
        return self._compute_value(cfg, self._compute_progress(generation))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _compute_progress(self, generation: int) -> float:
        if self.total_generations <= 1:
            return 0.0
        ratio = float(generation) / float(self.total_generations - 1)
        return max(0.0, min(1.0, ratio))

    def _compute_value(self, cfg: Dict[str, Any], progress: float) -> float:
        schedule = str(cfg.get("schedule", "constant")).lower()
        if schedule == "linear":
            start = float(self._resolve_contextual_value(cfg, "start", cfg.get("value", 0.0)))
            end = float(self._resolve_contextual_value(cfg, "end", start))
            value = start + (end - start) * progress
        elif schedule == "exponential":
            start = float(self._resolve_contextual_value(cfg, "start", cfg.get("value", 0.0)))
            end = float(self._resolve_contextual_value(cfg, "end", start))
            if start <= 0 or end <= 0:
                value = start + (end - start) * progress
            else:
                value = start * ((end / start) ** progress)
        elif schedule == "piecewise":
            value = self._compute_piecewise(cfg, progress)
        else:
            value = float(self._resolve_contextual_value(cfg, "value", cfg.get("default", 0.0)))

        min_val = self._resolve_contextual_value(cfg, "min")
        max_val = self._resolve_contextual_value(cfg, "max")
        value = _clamp(value, min_val, max_val)

        precision = self._resolve_contextual_value(cfg, "precision")
        if precision is not None:
            try:
                digits = int(precision)
                value = round(value, digits)
            except (TypeError, ValueError):
                pass

        return value

    def _compute_piecewise(self, cfg: Dict[str, Any], progress: float) -> float:
        pieces = self._resolve_contextual_value(cfg, "pieces")
        if pieces is None:
            pieces = cfg.get("pieces")
        if not pieces:
            fallback = self._resolve_contextual_value(cfg, "value")
            return float(fallback) if fallback is not None else 0.0

        normalized: list[tuple[float, float]] = []
        for piece in pieces:
            if not isinstance(piece, dict):
                continue
            threshold = piece.get("progress")
            if threshold is None:
                threshold = piece.get("until")
            try:
                threshold_f = float(threshold)
            except (TypeError, ValueError):
                threshold_f = 1.0

            piece_value = self._resolve_contextual_value(piece, "value")
            if piece_value is None:
                piece_value = piece.get("value")
            if piece_value is None:
                continue
            try:
                normalized.append((threshold_f, float(piece_value)))
            except (TypeError, ValueError):
                continue

        if not normalized:
            fallback = self._resolve_contextual_value(cfg, "value", 0.0)
            return float(fallback)

        normalized.sort(key=lambda item: item[0])
        for threshold, value in normalized:
            if progress <= threshold:
                return value
        return normalized[-1][1]

    def _resolve_contextual_value(self, cfg: Dict[str, Any], key: str, default: float | None = None) -> float | None:
        if self.is_targeted_attack:
            targeted_block = cfg.get("targeted", {})
            if isinstance(targeted_block, dict) and key in targeted_block:
                return targeted_block[key]
            targeted_key = f"targeted_{key}"
            if targeted_key in cfg:
                return cfg[targeted_key]
        else:
            non_targeted_block = cfg.get("non_targeted", {})
            if isinstance(non_targeted_block, dict) and key in non_targeted_block:
                return non_targeted_block[key]
            non_targeted_key = f"non_targeted_{key}"
            if non_targeted_key in cfg:
                return cfg[non_targeted_key]

        return cfg.get(key, default)
