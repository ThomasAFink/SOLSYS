"""Mean-anomaly helpers for frame-based animation."""

from __future__ import annotations

import numpy as np


def meanAnomalyAtFrame(
    initialMeanAnomalyRad: float | np.ndarray,
    orbitalPeriodDays: float | np.ndarray,
    frame: int,
    animationSpeed: float,
) -> np.ndarray:
    angularVelocityRad = 2 * np.pi / np.asarray(orbitalPeriodDays, dtype=float)
    return np.asarray(initialMeanAnomalyRad, dtype=float) - animationSpeed * frame * angularVelocityRad


def planetMeanAnomalyRad(planetOrbitalPeriodDays: float, frame: int, animationSpeed: float) -> float:
    return float(meanAnomalyAtFrame(0.0, planetOrbitalPeriodDays, frame, animationSpeed))
