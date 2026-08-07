"""Named asteroids and dwarf planets with real orbital elements."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solsys.physics.orbit_calculator import OrbitCalculator


@dataclass(frozen=True)
class FamousAsteroidOrbit:
    name: str
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    color: str
    diameterKm: float
    orbitalPeriodDays: float
    category: str
    ascendingNodeDeg: float = 0.0


class FamousAsteroidCatalog:
    """Named asteroids and dwarf planets with real orbital elements."""

    INNER_AXIS_SPAN_AU = 25.0
    KUIPER_AXIS_SPAN_MIN_AU = 12.0
    KUIPER_AXIS_SPAN_MAX_AU = 160.0
    INNER_CAMERA_AU = 12.0
    KUIPER_CAMERA_MIN_AU = 35.0
    KUIPER_CAMERA_MAX_AU = 75.0

    def __init__(self):
        self.asteroids: dict[str, FamousAsteroidOrbit] = self._buildCatalog()

    @staticmethod
    def _orbitalPeriodDays(semiMajorAxisAu: float) -> float:
        return semiMajorAxisAu**1.5 * 365.25

    def _asteroid(
        self,
        name: str,
        semiMajorAxisAu: float,
        eccentricity: float,
        inclinationDeg: float,
        color: str,
        diameterKm: float,
        category: str,
        ascendingNodeDeg: float = 0.0,
    ) -> FamousAsteroidOrbit:
        return FamousAsteroidOrbit(
            name=name,
            semiMajorAxisAu=semiMajorAxisAu,
            eccentricity=eccentricity,
            inclinationDeg=inclinationDeg,
            color=color,
            diameterKm=diameterKm,
            orbitalPeriodDays=self._orbitalPeriodDays(semiMajorAxisAu),
            category=category,
            ascendingNodeDeg=ascendingNodeDeg,
        )

    def _buildCatalog(self) -> dict[str, FamousAsteroidOrbit]:
        return {
            'Ceres': self._asteroid('Ceres', 2.767, 0.076, 10.59, '#C4A882', 939, 'main_belt'),
            'Vesta': self._asteroid('Vesta', 2.362, 0.089, 7.14, '#B8B8B8', 525, 'main_belt'),
            'Pallas': self._asteroid('Pallas', 2.773, 0.231, 34.84, '#A0A0A0', 512, 'main_belt'),
            'Hygiea': self._asteroid('Hygiea', 3.139, 0.117, 3.84, '#909090', 434, 'main_belt'),
            'Psyche': self._asteroid('Psyche', 2.921, 0.134, 3.10, '#D4AF37', 226, 'main_belt'),
            'Gaspra': self._asteroid('Gaspra', 2.210, 0.174, 4.11, '#9E9E9E', 18, 'main_belt'),
            'Ida': self._asteroid('Ida', 2.861, 0.140, 1.14, '#8C8C8C', 31, 'main_belt'),
            'Lutetia': self._asteroid('Lutetia', 2.436, 0.164, 3.06, '#A8A8A8', 121, 'main_belt'),
            'Bennu': self._asteroid('Bennu', 1.126, 0.204, 6.03, '#8B7355', 0.49, 'near_earth'),
            'Ryugu': self._asteroid('Ryugu', 1.190, 0.190, 5.88, '#7A6A55', 0.87, 'near_earth'),
            'Eros': self._asteroid('Eros', 1.458, 0.223, 10.83, '#B09070', 16.8, 'near_earth'),
            'Hektor': self._asteroid('Hektor', 5.257, 0.024, 18.60, '#A67B5B', 225, 'trojan'),
            'Arrokoth': self._asteroid('Arrokoth', 44.58, 0.042, 2.45, '#C9B896', 36, 'kuiper'),
            'Eris': self._asteroid('Eris', 68.0, 0.440, 44.04, '#E8E8E8', 2326, 'kuiper'),
            'Makemake': self._asteroid('Makemake', 45.79, 0.159, 28.96, '#D0D0D0', 1430, 'kuiper'),
            'Haumea': self._asteroid('Haumea', 43.13, 0.195, 28.21, '#CFCFCF', 1632, 'kuiper'),
        }

    @staticmethod
    def initialPhaseRad(asteroidName: str) -> float:
        return (sum(ord(character) for character in asteroidName) % 628) / 100.0

    def visibleForAxisSpanAu(self, axisSpanAu: float, category: str) -> bool:
        if category in {'main_belt', 'near_earth', 'trojan'}:
            return axisSpanAu <= self.INNER_AXIS_SPAN_AU
        if category == 'kuiper':
            return self.KUIPER_AXIS_SPAN_MIN_AU <= axisSpanAu <= self.KUIPER_AXIS_SPAN_MAX_AU
        return False

    def visibleForCameraAu(self, cameraDistanceAu: float, category: str) -> bool:
        if category in {'main_belt', 'near_earth', 'trojan'}:
            return cameraDistanceAu <= self.INNER_CAMERA_AU
        if category == 'kuiper':
            return self.KUIPER_CAMERA_MIN_AU <= cameraDistanceAu <= self.KUIPER_CAMERA_MAX_AU
        return False

    def positionAtMeanAnomaly(
        self,
        asteroid: FamousAsteroidOrbit,
        meanAnomalyRad: float | np.ndarray,
        orbitCalculator: OrbitCalculator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return orbitCalculator.ellipticalPosition(
            asteroid.semiMajorAxisAu,
            asteroid.eccentricity,
            asteroid.inclinationDeg,
            meanAnomalyRad,
            asteroid.ascendingNodeDeg,
        )

    def markerSize2d(self, asteroid: FamousAsteroidOrbit, markerScaleDivisor: float) -> int:
        return max(10, int(10 + asteroid.diameterKm / markerScaleDivisor))

    def markerSize3d(self, asteroid: FamousAsteroidOrbit, markerScaleDivisor: float = 600.0) -> int:
        return max(8, int(8 + asteroid.diameterKm / markerScaleDivisor))
