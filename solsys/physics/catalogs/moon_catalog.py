"""Major natural satellites with heliocentric offsets from parent planets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


@dataclass(frozen=True)
class MoonOrbit:
    name: str
    parentPlanet: str
    semiMajorAxisKm: float
    orbitalPeriodDays: float
    color: str
    diameterKm: int
    inclinationDeg: float = 0.0




class MoonCatalog:
    """Major natural satellites with heliocentric offsets from parent planet positions."""

    KM_PER_AU = 149597870.7
    # Real moon orbits are ~0.002 AU; exaggerate for visibility at solar-system zoom.
    DISPLAY_ORBIT_SCALE = 50.0
    MOON_VISIBLE_AXIS_SPAN_AU = 25.0
    MOON_VISIBLE_CAMERA_AU = 25.0

    def __init__(self):
        self.moons: Dict[str, MoonOrbit] = self._buildCatalog()
        self.moonsByParent: Dict[str, List[MoonOrbit]] = self._groupByParent()

    def _buildCatalog(self) -> Dict[str, MoonOrbit]:
        return {
            'Moon': MoonOrbit('Moon', 'Earth', 384_400, 27.3, 'lightgray', 3474),
            'Phobos': MoonOrbit('Phobos', 'Mars', 9_376, 0.32, 'tan', 22),
            'Deimos': MoonOrbit('Deimos', 'Mars', 23_460, 1.26, 'wheat', 12),
            'Io': MoonOrbit('Io', 'Jupiter', 421_800, 1.77, 'yellow', 3643),
            'Europa': MoonOrbit('Europa', 'Jupiter', 671_100, 3.55, 'whitesmoke', 3122),
            'Ganymede': MoonOrbit('Ganymede', 'Jupiter', 1_070_400, 7.15, 'silver', 5268),
            'Callisto': MoonOrbit('Callisto', 'Jupiter', 1_882_700, 16.69, 'darkgray', 4821),
            'Titan': MoonOrbit('Titan', 'Saturn', 1_221_870, 15.95, 'orange', 5149),
            'Enceladus': MoonOrbit('Enceladus', 'Saturn', 238_020, 1.37, 'white', 504),
            'Rhea': MoonOrbit('Rhea', 'Saturn', 527_108, 4.52, 'gainsboro', 1528),
            'Titania': MoonOrbit('Titania', 'Uranus', 436_300, 8.71, 'lightgray', 1577),
            'Oberon': MoonOrbit('Oberon', 'Uranus', 583_520, 13.46, 'darkgray', 1523),
            'Triton': MoonOrbit('Triton', 'Neptune', 354_759, 5.88, 'lightblue', 2707),
            'Charon': MoonOrbit('Charon', 'Pluto', 19_596, 6.39, 'gray', 1212, 0.0),
        }

    def _groupByParent(self) -> Dict[str, List[MoonOrbit]]:
        grouped: Dict[str, List[MoonOrbit]] = {}
        for moon in self.moons.values():
            grouped.setdefault(moon.parentPlanet, []).append(moon)
        return grouped

    def forPlanet(self, planetName: str) -> List[MoonOrbit]:
        return self.moonsByParent.get(planetName, [])

    def semiMajorAxisAu(self, moon: MoonOrbit) -> float:
        return moon.semiMajorAxisKm / self.KM_PER_AU

    def displayScaleForAxisSpanAu(self, axisSpanAu: float) -> float:
        if axisSpanAu > self.MOON_VISIBLE_AXIS_SPAN_AU:
            return 0.0
        if axisSpanAu <= 7.0:
            return self.DISPLAY_ORBIT_SCALE
        fadeSpanAu = self.MOON_VISIBLE_AXIS_SPAN_AU - 7.0
        return self.DISPLAY_ORBIT_SCALE * (self.MOON_VISIBLE_AXIS_SPAN_AU - axisSpanAu) / fadeSpanAu

    def displayScaleForCameraAu(self, cameraDistanceAu: float) -> float:
        if cameraDistanceAu > self.MOON_VISIBLE_CAMERA_AU:
            return 0.0
        if cameraDistanceAu <= 8.0:
            return self.DISPLAY_ORBIT_SCALE
        fadeSpanAu = self.MOON_VISIBLE_CAMERA_AU - 8.0
        return self.DISPLAY_ORBIT_SCALE * (self.MOON_VISIBLE_CAMERA_AU - cameraDistanceAu) / fadeSpanAu

    @staticmethod
    def initialPhaseRad(moonName: str) -> float:
        return (sum(ord(character) for character in moonName) % 628) / 100.0

    def displayOrbitRadiusAu(self, moon: MoonOrbit, displayOrbitScale: float) -> float:
        return self.semiMajorAxisAu(moon) * displayOrbitScale

    def offsetFromPlanet(
        self,
        moon: MoonOrbit,
        moonMeanAnomalyRad: float | np.ndarray,
        displayOrbitScale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orbitRadiusAu = self.displayOrbitRadiusAu(moon, displayOrbitScale)
        inclinationRad = np.radians(moon.inclinationDeg)
        offsetX = orbitRadiusAu * np.cos(moonMeanAnomalyRad)
        offsetY = orbitRadiusAu * np.sin(moonMeanAnomalyRad) * np.cos(inclinationRad)
        offsetZ = orbitRadiusAu * np.sin(moonMeanAnomalyRad) * np.sin(inclinationRad)
        return offsetX, offsetY, offsetZ

    def heliocentricPosition(
        self,
        planetPositionX: float | np.ndarray,
        planetPositionY: float | np.ndarray,
        planetPositionZ: float | np.ndarray,
        moon: MoonOrbit,
        moonMeanAnomalyRad: float | np.ndarray,
        displayOrbitScale: float = 1.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        offsetX, offsetY, offsetZ = self.offsetFromPlanet(
            moon, moonMeanAnomalyRad, displayOrbitScale
        )
        return (
            np.asarray(planetPositionX) + offsetX,
            np.asarray(planetPositionY) + offsetY,
            np.asarray(planetPositionZ) + offsetZ,
        )

    def moonOrbitRing2d(
        self,
        moon: MoonOrbit,
        planetPositionX: float,
        planetPositionY: float,
        displayOrbitScale: float,
        numPoints: int = 48,
    ) -> Tuple[np.ndarray, np.ndarray]:
        azimuthRad = np.linspace(0, 2 * np.pi, numPoints)
        orbitRadiusAu = self.displayOrbitRadiusAu(moon, displayOrbitScale)
        return (
            planetPositionX + orbitRadiusAu * np.cos(azimuthRad),
            planetPositionY + orbitRadiusAu * np.sin(azimuthRad),
        )

    def markerSize2d(self, moon: MoonOrbit, markerScaleDivisor: float) -> int:
        return max(3, int((6 + moon.diameterKm / markerScaleDivisor) / 2))

    def markerSize3d(self, moon: MoonOrbit, markerScaleDivisor: float = 800.0) -> int:
        return max(2, int((5 + moon.diameterKm / markerScaleDivisor) / 2))


