"""Shared solar-system visualization core: constants, orbits, and star catalog."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AstronomicalConstants:
    plutoSemiMajorAxis: float = 39.482
    plutoEccentricity: float = 0.2488
    asteroidBeltInnerAu: float = 2.2
    asteroidBeltOuterAu: float = 3.2
    kuiperBeltInnerAu: float = 30.0
    kuiperBeltOuterAu: float = 55.0
    jupiterSemiMajorAxisAu: float = 5.2
    jupiterInclinationDeg: float = 1.3
    jupiterEccentricity: float = 0.0489
    oortCloudInnerAu: float = 2000.0
    oortCloudOuterAu: float = 100000.0
    lightYearToAu: float = 63241.077
    oumuamuaEccentricity: float = 1.2011
    oumuamuaPerihelionAu: float = 0.2559
    oumuamuaInclinationDeg: float = 122.74
    oumuamuaLongitudeAscendingNodeDeg: float = 24.60
    oumuamuaArgumentOfPerihelionDeg: float = 241.69

    @property
    def plutoPerihelionAu(self) -> float:
        return self.plutoSemiMajorAxis * (1 - self.plutoEccentricity)

    @property
    def plutoAphelionAu(self) -> float:
        return self.plutoSemiMajorAxis * (1 + self.plutoEccentricity)


@dataclass(frozen=True)
class PlanetOrbit:
    name: str
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    color: str
    diameterKm: int
    orbitalPeriodDays: float


@dataclass(frozen=True)
class MoonOrbit:
    name: str
    parentPlanet: str
    semiMajorAxisKm: float
    orbitalPeriodDays: float
    color: str
    diameterKm: int
    inclinationDeg: float = 0.0


@dataclass(frozen=True)
class ViewDefinition:
    viewId: str
    axisMinAu: float
    axisMaxAu: float
    titleFontSize: int
    shortName: str
    title: str


class OrbitCalculator:
    """Keplerian and hyperbolic orbit math in heliocentric coordinates."""

    @staticmethod
    def parseRightAscensionToDegrees(rightAscension: str) -> Optional[float]:
        if not isinstance(rightAscension, str):
            return None
        match = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', rightAscension)
        if not match:
            return None
        hours, minutes, seconds = map(float, match.groups())
        return 15 * (hours + minutes / 60 + seconds / 3600)

    @staticmethod
    def parseRightAscensionAndDeclination(
        rightAscension: str, declination: str
    ) -> Tuple[Optional[float], Optional[float]]:
        if not isinstance(rightAscension, str) or not isinstance(declination, str):
            return None, None
        raMatch = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', rightAscension)
        declinationNormalized = declination.replace('−', '-').replace('–', '-')
        decMatch = re.match(
            r'([+-]?\d+)°\s*(\d+)′\s*(\d+(?:\.\d*)?)″', declinationNormalized
        )
        if not raMatch or not decMatch:
            return None, None
        raDegrees = 15 * (
            float(raMatch.group(1))
            + float(raMatch.group(2)) / 60
            + float(raMatch.group(3)) / 3600
        )
        declinationSign = -1 if decMatch.group(1).startswith('-') else 1
        declinationDegrees = declinationSign * (
            abs(float(decMatch.group(1)))
            + float(decMatch.group(2)) / 60
            + float(decMatch.group(3)) / 3600
        )
        return raDegrees, declinationDegrees

    @staticmethod
    def equatorialToCartesianAu(
        rightAscensionDeg: float, declinationDeg: float, distanceAu: float
    ) -> Tuple[float, float, float]:
        rightAscensionRad = np.radians(rightAscensionDeg)
        declinationRad = np.radians(declinationDeg)
        positionX = distanceAu * np.cos(declinationRad) * np.cos(rightAscensionRad)
        positionY = distanceAu * np.cos(declinationRad) * np.sin(rightAscensionRad)
        positionZ = distanceAu * np.sin(declinationRad)
        return positionX, positionY, positionZ

    @staticmethod
    def ellipticalOrbit2d(
        semiMajorAxisAu: float,
        eccentricity: float,
        inclinationDeg: float,
        numPoints: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        trueAnomaly = np.linspace(0, 2 * np.pi, numPoints)
        radiusAu = semiMajorAxisAu * (1 - eccentricity ** 2) / (
            1 + eccentricity * np.cos(trueAnomaly)
        )
        positionX = radiusAu * np.cos(trueAnomaly)
        positionY = radiusAu * np.sin(trueAnomaly) * np.cos(inclinationRad)
        return positionX, positionY

    @staticmethod
    def ellipticalOrbit3d(
        semiMajorAxisAu: float,
        eccentricity: float,
        inclinationDeg: float,
        numPoints: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        trueAnomaly = np.linspace(0, 2 * np.pi, numPoints)
        radiusAu = semiMajorAxisAu * (1 - eccentricity ** 2) / (
            1 + eccentricity * np.cos(trueAnomaly)
        )
        positionX = radiusAu * np.cos(trueAnomaly)
        positionY = radiusAu * np.sin(trueAnomaly) * np.cos(inclinationRad)
        positionZ = radiusAu * np.sin(trueAnomaly) * np.sin(inclinationRad)
        return positionX, positionY, positionZ

    @staticmethod
    def hyperbolicOrbit3d(
        perihelionAu: float,
        eccentricity: float,
        inclinationDeg: float,
        longitudeAscendingNodeDeg: float,
        argumentOfPerihelionDeg: float,
        numPoints: int = 1000,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        ascendingNodeRad = np.radians(longitudeAscendingNodeDeg)
        argumentOfPerihelionRad = np.radians(argumentOfPerihelionDeg)
        maxTrueAnomaly = np.arccos(-1 / eccentricity) - 1e-6
        trueAnomaly = np.linspace(-maxTrueAnomaly, maxTrueAnomaly, numPoints)
        radiusAu = perihelionAu * (1 + eccentricity) / (1 + eccentricity * np.cos(trueAnomaly))

        perifocalX = radiusAu * np.cos(trueAnomaly)
        perifocalY = radiusAu * np.sin(trueAnomaly)

        positionX = (
            (np.cos(ascendingNodeRad) * np.cos(argumentOfPerihelionRad)
             - np.sin(ascendingNodeRad) * np.sin(argumentOfPerihelionRad) * np.cos(inclinationRad))
            * perifocalX
            + (-np.cos(ascendingNodeRad) * np.sin(argumentOfPerihelionRad)
               - np.sin(ascendingNodeRad) * np.cos(argumentOfPerihelionRad) * np.cos(inclinationRad))
            * perifocalY
        )
        positionY = (
            (np.sin(ascendingNodeRad) * np.cos(argumentOfPerihelionRad)
             + np.cos(ascendingNodeRad) * np.sin(argumentOfPerihelionRad) * np.cos(inclinationRad))
            * perifocalX
            + (-np.sin(ascendingNodeRad) * np.sin(argumentOfPerihelionRad)
               + np.cos(ascendingNodeRad) * np.cos(argumentOfPerihelionRad) * np.cos(inclinationRad))
            * perifocalY
        )
        positionZ = (
            np.sin(argumentOfPerihelionRad) * np.sin(inclinationRad) * perifocalX
            + np.cos(argumentOfPerihelionRad) * np.sin(inclinationRad) * perifocalY
        )
        return positionX, positionY, positionZ

    @staticmethod
    def ellipticalPosition(
        semiMajorAxisAu: float,
        eccentricity: float,
        inclinationDeg: float,
        trueAnomalyRad: float | np.ndarray,
        ascendingNodeDeg: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        ascendingNodeRad = np.radians(ascendingNodeDeg)
        radiusAu = semiMajorAxisAu * (1 - eccentricity ** 2) / (
            1 + eccentricity * np.cos(trueAnomalyRad)
        )
        orbitPlaneX = radiusAu * np.cos(trueAnomalyRad)
        orbitPlaneY = radiusAu * np.sin(trueAnomalyRad)
        positionX = (
            orbitPlaneX * np.cos(ascendingNodeRad)
            - orbitPlaneY * np.cos(inclinationRad) * np.sin(ascendingNodeRad)
        )
        positionY = (
            orbitPlaneX * np.sin(ascendingNodeRad)
            + orbitPlaneY * np.cos(inclinationRad) * np.cos(ascendingNodeRad)
        )
        positionZ = orbitPlaneY * np.sin(inclinationRad)
        return positionX, positionY, positionZ

    @staticmethod
    def keplerianAngularVelocityRad(semiMajorAxisAu: float | np.ndarray) -> np.ndarray:
        """Angular speed (rad per day) using Kepler's third law with semi-major axis in AU."""
        semiMajorAxisArray = np.maximum(np.abs(np.asarray(semiMajorAxisAu, dtype=float)), 0.01)
        orbitalPeriodDays = semiMajorAxisArray ** 1.5 * 365.25
        return 2 * np.pi / orbitalPeriodDays

    @staticmethod
    def eclipticPosition2d(
        radiusAu: float | np.ndarray,
        inclinationDeg: float | np.ndarray,
        meanAnomalyRad: float | np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        positionX = radiusAu * np.cos(meanAnomalyRad)
        positionY = radiusAu * np.sin(meanAnomalyRad) * np.cos(inclinationRad)
        positionZ = radiusAu * np.sin(meanAnomalyRad) * np.sin(inclinationRad)
        return positionX, positionY, positionZ


class PlanetCatalog:
    def __init__(self, constants: AstronomicalConstants):
        self.constants = constants
        self.planets: Dict[str, PlanetOrbit] = self._buildCatalog()

    def _buildCatalog(self) -> Dict[str, PlanetOrbit]:
        constants = self.constants
        return {
            'Mercury': PlanetOrbit('Mercury', 0.387, 0.205, 7.0, 'gray', 4879, 88),
            'Venus': PlanetOrbit('Venus', 0.723, 0.007, 3.4, 'yellow', 12104, 224.7),
            'Earth': PlanetOrbit('Earth', 1.00, 0.017, 0.0, 'blue', 12742, 365.2),
            'Mars': PlanetOrbit('Mars', 1.52, 0.093, 1.85, 'red', 6779, 687),
            'Jupiter': PlanetOrbit('Jupiter', 5.20, 0.048, 1.3, 'orange', 139822, 4331),
            'Saturn': PlanetOrbit('Saturn', 9.58, 0.056, 2.49, 'gold', 116464, 10747),
            'Uranus': PlanetOrbit('Uranus', 19.22, 0.046, 0.77, 'lightblue', 50724, 30589),
            'Neptune': PlanetOrbit('Neptune', 30.05, 0.010, 1.77, 'blue', 49244, 59800),
            'Pluto': PlanetOrbit(
                'Pluto',
                constants.plutoSemiMajorAxis,
                constants.plutoEccentricity,
                17.16,
                'brown',
                2376,
                90560,
            ),
        }


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


class StarCatalog:
    def __init__(self, csvPath: str, constants: AstronomicalConstants):
        self.constants = constants
        self.starsDataFrame = self._loadStars(csvPath)

    def _loadStars(self, csvPath: str) -> pd.DataFrame:
        starsFrame = pd.read_csv(csvPath)
        starsFrame['distanceAu'] = starsFrame['Distance (ly)'] * self.constants.lightYearToAu
        coordinates = starsFrame.apply(
            lambda row: OrbitCalculator.parseRightAscensionAndDeclination(row['RA'], row['Dec']),
            axis=1,
        )
        starsFrame['rightAscensionDeg'] = coordinates.apply(lambda pair: pair[0])
        starsFrame['declinationDeg'] = coordinates.apply(lambda pair: pair[1])
        cartesianCoords = starsFrame.apply(
            lambda row: OrbitCalculator.equatorialToCartesianAu(
                row['rightAscensionDeg'], row['declinationDeg'], row['distanceAu']
            )
            if pd.notna(row['rightAscensionDeg']) and pd.notna(row['declinationDeg'])
            else (np.nan, np.nan, np.nan),
            axis=1,
        )
        starsFrame['positionX'] = cartesianCoords.apply(lambda coord: coord[0])
        starsFrame['positionY'] = cartesianCoords.apply(lambda coord: coord[1])
        starsFrame['positionZ'] = cartesianCoords.apply(lambda coord: coord[2])
        return starsFrame

    def vegaRow(self) -> pd.Series:
        return self.starsDataFrame[
            self.starsDataFrame['System'].str.startswith('Vega', na=False)
        ].iloc[0]

    def starsWithinLightYears(self, maxDistanceLy: float) -> pd.DataFrame:
        return self.starsDataFrame[
            (self.starsDataFrame['Distance (ly)'] <= maxDistanceLy)
            & self.starsDataFrame['positionX'].notna()
            & self.starsDataFrame['positionY'].notna()
        ]


class PointDensityConfig:
    """Scatter-point counts per belt/group for each zoom level."""

    DENSITIES_BY_VIEW: Dict[str, Tuple[int, int, int, int, int]] = {
        '0_inner_solar_system': (20000, 4000, 4000, 10000, 50000),
        '1_inner_solar_system_with_jupiter': (10000, 2000, 1000, 10000, 50000),
        '2_solar_system_with_kuiper_belt': (500, 20, 15, 10000, 50000),
        '3_solar_system_with_oort_cloud': (20, 10, 100, 100, 50000),
        '4_solar_system_with_alpha_centauri': (10, 5, 5, 50, 5000),
        '5_solar_system_with_nearest_stars_10': (2, 2, 2, 20, 2000),
        'inner_solar_system': (20000, 4000, 4000, 10000, 50000),
        'inner_solar_system_with_jupiter': (10000, 2000, 2000, 10000, 50000),
        'solar_system_with_kuiper_belt': (200, 100, 50, 10000, 50000),
        'solar_system_with_oort_cloud': (20, 10, 10, 100, 50000),
        'solar_system_with_alpha_centauri': (10, 5, 5, 50, 5000),
        'solar_system_with_nearest_stars_10': (2, 2, 2, 20, 2000),
        'solar_system_with_nearest_stars_25': (1, 1, 1, 10, 1000),
        'solar_system_with_nearest_stars_30': (1, 1, 1, 10, 1000),
        'default': (1, 1, 1, 10, 1000),
    }

    @classmethod
    def forView(cls, viewId: str) -> Dict[str, int]:
        densities = cls.DENSITIES_BY_VIEW.get(viewId, cls.DENSITIES_BY_VIEW['default'])
        return {
            'asteroidBelt': densities[0],
            'trojansAndGreeks': densities[1],
            'hildas': densities[2],
            'kuiperBelt': densities[3],
            'oortCloud': densities[4],
        }


class ViewRegistry:
    VIEWS_2D: List[ViewDefinition] = [
        ViewDefinition('0_inner_solar_system', -3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
        ViewDefinition('1_inner_solar_system_with_jupiter', -6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
        ViewDefinition('2_solar_system_with_kuiper_belt', -70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
        ViewDefinition('3_solar_system_with_oort_cloud', -100000, 100000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
        ViewDefinition('4_solar_system_with_alpha_centauri', -280000, 125000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
        ViewDefinition('5_solar_system_with_nearest_stars_10', -632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
        ViewDefinition('6_solar_system_with_nearest_stars_25', -1584189.9811, 1584189.9811, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years'),
        ViewDefinition('7_solar_system_with_nearest_stars_30', -1897232.3126, 1897232.3126, 80, 'solar_system_with_nearest_stars_30', 'Interstellar Neighbors Within 30 Light Years'),
    ]

    VIEWS_3D: List[ViewDefinition] = [
        ViewDefinition('0_inner_solar_system', -3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
        ViewDefinition('1_inner_solar_system_with_jupiter', -6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
        ViewDefinition('2_solar_system_with_kuiper_belt', -70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
        ViewDefinition('3_solar_system_with_oort_cloud', -100000, 100000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
        ViewDefinition('4_solar_system_with_alpha_centauri', -280000, 280000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
        ViewDefinition('5_solar_system_with_nearest_stars_10', -632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
        ViewDefinition('6_solar_system_with_nearest_stars_25', -1584188.9811, 1584188.9811, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years'),
        ViewDefinition('7_solar_system_with_nearest_stars_30', -1897232.3126, 1897232.3126, 80, 'solar_system_with_nearest_stars_30', 'Interstellar Neighbors Within 30 Light Years'),
    ]

    STAR_DISTANCE_LIMITS_LY: Dict[str, float] = {
        '5_solar_system_with_nearest_stars_10': 10,
        '6_solar_system_with_nearest_stars_25': 25.05,
        '7_solar_system_with_nearest_stars_30': 30,
        '4_solar_system_with_alpha_centauri': 5,
        'solar_system_with_nearest_stars_10': 10,
        'solar_system_with_nearest_stars_25': 25.05,
        'solar_system_with_nearest_stars_30': 30,
        'solar_system_with_alpha_centauri': 5,
    }

    @classmethod
    def axisLimitsForView(cls, viewId: str) -> Tuple[float, float]:
        for view in cls.VIEWS_2D:
            if view.viewId == viewId:
                return view.axisMinAu, view.axisMaxAu
        return -3.5, 3.5

    @classmethod
    def titleForView(cls, viewId: str) -> str:
        for view in cls.VIEWS_2D:
            if view.viewId == viewId:
                return view.title
        return 'Solar System'

    @classmethod
    def maxStarDistanceLy(cls, viewId: str) -> float:
        return cls.STAR_DISTANCE_LIMITS_LY.get(viewId, 25.05)


class BeltPointGenerator:
    """Random point clouds for asteroid belts and spherical shells."""

    @staticmethod
    def ring2d(innerRadiusAu: float, outerRadiusAu: float, numPoints: int) -> Tuple[np.ndarray, np.ndarray]:
        radiusAu = np.random.uniform(innerRadiusAu, outerRadiusAu, numPoints)
        azimuthRad = np.random.uniform(0, 2 * np.pi, numPoints)
        return radiusAu * np.cos(azimuthRad), radiusAu * np.sin(azimuthRad)

    @staticmethod
    def sphericalShell(
        innerRadiusAu: float,
        outerRadiusAu: float,
        shellThicknessAu: float,
        numPoints: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        azimuthRad = np.random.uniform(0, 2 * np.pi, numPoints)
        cosPolarAngle = np.random.uniform(-1, 1, numPoints)
        radiusAu = np.random.uniform(
            innerRadiusAu - shellThicknessAu / 2,
            outerRadiusAu + shellThicknessAu / 2,
            numPoints,
        )
        polarAngleRad = np.arccos(cosPolarAngle)
        positionX = radiusAu * np.sin(polarAngleRad) * np.cos(azimuthRad)
        positionY = radiusAu * np.sin(polarAngleRad) * np.sin(azimuthRad)
        positionZ = radiusAu * np.cos(polarAngleRad)
        return positionX, positionY, positionZ

    @staticmethod
    def jupiterLagrangeCloud2d(
        jupiterAngleRad: float,
        lagrangeOffsetRad: float,
        semiMajorAxisAu: float,
        radialSpreadAu: float,
        angularSpreadRad: float,
        numPoints: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        clusterCenterAngle = jupiterAngleRad + lagrangeOffsetRad
        radiusAu = np.random.uniform(
            semiMajorAxisAu - radialSpreadAu,
            semiMajorAxisAu + radialSpreadAu,
            numPoints,
        )
        azimuthRad = np.linspace(
            clusterCenterAngle - angularSpreadRad / 2,
            clusterCenterAngle + angularSpreadRad / 2,
            numPoints,
        )
        return radiusAu * np.cos(azimuthRad), radiusAu * np.sin(azimuthRad)
