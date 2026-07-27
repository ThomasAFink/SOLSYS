"""Keplerian and hyperbolic orbit math in heliocentric coordinates."""

from __future__ import annotations

import re

import numpy as np


class OrbitCalculator:
    """Keplerian and hyperbolic orbit math in heliocentric coordinates."""

    @staticmethod
    def parseRightAscensionToDegrees(rightAscension: str) -> float | None:
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
    ) -> tuple[float | None, float | None]:
        if not isinstance(rightAscension, str) or not isinstance(declination, str):
            return None, None
        raMatch = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', rightAscension)
        declinationNormalized = declination.replace('−', '-').replace('–', '-')
        decMatch = re.match(r'([+-]?\d+)°\s*(\d+)′\s*(\d+(?:\.\d*)?)″', declinationNormalized)
        if not raMatch or not decMatch:
            return None, None
        raDegrees = 15 * (
            float(raMatch.group(1)) + float(raMatch.group(2)) / 60 + float(raMatch.group(3)) / 3600
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
    ) -> tuple[float, float, float]:
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
    ) -> tuple[np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        trueAnomaly = np.linspace(0, 2 * np.pi, numPoints)
        radiusAu = (
            semiMajorAxisAu * (1 - eccentricity**2) / (1 + eccentricity * np.cos(trueAnomaly))
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        trueAnomaly = np.linspace(0, 2 * np.pi, numPoints)
        radiusAu = (
            semiMajorAxisAu * (1 - eccentricity**2) / (1 + eccentricity * np.cos(trueAnomaly))
        )
        positionX = radiusAu * np.cos(trueAnomaly)
        positionY = radiusAu * np.sin(trueAnomaly) * np.cos(inclinationRad)
        positionZ = radiusAu * np.sin(trueAnomaly) * np.sin(inclinationRad)
        return positionX, positionY, positionZ

    @staticmethod
    def hyperbolicAnomalyFromTrueAnomaly(
        trueAnomalyRad: float | np.ndarray, eccentricity: float
    ) -> np.ndarray:
        """Hyperbolic eccentric anomaly F from true anomaly ν."""
        trueAnomaly = np.asarray(trueAnomalyRad, dtype=float)
        sinhF = (np.sqrt(eccentricity**2 - 1.0) * np.sin(trueAnomaly)) / (
            1.0 + eccentricity * np.cos(trueAnomaly)
        )
        return np.arcsinh(sinhF)

    @staticmethod
    def hyperbolicMeanAnomalyFromTrueAnomaly(
        trueAnomalyRad: float | np.ndarray, eccentricity: float
    ) -> np.ndarray:
        """Mean anomaly M = e sinh F - F (proportional to time from perihelion)."""
        hyperbolicAnomaly = OrbitCalculator.hyperbolicAnomalyFromTrueAnomaly(
            trueAnomalyRad, eccentricity
        )
        return eccentricity * np.sinh(hyperbolicAnomaly) - hyperbolicAnomaly

    @staticmethod
    def hyperbolicTrueAnomalyFromMeanAnomaly(
        meanAnomalyRad: float | np.ndarray, eccentricity: float
    ) -> np.ndarray:
        """Invert M = e sinh F - F, then convert F → true anomaly ν."""
        meanAnomaly = np.asarray(meanAnomalyRad, dtype=float)
        hyperbolicAnomaly = np.array(meanAnomaly, copy=True, dtype=float)
        for _ in range(16):
            residual = eccentricity * np.sinh(hyperbolicAnomaly) - hyperbolicAnomaly - meanAnomaly
            derivative = eccentricity * np.cosh(hyperbolicAnomaly) - 1.0
            hyperbolicAnomaly = hyperbolicAnomaly - residual / np.maximum(derivative, 1e-12)
        sinhF = np.sinh(hyperbolicAnomaly)
        coshF = np.cosh(hyperbolicAnomaly)
        sinNu = (np.sqrt(eccentricity**2 - 1.0) * sinhF) / (eccentricity * coshF - 1.0)
        cosNu = (eccentricity - coshF) / (eccentricity * coshF - 1.0)
        return np.arctan2(sinNu, cosNu)

    @staticmethod
    def hyperbolicPosition(
        perihelionAu: float,
        eccentricity: float,
        inclinationDeg: float,
        longitudeAscendingNodeDeg: float,
        argumentOfPerihelionDeg: float,
        trueAnomalyRad: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Heliocentric position on a hyperbola at true anomaly ν."""
        inclinationRad = np.radians(inclinationDeg)
        ascendingNodeRad = np.radians(longitudeAscendingNodeDeg)
        argumentOfPerihelionRad = np.radians(argumentOfPerihelionDeg)
        trueAnomaly = np.asarray(trueAnomalyRad, dtype=float)
        radiusAu = perihelionAu * (1 + eccentricity) / (1 + eccentricity * np.cos(trueAnomaly))

        perifocalX = radiusAu * np.cos(trueAnomaly)
        perifocalY = radiusAu * np.sin(trueAnomaly)

        positionX = (
            np.cos(ascendingNodeRad) * np.cos(argumentOfPerihelionRad)
            - np.sin(ascendingNodeRad) * np.sin(argumentOfPerihelionRad) * np.cos(inclinationRad)
        ) * perifocalX + (
            -np.cos(ascendingNodeRad) * np.sin(argumentOfPerihelionRad)
            - np.sin(ascendingNodeRad) * np.cos(argumentOfPerihelionRad) * np.cos(inclinationRad)
        ) * perifocalY
        positionY = (
            np.sin(ascendingNodeRad) * np.cos(argumentOfPerihelionRad)
            + np.cos(ascendingNodeRad) * np.sin(argumentOfPerihelionRad) * np.cos(inclinationRad)
        ) * perifocalX + (
            -np.sin(ascendingNodeRad) * np.sin(argumentOfPerihelionRad)
            + np.cos(ascendingNodeRad) * np.cos(argumentOfPerihelionRad) * np.cos(inclinationRad)
        ) * perifocalY
        positionZ = (
            np.sin(argumentOfPerihelionRad) * np.sin(inclinationRad) * perifocalX
            + np.cos(argumentOfPerihelionRad) * np.sin(inclinationRad) * perifocalY
        )
        return positionX, positionY, positionZ

    @staticmethod
    def hyperbolicOrbit3d(
        perihelionAu: float,
        eccentricity: float,
        inclinationDeg: float,
        longitudeAscendingNodeDeg: float,
        argumentOfPerihelionDeg: float,
        numPoints: int = 1000,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        maxTrueAnomaly = np.arccos(-1 / eccentricity) - 1e-6
        trueAnomaly = np.linspace(-maxTrueAnomaly, maxTrueAnomaly, numPoints)
        return OrbitCalculator.hyperbolicPosition(
            perihelionAu,
            eccentricity,
            inclinationDeg,
            longitudeAscendingNodeDeg,
            argumentOfPerihelionDeg,
            trueAnomaly,
        )

    @staticmethod
    def ellipticalPosition(
        semiMajorAxisAu: float,
        eccentricity: float,
        inclinationDeg: float,
        trueAnomalyRad: float | np.ndarray,
        ascendingNodeDeg: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        ascendingNodeRad = np.radians(ascendingNodeDeg)
        radiusAu = (
            semiMajorAxisAu * (1 - eccentricity**2) / (1 + eccentricity * np.cos(trueAnomalyRad))
        )
        orbitPlaneX = radiusAu * np.cos(trueAnomalyRad)
        orbitPlaneY = radiusAu * np.sin(trueAnomalyRad)
        positionX = orbitPlaneX * np.cos(ascendingNodeRad) - orbitPlaneY * np.cos(
            inclinationRad
        ) * np.sin(ascendingNodeRad)
        positionY = orbitPlaneX * np.sin(ascendingNodeRad) + orbitPlaneY * np.cos(
            inclinationRad
        ) * np.cos(ascendingNodeRad)
        positionZ = orbitPlaneY * np.sin(inclinationRad)
        return positionX, positionY, positionZ

    @staticmethod
    def keplerianAngularVelocityRad(semiMajorAxisAu: float | np.ndarray) -> np.ndarray:
        """Angular speed (rad per day) using Kepler's third law with semi-major axis in AU."""
        semiMajorAxisArray = np.maximum(np.abs(np.asarray(semiMajorAxisAu, dtype=float)), 0.01)
        orbitalPeriodDays = semiMajorAxisArray**1.5 * 365.25
        return 2 * np.pi / orbitalPeriodDays

    @staticmethod
    def eclipticPosition2d(
        radiusAu: float | np.ndarray,
        inclinationDeg: float | np.ndarray,
        meanAnomalyRad: float | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(inclinationDeg)
        positionX = radiusAu * np.cos(meanAnomalyRad)
        positionY = radiusAu * np.sin(meanAnomalyRad) * np.cos(inclinationRad)
        positionZ = radiusAu * np.sin(meanAnomalyRad) * np.sin(inclinationRad)
        return positionX, positionY, positionZ
