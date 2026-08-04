"""Randomized asteroid fields with Keplerian motion and Jupiter-coupled groups."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solsys.motion.mean_anomaly import meanAnomalyAtFrame
from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.belt_point_generator import BeltPointGenerator
from solsys.physics.orbit_calculator import OrbitCalculator

LAGRANGE_OFFSET_RAD = np.pi / 3
BELT_SHELL_THICKNESS_AU = 0.05
KUIPER_SHELL_THICKNESS_AU = 0.05
OORT_SHELL_THICKNESS_AU = BELT_SHELL_THICKNESS_AU * 5


@dataclass(frozen=True)
class AsteroidPopulationCounts:
    asteroidBelt: int = 800
    hildas: int = 240
    trojansAndGreeks: int = 150
    kuiperBelt: int = 2000
    oortCloud: int = 6000


class AnimatedAsteroidPopulation:
    """Randomized asteroid fields with Keplerian motion and Jupiter-coupled groups."""

    def __init__(
        self,
        constants: AstronomicalConstants,
        counts: AsteroidPopulationCounts,
        includeKuiperAndOort: bool = False,
        useSphericalShell3d: bool = False,
    ):
        self.constants = constants
        self.orbitCalculator = OrbitCalculator()
        self.beltGenerator = BeltPointGenerator()
        self.useSphericalShell3d = useSphericalShell3d
        if useSphericalShell3d:
            self._initAsteroidBeltShell(counts.asteroidBelt)
        else:
            self._initAsteroidBelt(counts.asteroidBelt)
        self._initHildas(counts.hildas)
        self._initJupiterLagrangeClouds(counts.trojansAndGreeks)
        if includeKuiperAndOort:
            if useSphericalShell3d:
                self._initKuiperBeltShell(counts.kuiperBelt)
                self._initOortCloudShell(counts.oortCloud)
            else:
                self._initKuiperBelt(counts.kuiperBelt)
                self._initOortCloud(counts.oortCloud)

    @staticmethod
    def _cartesianToSpherical(
        positionX: np.ndarray, positionY: np.ndarray, positionZ: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        radiusAu = np.sqrt(positionX**2 + positionY**2 + positionZ**2)
        azimuthRad = np.arctan2(positionY, positionX)
        polarAngleRad = np.arccos(np.clip(positionZ / np.maximum(radiusAu, 1e-9), -1.0, 1.0))
        return radiusAu, azimuthRad, polarAngleRad

    @staticmethod
    def _sphericalToCartesian(
        radiusAu: np.ndarray, azimuthRad: np.ndarray, polarAngleRad: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        positionX = radiusAu * np.sin(polarAngleRad) * np.cos(azimuthRad)
        positionY = radiusAu * np.sin(polarAngleRad) * np.sin(azimuthRad)
        positionZ = radiusAu * np.cos(polarAngleRad)
        return positionX, positionY, positionZ

    def _initAsteroidBeltShell(self, numPoints: int) -> None:
        constants = self.constants
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            constants.asteroidBeltInnerAu,
            constants.asteroidBeltOuterAu,
            BELT_SHELL_THICKNESS_AU,
            numPoints,
        )
        (
            self.beltShellRadiusAu,
            self.beltShellInitialAzimuthRad,
            self.beltShellPolarAngleRad,
        ) = self._cartesianToSpherical(positionX, positionY, positionZ)

    def _initKuiperBeltShell(self, numPoints: int) -> None:
        constants = self.constants
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            constants.kuiperBeltInnerAu,
            constants.kuiperBeltOuterAu,
            KUIPER_SHELL_THICKNESS_AU,
            numPoints,
        )
        (
            self.kuiperShellRadiusAu,
            self.kuiperShellInitialAzimuthRad,
            self.kuiperShellPolarAngleRad,
        ) = self._cartesianToSpherical(positionX, positionY, positionZ)

    def _initOortCloudShell(self, numPoints: int) -> None:
        constants = self.constants
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            constants.oortCloudInnerAu,
            constants.oortCloudOuterAu,
            OORT_SHELL_THICKNESS_AU,
            numPoints,
        )
        (
            self.oortShellRadiusAu,
            self.oortShellInitialAzimuthRad,
            self.oortShellPolarAngleRad,
        ) = self._cartesianToSpherical(positionX, positionY, positionZ)

    def _initAsteroidBelt(self, numPoints: int) -> None:
        constants = self.constants
        self.beltMeanAnomalyRad = np.random.uniform(0, 2 * np.pi, numPoints)
        self.beltSemiMajorAxisAu = np.random.uniform(
            constants.asteroidBeltInnerAu,
            constants.asteroidBeltOuterAu,
            numPoints,
        )
        self.beltEccentricity = np.random.uniform(0.0, 0.15, numPoints)
        self.beltInclinationDeg = np.random.uniform(-10.0, 10.0, numPoints)
        self.beltAscendingNodeDeg = np.random.uniform(0.0, 360.0, numPoints)
        self.beltPhaseRad = np.random.uniform(0, 2 * np.pi, numPoints)
        self.beltOscillationAmplitude = np.random.uniform(0.1, 0.3, numPoints)
        self.beltOscillationFrequency = np.random.uniform(0.01, 0.03, numPoints)
        self.beltRadialVariationAu = np.random.uniform(0.05, 0.15, numPoints)

    @staticmethod
    def _arcAnglesAround(
        centerAnglesRad: list[float],
        numPoints: int,
        halfArcRad: float,
    ) -> np.ndarray:
        """Fill long arcs around each center so groups wrap the orbit with clear gaps."""
        pointsPerArc = max(numPoints // max(len(centerAnglesRad), 1), 1)
        angles: list[float] = []
        for centerRad in centerAnglesRad:
            # Triangle-ish density: denser near the arc center, still spanning the full arc.
            offsets = np.random.uniform(-1.0, 1.0, pointsPerArc)
            offsets = np.sign(offsets) * (np.abs(offsets) ** 1.35) * halfArcRad
            angles.extend((centerRad + offsets) % (2.0 * np.pi))
        return np.asarray(angles, dtype=float)

    def _initHildas(self, numPoints: int) -> None:
        # Three Hilda groups ~120° apart: long arcs around the ring, gaps between them.
        clusterAnglesRad = [0.0, 2 * np.pi / 3, 4 * np.pi / 3]
        self.hildaMeanAnomalyRad = self._arcAnglesAround(
            clusterAnglesRad, numPoints, halfArcRad=np.radians(42.0)
        )
        innerRadiusAu = self.constants.asteroidBeltOuterAu + 0.25
        outerRadiusAu = self.constants.jupiterSemiMajorAxisAu - 0.25
        self.hildaSemiMajorAxisAu = np.random.uniform(
            innerRadiusAu, outerRadiusAu, len(self.hildaMeanAnomalyRad)
        )
        # Match the volumetric feel of the spherical-shell main belt / Kuiper.
        inclinationSpreadDeg = 28.0 if self.useSphericalShell3d else 12.0
        self.hildaInclinationDeg = np.random.uniform(
            -inclinationSpreadDeg, inclinationSpreadDeg, len(self.hildaMeanAnomalyRad)
        )
        self.hildaOscillationAmplitude = np.random.uniform(0.1, 0.3, len(self.hildaMeanAnomalyRad))
        self.hildaPhaseRad = np.random.uniform(0, 2 * np.pi, len(self.hildaMeanAnomalyRad))

    def _initJupiterLagrangeClouds(self, numPoints: int) -> None:
        # L4/L5 clouds as long arcs on Jupiter's orbit — wrap most of the circle, keep
        # empty gaps around Jupiter (0°) and the far side between the two swarms.
        self.trojanMeanAnomalyOffsetRad = self._arcAnglesAround(
            [LAGRANGE_OFFSET_RAD], numPoints, halfArcRad=np.radians(48.0)
        )
        self.greekMeanAnomalyOffsetRad = self._arcAnglesAround(
            [-LAGRANGE_OFFSET_RAD], numPoints, halfArcRad=np.radians(48.0)
        )
        # Unwrap greek offsets back near -60° (modulo can park them near +300°).
        self.greekMeanAnomalyOffsetRad = (
            (self.greekMeanAnomalyOffsetRad + np.pi) % (2.0 * np.pi)
        ) - np.pi
        radialSpreadAu = 0.55
        jupiterSemiMajorAxisAu = self.constants.jupiterSemiMajorAxisAu
        trojanCount = len(self.trojanMeanAnomalyOffsetRad)
        greekCount = len(self.greekMeanAnomalyOffsetRad)
        self.trojanSemiMajorAxisAu = np.random.uniform(
            jupiterSemiMajorAxisAu - radialSpreadAu,
            jupiterSemiMajorAxisAu + radialSpreadAu,
            trojanCount,
        )
        self.greekSemiMajorAxisAu = np.random.uniform(
            jupiterSemiMajorAxisAu - radialSpreadAu,
            jupiterSemiMajorAxisAu + radialSpreadAu,
            greekCount,
        )
        inclinationSpreadDeg = 22.0 if self.useSphericalShell3d else 8.0
        verticalSpreadAu = 0.45 if self.useSphericalShell3d else 0.2
        self.trojanInclinationDeg = np.random.uniform(
            -inclinationSpreadDeg, inclinationSpreadDeg, trojanCount
        )
        self.greekInclinationDeg = np.random.uniform(
            -inclinationSpreadDeg, inclinationSpreadDeg, greekCount
        )
        self.trojanLibrationAmplitude = np.random.uniform(0.05, 0.15, trojanCount)
        self.greekLibrationAmplitude = np.random.uniform(0.05, 0.15, greekCount)
        self.trojanPhaseRad = np.random.uniform(0, 2 * np.pi, trojanCount)
        self.greekPhaseRad = np.random.uniform(0, 2 * np.pi, greekCount)
        self.trojanVerticalOffsetAu = np.random.uniform(
            -verticalSpreadAu, verticalSpreadAu, trojanCount
        )
        self.greekVerticalOffsetAu = np.random.uniform(
            -verticalSpreadAu, verticalSpreadAu, greekCount
        )

    def _initKuiperBelt(self, numPoints: int) -> None:
        constants = self.constants
        self.kuiperMeanAnomalyRad = np.random.uniform(0, 2 * np.pi, numPoints)
        self.kuiperSemiMajorAxisAu = np.random.uniform(
            constants.kuiperBeltInnerAu,
            constants.kuiperBeltOuterAu,
            numPoints,
        )
        self.kuiperEccentricity = np.random.uniform(0.02, 0.2, numPoints)
        self.kuiperInclinationDeg = np.random.uniform(-20.0, 20.0, numPoints)
        self.kuiperAscendingNodeDeg = np.random.uniform(0.0, 360.0, numPoints)

    def _initOortCloud(self, numPoints: int) -> None:
        constants = self.constants
        self.oortAzimuthRad = np.random.uniform(0, 2 * np.pi, numPoints)
        self.oortPolarAngleRad = np.arccos(2 * np.random.uniform(0, 1, numPoints) - 1)
        self.oortRadiusAu = (
            np.random.power(0.5, numPoints)
            * (constants.oortCloudOuterAu - constants.oortCloudInnerAu)
            + constants.oortCloudInnerAu
        )
        self.oortRotationRad = np.random.uniform(0, 2 * np.pi, numPoints)

    def _shellPositionsFromSpherical(
        self,
        radiusAu: np.ndarray,
        initialAzimuthRad: np.ndarray,
        polarAngleRad: np.ndarray,
        frame: int,
        animationSpeed: float,
        angularVelocityScale: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        angularVelocityRad = (
            OrbitCalculator.keplerianAngularVelocityRad(radiusAu) * angularVelocityScale
        )
        azimuthRad = initialAzimuthRad - animationSpeed * frame * angularVelocityRad
        return self._sphericalToCartesian(radiusAu, azimuthRad, polarAngleRad)

    def _asteroidBeltShellPositions(
        self, frame: int, animationSpeed: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._shellPositionsFromSpherical(
            self.beltShellRadiusAu,
            self.beltShellInitialAzimuthRad,
            self.beltShellPolarAngleRad,
            frame,
            animationSpeed,
        )

    def _kuiperBeltShellPositions(
        self, frame: int, animationSpeed: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._shellPositionsFromSpherical(
            self.kuiperShellRadiusAu,
            self.kuiperShellInitialAzimuthRad,
            self.kuiperShellPolarAngleRad,
            frame,
            animationSpeed,
        )

    def _oortCloudShellPositions(
        self, frame: int, animationSpeed: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._shellPositionsFromSpherical(
            self.oortShellRadiusAu,
            self.oortShellInitialAzimuthRad,
            self.oortShellPolarAngleRad,
            frame,
            animationSpeed,
            angularVelocityScale=0.0005,
        )

    def asteroidBeltPositions(
        self, frame: int, animationSpeed: float, ecliptic2d: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.useSphericalShell3d and not ecliptic2d:
            return self._asteroidBeltShellPositions(frame, animationSpeed)
        orbitalPeriodDays = self.beltSemiMajorAxisAu**1.5 * 365.25
        meanAnomalyRad = meanAnomalyAtFrame(
            self.beltMeanAnomalyRad, orbitalPeriodDays, frame, animationSpeed
        )
        eccentricRadiusAu = self.beltSemiMajorAxisAu * (
            1 + self.beltEccentricity * np.cos(meanAnomalyRad)
        )
        radialOscillationAu = self.beltRadialVariationAu * np.sin(
            frame * self.beltOscillationFrequency + self.beltPhaseRad
        )
        radiusAu = eccentricRadiusAu + radialOscillationAu
        orbitalPerturbationRad = self.beltOscillationAmplitude * np.cos(
            frame * 0.03 + self.beltPhaseRad
        )
        meanAnomalyWithPerturbationRad = meanAnomalyRad + orbitalPerturbationRad
        if ecliptic2d:
            return self.orbitCalculator.eclipticPosition2d(
                radiusAu, self.beltInclinationDeg, meanAnomalyWithPerturbationRad
            )
        return self.orbitCalculator.ellipticalPosition(
            radiusAu,
            np.zeros_like(radiusAu),
            self.beltInclinationDeg,
            meanAnomalyWithPerturbationRad,
            self.beltAscendingNodeDeg,
        )

    def hildaPositions(
        self,
        frame: int,
        jupiterMeanAnomalyRad: float,
        animationSpeed: float,
        ecliptic2d: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        del animationSpeed
        # 3:2 resonance: three Hilda orbits per two Jupiter orbits → angular rate 3/2
        meanAnomalyRad = self.hildaMeanAnomalyRad + jupiterMeanAnomalyRad * (3 / 2)
        oscillationAu = self.hildaOscillationAmplitude * np.sin(frame * 0.1 + self.hildaPhaseRad)
        radiusAu = self.hildaSemiMajorAxisAu + oscillationAu
        if ecliptic2d:
            return self.orbitCalculator.eclipticPosition2d(
                radiusAu, self.hildaInclinationDeg, meanAnomalyRad
            )
        return self.orbitCalculator.ellipticalPosition(
            radiusAu,
            np.zeros_like(radiusAu),
            self.hildaInclinationDeg,
            meanAnomalyRad,
        )

    def trojanPositions(
        self, frame: int, jupiterMeanAnomalyRad: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        librationRad = self.trojanLibrationAmplitude * np.sin(frame * 0.05 + self.trojanPhaseRad)
        meanAnomalyRad = self.trojanMeanAnomalyOffsetRad + jupiterMeanAnomalyRad + librationRad
        positionX, positionY, positionZ = self.orbitCalculator.ellipticalPosition(
            self.trojanSemiMajorAxisAu,
            np.zeros_like(self.trojanSemiMajorAxisAu),
            self.trojanInclinationDeg,
            meanAnomalyRad,
        )
        positionZ = positionZ + self.trojanVerticalOffsetAu
        return positionX, positionY, positionZ

    def greekPositions(
        self, frame: int, jupiterMeanAnomalyRad: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        librationRad = self.greekLibrationAmplitude * np.sin(
            frame * 0.05 + self.greekPhaseRad + np.pi
        )
        meanAnomalyRad = self.greekMeanAnomalyOffsetRad + jupiterMeanAnomalyRad + librationRad
        positionX, positionY, positionZ = self.orbitCalculator.ellipticalPosition(
            self.greekSemiMajorAxisAu,
            np.zeros_like(self.greekSemiMajorAxisAu),
            self.greekInclinationDeg,
            meanAnomalyRad,
        )
        positionZ = positionZ + self.greekVerticalOffsetAu
        return positionX, positionY, positionZ

    def kuiperBeltPositions(
        self, frame: int, animationSpeed: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.useSphericalShell3d:
            return self._kuiperBeltShellPositions(frame, animationSpeed)
        orbitalPeriodDays = self.kuiperSemiMajorAxisAu**1.5 * 365.25
        meanAnomalyRad = meanAnomalyAtFrame(
            self.kuiperMeanAnomalyRad, orbitalPeriodDays, frame, animationSpeed
        )
        semiMajorAxisAu = self.kuiperSemiMajorAxisAu * (
            1 + self.kuiperEccentricity * np.cos(meanAnomalyRad)
        )
        return self.orbitCalculator.ellipticalPosition(
            semiMajorAxisAu,
            np.zeros_like(semiMajorAxisAu),
            self.kuiperInclinationDeg,
            meanAnomalyRad,
            self.kuiperAscendingNodeDeg,
        )

    def oortCloudPositions(
        self, frame: int, animationSpeed: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.useSphericalShell3d:
            return self._oortCloudShellPositions(frame, animationSpeed)
        angularVelocity = OrbitCalculator.keplerianAngularVelocityRad(self.oortRadiusAu) * 0.0005
        rotationRad = self.oortRotationRad - animationSpeed * frame * angularVelocity
        azimuthRad = self.oortAzimuthRad + rotationRad
        positionX = self.oortRadiusAu * np.sin(self.oortPolarAngleRad) * np.cos(azimuthRad)
        positionY = self.oortRadiusAu * np.sin(self.oortPolarAngleRad) * np.sin(azimuthRad)
        positionZ = self.oortRadiusAu * np.cos(self.oortPolarAngleRad)
        return positionX, positionY, positionZ
