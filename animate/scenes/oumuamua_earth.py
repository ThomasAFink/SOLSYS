"""ʻOumuamua–Earth closest-approach flyby animation (Sol heliocentric)."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from solsys.physics import AstronomicalConstants, OrbitCalculator, PlanetCatalog

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 160  # was 480; 3× faster overall
TRAJECTORY_SAMPLES = 1200
# Hold on closest-approach frame so the miss distance is readable.
CLOSEST_APPROACH_HOLD_FRAMES = 12  # was 36; keep hold share of the GIF


class OumuamuaEarthAnimator:
    """Side-view (X–Z) heliocentric flyby so ecliptic inclination is visible."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
    ):
        self.constants = AstronomicalConstants()
        self.orbitCalculator = OrbitCalculator()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.orbitColor = '#C8C8C8' if self.isDark else '#505050'
        self.labelColor = '#F2F2F2' if self.isDark else '#202020'
        self.oumuamuaColor = '#E0E0E0' if self.isDark else 'darkred'
        self.earthColor = self.planetCatalog.planets['Earth'].color
        self.sunColor = '#F6D56A'

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)
        self._prepareTrajectory()

    def _prepareTrajectory(self) -> None:
        constants = self.constants
        eccentricity = constants.oumuamuaEccentricity
        maxTrueAnomaly = float(np.arccos(-1.0 / eccentricity) - 1e-6)
        # Focus on the inner-system passage, but keep enough wing to read the hyperbola.
        trueAnomalySpan = min(maxTrueAnomaly, np.radians(125.0))
        meanAnomalySpan = float(
            np.abs(
                self.orbitCalculator.hyperbolicMeanAnomalyFromTrueAnomaly(
                    trueAnomalySpan, eccentricity
                )
            )
        )
        # Equal steps in hyperbolic mean anomaly ≈ equal steps in time.
        self.meanAnomalies = np.linspace(-meanAnomalySpan, meanAnomalySpan, TRAJECTORY_SAMPLES)
        self.trueAnomalies = self.orbitCalculator.hyperbolicTrueAnomalyFromMeanAnomaly(
            self.meanAnomalies, eccentricity
        )
        pathX, pathY, pathZ = self.orbitCalculator.hyperbolicPosition(
            constants.oumuamuaPerihelionAu,
            eccentricity,
            constants.oumuamuaInclinationDeg,
            constants.oumuamuaLongitudeAscendingNodeDeg,
            constants.oumuamuaArgumentOfPerihelionDeg,
            self.trueAnomalies,
        )
        self.pathX = np.asarray(pathX, dtype=float)
        self.pathY = np.asarray(pathY, dtype=float)
        self.pathZ = np.asarray(pathZ, dtype=float)

        # Time from perihelion (years): M = n t with n = sqrt(μ / a³), μ = 4π² AU³/yr².
        semiMajorAxisAu = constants.oumuamuaPerihelionAu / (eccentricity - 1.0)
        meanMotionRadPerYear = float(np.sqrt((4.0 * np.pi**2) / semiMajorAxisAu**3))
        self.timeYears = self.meanAnomalies / meanMotionRadPerYear

        earth = self.planetCatalog.planets['Earth']
        self.earthSemiMajorAxisAu = earth.semiMajorAxisAu
        # Geometric closest approach to Earth's orbit in XY, then phase Earth there.
        # (After perihelion / outbound — matches the historical Oct 2017 Earth flyby ordering.)
        radialAu = np.hypot(self.pathX, self.pathY)
        perihelionIndex = int(np.argmin(np.sqrt(self.pathX**2 + self.pathY**2 + self.pathZ**2)))
        outbound = np.arange(perihelionIndex, len(radialAu))
        distanceToEarthOrbit = np.abs(radialAu[outbound] - earth.semiMajorAxisAu)
        closestIndex = int(outbound[int(np.argmin(distanceToEarthOrbit))])
        self.closestIndex = closestIndex
        oumuamuaAtClosest = np.array(
            [self.pathX[closestIndex], self.pathY[closestIndex], self.pathZ[closestIndex]]
        )
        xyNorm = float(np.hypot(oumuamuaAtClosest[0], oumuamuaAtClosest[1])) or 1.0
        earthAtClosestX = earth.semiMajorAxisAu * oumuamuaAtClosest[0] / xyNorm
        earthAtClosestY = earth.semiMajorAxisAu * oumuamuaAtClosest[1] / xyNorm
        self.earthAngleAtClosestRad = float(np.arctan2(earthAtClosestY, earthAtClosestX))
        self.timeYearsAtClosest = float(self.timeYears[closestIndex])
        self.geometricMissAu = float(
            np.linalg.norm(oumuamuaAtClosest - np.array([earthAtClosestX, earthAtClosestY, 0.0]))
        )
        self.missDistanceAu = constants.oumuamuaEarthClosestApproachAu
        self.missDistanceKm = self.missDistanceAu * constants.auToKm

        approachFrames = ANIMATION_FRAMES - CLOSEST_APPROACH_HOLD_FRAMES
        before = max(1, int(approachFrames * closestIndex / max(len(self.trueAnomalies) - 1, 1)))
        after = max(1, approachFrames - before)
        self.frameToSample = np.concatenate(
            [
                np.linspace(0, closestIndex, before, endpoint=False, dtype=int),
                np.full(CLOSEST_APPROACH_HOLD_FRAMES, closestIndex, dtype=int),
                np.linspace(
                    closestIndex, len(self.trueAnomalies) - 1, after, endpoint=True, dtype=int
                ),
            ]
        )
        self.animationFrames = len(self.frameToSample)

        # Side view: X horizontal, Z vertical (looking along −Y). Planet orbits sit on Z≈0.
        earthOrbitAngle = np.linspace(0, 2 * np.pi, 360)
        self.earthOrbitX = earth.semiMajorAxisAu * np.cos(earthOrbitAngle)
        self.earthOrbitZ = np.zeros_like(self.earthOrbitX)

        self.contextOrbits: list[tuple[np.ndarray, np.ndarray]] = []
        for name in ('Mercury', 'Venus', 'Mars'):
            planet = self.planetCatalog.planets[name]
            angle = np.linspace(0, 2 * np.pi, 240)
            orbitX, _, orbitZ = self.orbitCalculator.ellipticalPosition(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                angle,
            )
            self.contextOrbits.append((np.asarray(orbitX), np.asarray(orbitZ)))

        maxX = float(np.max(np.abs(self.pathX)))
        maxZ = float(np.max(np.abs(self.pathZ)))
        self.axisLimitX = max(2.2, maxX * 1.08)
        self.axisLimitZ = max(1.6, maxZ * 1.15)

    def _earthPositionXz(self, sampleIndex: int) -> tuple[float, float]:
        """Earth on its orbit, phased so it meets the flyby geometry at closest approach."""
        timeYears = float(self.timeYears[sampleIndex])
        # 2π rad/year for a≈1 AU circular Earth.
        earthAngleRad = self.earthAngleAtClosestRad + 2.0 * np.pi * (
            timeYears - self.timeYearsAtClosest
        )
        return (
            self.earthSemiMajorAxisAu * float(np.cos(earthAngleRad)),
            0.0,
        )

    def update(self, frame: int):
        sampleIndex = int(self.frameToSample[frame])
        oumuamuaX = self.pathX[sampleIndex]
        oumuamuaZ = self.pathZ[sampleIndex]
        earthX, earthZ = self._earthPositionXz(sampleIndex)
        atClosest = sampleIndex == self.closestIndex

        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitX, self.axisLimitX)
        self.axes.set_ylim(-self.axisLimitZ, self.axisLimitZ)
        self.axes.set_title(
            "'Oumuamua flyby — closest approach to Earth (side view, X–Z)",
            color=self.labelColor,
            pad=14,
        )

        # Ecliptic reference.
        self.axes.axhline(0.0, color=self.orbitColor, linewidth=0.6, alpha=0.35, zorder=1)
        self.axes.text(
            self.axisLimitX * 0.62,
            self.axisLimitZ * 0.04,
            'ecliptic',
            color=self.labelColor,
            fontsize=7,
            alpha=0.55,
        )

        for orbitX, orbitZ in self.contextOrbits:
            self.axes.plot(orbitX, orbitZ, color=self.orbitColor, linewidth=0.5, alpha=0.35)
        self.axes.plot(
            self.earthOrbitX,
            self.earthOrbitZ,
            color=self.earthColor,
            linewidth=1.0,
            alpha=0.55,
        )
        self.axes.plot(
            self.pathX,
            self.pathZ,
            color=self.oumuamuaColor,
            linewidth=1.1,
            alpha=0.55,
            linestyle='--',
        )

        self.axes.scatter([0], [0], s=220, color=self.sunColor, zorder=5)
        self.axes.text(0.08, -0.18, 'Sun', color=self.labelColor, fontsize=8)

        self.axes.scatter([earthX], [earthZ], s=70, color=self.earthColor, zorder=6)
        self.axes.text(earthX + 0.08, earthZ + 0.08, 'Earth', color=self.labelColor, fontsize=8)

        self.axes.scatter([oumuamuaX], [oumuamuaZ], s=22, color=self.oumuamuaColor, zorder=7)
        self.axes.text(
            oumuamuaX + 0.08,
            oumuamuaZ - 0.12,
            "'Oumuamua",
            color=self.oumuamuaColor,
            fontsize=8,
        )

        if atClosest:
            self.axes.plot(
                [oumuamuaX, earthX],
                [oumuamuaZ, earthZ],
                color=self.labelColor,
                linewidth=1.2,
                alpha=0.9,
                zorder=4,
            )
            midX = 0.5 * (oumuamuaX + earthX)
            midZ = 0.5 * (oumuamuaZ + earthZ)
            self.axes.text(
                midX + 0.05,
                midZ + 0.12,
                f'Closest approach\n{self.missDistanceAu:.3f} AU\n({self.missDistanceKm / 1e6:.1f}×10⁶ km)',
                color=self.labelColor,
                fontsize=9,
                ha='left',
                va='bottom',
                bbox={
                    'boxstyle': 'round,pad=0.25',
                    'facecolor': '#111111' if self.isDark else '#FFFFFF',
                    'edgecolor': self.labelColor,
                    'alpha': 0.8,
                },
            )

        self.axes.text(
            -self.axisLimitX * 0.95,
            -self.axisLimitZ * 0.92,
            'Side view (X–Z) · Earth orbits in sync with flyby time · ecliptic edge-on',
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )
        return []

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath) or '.', exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.animationFrames,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        self.figure.set_size_inches(*self.figureSizeInches)
        self.figure.set_dpi(self.dpi)
        animation.save(outputPath, writer=PillowWriter(fps=ANIMATION_FPS))
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderOumuamuaEarthAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
) -> None:
    outputDirectory = 'output/animate/oumuamua'
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{outputDirectory}/oumuamua_earth_flyby_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = OumuamuaEarthAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
        )
        animator.saveGif(outputPath)
    print("'Oumuamua–Earth animations completed!")
