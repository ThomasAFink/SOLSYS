"""Interstellar-object hyperbolic passage animations (1I / 2I / 3I)."""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from solsys.physics import AstronomicalConstants, OrbitCalculator, PlanetCatalog
from solsys.physics.catalogs.interstellar_object_catalog import (
    DEFAULT_INTERSTELLAR_OBJECTS_CSV,
    InterstellarObject,
    InterstellarObjectCatalog,
)

ViewName = Literal['side', 'oblique']
HighlightMode = Literal['earth_flyby', 'perihelion']

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 160
TRAJECTORY_SAMPLES = 1200
CLOSEST_APPROACH_HOLD_FRAMES = 12

OBLIQUE_ELEVATION_DEG = 26.0
OBLIQUE_AZIMUTH_OFFSET_DEG = 125.0

OUTPUT_DIRECTORY = 'output/animate/interstellar_objects'


class InterstellarObjectAnimator:
    """Edge-on or oblique heliocentric scene for one interstellar visitor."""

    def __init__(
        self,
        interstellarObject: InterstellarObject,
        view: ViewName = 'side',
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
    ):
        if view not in ('side', 'oblique'):
            raise ValueError(f'Unknown view: {view!r}')
        if interstellarObject.highlight not in ('earth_flyby', 'perihelion'):
            raise ValueError(f'Unknown highlight: {interstellarObject.highlight!r}')

        self.visitor = interstellarObject
        self.view = view
        self.highlight: HighlightMode = interstellarObject.highlight  # type: ignore[assignment]
        self.constants = AstronomicalConstants()
        self.orbitCalculator = OrbitCalculator()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.orbitColor = '#C8C8C8' if self.isDark else '#505050'
        self.labelColor = '#F2F2F2' if self.isDark else '#202020'
        self.visitorColor = interstellarObject.colorForStyle(self.isDark)
        self.earthColor = self.planetCatalog.planets['Earth'].color
        self.sunColor = '#F6D56A'

        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi)
        if view == 'oblique':
            self.axes = self.figure.add_subplot(111, projection='3d')
        else:
            self.axes = self.figure.add_subplot(111)
        self._prepareTrajectory()

    @staticmethod
    def _trueAnomalyAtRadiusAu(perihelionAu: float, eccentricity: float, radiusAu: float) -> float:
        """True anomaly ν where r(ν) = radiusAu on the outbound/inbound wing."""
        cosNu = (perihelionAu * (1.0 + eccentricity) / radiusAu - 1.0) / eccentricity
        cosNu = float(np.clip(cosNu, -1.0 + 1e-9, 1.0 - 1e-9))
        return float(np.arccos(cosNu))

    def _prepareTrajectory(self) -> None:
        visitor = self.visitor
        eccentricity = visitor.eccentricity
        maxTrueAnomaly = float(np.arccos(-1.0 / eccentricity) - 1e-6)
        # Cap by published angle and by radius so high-e wings do not explode to infinity.
        radiusLimitedNu = self._trueAnomalyAtRadiusAu(
            visitor.perihelionAu, eccentricity, visitor.maxHeliocentricAu
        )
        trueAnomalySpan = min(
            maxTrueAnomaly,
            np.radians(visitor.trueAnomalySpanDeg),
            radiusLimitedNu,
        )
        meanAnomalySpan = float(
            np.abs(
                self.orbitCalculator.hyperbolicMeanAnomalyFromTrueAnomaly(
                    trueAnomalySpan, eccentricity
                )
            )
        )
        self.meanAnomalies = np.linspace(-meanAnomalySpan, meanAnomalySpan, TRAJECTORY_SAMPLES)
        self.trueAnomalies = self.orbitCalculator.hyperbolicTrueAnomalyFromMeanAnomaly(
            self.meanAnomalies, eccentricity
        )
        pathX, pathY, pathZ = self.orbitCalculator.hyperbolicPosition(
            visitor.perihelionAu,
            eccentricity,
            visitor.inclinationDeg,
            visitor.longitudeAscendingNodeDeg,
            visitor.argumentOfPerihelionDeg,
            self.trueAnomalies,
        )
        self.pathX = np.asarray(pathX, dtype=float)
        self.pathY = np.asarray(pathY, dtype=float)
        self.pathZ = np.asarray(pathZ, dtype=float)

        # |a| = q / (e − 1); M = n t with n = sqrt(μ / |a|³), μ = 4π² AU³/yr².
        semiMajorAxisAu = visitor.perihelionAu / (eccentricity - 1.0)
        meanMotionRadPerYear = float(np.sqrt((4.0 * np.pi**2) / semiMajorAxisAu**3))
        self.timeYears = self.meanAnomalies / meanMotionRadPerYear

        earth = self.planetCatalog.planets['Earth']
        self.earthSemiMajorAxisAu = earth.semiMajorAxisAu
        radialAu = np.hypot(self.pathX, self.pathY)
        perihelionIndex = int(np.argmin(np.sqrt(self.pathX**2 + self.pathY**2 + self.pathZ**2)))
        self.perihelionIndex = perihelionIndex

        if self.highlight == 'earth_flyby':
            # After perihelion / outbound — matches historical ʻOumuamua Earth flyby ordering.
            outbound = np.arange(perihelionIndex, len(radialAu))
            distanceToEarthOrbit = np.abs(radialAu[outbound] - earth.semiMajorAxisAu)
            eventIndex = int(outbound[int(np.argmin(distanceToEarthOrbit))])
            publishedMiss = visitor.earthClosestApproachAu
            if publishedMiss is None:
                raise ValueError(
                    f'{visitor.objectId} earth_flyby highlight needs earth_closest_approach_au'
                )
            self.calloutAu = publishedMiss
            self.calloutLabel = 'Closest approach'
        else:
            eventIndex = perihelionIndex
            self.calloutAu = visitor.perihelionAu
            self.calloutLabel = 'Perihelion'

        self.eventIndex = eventIndex
        visitorAtEvent = np.array(
            [self.pathX[eventIndex], self.pathY[eventIndex], self.pathZ[eventIndex]]
        )
        xyNorm = float(np.hypot(visitorAtEvent[0], visitorAtEvent[1])) or 1.0
        earthAtEventX = earth.semiMajorAxisAu * visitorAtEvent[0] / xyNorm
        earthAtEventY = earth.semiMajorAxisAu * visitorAtEvent[1] / xyNorm
        self.earthAngleAtEventRad = float(np.arctan2(earthAtEventY, earthAtEventX))
        self.timeYearsAtEvent = float(self.timeYears[eventIndex])
        self.calloutKm = self.calloutAu * self.constants.auToKm
        self.showEarthMissLine = self.highlight == 'earth_flyby'

        approachFrames = ANIMATION_FRAMES - CLOSEST_APPROACH_HOLD_FRAMES
        before = max(1, int(approachFrames * eventIndex / max(len(self.trueAnomalies) - 1, 1)))
        after = max(1, approachFrames - before)
        self.frameToSample = np.concatenate(
            [
                np.linspace(0, eventIndex, before, endpoint=False, dtype=int),
                np.full(CLOSEST_APPROACH_HOLD_FRAMES, eventIndex, dtype=int),
                np.linspace(
                    eventIndex, len(self.trueAnomalies) - 1, after, endpoint=True, dtype=int
                ),
            ]
        )
        self.animationFrames = len(self.frameToSample)

        earthOrbitAngle = np.linspace(0, 2 * np.pi, 360)
        self.earthOrbitX = earth.semiMajorAxisAu * np.cos(earthOrbitAngle)
        self.earthOrbitY = earth.semiMajorAxisAu * np.sin(earthOrbitAngle)
        self.earthOrbitZ = np.zeros_like(self.earthOrbitX)

        contextNames = ['Mercury', 'Venus', 'Mars']
        if visitor.perihelionAu >= 1.5:
            contextNames.append('Jupiter')
        self.contextOrbits: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for name in contextNames:
            planet = self.planetCatalog.planets[name]
            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=240,
            )
            self.contextOrbits.append((np.asarray(orbitX), np.asarray(orbitY), np.asarray(orbitZ)))

        if self.view == 'side':
            maxX = float(np.max(np.abs(self.pathX)))
            maxZ = float(np.max(np.abs(self.pathZ)))
            floorX = 2.2 if visitor.perihelionAu < 1.5 else max(3.0, visitor.perihelionAu * 1.4)
            self.axisLimitX = max(floorX, maxX * 1.08)
            self.axisLimitZ = max(1.6, maxZ * 1.15)
        else:
            maxExtent = float(
                np.max(
                    np.abs(
                        np.concatenate(
                            [
                                self.pathX,
                                self.pathY,
                                self.pathZ,
                                self.earthOrbitX,
                                self.earthOrbitY,
                            ]
                        )
                    )
                )
            )
            self.axisLimit = maxExtent * 1.06
            eventAzimuthDeg = float(np.degrees(np.arctan2(visitorAtEvent[1], visitorAtEvent[0])))
            self.obliqueAzimuthDeg = eventAzimuthDeg + OBLIQUE_AZIMUTH_OFFSET_DEG

    def _earthPosition(self, sampleIndex: int) -> tuple[float, float, float]:
        timeYears = float(self.timeYears[sampleIndex])
        earthAngleRad = self.earthAngleAtEventRad + 2.0 * np.pi * (
            timeYears - self.timeYearsAtEvent
        )
        return (
            self.earthSemiMajorAxisAu * float(np.cos(earthAngleRad)),
            self.earthSemiMajorAxisAu * float(np.sin(earthAngleRad)),
            0.0,
        )

    def _title(self) -> str:
        name = self.visitor.displayName
        if self.highlight == 'earth_flyby':
            focus = 'closest approach to Earth'
        else:
            focus = 'perihelion passage'
        if self.view == 'side':
            return f'{name} — {focus} (side view, X–Z)'
        return f'{name} — {focus} (oblique view)'

    def _footer(self) -> str:
        designation = self.visitor.designation
        if self.view == 'side':
            return f'{designation} · side view (X–Z) · catalog-driven hyperbolic trajectory'
        return f'{designation} · oblique view · looking down onto the ecliptic'

    def _calloutText(self) -> str:
        return f'{self.calloutLabel}\n{self.calloutAu:.3f} AU\n({self.calloutKm / 1e6:.1f}×10⁶ km)'

    def update(self, frame: int):
        if self.view == 'oblique':
            return self._updateOblique(frame)
        return self._updateSide(frame)

    def _updateSide(self, frame: int):
        sampleIndex = int(self.frameToSample[frame])
        visitorX = self.pathX[sampleIndex]
        visitorZ = self.pathZ[sampleIndex]
        earthX, _, _ = self._earthPosition(sampleIndex)
        earthZ = 0.0
        atEvent = sampleIndex == self.eventIndex

        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitX, self.axisLimitX)
        self.axes.set_ylim(-self.axisLimitZ, self.axisLimitZ)
        self.axes.set_title(self._title(), color=self.labelColor, pad=14)

        self.axes.axhline(0.0, color=self.orbitColor, linewidth=0.6, alpha=0.35, zorder=1)
        self.axes.text(
            self.axisLimitX * 0.62,
            self.axisLimitZ * 0.04,
            'ecliptic',
            color=self.labelColor,
            fontsize=7,
            alpha=0.55,
        )

        for orbitX, _, orbitZ in self.contextOrbits:
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
            color=self.visitorColor,
            linewidth=1.1,
            alpha=0.55,
            linestyle='--',
        )

        self.axes.scatter([0], [0], s=220, color=self.sunColor, zorder=5)
        self.axes.text(0.08, -0.18, 'Sun', color=self.labelColor, fontsize=8)

        self.axes.scatter([earthX], [earthZ], s=70, color=self.earthColor, zorder=6)
        self.axes.text(earthX + 0.08, earthZ + 0.08, 'Earth', color=self.labelColor, fontsize=8)

        self.axes.scatter([visitorX], [visitorZ], s=22, color=self.visitorColor, zorder=7)
        self.axes.text(
            visitorX + 0.08,
            visitorZ - 0.12,
            self.visitor.displayName,
            color=self.visitorColor,
            fontsize=8,
        )

        if atEvent:
            if self.showEarthMissLine:
                self.axes.plot(
                    [visitorX, earthX],
                    [visitorZ, earthZ],
                    color=self.labelColor,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=4,
                )
                textX = 0.5 * (visitorX + earthX) + 0.05
                textZ = 0.5 * (visitorZ + earthZ) + 0.12
            else:
                textX = visitorX + 0.12
                textZ = visitorZ + 0.12
            self.axes.text(
                textX,
                textZ,
                self._calloutText(),
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
            self._footer(),
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )
        return []

    def _applyObliqueAxes(self) -> None:
        limit = self.axisLimit
        self.axes.set_xlim(-limit, limit)
        self.axes.set_ylim(-limit, limit)
        self.axes.set_zlim(-limit, limit)
        self.axes.view_init(elev=OBLIQUE_ELEVATION_DEG, azim=self.obliqueAzimuthDeg)
        self.axes.set_axis_off()
        self.axes.set_position((0.0, 0.0, 1.0, 1.0))
        self.axes.set_box_aspect((1, 1, 1), zoom=1.65)
        self.axes.set_title(self._title(), color=self.labelColor, pad=8, y=0.98)

    def _updateOblique(self, frame: int):
        sampleIndex = int(self.frameToSample[frame])
        visitorX = self.pathX[sampleIndex]
        visitorY = self.pathY[sampleIndex]
        visitorZ = self.pathZ[sampleIndex]
        earthX, earthY, earthZ = self._earthPosition(sampleIndex)
        atEvent = sampleIndex == self.eventIndex

        self.axes.clear()
        self._applyObliqueAxes()

        for orbitX, orbitY, orbitZ in self.contextOrbits:
            self.axes.plot(orbitX, orbitY, orbitZ, color=self.orbitColor, linewidth=0.5, alpha=0.35)
        self.axes.plot(
            self.earthOrbitX,
            self.earthOrbitY,
            self.earthOrbitZ,
            color=self.earthColor,
            linewidth=1.0,
            alpha=0.55,
        )
        self.axes.plot(
            self.pathX,
            self.pathY,
            self.pathZ,
            color=self.visitorColor,
            linewidth=1.1,
            alpha=0.55,
            linestyle='--',
        )

        self.axes.scatter([0], [0], [0], s=220, color=self.sunColor, depthshade=False, zorder=5)
        self.axes.text(0.08, 0.0, -0.18, 'Sun', color=self.labelColor, fontsize=8)

        self.axes.scatter(
            [earthX], [earthY], [earthZ], s=70, color=self.earthColor, depthshade=False, zorder=6
        )
        self.axes.text(
            earthX + 0.08, earthY + 0.08, earthZ + 0.08, 'Earth', color=self.labelColor, fontsize=8
        )

        self.axes.scatter(
            [visitorX],
            [visitorY],
            [visitorZ],
            s=22,
            color=self.visitorColor,
            depthshade=False,
            zorder=7,
        )
        self.axes.text(
            visitorX + 0.08,
            visitorY + 0.08,
            visitorZ - 0.12,
            self.visitor.displayName,
            color=self.visitorColor,
            fontsize=8,
        )

        if atEvent:
            if self.showEarthMissLine:
                self.axes.plot(
                    [visitorX, earthX],
                    [visitorY, earthY],
                    [visitorZ, earthZ],
                    color=self.labelColor,
                    linewidth=1.2,
                    alpha=0.9,
                    zorder=4,
                )
                textX = 0.5 * (visitorX + earthX) + 0.05
                textY = 0.5 * (visitorY + earthY) + 0.05
                textZ = 0.5 * (visitorZ + earthZ) + 0.12
            else:
                textX = visitorX + 0.12
                textY = visitorY + 0.12
                textZ = visitorZ + 0.12
            self.axes.text(
                textX,
                textY,
                textZ,
                self._calloutText(),
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

        self.axes.text2D(
            0.03,
            0.03,
            self._footer(),
            transform=self.axes.transAxes,
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
        saveKwargs = {}
        if self.view == 'oblique':
            saveKwargs['savefig_kwargs'] = {
                'pad_inches': 0,
                'facecolor': self.figure.get_facecolor(),
            }
        animation.save(outputPath, writer=PillowWriter(fps=ANIMATION_FPS), **saveKwargs)
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def _filenameStem(objectId: str, view: ViewName) -> str:
    return f'{objectId}_{view}'


def renderInterstellarObjectAnimations(
    objectIds: tuple[str, ...] | None = None,
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    csvPath: str = DEFAULT_INTERSTELLAR_OBJECTS_CSV,
) -> None:
    catalog = InterstellarObjectCatalog(csvPath)
    selectedIds = objectIds if objectIds is not None else tuple(catalog.listObjectIds())
    for objectId in selectedIds:
        visitor = catalog.load(objectId)
        for view in ('side', 'oblique'):
            filenameStem = _filenameStem(objectId, view)  # type: ignore[arg-type]
            for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
                outputPath = f'{OUTPUT_DIRECTORY}/{filenameStem}_{themeName}.gif'
                print(f'Rendering {outputPath}...')
                animator = InterstellarObjectAnimator(
                    visitor,
                    view=view,  # type: ignore[arg-type]
                    style=styleName,
                    figureSizeInches=figureSizeInches,
                    dpi=dpi,
                )
                animator.saveGif(outputPath)
    print('Interstellar-object animations completed!')
