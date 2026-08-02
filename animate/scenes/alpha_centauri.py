"""Alpha Centauri top-down animations (A–B binary + wide triple + Proxima planets)."""

from __future__ import annotations

import os
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from solsys.physics import OrbitCalculator
from solsys.physics.catalogs.system_catalog import StarSystem, StellarOrbit, SystemCatalog

from .exoplanet_system import (
    ExoplanetSystemSceneConfig,
    bodyPositionInOrbitalPlane,
    orbitPathInOrbitalPlane,
    renderExoplanetSystemAnimations,
)

ViewName = Literal['ab_binary', 'system_wide']

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 600
ANIMATION_SPEED_AB = 90.0  # ~80 yr binary; need a large multiplier to read motion
ANIMATION_SPEED_WIDE = 120000.0  # Proxima’s ~550 kyr orbit around AB

# Face-on schematic: plot in the binary orbital plane (ignore sky inclination).
AB_AXIS_LIMIT_AU = 28.0
SYSTEM_WIDE_AXIS_LIMIT_AU = 14000.0

STAR_COLORS = {
    'primary': '#F6D56A',
    'secondary': '#E8A05A',
    'wide_companion': '#E07060',
}

PROXIMA_PLANETS_CONFIG = ExoplanetSystemSceneConfig(
    hostStarRole='wide_companion',
    title='Proxima Centauri planets (Alpha Centauri system)',
    starLabel='Proxima',
    starColor=STAR_COLORS['wide_companion'],
    minAxisLimitAu=0.22,
    planetIdAxisLimits={'proxima_c': 2.2},
    animationSpeed=0.25,
    footerNote='A–B binary ~8.7 kau away · system_id=alpha_centauri',
)


class AlphaCentauriAnimator:
    """Top-down Keplerian scene for Alpha Centauri A–B / wide triple views."""

    def __init__(
        self,
        system: StarSystem,
        view: ViewName = 'ab_binary',
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
    ):
        if system.systemId != 'alpha_centauri':
            raise ValueError(f'Expected alpha_centauri, got {system.systemId!r}')
        if view not in ('ab_binary', 'system_wide'):
            raise ValueError(f'Unknown view: {view!r}')

        self.system = system
        self.view = view
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.orbitCalculator = OrbitCalculator()
        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.orbitColor = '#D0D0D0' if self.isDark else '#404040'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)
        self.animationFrames = ANIMATION_FRAMES
        if view == 'ab_binary':
            self.animationSpeed = ANIMATION_SPEED_AB
            self._initAbBinary()
        else:
            self.animationSpeed = ANIMATION_SPEED_WIDE
            self._initSystemWide()

    def _initAbBinary(self) -> None:
        self.primaryOrbit = self._requireOrbit('primary')
        self.secondaryOrbit = self._requireOrbit('secondary')
        self.proximaOrbit = self._requireOrbit('wide_companion')
        self.axisLimitAu = AB_AXIS_LIMIT_AU
        self.title = 'Alpha Centauri A–B (orbital plane)'

    def _initSystemWide(self) -> None:
        self.primaryOrbit = self._requireOrbit('primary')
        self.secondaryOrbit = self._requireOrbit('secondary')
        self.proximaOrbit = self._requireOrbit('wide_companion')
        self.axisLimitAu = max(
            SYSTEM_WIDE_AXIS_LIMIT_AU,
            self.proximaOrbit.semiMajorAxisAu * (1.0 + self.proximaOrbit.eccentricity) * 1.15,
        )
        self.title = 'Alpha Centauri system (A–B + Proxima)'

    def _requireOrbit(self, role: str) -> StellarOrbit:
        for orbit in self.system.stellarOrbits:
            if orbit.role == role:
                return orbit
        raise KeyError(f'No stellar orbit with role={role!r}')

    def _starLabel(self, orbit: StellarOrbit) -> str:
        member = self.system.starByUuid(orbit.starUuid)
        if member is None:
            return orbit.role
        name = member.starName.replace('\xa0', ' ')
        if 'Rigil' in name or orbit.role == 'primary':
            return 'α Cen A'
        if 'Toliman' in name or orbit.role == 'secondary':
            return 'α Cen B'
        if 'Proxima' in name or orbit.role == 'wide_companion':
            return 'Proxima'
        return name.split('(')[0].strip() or orbit.role

    def _bodyPosition(
        self,
        semiMajorAxisAu: float,
        eccentricity: float,
        periodDays: float,
        argumentPeriapsisDeg: float,
        meanAnomalyDegEpoch: float,
        frame: int,
        speedScale: float,
    ) -> tuple[float, float]:
        return bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            semiMajorAxisAu,
            eccentricity,
            periodDays,
            argumentPeriapsisDeg,
            meanAnomalyDegEpoch,
            frame,
            speedScale,
        )

    def _orbitPath(
        self, semiMajorAxisAu: float, eccentricity: float, argumentPeriapsisDeg: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return orbitPathInOrbitalPlane(
            self.orbitCalculator, semiMajorAxisAu, eccentricity, argumentPeriapsisDeg
        )

    def update(self, frame: int):
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_ylim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_title(self.title, color=self.labelColor, pad=16)

        if self.view == 'ab_binary':
            self._drawAbBinaryFrame(frame)
        else:
            self._drawSystemWideFrame(frame)
        return []

    def _drawAbBinaryFrame(self, frame: int) -> None:
        for orbit, markerSize in (
            (self.primaryOrbit, 220),
            (self.secondaryOrbit, 160),
        ):
            pathX, pathY = self._orbitPath(
                orbit.semiMajorAxisAu, orbit.eccentricity, orbit.argumentPeriapsisDeg
            )
            self.axes.plot(pathX, pathY, color=self.orbitColor, linewidth=0.9, alpha=0.55)
            positionX, positionY = self._bodyPosition(
                orbit.semiMajorAxisAu,
                orbit.eccentricity,
                orbit.periodDays,
                orbit.argumentPeriapsisDeg,
                orbit.meanAnomalyDegEpoch,
                frame,
                self.animationSpeed,
            )
            color = STAR_COLORS.get(orbit.role, '#FFFFFF')
            self.axes.scatter([positionX], [positionY], s=markerSize, color=color, zorder=5)
            self.axes.text(
                positionX + 0.8,
                positionY + 0.8,
                self._starLabel(orbit),
                color=self.labelColor,
                fontsize=9,
            )

        # Direction to Proxima (true scale ~8.7 kau — far outside this ±28 AU frame).
        proximaX, proximaY = self._bodyPosition(
            self.proximaOrbit.semiMajorAxisAu,
            self.proximaOrbit.eccentricity,
            self.proximaOrbit.periodDays,
            self.proximaOrbit.argumentPeriapsisDeg,
            self.proximaOrbit.meanAnomalyDegEpoch,
            frame,
            ANIMATION_SPEED_WIDE,
        )
        directionNorm = np.hypot(proximaX, proximaY) or 1.0
        edgeX = self.axisLimitAu * 0.92 * proximaX / directionNorm
        edgeY = self.axisLimitAu * 0.92 * proximaY / directionNorm
        self.axes.annotate(
            '',
            xy=(edgeX, edgeY),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->', color=STAR_COLORS['wide_companion'], lw=1.0, alpha=0.7
            ),
        )
        self.axes.text(
            edgeX * 0.72,
            edgeY * 0.72,
            'Proxima →',
            color=STAR_COLORS['wide_companion'],
            fontsize=8,
            alpha=0.85,
        )

        self.axes.scatter([0], [0], s=18, color=self.labelColor, alpha=0.5, zorder=4)
        self.axes.text(0.6, -1.4, 'AB barycenter', color=self.labelColor, fontsize=7, alpha=0.7)
        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.92,
            'Proxima ~8.7 kau off-frame (same system) · see system_wide GIF',
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )

    def _drawSystemWideFrame(self, frame: int) -> None:
        pathX, pathY = self._orbitPath(
            self.proximaOrbit.semiMajorAxisAu,
            self.proximaOrbit.eccentricity,
            self.proximaOrbit.argumentPeriapsisDeg,
        )
        self.axes.plot(pathX, pathY, color=self.orbitColor, linewidth=0.8, alpha=0.45)

        # A–B unresolved at this scale — single barycenter marker.
        self.axes.scatter([0], [0], s=110, color=STAR_COLORS['primary'], zorder=5)
        self.axes.text(
            200, 350, 'α Cen A–B\n(unresolved at this scale)', color=self.labelColor, fontsize=8
        )

        proximaX, proximaY = self._bodyPosition(
            self.proximaOrbit.semiMajorAxisAu,
            self.proximaOrbit.eccentricity,
            self.proximaOrbit.periodDays,
            self.proximaOrbit.argumentPeriapsisDeg,
            self.proximaOrbit.meanAnomalyDegEpoch,
            frame,
            self.animationSpeed,
        )
        self.axes.scatter(
            [proximaX], [proximaY], s=120, color=STAR_COLORS['wide_companion'], zorder=6
        )
        self.axes.text(
            proximaX + self.axisLimitAu * 0.02,
            proximaY + self.axisLimitAu * 0.02,
            'Proxima',
            color=self.labelColor,
            fontsize=9,
        )
        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.92,
            f'Wide triple · Proxima a≈{self.proximaOrbit.semiMajorAxisAu:g} AU · system_id=alpha_centauri',
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )

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


def renderAlphaCentauriAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    catalog = SystemCatalog(starsCsvPath=starsCsvPath)
    system = catalog.load('alpha_centauri')
    outputDirectory = 'output/animate/alpha_centauri'
    for view, filenameStem in (
        ('ab_binary', 'alpha_centauri_ab'),
        ('system_wide', 'alpha_centauri_system'),
    ):
        for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
            outputPath = f'{outputDirectory}/{filenameStem}_{themeName}.gif'
            print(f'Rendering {outputPath}...')
            animator = AlphaCentauriAnimator(
                system,
                view=view,  # type: ignore[arg-type]
                style=styleName,
                figureSizeInches=figureSizeInches,
                dpi=dpi,
            )
            animator.saveGif(outputPath)

    renderExoplanetSystemAnimations(
        systemId='alpha_centauri',
        filenameStem='proxima_planets',
        outputDirectory=outputDirectory,
        config=PROXIMA_PLANETS_CONFIG,
        figureSizeInches=figureSizeInches,
        dpi=dpi,
        starsCsvPath=starsCsvPath,
    )
    print('Alpha Centauri animations completed!')
