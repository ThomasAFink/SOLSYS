"""Tabby's Star dust-cloud dimming animation (Boyajian's Star / KIC 8462852)."""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from solsys.physics import OrbitCalculator
from solsys.physics.catalogs.system_catalog import SystemCatalog

from .exoplanet_system import bodyPositionInOrbitalPlane, orbitPathInOrbitalPlane

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480
ANIMATION_SPEED = 1.8
AXIS_LIMIT_AU = 7.5
STAR_COLOR = '#F5E6A3'
LOS_HALF_WIDTH_AU = 0.22


@dataclass(frozen=True)
class DustClump:
    """Schematic eccentric dust / debris clump (not a planet)."""

    name: str
    semiMajorAxisAu: float
    eccentricity: float
    periodDays: float
    argumentPeriapsisDeg: float
    meanAnomalyDegEpoch: float
    opticalDepth: float
    sizeAu: float
    color: str


# Uneven debris on eccentric paths — schematic of the dust-cloud dimming story.
DUST_CLUMPS = (
    DustClump('clump A', 2.4, 0.62, 900.0, 10.0, 20.0, 0.38, 0.38, '#C4B59A'),
    DustClump('clump B', 3.6, 0.72, 1600.0, 200.0, 140.0, 0.55, 0.52, '#B8A888'),
    DustClump('clump C', 4.8, 0.48, 2400.0, 95.0, 260.0, 0.22, 0.45, '#A89878'),
    DustClump('clump D', 5.6, 0.58, 3100.0, 310.0, 40.0, 0.30, 0.60, '#D0C4A8'),
)


class TabbysStarAnimator:
    """Face-on schematic: F-star + eccentric dust clumps crossing the Earth line of sight."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
    ):
        catalog = SystemCatalog(starsCsvPath=starsCsvPath)
        self.system = catalog.load('tabbys_star')
        if self.system.planets:
            raise ValueError("Tabby's Star scene expects no catalog planets")
        if not self.system.stars:
            raise ValueError("Tabby's Star catalog is missing the host star")

        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.orbitCalculator = OrbitCalculator()
        self.animationFrames = ANIMATION_FRAMES
        self.animationSpeed = ANIMATION_SPEED
        self.axisLimitAu = AXIS_LIMIT_AU
        self.clumps = DUST_CLUMPS

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.orbitColor = '#888888' if self.isDark else '#606060'
        self.dustEdgeColor = '#E8DCC0' if self.isDark else '#6A5A40'

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)
        self.fluxHistory = np.array(
            [self._relativeFlux(frame) for frame in range(self.animationFrames)],
            dtype=float,
        )

    def _clumpPosition(self, clump: DustClump, frame: int) -> tuple[float, float]:
        return bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            clump.semiMajorAxisAu,
            clump.eccentricity,
            clump.periodDays,
            clump.argumentPeriapsisDeg,
            clump.meanAnomalyDegEpoch,
            frame,
            self.animationSpeed,
        )

    def _opticalDepthOnLineOfSight(self, frame: int) -> float:
        """Earth looks in from +X; material at x>0 near the x-axis dims the star."""
        total = 0.0
        for clump in self.clumps:
            positionX, positionY = self._clumpPosition(clump, frame)
            if positionX <= 0.0:
                continue
            if abs(positionY) > max(LOS_HALF_WIDTH_AU, clump.sizeAu * 0.85):
                continue
            # Stronger when closer to the star along the ray.
            distanceWeight = 1.0 / (1.0 + 0.15 * positionX)
            total += clump.opticalDepth * distanceWeight
        return float(total)

    def _relativeFlux(self, frame: int) -> float:
        return float(np.exp(-self._opticalDepthOnLineOfSight(frame)))

    def update(self, frame: int):
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_ylim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_title(
            "Tabby's Star — uneven dust dimming (schematic)",
            color=self.labelColor,
            pad=16,
        )

        # Faint guide orbits for the debris paths.
        for clump in self.clumps:
            pathX, pathY = orbitPathInOrbitalPlane(
                self.orbitCalculator,
                clump.semiMajorAxisAu,
                clump.eccentricity,
                clump.argumentPeriapsisDeg,
            )
            self.axes.plot(pathX, pathY, color=self.orbitColor, linewidth=0.6, alpha=0.28)

        # Line of sight from Earth (Kepler) toward the star.
        self.axes.plot(
            [0.4, self.axisLimitAu * 0.95],
            [0.0, 0.0],
            color=self.labelColor,
            linestyle='--',
            linewidth=1.0,
            alpha=0.55,
        )
        self.axes.text(
            self.axisLimitAu * 0.42,
            self.axisLimitAu * 0.06,
            'line of sight → Earth (Kepler)',
            color=self.labelColor,
            fontsize=8,
            alpha=0.75,
        )

        flux = self._relativeFlux(frame)
        starAlpha = 0.35 + 0.65 * flux
        starSize = 180 + 220 * flux
        self.axes.scatter(
            [0],
            [0],
            s=starSize,
            color=STAR_COLOR,
            alpha=starAlpha,
            zorder=5,
            edgecolors=self.dustEdgeColor,
            linewidths=0.6,
        )
        self.axes.text(
            0.25,
            0.35,
            "Tabby's Star\n(F3V)",
            color=self.labelColor,
            fontsize=9,
            zorder=6,
        )

        for clump in self.clumps:
            positionX, positionY = self._clumpPosition(clump, frame)
            onLos = positionX > 0.0 and abs(positionY) <= max(
                LOS_HALF_WIDTH_AU, clump.sizeAu * 0.85
            )
            patch = Ellipse(
                (positionX, positionY),
                width=clump.sizeAu * 2.2,
                height=clump.sizeAu * 1.2,
                angle=np.degrees(np.arctan2(positionY, positionX)),
                facecolor=clump.color,
                edgecolor=self.dustEdgeColor if onLos else 'none',
                linewidth=1.2 if onLos else 0.0,
                alpha=0.55 if onLos else 0.32,
                zorder=4,
            )
            self.axes.add_patch(patch)

        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.78,
            f'relative flux {flux:.2f}  ·  dust / debris (not planets)',
            color=self.labelColor,
            fontsize=8,
            alpha=0.8,
        )
        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.92,
            "Boyajian's Star / KIC 8462852 · ~1470 ly · system_id=tabbys_star",
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )

        inset = self.axes.inset_axes([0.08, 0.12, 0.34, 0.18])
        inset.plot(
            np.arange(self.animationFrames),
            self.fluxHistory,
            color='#7EB6FF' if self.isDark else '#204080',
            linewidth=1.0,
        )
        inset.axvline(frame, color=self.labelColor, linewidth=0.8, alpha=0.7)
        inset.set_ylim(0.55, 1.05)
        inset.set_xlim(0, self.animationFrames - 1)
        inset.set_xticks([])
        inset.set_yticks([0.6, 0.8, 1.0])
        inset.tick_params(colors=self.labelColor, labelsize=6)
        inset.set_title('schematic light curve', color=self.labelColor, fontsize=7, pad=2)
        for spine in inset.spines.values():
            spine.set_color(self.labelColor)
            spine.set_alpha(0.5)
        inset.set_facecolor('#101010' if self.isDark else '#F7F7F7')
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


def renderTabbysStarAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    outputDirectory = 'output/animate/tabbys_star'
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{outputDirectory}/tabbys_star_dust_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TabbysStarAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
        )
        animator.saveGif(outputPath)
    print("Tabby's Star dust animations completed!")
