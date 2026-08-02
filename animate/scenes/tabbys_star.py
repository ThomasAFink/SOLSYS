"""Tabby's Star dust-cloud dimming animation (Boyajian's Star / KIC 8462852)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse
from solsys.physics.catalogs.system_catalog import SystemCatalog

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480
AXIS_LIMIT_AU = 6.5
STAR_COLOR = '#F5E6A3'
DEFAULT_LIGHTCURVE_CSV = 'data/tabbys_star_lightcurve.csv'
MIN_DIP_DEPTH = 0.012
MIN_DIP_SEPARATION_FRAMES = 18
MAX_ORBITING_CLUMPS = 8
REFERENCE_DIP_DEPTH = 0.20
LOS_ANGLE_RAD = 0.0  # +X axis toward Earth


def loadKeplerLightCurve(
    csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csvPath, comment='#')
    return (
        np.asarray(frame['bkjd_day'], dtype=float),
        np.asarray(frame['relative_flux'], dtype=float),
    )


@dataclass(frozen=True)
class OrbitingDustClump:
    """One debris clump on a circular orbit; crosses the LOS at crossingFrame."""

    crossingFrame: int
    orbitRadiusAu: float
    sizeAu: float
    color: str
    dipDepth: float


def resampleFluxToFrames(
    keplerFlux: np.ndarray,
    animationFrames: int,
) -> np.ndarray:
    sourceX = np.linspace(0.0, 1.0, len(keplerFlux))
    targetX = np.linspace(0.0, 1.0, animationFrames)
    return np.interp(targetX, sourceX, keplerFlux)


def findDipCrossingFrames(
    fluxByFrame: np.ndarray,
    minDipDepth: float = MIN_DIP_DEPTH,
    minSeparationFrames: int = MIN_DIP_SEPARATION_FRAMES,
    maxClumps: int = MAX_ORBITING_CLUMPS,
) -> list[tuple[int, float]]:
    """Return (frame, dip_depth) for significant local minima, deepest first then time-ordered."""
    flux = np.asarray(fluxByFrame, dtype=float)
    if len(flux) < 3:
        return []

    candidateFrames: list[int] = []
    for index in range(1, len(flux) - 1):
        if flux[index] <= flux[index - 1] and flux[index] <= flux[index + 1]:
            depth = 1.0 - float(flux[index])
            if depth >= minDipDepth:
                candidateFrames.append(index)

    # Keep deepest in each local cluster.
    selected: list[tuple[int, float]] = []
    for frame in sorted(candidateFrames, key=lambda item: 1.0 - flux[item], reverse=True):
        depth = 1.0 - float(flux[frame])
        if any(abs(frame - keptFrame) < minSeparationFrames for keptFrame, _ in selected):
            continue
        selected.append((frame, depth))
        if len(selected) >= maxClumps:
            break

    selected.sort(key=lambda item: item[0])
    return selected


def buildOrbitingClumps(
    dipEvents: list[tuple[int, float]],
) -> tuple[OrbitingDustClump, ...]:
    if not dipEvents:
        return ()

    colors = (
        '#D2C09A',
        '#C4B59A',
        '#B8A888',
        '#A89878',
        '#E0D2B0',
        '#C9B892',
        '#B0A07A',
        '#D8CBA8',
    )
    radii = np.linspace(2.0, 5.0, len(dipEvents))
    clumps: list[OrbitingDustClump] = []
    for index, ((crossingFrame, dipDepth), radius) in enumerate(zip(dipEvents, radii, strict=True)):
        strength = float(np.clip(dipDepth / REFERENCE_DIP_DEPTH, 0.15, 1.0))
        sizeAu = 0.22 + 1.05 * strength
        clumps.append(
            OrbitingDustClump(
                crossingFrame=crossingFrame,
                orbitRadiusAu=float(radius),
                sizeAu=sizeAu,
                color=colors[index % len(colors)],
                dipDepth=dipDepth,
            )
        )
    return tuple(clumps)


class TabbysStarAnimator:
    """F-star + orbiting dust clumps that cross the Earth LOS on Kepler dip frames."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
        lightcurveCsvPath: str = DEFAULT_LIGHTCURVE_CSV,
    ):
        catalog = SystemCatalog(starsCsvPath=starsCsvPath)
        self.system = catalog.load('tabbys_star')
        if self.system.planets:
            raise ValueError("Tabby's Star scene expects no catalog planets")
        if not self.system.stars:
            raise ValueError("Tabby's Star catalog is missing the host star")

        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.axisLimitAu = AXIS_LIMIT_AU
        self.keplerTimeBkjd, self.keplerFlux = loadKeplerLightCurve(lightcurveCsvPath)
        self.fluxByFrame = resampleFluxToFrames(self.keplerFlux, self.animationFrames)
        self.timeByFrame = resampleFluxToFrames(self.keplerTimeBkjd, self.animationFrames)
        dipEvents = findDipCrossingFrames(self.fluxByFrame)
        self.clumps = buildOrbitingClumps(dipEvents)

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.guideColor = '#777777' if self.isDark else '#888888'
        self.dustEdgeColor = '#E8DCC0' if self.isDark else '#6A5A40'

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)

    def _clumpAngleRad(self, clump: OrbitingDustClump, frame: int) -> float:
        # One full orbit per GIF; angle 0 (+X / LOS) exactly at the assigned dip frame.
        return LOS_ANGLE_RAD + 2.0 * np.pi * (frame - clump.crossingFrame) / self.animationFrames

    def _clumpPosition(self, clump: OrbitingDustClump, frame: int) -> tuple[float, float, float]:
        angle = self._clumpAngleRad(clump, frame)
        return (
            float(clump.orbitRadiusAu * np.cos(angle)),
            float(clump.orbitRadiusAu * np.sin(angle)),
            float(angle),
        )

    def _onLineOfSight(self, positionX: float, positionY: float, sizeAu: float) -> bool:
        return positionX > 0.0 and abs(positionY) <= max(0.35, sizeAu * 0.75)

    def update(self, frame: int):
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_ylim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_title(
            "Tabby's Star — orbiting dust crosses LOS on Kepler dips",
            color=self.labelColor,
            pad=16,
        )

        # Orbit guides for each clump radius.
        angles = np.linspace(0.0, 2.0 * np.pi, 240)
        for clump in self.clumps:
            self.axes.plot(
                clump.orbitRadiusAu * np.cos(angles),
                clump.orbitRadiusAu * np.sin(angles),
                color=self.guideColor,
                linewidth=0.55,
                alpha=0.22,
            )

        self.axes.plot(
            [0.35, self.axisLimitAu * 0.95],
            [0.0, 0.0],
            color=self.labelColor,
            linestyle='--',
            linewidth=1.0,
            alpha=0.55,
        )
        self.axes.text(
            self.axisLimitAu * 0.40,
            self.axisLimitAu * 0.07,
            'line of sight → Earth (Kepler)',
            color=self.labelColor,
            fontsize=8,
            alpha=0.75,
        )

        flux = float(self.fluxByFrame[frame])
        starAlpha = 0.35 + 0.65 * np.clip(flux, 0.0, 1.0)
        starSize = 180 + 220 * np.clip(flux, 0.0, 1.0)
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
            0.28,
            0.40,
            "Tabby's Star\n(F3V)",
            color=self.labelColor,
            fontsize=9,
            zorder=6,
        )

        for clump in self.clumps:
            positionX, positionY, angle = self._clumpPosition(clump, frame)
            onLos = self._onLineOfSight(positionX, positionY, clump.sizeAu)
            alpha = 0.62 if onLos else 0.34
            self.axes.add_patch(
                Ellipse(
                    (positionX, positionY),
                    width=clump.sizeAu * 2.3,
                    height=clump.sizeAu * 1.15,
                    angle=np.degrees(angle),
                    facecolor=clump.color,
                    edgecolor=self.dustEdgeColor if onLos else 'none',
                    linewidth=1.2 if onLos else 0.0,
                    alpha=alpha,
                    zorder=4,
                )
            )

        cursorTime = float(self.timeByFrame[frame])
        dipPercent = max(0.0, (1.0 - flux) * 100.0)
        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.78,
            f'Kepler flux {flux:.2f} ({dipPercent:.0f}% dip) at BKJD {cursorTime:.0f}',
            color=self.labelColor,
            fontsize=8,
            alpha=0.8,
        )
        self.axes.text(
            -self.axisLimitAu * 0.95,
            -self.axisLimitAu * 0.92,
            f'{len(self.clumps)} orbiting clumps · larger = deeper dips · cross LOS at dip times',
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )

        inset = self.axes.inset_axes([0.08, 0.12, 0.38, 0.22])
        curveColor = '#7EB6FF' if self.isDark else '#204080'
        inset.plot(self.keplerTimeBkjd, self.keplerFlux, color=curveColor, linewidth=0.9)
        inset.axvline(cursorTime, color=self.labelColor, linewidth=0.9, alpha=0.8)
        # Mark scheduled crossing times on the inset.
        for clump in self.clumps:
            inset.axvline(
                float(self.timeByFrame[clump.crossingFrame]),
                color=clump.color,
                linewidth=0.7,
                alpha=0.55,
            )
        inset.set_ylim(0.75, 1.03)
        inset.set_xlim(float(self.keplerTimeBkjd[0]), float(self.keplerTimeBkjd[-1]))
        inset.set_xticks([])
        inset.set_yticks([0.8, 0.9, 1.0])
        inset.tick_params(colors=self.labelColor, labelsize=6)
        inset.set_title(
            'Kepler light curve (downsampled)',
            color=self.labelColor,
            fontsize=7,
            pad=2,
        )
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
