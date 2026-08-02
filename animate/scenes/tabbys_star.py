"""Tabby's Star dust-cloud dimming animation (Boyajian's Star / KIC 8462852)."""

from __future__ import annotations

import os
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
DUST_COLOR = '#C9B892'
DEFAULT_LIGHTCURVE_CSV = 'data/tabbys_star_lightcurve.csv'
# Scale observed dips (~0–0.20) onto occulting-cloud strength.
REFERENCE_DIP_DEPTH = 0.20


def loadKeplerLightCurve(
    csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.read_csv(csvPath, comment='#')
    return (
        np.asarray(frame['bkjd_day'], dtype=float),
        np.asarray(frame['relative_flux'], dtype=float),
    )


class TabbysStarAnimator:
    """F-star + LOS dust sized to the Kepler light curve (no free-roaming blobs)."""

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

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.guideColor = '#777777' if self.isDark else '#888888'
        self.dustEdgeColor = '#E8DCC0' if self.isDark else '#6A5A40'

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)

    def _lightcurveIndex(self, frame: int) -> int:
        if self.animationFrames <= 1:
            return 0
        scale = (len(self.keplerFlux) - 1) / (self.animationFrames - 1)
        return int(round(frame * scale))

    def _keplerFluxAtFrame(self, frame: int) -> float:
        return float(self.keplerFlux[self._lightcurveIndex(frame)])

    def _dipStrength(self, flux: float) -> float:
        depth = max(0.0, 1.0 - float(flux))
        return float(np.clip(depth / REFERENCE_DIP_DEPTH, 0.0, 1.0))

    def _drawGuideRing(self) -> None:
        angles = np.linspace(0.0, 2.0 * np.pi, 240)
        for radius in (2.0, 3.2, 4.5):
            self.axes.plot(
                radius * np.cos(angles),
                radius * np.sin(angles),
                color=self.guideColor,
                linewidth=0.6,
                alpha=0.2,
            )

    def _drawOccultingDust(self, strength: float) -> None:
        if strength < 0.02:
            return
        # Stable LOS placement; only size/alpha change with the Kepler dip.
        mainX = 2.8
        mainSize = 0.20 + 1.35 * strength
        mainAlpha = 0.15 + 0.65 * strength
        self.axes.add_patch(
            Ellipse(
                (mainX, 0.0),
                width=mainSize * 2.6,
                height=mainSize * 1.2,
                angle=0.0,
                facecolor=DUST_COLOR,
                edgecolor=self.dustEdgeColor,
                linewidth=1.0,
                alpha=mainAlpha,
                zorder=4,
            )
        )
        if strength > 0.35:
            frag = (strength - 0.35) / 0.65
            self.axes.add_patch(
                Ellipse(
                    (mainX + 1.15, 0.12 * frag),
                    width=(0.20 + 0.70 * frag) * 2.1,
                    height=(0.20 + 0.70 * frag) * 1.0,
                    angle=8.0,
                    facecolor=DUST_COLOR,
                    edgecolor=self.dustEdgeColor,
                    linewidth=0.8,
                    alpha=0.12 + 0.45 * frag,
                    zorder=4,
                )
            )

    def update(self, frame: int):
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_ylim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_title(
            "Tabby's Star — Kepler dips explained as LOS dust",
            color=self.labelColor,
            pad=16,
        )

        self._drawGuideRing()

        # Line of sight from Earth (Kepler) toward the star.
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

        flux = self._keplerFluxAtFrame(frame)
        strength = self._dipStrength(flux)
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

        self._drawOccultingDust(strength)

        cursorTime = float(self.keplerTimeBkjd[self._lightcurveIndex(frame)])
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
            'LOS dust grows with observed dips · schematic geometry · system_id=tabbys_star',
            color=self.labelColor,
            fontsize=7,
            alpha=0.65,
        )

        inset = self.axes.inset_axes([0.08, 0.12, 0.38, 0.22])
        curveColor = '#7EB6FF' if self.isDark else '#204080'
        inset.plot(self.keplerTimeBkjd, self.keplerFlux, color=curveColor, linewidth=0.9)
        inset.axvline(cursorTime, color=self.labelColor, linewidth=0.9, alpha=0.8)
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
