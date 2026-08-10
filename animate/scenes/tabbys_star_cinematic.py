"""Tabby's Star lightcurve cinema — Kepler flux drives the cut; dust occults the star.

Not a Sol→destination odyssey. Line-of-sight view so clumps silhouette against the
stellar disk when the resampled Kepler playhead hits a dip.
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Ellipse
from solsys.physics.catalogs.system_catalog import SystemCatalog

from animate.scenes.tabbys_star import (
    DEFAULT_LIGHTCURVE_CSV,
    OrbitingDustClump,
    buildOrbitingClumps,
    findDipCrossingFrames,
    loadKeplerLightCurve,
    sampleSeriesToFrames,
)

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 560
STAR_COLOR = '#F5E6A3'
# Schematic stellar disk (exaggerated vs physical R★ so clumps can read as occultors).
STAR_RADIUS_AU = 1.35
SKY_HALF_AU = 3.6
OUTPUT_DIRECTORY = 'output/animate/tabbys_star/cinematic'


class TabbysStarCinematicAnimator:
    """Lightcurve spine + Earth-looking-in view of dust crossing the stellar disk."""

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
            raise ValueError("Tabby's Star cinematic expects no catalog planets")
        if not self.system.stars:
            raise ValueError("Tabby's Star catalog is missing the host star")

        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.starRadiusAu = STAR_RADIUS_AU
        self.skyHalfAu = SKY_HALF_AU

        self.keplerTimeBkjd, self.keplerFlux = loadKeplerLightCurve(lightcurveCsvPath)
        self.fluxByFrame = sampleSeriesToFrames(
            self.keplerFlux, self.animationFrames, reduceMin=True
        )
        self.timeByFrame = sampleSeriesToFrames(self.keplerTimeBkjd, self.animationFrames)
        self.dipEvents = findDipCrossingFrames(self.fluxByFrame)
        self.clumps = buildOrbitingClumps(self.dipEvents)
        self.deepestCrossingFrame = (
            max(self.dipEvents, key=lambda item: item[1])[0] if self.dipEvents else None
        )

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.guideColor = '#777777' if self.isDark else '#888888'
        self.dustEdgeColor = '#E8DCC0' if self.isDark else '#6A5A40'
        self.curveColor = '#7EB6FF' if self.isDark else '#204080'
        self.stageFace = '#0B0B0F' if self.isDark else '#E8EEF4'
        self.lcFace = '#101018' if self.isDark else '#F3F6F9'

        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi)
        grid = GridSpec(
            2,
            1,
            figure=self.figure,
            height_ratios=[2.55, 1.0],
            hspace=0.10,
            left=0.06,
            right=0.94,
            top=0.92,
            bottom=0.07,
        )
        self.skyAxes = self.figure.add_subplot(grid[0])
        self.lcAxes = self.figure.add_subplot(grid[1])

    def _clumpAngleRad(self, clump: OrbitingDustClump, frame: int) -> float:
        # Angle 0 = on the Earth-facing limb (screen center); matches dip frame.
        return 2.0 * np.pi * (frame - clump.crossingFrame) / self.animationFrames

    def _clumpOrbitXy(self, clump: OrbitingDustClump, frame: int) -> tuple[float, float]:
        """Orbital plane: +X toward Earth (camera), +Y across the sky."""
        angle = self._clumpAngleRad(clump, frame)
        return (
            float(clump.orbitRadiusAu * np.cos(angle)),
            float(clump.orbitRadiusAu * np.sin(angle)),
        )

    def _clumpScreenState(
        self, clump: OrbitingDustClump, frame: int
    ) -> tuple[float, float, bool, bool]:
        """Return (screenX, screenY, inFront, occulting).

        Camera looks from +X toward the star at the origin. Screen uses orbital Y
        as horizontal offset; clumps with x>0 are between Earth and the star.
        """
        positionX, positionY = self._clumpOrbitXy(clump, frame)
        inFront = positionX > 0.0
        screenX = positionY
        screenY = 0.12 * np.sin(self._clumpAngleRad(clump, frame) * 2.0)
        occulting = (
            inFront
            and abs(screenX) <= self.starRadiusAu + clump.sizeAu * 0.55
            and abs(screenY) <= self.starRadiusAu * 0.85
        )
        return screenX, screenY, inFront, occulting

    def _skyHalfWidth(self, frame: int) -> float:
        """Slight push-in around the deepest dip so the occultation fills the stage."""
        if self.deepestCrossingFrame is None:
            return self.skyHalfAu
        distance = abs(frame - self.deepestCrossingFrame)
        if distance > 28:
            return self.skyHalfAu
        blend = 1.0 - (distance / 28.0)
        blend = blend * blend * (3.0 - 2.0 * blend)
        return float(self.skyHalfAu * (1.0 - 0.22 * blend))

    def _caption(self, frame: int) -> tuple[str, str]:
        flux = float(self.fluxByFrame[frame])
        dipPercent = max(0.0, (1.0 - flux) * 100.0)
        occulting = any(self._clumpScreenState(clump, frame)[3] for clump in self.clumps)
        if self.deepestCrossingFrame is not None and abs(frame - self.deepestCrossingFrame) <= 14:
            return (
                "Tabby's Star · deepest dip in window",
                f'Debris crosses the line of sight · Kepler −{dipPercent:.0f}%',
            )
        if occulting or dipPercent >= 1.5:
            return (
                "Tabby's Star · dust occultation",
                f'Uneven circumstellar debris · relative flux {flux:.3f} (−{dipPercent:.0f}%)',
            )
        if dipPercent >= 0.6:
            return (
                "Tabby's Star · irregular dimming",
                'Not a periodic planet transit · clump train on real Kepler times',
            )
        return (
            "Tabby's Star · Kepler photometry",
            'KIC 8462852 (F3V) · lightcurve is the timeline',
        )

    def update(self, frame: int):
        self.skyAxes.clear()
        self.lcAxes.clear()

        half = self._skyHalfWidth(frame)
        self.skyAxes.set_xlim(-half, half)
        self.skyAxes.set_ylim(-half * 0.72, half * 0.72)
        self.skyAxes.set_aspect('equal')
        self.skyAxes.axis('off')
        self.skyAxes.set_facecolor(self.stageFace)

        flux = float(self.fluxByFrame[frame])
        cursorTime = float(self.timeByFrame[frame])
        title, subtitle = self._caption(frame)
        self.figure.suptitle(title, color=self.labelColor, fontsize=14, y=0.97)
        self.figure.text(
            0.5,
            0.935,
            subtitle,
            ha='center',
            va='top',
            color=self.labelColor,
            fontsize=9,
            alpha=0.82,
        )

        # Soft limb glow, then photosphere dimmed by measured flux.
        glow = Circle(
            (0.0, 0.0),
            radius=self.starRadiusAu * 1.18,
            facecolor=STAR_COLOR,
            edgecolor='none',
            alpha=0.12 + 0.10 * flux,
            zorder=2,
        )
        star = Circle(
            (0.0, 0.0),
            radius=self.starRadiusAu,
            facecolor=STAR_COLOR,
            edgecolor=self.dustEdgeColor,
            linewidth=1.1,
            alpha=0.28 + 0.72 * np.clip(flux, 0.0, 1.0),
            zorder=3,
        )
        self.skyAxes.add_patch(glow)
        self.skyAxes.add_patch(star)
        self.skyAxes.text(
            0.0,
            -self.starRadiusAu * 1.35,
            "Tabby's Star (F3V) · view from Earth",
            ha='center',
            color=self.labelColor,
            fontsize=8,
            alpha=0.7,
            zorder=6,
        )

        # Faint orbit ticks (impact-parameter lane).
        self.skyAxes.axhline(0.0, color=self.guideColor, linewidth=0.6, alpha=0.25, zorder=1)
        self.skyAxes.text(
            half * 0.62,
            half * 0.52,
            'line of sight\n(Kepler → star)',
            ha='center',
            color=self.labelColor,
            fontsize=7,
            alpha=0.55,
            zorder=6,
        )

        # Draw behind-star clumps first (dim), then in-front occultors on top of the disk.
        ordered = sorted(
            self.clumps,
            key=lambda clump: self._clumpOrbitXy(clump, frame)[0],
        )
        for clump in ordered:
            screenX, screenY, inFront, occulting = self._clumpScreenState(clump, frame)
            # Hide clumps far off-screen.
            if abs(screenX) > half * 1.15:
                continue
            angleDeg = np.degrees(self._clumpAngleRad(clump, frame))
            if inFront:
                alpha = 0.78 if occulting else 0.45
                zorder = 5 if occulting else 4
                edge = self.dustEdgeColor
                width = clump.sizeAu * (2.5 if occulting else 2.1)
                height = clump.sizeAu * (1.25 if occulting else 1.05)
            else:
                # Behind the star — only show if not covered by the disk.
                if abs(screenX) < self.starRadiusAu * 0.9:
                    continue
                alpha = 0.22
                zorder = 2
                edge = 'none'
                width = clump.sizeAu * 1.8
                height = clump.sizeAu * 0.9
            self.skyAxes.add_patch(
                Ellipse(
                    (screenX, screenY),
                    width=width,
                    height=height,
                    angle=angleDeg * 0.15,
                    facecolor=clump.color,
                    edgecolor=edge,
                    linewidth=1.35 if occulting else 0.0,
                    alpha=alpha,
                    zorder=zorder,
                )
            )
            if occulting:
                self.skyAxes.text(
                    screenX,
                    screenY + clump.sizeAu * 0.95,
                    f'−{clump.dipDepth * 100.0:.0f}%',
                    ha='center',
                    color=self.labelColor,
                    fontsize=8,
                    fontweight='bold',
                    zorder=7,
                )

        # Hero lightcurve — the edit spine.
        self.lcAxes.set_facecolor(self.lcFace)
        self.lcAxes.plot(
            self.keplerTimeBkjd,
            self.keplerFlux,
            color=self.curveColor,
            linewidth=1.15,
            solid_capstyle='round',
        )
        for clump in self.clumps:
            self.lcAxes.axvline(
                float(self.timeByFrame[clump.crossingFrame]),
                color=clump.color,
                linewidth=0.8,
                alpha=0.55,
            )
        self.lcAxes.axvline(cursorTime, color=self.labelColor, linewidth=1.35, alpha=0.95)
        self.lcAxes.scatter(
            [cursorTime],
            [flux],
            s=36,
            color=self.labelColor,
            zorder=5,
            edgecolors=self.curveColor,
            linewidths=0.8,
        )
        self.lcAxes.set_xlim(float(self.keplerTimeBkjd[0]), float(self.keplerTimeBkjd[-1]))
        self.lcAxes.set_ylim(0.75, 1.03)
        self.lcAxes.set_ylabel('Relative flux', color=self.labelColor, fontsize=8)
        self.lcAxes.set_xlabel('Kepler BKJD (days)', color=self.labelColor, fontsize=8)
        self.lcAxes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in self.lcAxes.spines.values():
            spine.set_color(self.labelColor)
            spine.set_alpha(0.45)
        self.lcAxes.set_title(
            'Kepler light curve · playhead = camera time',
            color=self.labelColor,
            fontsize=9,
            pad=6,
            loc='left',
        )
        dipPercent = max(0.0, (1.0 - flux) * 100.0)
        self.lcAxes.text(
            0.99,
            0.92,
            f'{flux:.3f}  (−{dipPercent:.1f}%)',
            transform=self.lcAxes.transAxes,
            ha='right',
            va='top',
            color=self.labelColor,
            fontsize=10,
            fontweight='bold',
            zorder=7,
        )
        self.lcAxes.text(
            0.99,
            0.78,
            f'BKJD {cursorTime:.0f}',
            transform=self.lcAxes.transAxes,
            ha='right',
            va='top',
            color=self.labelColor,
            fontsize=7,
            alpha=0.75,
            zorder=7,
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


def renderTabbysStarCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{OUTPUT_DIRECTORY}/tabbys_star_cinematic_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TabbysStarCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
        )
        animator.saveGif(outputPath)
    print("Tabby's Star lightcurve cinema completed!")
