"""Tabby's Star lightcurve cinema — Kepler flux drives the edit (#73).

Not a Sol→destination odyssey. The measurement timeline is the spine:
Blender F3V photosphere + soft debris sprites occult the disk on real dips.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from PIL import Image
from solsys.physics.catalogs.system_catalog import SystemCatalog

from animate.blender_body_sprites import BlenderBodySpriteAtlas
from animate.debris_sprites import DEBRIS_CLUMPS, debrisSpriteAvailable, loadDebrisSprite
from animate.scenes.tabbys_star import (
    DEFAULT_LIGHTCURVE_CSV,
    REFERENCE_DIP_DEPTH,
    OrbitingDustClump,
    buildOrbitingClumps,
    findDipCrossingFrames,
    loadKeplerLightCurve,
    sampleSeriesToFrames,
)

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 110
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480
TABBYS_CATALOG_NAME = "Tabby's Star"
STAR_DISPLAY_RESOLUTION = 512
# Photosphere panel uses normalized coords; star disk radius in those units.
STAR_DISK_RADIUS = 0.38
# Fixed framing — no dip push-in (that read as the star swelling).
STAR_PANEL_HALF_WIDTH = 0.72


def _debrisCatalogForIndex(index: int) -> str:
    return DEBRIS_CLUMPS[index % len(DEBRIS_CLUMPS)].catalogName


def _losWeight(clump: OrbitingDustClump, frame: int, animationFrames: int) -> float:
    """1 at LOS crossing; soft falloff so occultation reads around the dip."""
    angle = 2.0 * np.pi * (frame - clump.crossingFrame) / animationFrames
    # Angular distance to +X (LOS).
    delta = abs((angle + np.pi) % (2.0 * np.pi) - np.pi)
    # ~±35° window around the crossing.
    return float(np.clip(1.0 - delta / 0.62, 0.0, 1.0))


def _clumpDiskOffset(
    clump: OrbitingDustClump, frame: int, animationFrames: int
) -> tuple[float, float]:
    """Map orbit phase near LOS to an on-disk (x, y) offset in panel coords."""
    angle = 2.0 * np.pi * (frame - clump.crossingFrame) / animationFrames
    # Chord across the disk: horizontal sweep through mid-transit.
    chord = np.sin(angle)  # 0 at exact LOS when angle≈0
    impact = 0.22 * np.cos(angle * 0.5 + clump.orbitRadiusAu)
    return (float(chord * STAR_DISK_RADIUS * 0.95), float(impact * STAR_DISK_RADIUS))


def _dipStrength(clump: OrbitingDustClump) -> float:
    return float(np.clip(clump.dipDepth / REFERENCE_DIP_DEPTH, 0.18, 1.0))


class TabbysStarCinematicAnimator:
    """Lightcurve-led Tabby's episode: photosphere occultation + Kepler playhead."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
        lightcurveCsvPath: str = DEFAULT_LIGHTCURVE_CSV,
        *,
        requireBlenderBody: bool = True,
    ):
        catalog = SystemCatalog(starsCsvPath=starsCsvPath)
        self.system = catalog.load('tabbys_star')
        if self.system.planets:
            raise ValueError("Tabby's cinematic expects no catalog planets")
        if not self.system.stars:
            raise ValueError("Tabby's Star catalog is missing the host star")

        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.keplerTimeBkjd, self.keplerFlux = loadKeplerLightCurve(lightcurveCsvPath)
        self.fluxByFrame = sampleSeriesToFrames(
            self.keplerFlux, self.animationFrames, reduceMin=True
        )
        self.timeByFrame = sampleSeriesToFrames(self.keplerTimeBkjd, self.animationFrames)
        dipEvents = findDipCrossingFrames(self.fluxByFrame)
        self.clumps = buildOrbitingClumps(dipEvents)
        if not self.clumps:
            raise ValueError('No Kepler dip events found for Tabby cinematic clumps')
        self.deepestClump = max(self.clumps, key=lambda item: item.dipDepth)

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.theme = 'dark' if self.isDark else 'light'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.curveColor = '#7EB6FF' if self.isDark else '#204080'
        self.panelFace = '#050508' if self.isDark else '#F4F2EC'
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, facecolor=self.panelFace)
        grid = GridSpec(
            2,
            1,
            figure=self.figure,
            height_ratios=[1.35, 1.0],
            hspace=0.08,
            left=0.06,
            right=0.94,
            top=0.94,
            bottom=0.08,
        )
        self.starAxes = self.figure.add_subplot(grid[0])
        self.lcAxes = self.figure.add_subplot(grid[1])

        self.atlas = BlenderBodySpriteAtlas(self.theme)
        if requireBlenderBody and not self.atlas.hasBody(TABBYS_CATALOG_NAME):
            raise FileNotFoundError(
                f'Missing Blender spin for {TABBYS_CATALOG_NAME!r}. Run:\n'
                f'  render.py blender --body "Tabby\'s Star" --spin --theme all'
            )
        self._debrisCache: dict[str, np.ndarray | None] = {}
        for index, _clump in enumerate(self.clumps):
            name = _debrisCatalogForIndex(index)
            if name not in self._debrisCache:
                if not debrisSpriteAvailable(name):
                    raise FileNotFoundError(f'Missing debris sprite {name!r} (issue #78)')
                self._debrisCache[name] = loadDebrisSprite(name, size=320)

    def _compositeOccultation(self, starRgba: np.ndarray, frame: int) -> np.ndarray:
        """Alpha-composite dip-scaled debris in front of the photosphere."""
        canvas = starRgba.copy()
        height, width = canvas.shape[:2]
        for index, clump in enumerate(self.clumps):
            weight = _losWeight(clump, frame, self.animationFrames)
            if weight < 0.04:
                continue
            sprite = self._debrisCache[_debrisCatalogForIndex(index)]
            if sprite is None:
                continue
            strength = _dipStrength(clump)
            # Deeper dips → larger, more opaque cover.
            span = int(np.clip(width * (0.28 + 0.55 * strength), width * 0.25, width * 0.85))
            pil = Image.fromarray((np.clip(sprite, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
            pil = pil.resize((span, span), Image.Resampling.LANCZOS)
            clumpRgba = np.asarray(pil, dtype=np.float32) / 255.0
            # Soften opacity with LOS weight; keep cores dark for contrast.
            alphaScale = (0.45 + 0.55 * strength) * (0.35 + 0.65 * weight)
            clumpRgba = clumpRgba.copy()
            clumpRgba[..., 3] *= alphaScale

            offsetX, offsetY = _clumpDiskOffset(clump, frame, self.animationFrames)
            cx = int(width * (0.5 + offsetX / (2.0 * STAR_DISK_RADIUS) * 0.85))
            cy = int(height * (0.5 - offsetY / (2.0 * STAR_DISK_RADIUS) * 0.85))
            ch, cw = clumpRgba.shape[:2]
            x0, y0 = cx - cw // 2, cy - ch // 2
            x1, y1 = x0 + cw, y0 + ch
            xs0, ys0 = max(0, x0), max(0, y0)
            xs1, ys1 = min(width, x1), min(height, y1)
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            cx0, cy0 = xs0 - x0, ys0 - y0
            region = canvas[ys0:ys1, xs0:xs1]
            patch = clumpRgba[cy0 : cy0 + (ys1 - ys0), cx0 : cx0 + (xs1 - xs0)]
            # Only occult where the photosphere is present (star alpha).
            starAlpha = region[..., 3:4]
            alpha = patch[..., 3:4] * starAlpha
            region[..., :3] = patch[..., :3] * alpha + region[..., :3] * (1.0 - alpha)
            region[..., 3:4] = np.clip(
                region[..., 3:4] + alpha * (1.0 - region[..., 3:4]), 0.0, 1.0
            )
        return canvas

    def update(self, frame: int):
        self.starAxes.clear()
        self.lcAxes.clear()
        for axes in (self.starAxes, self.lcAxes):
            axes.set_facecolor(self.panelFace)
            for spine in axes.spines.values():
                spine.set_visible(False)

        half = STAR_PANEL_HALF_WIDTH
        self.starAxes.set_xlim(-half, half)
        self.starAxes.set_ylim(-half, half)
        self.starAxes.set_aspect('equal')
        self.starAxes.axis('off')

        star = self.atlas.bodyFrame(TABBYS_CATALOG_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION)
        if star is None:
            # Schematic fallback only if spin missing and requireBlenderBody was False.
            disk = plt.Circle((0, 0), STAR_DISK_RADIUS, color='#F8F0D8', alpha=0.95)
            self.starAxes.add_patch(disk)
        else:
            occulted = self._compositeOccultation(star, frame)
            extent = [-STAR_DISK_RADIUS, STAR_DISK_RADIUS, -STAR_DISK_RADIUS, STAR_DISK_RADIUS]
            self.starAxes.imshow(
                occulted,
                extent=extent,
                origin='upper',
                interpolation='bilinear',
                zorder=3,
            )

        flux = float(self.fluxByFrame[frame])
        dipPercent = max(0.0, (1.0 - flux) * 100.0)
        cursorTime = float(self.timeByFrame[frame])
        caption = 'KIC 8462852 · Kepler relative flux'
        if dipPercent >= 1.2:
            caption = f'Dip · {dipPercent:.0f}% · debris crossing the line of sight'
        self.starAxes.set_title(
            "Tabby's Star — lightcurve cinema",
            color=self.labelColor,
            fontsize=14,
            pad=10,
        )
        self.starAxes.text(
            0.0,
            -half * 0.92,
            caption,
            color=self.labelColor,
            fontsize=9,
            ha='center',
            alpha=0.85,
            zorder=6,
        )
        self.starAxes.text(
            half * 0.92,
            half * 0.88,
            'F3V',
            color=self.labelColor,
            fontsize=8,
            ha='right',
            alpha=0.55,
        )

        # Kepler strip — primary timeline plane (not a corner inset).
        self.lcAxes.plot(
            self.keplerTimeBkjd,
            self.keplerFlux,
            color=self.curveColor,
            linewidth=1.15,
            zorder=2,
        )
        self.lcAxes.axvline(cursorTime, color=self.labelColor, linewidth=1.2, alpha=0.9, zorder=4)
        for index, clump in enumerate(self.clumps):
            markTime = float(self.timeByFrame[clump.crossingFrame])
            self.lcAxes.axvline(
                markTime,
                color=('#C4A574', '#9A8A78', '#B07850')[index % 3],
                linewidth=0.8,
                alpha=0.45,
                zorder=3,
            )
        self.lcAxes.set_xlim(float(self.keplerTimeBkjd[0]), float(self.keplerTimeBkjd[-1]))
        self.lcAxes.set_ylim(0.75, 1.03)
        self.lcAxes.set_ylabel('Relative flux', color=self.labelColor, fontsize=9)
        self.lcAxes.set_xlabel('BKJD', color=self.labelColor, fontsize=9)
        self.lcAxes.tick_params(colors=self.labelColor, labelsize=7)
        self.lcAxes.set_title(
            f'Playhead · BKJD {cursorTime:.0f} · flux {flux:.3f}',
            color=self.labelColor,
            fontsize=9,
            loc='left',
            pad=6,
        )
        for spine in self.lcAxes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)
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
    outputDirectory = Path('output/animate/tabbys_star/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'tabbys_star_cinematic_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TabbysStarCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
            requireBlenderBody=True,
        )
        animator.saveGif(str(outputPath))
    print("Tabby's Star lightcurve cinema completed!")
