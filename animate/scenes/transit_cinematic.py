"""TRAPPIST-1 b transit cinema — real TESS photometry, revealed by folding (#95).

Sister grammar to Tabby's cinema (#73): stay with the star, let the measured
flux be the spine. The honest difference from a textbook transit diagram is
that a single transit here is *invisible* — TRAPPIST-1's 0.74% dip sits under
1.37% point-to-point scatter. Only stacking every transit on the period brings
the planet out, so the fold is the reveal the film is built around.

Every flux value plotted is observed: TESS Sector 70, 2-minute PDCSAP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from PIL import Image

from animate.blender_body_sprites import BlenderBodySpriteAtlas, diskRadiusFraction

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
# The photosphere fills the hero panel, so every frame differs across a large
# area; this dpi keeps the GIF inside the gallery's size budget.
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

DEFAULT_LIGHTCURVE_CSV = 'data/trappist_1_tess_lightcurve.csv'
TRAPPIST_1_CATALOG_NAME = 'TRAPPIST-1'
TRANSITING_PLANET_NAME = 'TRAPPIST-1 b'

# Measured from the committed light curve by box-least-squares, no catalog
# input — see the provenance header in the CSV.
TESS_PERIOD_DAYS = 1.510919
TESS_MID_TRANSIT_BTJD = 3209.809538
# Published values for comparison (NASA Exoplanet Archive pscomppars).
PUBLISHED_PERIOD_DAYS = 1.510826
PUBLISHED_RADIUS_RATIO = 0.08590
PUBLISHED_IMPACT_PARAMETER = 0.095
PUBLISHED_DURATION_DAYS = 0.6010 / 24.0
PUBLISHED_DEPTH = PUBLISHED_RADIUS_RATIO**2

DISPLAY_BIN_MINUTES = 10.0
FOLD_BIN_MINUTES = 10.0
FOLD_HALF_WINDOW_DAYS = 3.0 / 24.0

STAR_DISPLAY_RESOLUTION = 512
STAR_DISK_RADIUS = 0.76
# Blender sprites carry transparent margin, so the photosphere covers only part
# of the frame; the panel is tightened to what the disk actually fills.
STAR_PANEL_HALF_WIDTH = 0.50
# Silhouette, not a lit globe: a transiting planet shows us its night side.
SILHOUETTE_RGB_SCALE = 0.12

# Act structure: watch the raw stream, fold it, then read the stacked dip.
STREAM_FRAMES = 250
FOLD_FRAMES = 60
STREAM_FLUX_LIMITS = (0.950, 1.050)
FOLD_FLUX_LIMITS = (0.985, 1.012)
# Real transits are ~2% of the window, so an even playhead would skim past them.
TRANSIT_TIME_BOOST = 26.0
TRANSIT_SHOULDER = 0.8


@dataclass(frozen=True)
class FoldedProfile:
    """Phase-binned stack of every observed transit."""

    phaseHours: np.ndarray
    flux: np.ndarray
    error: np.ndarray
    depth: float
    depthError: float
    transitCount: int


def loadTessLightCurve(
    csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed TESS time (BTJD) and detrended relative flux."""
    frame = pd.read_csv(csvPath, comment='#')
    return (
        np.asarray(frame['btjd_day'], dtype=float),
        np.asarray(frame['detrended_flux'], dtype=float),
    )


def binSeries(
    time: np.ndarray, flux: np.ndarray, binMinutes: float
) -> tuple[np.ndarray, np.ndarray]:
    """Average onto fixed-width bins, dropping empties (the downlink gap)."""
    width = binMinutes / 1440.0
    index = np.floor((time - time.min()) / width).astype(int)
    counts = np.bincount(index)
    sumsTime = np.bincount(index, weights=time)
    sumsFlux = np.bincount(index, weights=flux)
    filled = counts > 0
    return sumsTime[filled] / counts[filled], sumsFlux[filled] / counts[filled]


def transitPhase(
    time: np.ndarray,
    *,
    periodDays: float = TESS_PERIOD_DAYS,
    midTransitBtjd: float = TESS_MID_TRANSIT_BTJD,
) -> np.ndarray:
    """Signed days from the nearest mid-transit."""
    return (time - midTransitBtjd + 0.5 * periodDays) % periodDays - 0.5 * periodDays


def observedTransitTimes(
    time: np.ndarray,
    *,
    periodDays: float = TESS_PERIOD_DAYS,
    midTransitBtjd: float = TESS_MID_TRANSIT_BTJD,
    durationDays: float = PUBLISHED_DURATION_DAYS,
) -> tuple[float, ...]:
    """Mid-transit times the sector actually covers with data."""
    first = np.ceil((time.min() - midTransitBtjd) / periodDays)
    last = np.floor((time.max() - midTransitBtjd) / periodDays)
    mids = midTransitBtjd + np.arange(first, last + 1) * periodDays
    return tuple(float(mid) for mid in mids if np.any(np.abs(time - mid) < durationDays * 0.5))


def foldedProfile(
    time: np.ndarray,
    flux: np.ndarray,
    *,
    periodDays: float = TESS_PERIOD_DAYS,
    midTransitBtjd: float = TESS_MID_TRANSIT_BTJD,
    binMinutes: float = FOLD_BIN_MINUTES,
    halfWindowDays: float = FOLD_HALF_WINDOW_DAYS,
    durationDays: float = PUBLISHED_DURATION_DAYS,
) -> FoldedProfile:
    """Stack every transit on the period; the dip only exists after this step."""
    phase = transitPhase(time, periodDays=periodDays, midTransitBtjd=midTransitBtjd)
    inside = np.abs(phase) <= halfWindowDays
    phase, stacked = phase[inside], flux[inside]

    width = binMinutes / 1440.0
    edges = np.arange(-halfWindowDays, halfWindowDays + width, width)
    index = np.clip(np.digitize(phase, edges) - 1, 0, len(edges) - 2)
    counts = np.bincount(index, minlength=len(edges) - 1)
    sums = np.bincount(index, weights=stacked, minlength=len(edges) - 1)
    squares = np.bincount(index, weights=stacked**2, minlength=len(edges) - 1)
    filled = counts > 1
    means = sums[filled] / counts[filled]
    variance = np.maximum(squares[filled] / counts[filled] - means**2, 0.0)
    errors = np.sqrt(variance / counts[filled])
    centers = ((edges[:-1] + edges[1:]) * 0.5)[filled]

    inTransit = np.abs(phase) < durationDays * 0.4
    outTransit = np.abs(phase) > durationDays
    baseline = float(stacked[outTransit].mean())
    depth = baseline - float(stacked[inTransit].mean())
    depthError = float(stacked[outTransit].std() / np.sqrt(inTransit.sum()))
    return FoldedProfile(
        phaseHours=centers * 24.0,
        flux=means,
        error=errors,
        depth=depth,
        depthError=depthError,
        transitCount=len(
            observedTransitTimes(time, periodDays=periodDays, midTransitBtjd=midTransitBtjd)
        ),
    )


def transitWeight(
    time: np.ndarray,
    midTimes: tuple[float, ...],
    *,
    durationDays: float = PUBLISHED_DURATION_DAYS,
    shoulder: float = TRANSIT_SHOULDER,
) -> np.ndarray:
    """1 inside a transit, tapering to 0 across a shoulder either side."""
    weight = np.zeros_like(time)
    half = max(durationDays * 0.5, 1e-9)
    for mid in midTimes:
        distance = np.abs(time - mid) / half
        taper = 0.5 * (1.0 + np.cos(np.pi * np.clip((distance - 1.0) / shoulder, 0.0, 1.0)))
        weight = np.maximum(weight, np.where(distance <= 1.0, 1.0, taper))
    return weight


def warpedTimeByFrame(
    time: np.ndarray,
    weight: np.ndarray,
    frames: int,
    *,
    boost: float = TRANSIT_TIME_BOOST,
) -> np.ndarray:
    """Playhead times that linger on transits and skim the quiet baseline."""
    spend = 1.0 + boost * np.asarray(weight, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(spend[:-1] * np.diff(time))))
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.linspace(time[0], time[-1], frames)
    return np.interp(np.linspace(0.0, total, frames), cumulative, time)


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


class TransitCinematicAnimator:
    """Three acts: watch the raw TESS stream, fold it, read the stacked dip."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
        *,
        requireBlenderBody: bool = True,
    ):
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES

        self.time, self.flux = loadTessLightCurve(lightcurveCsvPath)
        self.binnedTime, self.binnedFlux = binSeries(self.time, self.flux, DISPLAY_BIN_MINUTES)
        self.midTimes = observedTransitTimes(self.time)
        if len(self.midTimes) < 5:
            raise ValueError('Light curve covers too few transits to fold')
        self.profile = foldedProfile(self.time, self.flux)
        self.pointScatter = float(np.std(np.diff(self.flux)) / np.sqrt(2.0))

        self.streamTimeByFrame = warpedTimeByFrame(
            self.binnedTime,
            transitWeight(self.binnedTime, self.midTimes),
            STREAM_FRAMES,
        )
        revealFrames = self.animationFrames - STREAM_FRAMES - FOLD_FRAMES
        phaseGrid = np.linspace(-FOLD_HALF_WINDOW_DAYS, FOLD_HALF_WINDOW_DAYS, 2000)
        self.revealPhaseByFrame = warpedTimeByFrame(
            phaseGrid,
            transitWeight(phaseGrid, (0.0,)),
            revealFrames,
        )

        # Normalized x for both layouts, so the fold can interpolate between them.
        span = self.binnedTime.max() - self.binnedTime.min()
        self.streamX = (self.binnedTime - self.binnedTime.min()) / span
        binnedPhase = transitPhase(self.binnedTime)
        self.foldX = (binnedPhase + FOLD_HALF_WINDOW_DAYS) / (2.0 * FOLD_HALF_WINDOW_DAYS)
        self.foldable = np.abs(binnedPhase) <= FOLD_HALF_WINDOW_DAYS

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.theme = 'dark' if self.isDark else 'light'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.pointColor = '#7EB6FF' if self.isDark else '#204080'
        self.accentColor = '#FFB570' if self.isDark else '#B4540A'
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
        if requireBlenderBody:
            missing = [
                name
                for name in (TRAPPIST_1_CATALOG_NAME, TRANSITING_PLANET_NAME)
                if not self.atlas.hasBody(name)
            ]
            if missing:
                commands = '\n'.join(
                    f'  render.py blender --body "{name}" --spin --theme all' for name in missing
                )
                raise FileNotFoundError(f'Missing Blender spins for {missing}. Run:\n{commands}')
        self._diskFractions: dict[str, float] = {}

    # ---- act bookkeeping -------------------------------------------------

    def act(self, frame: int) -> str:
        if frame < STREAM_FRAMES:
            return 'stream'
        if frame < STREAM_FRAMES + FOLD_FRAMES:
            return 'fold'
        return 'reveal'

    def foldProgress(self, frame: int) -> float:
        """0 while streaming, 1 once every transit is stacked."""
        return smoothStep((frame - STREAM_FRAMES) / FOLD_FRAMES)

    def phaseOffsetDays(self, frame: int) -> float:
        """Signed days from mid-transit for whatever the playhead is over."""
        act = self.act(frame)
        if act == 'stream':
            return float(transitPhase(np.array([self.streamTimeByFrame[frame]]))[0])
        if act == 'fold':
            return float(FOLD_HALF_WINDOW_DAYS)  # parked off-transit during the fold
        return float(self.revealPhaseByFrame[frame - STREAM_FRAMES - FOLD_FRAMES])

    def inTransit(self, frame: int) -> bool:
        return abs(self.phaseOffsetDays(frame)) <= PUBLISHED_DURATION_DAYS * 0.5

    # ---- star panel ------------------------------------------------------

    def _diskFraction(self, catalogName: str) -> float:
        if catalogName not in self._diskFractions:
            sprite = self.atlas.bodyFrame(catalogName, 0, resolution=STAR_DISPLAY_RESOLUTION)
            self._diskFractions[catalogName] = 1.0 if sprite is None else diskRadiusFraction(sprite)
        return self._diskFractions[catalogName]

    def _planetSilhouette(self, frame: int, diskPixels: float) -> np.ndarray:
        """Planet spin frame darkened to a night-side disk (falls back to a dot)."""
        sprite = self.atlas.bodyFrame(
            TRANSITING_PLANET_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION
        )
        if sprite is None:
            span = max(int(round(diskPixels)), 4)
            radius = span * 0.5
            yy, xx = np.mgrid[0:span, 0:span]
            distance = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2)
            disk = np.zeros((span, span, 4), dtype=np.float32)
            disk[..., 3] = np.clip(radius - distance, 0.0, 1.0)
            return disk
        # Scale so the planet's own disk — not its transparent margin — matches
        # the size the published radius ratio demands.
        span = max(int(round(diskPixels / self._diskFraction(TRANSITING_PLANET_NAME))), 6)
        pil = Image.fromarray((np.clip(sprite, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
        pil = pil.resize((span, span), Image.Resampling.LANCZOS)
        disk = np.asarray(pil, dtype=np.float32) / 255.0
        disk[..., :3] *= SILHOUETTE_RGB_SCALE
        return disk

    def _compositeTransit(self, starRgba: np.ndarray, frame: int) -> np.ndarray:
        """Alpha-composite b in front of the photosphere on transit frames."""
        canvas = starRgba.copy()
        offset = self.phaseOffsetDays(frame) / (PUBLISHED_DURATION_DAYS * 0.5)
        if abs(offset) > 1.0:
            return canvas
        height, width = canvas.shape[:2]
        starRadiusPixels = width * 0.5 * self._diskFraction(TRAPPIST_1_CATALOG_NAME)
        disk = self._planetSilhouette(frame, 2.0 * PUBLISHED_RADIUS_RATIO * starRadiusPixels)
        # Chord across the disk: contact to contact in stellar radii.
        chordHalfSpan = float(
            np.sqrt((1.0 + PUBLISHED_RADIUS_RATIO) ** 2 - PUBLISHED_IMPACT_PARAMETER**2)
        )
        centerX = int(round(width * 0.5 + offset * chordHalfSpan * starRadiusPixels))
        centerY = int(round(height * 0.5 - PUBLISHED_IMPACT_PARAMETER * starRadiusPixels))
        diskHeight, diskWidth = disk.shape[:2]
        x0, y0 = centerX - diskWidth // 2, centerY - diskHeight // 2
        xs0, ys0 = max(0, x0), max(0, y0)
        xs1, ys1 = min(width, x0 + diskWidth), min(height, y0 + diskHeight)
        if xs1 <= xs0 or ys1 <= ys0:
            return canvas
        region = canvas[ys0:ys1, xs0:xs1]
        patch = disk[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0]
        # Only occult where the photosphere is present (star alpha).
        alpha = patch[..., 3:4] * region[..., 3:4]
        region[..., :3] = patch[..., :3] * alpha + region[..., :3] * (1.0 - alpha)
        region[..., 3:4] = np.clip(region[..., 3:4] + alpha * (1.0 - region[..., 3:4]), 0.0, 1.0)
        return canvas

    # ---- captions --------------------------------------------------------

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        if act == 'stream':
            if self.inTransit(frame):
                return (
                    f'TRAPPIST-1 b is transiting right now · {PUBLISHED_DEPTH * 100:.2f}% deep · '
                    f'and the scatter is {self.pointScatter * 100:.2f}%'
                )
            return 'TESS Sector 70 · 2-minute PDCSAP · every point observed'
        if act == 'fold':
            return (
                f'Stack all {self.profile.transitCount} transits on the '
                f'{TESS_PERIOD_DAYS:.4f} d period'
            )
        if self.inTransit(frame):
            # Limb darkening puts the deepest part of a real transit below the
            # geometric (Rp/R*)^2, since the planet covers the bright middle.
            return (
                f'There it is · stacked depth {self.profile.depth * 100:.2f}% '
                f'± {self.profile.depthError * 100:.2f}% · limb darkening deepens the '
                f'geometric {PUBLISHED_DEPTH * 100:.2f}%'
            )
        return (
            f'Folded · {FOLD_BIN_MINUTES:.0f}-minute phase bins · '
            f'{self.profile.transitCount} transits stacked'
        )

    # ---- frame -----------------------------------------------------------

    def _drawStarPanel(self, frame: int) -> None:
        half = STAR_PANEL_HALF_WIDTH
        self.starAxes.set_xlim(-half, half)
        self.starAxes.set_ylim(-half, half)
        self.starAxes.set_aspect('equal')
        self.starAxes.axis('off')

        star = self.atlas.bodyFrame(
            TRAPPIST_1_CATALOG_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION
        )
        if star is None:
            self.starAxes.add_patch(plt.Circle((0, 0), STAR_DISK_RADIUS, color='#FF6B4A'))
        else:
            extent = [-STAR_DISK_RADIUS, STAR_DISK_RADIUS, -STAR_DISK_RADIUS, STAR_DISK_RADIUS]
            self.starAxes.imshow(
                self._compositeTransit(star, frame),
                extent=extent,
                origin='upper',
                interpolation='bilinear',
                zorder=3,
            )
        self.starAxes.set_title(
            'TRAPPIST-1 b — the transit you cannot see until you fold it',
            color=self.labelColor,
            fontsize=14,
            pad=10,
        )
        self.starAxes.text(
            0.5,
            0.015,
            self.caption(frame),
            transform=self.starAxes.transAxes,
            color=self.labelColor,
            fontsize=9,
            ha='center',
            alpha=0.85,
            zorder=6,
        )
        self.starAxes.text(
            0.985,
            0.965,
            'M8V · TIC 278892590',
            transform=self.starAxes.transAxes,
            color=self.labelColor,
            fontsize=8,
            ha='right',
            va='top',
            alpha=0.55,
        )

    def pointPositions(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Normalized x and alpha per point, morphing from time to phase.

        Points outside the fold window fade out — the fold keeps the windows
        around each transit and throws the quiet baseline away.
        """
        fold = self.foldProgress(frame)
        x = self.streamX + (self.foldX - self.streamX) * fold
        alpha = np.where(self.foldable, 0.75, 0.75 * (1.0 - fold))
        return x, alpha

    def _drawLightcurvePanel(self, frame: int) -> None:
        fold = self.foldProgress(frame)
        x, alpha = self.pointPositions(frame)
        colors = np.zeros((len(x), 4))
        rgb = plt.matplotlib.colors.to_rgb(self.pointColor)
        colors[:, :3] = rgb
        colors[:, 3] = alpha
        self.lcAxes.scatter(x, self.binnedFlux, s=3.0, c=colors, linewidths=0.0, zorder=2)

        low = STREAM_FLUX_LIMITS[0] + (FOLD_FLUX_LIMITS[0] - STREAM_FLUX_LIMITS[0]) * fold
        high = STREAM_FLUX_LIMITS[1] + (FOLD_FLUX_LIMITS[1] - STREAM_FLUX_LIMITS[1]) * fold
        self.lcAxes.set_xlim(0.0, 1.0)
        self.lcAxes.set_ylim(low, high)

        if fold >= 1.0:
            profileX = (self.profile.phaseHours / 24.0 + FOLD_HALF_WINDOW_DAYS) / (
                2.0 * FOLD_HALF_WINDOW_DAYS
            )
            self.lcAxes.errorbar(
                profileX,
                self.profile.flux,
                yerr=self.profile.error,
                fmt='o-',
                color=self.accentColor,
                markersize=3.5,
                linewidth=1.2,
                elinewidth=0.8,
                capsize=0.0,
                zorder=4,
            )
            self.lcAxes.axhline(
                1.0 - PUBLISHED_DEPTH,
                color=self.labelColor,
                linestyle='--',
                linewidth=0.9,
                alpha=0.45,
                zorder=3,
            )

        playhead = self._playheadX(frame)
        if playhead is not None:
            self.lcAxes.axvline(playhead, color=self.labelColor, linewidth=1.2, alpha=0.9, zorder=5)

        self._drawAxisLabels(fold)

    def _playheadX(self, frame: int) -> float | None:
        act = self.act(frame)
        if act == 'stream':
            span = self.binnedTime.max() - self.binnedTime.min()
            return float((self.streamTimeByFrame[frame] - self.binnedTime.min()) / span)
        if act == 'fold':
            return None
        phase = self.revealPhaseByFrame[frame - STREAM_FRAMES - FOLD_FRAMES]
        return float((phase + FOLD_HALF_WINDOW_DAYS) / (2.0 * FOLD_HALF_WINDOW_DAYS))

    def _drawAxisLabels(self, fold: float) -> None:
        if fold < 0.5:
            ticks = np.linspace(0.0, 1.0, 6)
            values = self.binnedTime.min() + ticks * (self.binnedTime.max() - self.binnedTime.min())
            labels = [f'{value:.0f}' for value in values]
            xlabel = 'BTJD (days)'
            labelAlpha = 1.0 - 2.0 * fold
        else:
            hours = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
            ticks = (hours / 24.0 + FOLD_HALF_WINDOW_DAYS) / (2.0 * FOLD_HALF_WINDOW_DAYS)
            labels = [f'{hour:.0f}' for hour in hours]
            xlabel = 'hours from mid-transit'
            labelAlpha = 2.0 * fold - 1.0
        self.lcAxes.set_xticks(ticks)
        self.lcAxes.set_xticklabels(labels, alpha=max(labelAlpha, 0.0))
        self.lcAxes.set_xlabel(
            xlabel, color=self.labelColor, fontsize=9, alpha=max(labelAlpha, 0.0)
        )
        self.lcAxes.set_ylabel('Relative flux', color=self.labelColor, fontsize=9)
        self.lcAxes.tick_params(colors=self.labelColor, labelsize=7)
        self.lcAxes.text(
            1.0,
            -0.145,
            'TESS Sector 70 · MAST · detrended, flares and gap left in',
            transform=self.lcAxes.transAxes,
            color=self.labelColor,
            fontsize=7,
            ha='right',
            alpha=0.6,
        )
        for spine in self.lcAxes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)

    def update(self, frame: int):
        self.starAxes.clear()
        self.lcAxes.clear()
        for axes in (self.starAxes, self.lcAxes):
            axes.set_facecolor(self.panelFace)
            for spine in axes.spines.values():
                spine.set_visible(False)
        self._drawStarPanel(frame)
        self._drawLightcurvePanel(frame)
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


def renderTransitCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> None:
    outputDirectory = Path('output/animate/trappist_1/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'trappist_1_transit_cinematic_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TransitCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            lightcurveCsvPath=lightcurveCsvPath,
            requireBlenderBody=True,
        )
        animator.saveGif(str(outputPath))
    print('TRAPPIST-1 transit cinema completed!')


__all__ = [
    'FoldedProfile',
    'TransitCinematicAnimator',
    'binSeries',
    'diskRadiusFraction',
    'foldedProfile',
    'loadTessLightCurve',
    'observedTransitTimes',
    'renderTransitCinematicAnimations',
    'transitPhase',
    'transitWeight',
    'warpedTimeByFrame',
]
