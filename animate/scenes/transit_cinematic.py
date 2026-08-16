"""TRAPPIST-1 transit lightcurve cinema — periodic planet dips drive the edit (#95).

Sister grammar to Tabby's cinema (#73): stay with the star, let the flux
timeline be the spine. The difference is the explanation — here the dips are
*periodic* and a planet silhouette crosses the photosphere on every one.

The flux strip is a transit model built from catalog periods and radii, not
photometry: depth is (Rp/R*)^2 and duration follows the chord geometry, but
the epochs are illustrative so one window shows a representative train.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from PIL import Image
from solsys.physics.catalogs.system_catalog import StarSystem, SystemCatalog

from animate.blender_body_sprites import BlenderBodySpriteAtlas

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
# The photosphere fills the hero panel, so every frame differs across a large
# area; this dpi keeps the GIF inside the gallery's size budget.
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

TRAPPIST_1_SYSTEM_ID = 'trappist_1'
TRAPPIST_1_CATALOG_NAME = 'TRAPPIST-1'
# R* ~ 0.1192 R_sun — same value as the Blender host pack (#88).
TRAPPIST_1_STAR_RADIUS_KM = 83_000.0
AU_KM = 149_597_870.7

STAR_DISPLAY_RESOLUTION = 512
STAR_DISK_RADIUS = 0.76
# Blender sprites carry transparent margin, so the photosphere covers only part
# of the frame; the panel is tightened to what the disk actually fills.
STAR_PANEL_HALF_WIDTH = 0.50
# Silhouette, not a lit globe: a transiting planet shows us its night side.
SILHOUETTE_RGB_SCALE = 0.12

WINDOW_DAYS = 8.0
MODEL_SAMPLES = 16_000
# Frames spent per unit time inside a transit vs the quiet baseline. Real
# transits are ~0.5% of the window, so an even playhead would skip them.
TRANSIT_TIME_BOOST = 110.0
# Taper width either side of a transit, in half-durations.
TRANSIT_SHOULDER = 0.6

# Illustrative first-transit epochs (days into the window), tuned so the train
# shows singles, a near-pair, and one genuine b+c overlap.
FIRST_TRANSIT_DAYS: dict[str, float] = {
    'TRAPPIST-1 b': 0.55,
    'TRAPPIST-1 c': 1.155,
    'TRAPPIST-1 d': 2.45,
    'TRAPPIST-1 e': 3.30,
    'TRAPPIST-1 f': 5.10,
    'TRAPPIST-1 g': 6.60,
    'TRAPPIST-1 h': 4.05,
}
# Illustrative impact parameters (fraction of R*, signed for north/south chord).
IMPACT_PARAMETERS: dict[str, float] = {
    'TRAPPIST-1 b': 0.10,
    'TRAPPIST-1 c': 0.30,
    'TRAPPIST-1 d': -0.20,
    'TRAPPIST-1 e': 0.36,
    'TRAPPIST-1 f': -0.32,
    'TRAPPIST-1 g': 0.24,
    'TRAPPIST-1 h': -0.40,
}


@dataclass(frozen=True)
class TransitingPlanet:
    """One transiting world: depth and durations from catalog radius + period."""

    name: str
    shortName: str
    periodDays: float
    radiusRatio: float
    impactParameter: float
    totalDurationDays: float
    flatDurationDays: float
    midTimesDays: tuple[float, ...]
    color: str

    @property
    def depth(self) -> float:
        """Uniform-disk transit depth (Rp/R*)^2."""
        return self.radiusRatio**2

    def chordHalfSpan(self) -> float:
        """Projected star-center distance at contact, in stellar radii."""
        return float(np.sqrt(max((1.0 + self.radiusRatio) ** 2 - self.impactParameter**2, 1e-9)))

    def nearestMidTime(self, timeDays: float) -> float:
        return min(self.midTimesDays, key=lambda mid: abs(mid - timeDays))

    def phaseOffset(self, timeDays: float) -> float:
        """Signed offset from the nearest mid-transit, in half-durations."""
        half = max(self.totalDurationDays * 0.5, 1e-9)
        return float((timeDays - self.nearestMidTime(timeDays)) / half)

    def inTransit(self, timeDays: float) -> bool:
        return abs(self.phaseOffset(timeDays)) <= 1.0

    def fluxAt(self, timeDays: np.ndarray | float) -> np.ndarray:
        """Trapezoid transit: flat bottom between second and third contact."""
        times = np.atleast_1d(np.asarray(timeDays, dtype=float))
        drop = np.zeros_like(times)
        halfTotal = self.totalDurationDays * 0.5
        halfFlat = self.flatDurationDays * 0.5
        for mid in self.midTimesDays:
            offset = np.abs(times - mid)
            inside = offset <= halfTotal
            if not np.any(inside):
                continue
            # 1 on the flat bottom, ramping to 0 across ingress / egress.
            ramp = np.clip((halfTotal - offset) / max(halfTotal - halfFlat, 1e-9), 0.0, 1.0)
            drop = np.maximum(drop, np.where(inside, ramp, 0.0) * self.depth)
        return drop


def diskRadiusFraction(rgba: np.ndarray) -> float:
    """Opaque disk radius in a Blender sprite, as a fraction of its half-width.

    Sprites are rendered with transparent margin around the body, so on-disk
    geometry has to be measured rather than assumed to fill the frame.
    """
    alpha = np.asarray(rgba)[..., 3]
    height, width = alpha.shape
    solid = alpha > 0.5
    if not solid.any():
        return 1.0
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((xx - (width - 1) * 0.5) ** 2 + (yy - (height - 1) * 0.5) ** 2)
    return float(radius[solid].max() / (min(width, height) * 0.5))


def _transitDurationsDays(
    periodDays: float,
    semiMajorAxisKm: float,
    radiusRatio: float,
    impactParameter: float,
) -> tuple[float, float]:
    """Total (first-to-fourth contact) and flat (second-to-third) durations."""
    starRadiiPerOrbit = TRAPPIST_1_STAR_RADIUS_KM / semiMajorAxisKm
    outer = (1.0 + radiusRatio) ** 2 - impactParameter**2
    inner = (1.0 - radiusRatio) ** 2 - impactParameter**2
    scale = periodDays / np.pi * starRadiiPerOrbit
    total = float(scale * np.sqrt(max(outer, 0.0)))
    flat = float(scale * np.sqrt(max(inner, 0.0)))
    return total, flat


def buildTransitingPlanets(
    system: StarSystem,
    *,
    windowDays: float = WINDOW_DAYS,
) -> tuple[TransitingPlanet, ...]:
    """Turn catalog planets into transit events across one observing window."""
    palette = ('#7EB6FF', '#FFC46B', '#8FE3A2', '#FF8FA3', '#C9A6FF', '#6FD8D8', '#E0C46C')
    planets: list[TransitingPlanet] = []
    for index, planet in enumerate(sorted(system.planets, key=lambda item: item.semiMajorAxisAu)):
        radiusRatio = (planet.diameterKm * 0.5) / TRAPPIST_1_STAR_RADIUS_KM
        impact = IMPACT_PARAMETERS.get(planet.name, 0.0)
        total, flat = _transitDurationsDays(
            planet.orbitalPeriodDays,
            planet.semiMajorAxisAu * AU_KM,
            radiusRatio,
            impact,
        )
        first = FIRST_TRANSIT_DAYS.get(planet.name, 0.0)
        midTimes = tuple(
            float(first + step * planet.orbitalPeriodDays)
            for step in range(int(np.ceil(windowDays / planet.orbitalPeriodDays)) + 1)
            if first + step * planet.orbitalPeriodDays <= windowDays
        )
        if not midTimes:
            continue
        planets.append(
            TransitingPlanet(
                name=planet.name,
                shortName=planet.name.replace('TRAPPIST-1 ', ''),
                periodDays=float(planet.orbitalPeriodDays),
                radiusRatio=float(radiusRatio),
                impactParameter=float(impact),
                totalDurationDays=total,
                flatDurationDays=flat,
                midTimesDays=midTimes,
                color=palette[index % len(palette)],
            )
        )
    return tuple(planets)


def modelFlux(planets: tuple[TransitingPlanet, ...], timeDays: np.ndarray) -> np.ndarray:
    """Combined relative flux; overlapping transits stack their depths."""
    times = np.asarray(timeDays, dtype=float)
    drop = np.zeros_like(times)
    for planet in planets:
        drop = drop + planet.fluxAt(times)
    return 1.0 - drop


def transitWeight(
    planets: tuple[TransitingPlanet, ...],
    timeDays: np.ndarray,
    *,
    shoulder: float = TRANSIT_SHOULDER,
) -> np.ndarray:
    """1 inside any transit, tapering to 0 across a shoulder either side.

    The taper keeps the playhead from changing speed abruptly at ingress.
    """
    times = np.asarray(timeDays, dtype=float)
    weight = np.zeros_like(times)
    for planet in planets:
        half = max(planet.totalDurationDays * 0.5, 1e-9)
        for mid in planet.midTimesDays:
            distance = np.abs(times - mid) / half
            taper = 0.5 * (1.0 + np.cos(np.pi * np.clip((distance - 1.0) / shoulder, 0.0, 1.0)))
            weight = np.maximum(weight, np.where(distance <= 1.0, 1.0, taper))
    return weight


def warpedTimeByFrame(
    timeDays: np.ndarray,
    weight: np.ndarray,
    animationFrames: int,
    *,
    boost: float = TRANSIT_TIME_BOOST,
) -> np.ndarray:
    """Playhead times that linger on transits and skim the flat baseline.

    Frame spacing is proportional to 1 / (1 + boost * weight), so the edit
    spends most of its frames on the events without altering the curve.
    """
    times = np.asarray(timeDays, dtype=float)
    spend = 1.0 + boost * np.asarray(weight, dtype=float)
    cumulative = np.concatenate(([0.0], np.cumsum(spend[:-1] * np.diff(times))))
    total = float(cumulative[-1])
    if total <= 0.0:
        return np.linspace(times[0], times[-1], animationFrames)
    targets = np.linspace(0.0, total, animationFrames)
    return np.interp(targets, cumulative, times)


class TransitCinematicAnimator:
    """Transit-led TRAPPIST-1 episode: planet silhouettes + a model flux playhead."""

    def __init__(
        self,
        system: StarSystem | None = None,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
        *,
        requireBlenderBody: bool = True,
    ):
        if system is None:
            system = SystemCatalog(starsCsvPath=starsCsvPath).load(TRAPPIST_1_SYSTEM_ID)
        if system.systemId != TRAPPIST_1_SYSTEM_ID:
            raise ValueError(f'Expected {TRAPPIST_1_SYSTEM_ID}, got {system.systemId!r}')
        if not system.planets:
            raise ValueError('Transit cinema needs at least one catalog planet')

        self.system = system
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.planets = buildTransitingPlanets(system)
        if not self.planets:
            raise ValueError('No transits fall inside the observing window')

        self.modelTimeDays = np.linspace(0.0, WINDOW_DAYS, MODEL_SAMPLES)
        self.modelFluxSeries = modelFlux(self.planets, self.modelTimeDays)
        self.timeByFrame = warpedTimeByFrame(
            self.modelTimeDays,
            transitWeight(self.planets, self.modelTimeDays),
            self.animationFrames,
        )

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
        if requireBlenderBody:
            required = (TRAPPIST_1_CATALOG_NAME, *(planet.name for planet in self.planets))
            missing = [name for name in required if not self.atlas.hasBody(name)]
            if missing:
                commands = '\n'.join(
                    f'  render.py blender --body "{name}" --spin --theme all' for name in missing
                )
                raise FileNotFoundError(f'Missing Blender spins for {missing}. Run:\n{commands}')
        self._diskFractions: dict[str, float] = {}

    def transitingNow(self, frame: int) -> tuple[TransitingPlanet, ...]:
        timeDays = float(self.timeByFrame[frame])
        return tuple(planet for planet in self.planets if planet.inTransit(timeDays))

    def _diskFraction(self, catalogName: str) -> float:
        if catalogName not in self._diskFractions:
            sprite = self.atlas.bodyFrame(catalogName, 0, resolution=STAR_DISPLAY_RESOLUTION)
            self._diskFractions[catalogName] = 1.0 if sprite is None else diskRadiusFraction(sprite)
        return self._diskFractions[catalogName]

    def _planetSilhouette(self, planet: TransitingPlanet, frame: int, diskPixels: float):
        """Planet spin frame darkened to a night-side disk (falls back to a dot)."""
        sprite = self.atlas.bodyFrame(planet.name, frame, resolution=STAR_DISPLAY_RESOLUTION)
        if sprite is None:
            span = max(int(round(diskPixels)), 4)
            radius = span * 0.5
            yy, xx = np.mgrid[0:span, 0:span]
            distance = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2)
            disk = np.zeros((span, span, 4), dtype=np.float32)
            disk[..., 3] = np.clip(radius - distance, 0.0, 1.0)
            return disk
        # Scale so the planet's own disk — not its transparent margin — matches
        # the size the depth demands.
        span = max(int(round(diskPixels / self._diskFraction(planet.name))), 6)
        pil = Image.fromarray((np.clip(sprite, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
        pil = pil.resize((span, span), Image.Resampling.LANCZOS)
        disk = np.asarray(pil, dtype=np.float32) / 255.0
        disk[..., :3] *= SILHOUETTE_RGB_SCALE
        return disk

    def _compositeTransits(self, starRgba: np.ndarray, frame: int) -> np.ndarray:
        """Alpha-composite transiting planets in front of the photosphere."""
        canvas = starRgba.copy()
        height, width = canvas.shape[:2]
        starRadiusPixels = width * 0.5 * self._diskFraction(TRAPPIST_1_CATALOG_NAME)
        timeDays = float(self.timeByFrame[frame])
        for planet in self.planets:
            offset = planet.phaseOffset(timeDays)
            if abs(offset) > 1.0:
                continue
            disk = self._planetSilhouette(
                planet, frame, 2.0 * planet.radiusRatio * starRadiusPixels
            )
            # Chord across the disk: contact to contact in stellar radii.
            x = offset * planet.chordHalfSpan()
            centerX = int(round(width * 0.5 + x * starRadiusPixels))
            centerY = int(round(height * 0.5 - planet.impactParameter * starRadiusPixels))
            diskHeight, diskWidth = disk.shape[:2]
            x0, y0 = centerX - diskWidth // 2, centerY - diskHeight // 2
            xs0, ys0 = max(0, x0), max(0, y0)
            xs1, ys1 = min(width, x0 + diskWidth), min(height, y0 + diskHeight)
            if xs1 <= xs0 or ys1 <= ys0:
                continue
            region = canvas[ys0:ys1, xs0:xs1]
            patch = disk[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0]
            # Only occult where the photosphere is present (star alpha).
            alpha = patch[..., 3:4] * region[..., 3:4]
            region[..., :3] = patch[..., :3] * alpha + region[..., :3] * (1.0 - alpha)
            region[..., 3:4] = np.clip(
                region[..., 3:4] + alpha * (1.0 - region[..., 3:4]), 0.0, 1.0
            )
        return canvas

    def _caption(self, frame: int) -> str:
        active = self.transitingNow(frame)
        if not active:
            return 'Quiet baseline · relative flux from the model transit train'
        timeDays = float(self.timeByFrame[frame])
        if len(active) == 1:
            planet = active[0]
            index = planet.midTimesDays.index(planet.nearestMidTime(timeDays)) + 1
            return (
                f'{planet.name} in transit · depth {planet.depth * 100.0:.2f}% · '
                f'P = {planet.periodDays:.2f} d · transit {index} of {len(planet.midTimesDays)}'
            )
        names = ' + '.join(planet.shortName for planet in active)
        depth = sum(planet.depth for planet in active) * 100.0
        return f'TRAPPIST-1 {names} transiting together · combined depth {depth:.2f}%'

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

        star = self.atlas.bodyFrame(
            TRAPPIST_1_CATALOG_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION
        )
        if star is None:
            self.starAxes.add_patch(plt.Circle((0, 0), STAR_DISK_RADIUS, color='#FF6B4A'))
        else:
            extent = [-STAR_DISK_RADIUS, STAR_DISK_RADIUS, -STAR_DISK_RADIUS, STAR_DISK_RADIUS]
            self.starAxes.imshow(
                self._compositeTransits(star, frame),
                extent=extent,
                origin='upper',
                interpolation='bilinear',
                zorder=3,
            )

        timeDays = float(self.timeByFrame[frame])
        flux = float(modelFlux(self.planets, np.array([timeDays]))[0])
        self.starAxes.set_title(
            'TRAPPIST-1 — transit cinema',
            color=self.labelColor,
            fontsize=14,
            pad=10,
        )
        self.starAxes.text(
            0.5,
            0.015,
            self._caption(frame),
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
            'M8V · seven transiting worlds',
            transform=self.starAxes.transAxes,
            color=self.labelColor,
            fontsize=8,
            ha='right',
            va='top',
            alpha=0.55,
        )

        self.lcAxes.plot(
            self.modelTimeDays,
            self.modelFluxSeries,
            color=self.curveColor,
            linewidth=1.15,
            zorder=2,
        )
        self.lcAxes.axvline(timeDays, color=self.labelColor, linewidth=1.2, alpha=0.9, zorder=4)
        for planet in self.planets:
            for mid in planet.midTimesDays:
                self.lcAxes.axvline(
                    mid,
                    color=planet.color,
                    linewidth=0.8,
                    alpha=0.4,
                    zorder=3,
                )
        self.lcAxes.set_xlim(0.0, WINDOW_DAYS)
        self.lcAxes.set_ylim(0.9835, 1.0015)
        self.lcAxes.set_ylabel('Relative flux', color=self.labelColor, fontsize=9)
        self.lcAxes.set_xlabel('Days', color=self.labelColor, fontsize=9)
        self.lcAxes.tick_params(colors=self.labelColor, labelsize=7)
        self.lcAxes.set_title(
            f'Playhead · day {timeDays:.2f} · flux {flux:.4f}',
            color=self.labelColor,
            fontsize=9,
            loc='left',
            pad=6,
        )
        self.lcAxes.text(
            1.0,
            -0.145,
            'Model: catalog periods + radii · illustrative epochs (not photometry)',
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
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    outputDirectory = Path('output/animate/trappist_1/cinematic')
    catalog = SystemCatalog(starsCsvPath=starsCsvPath)
    system = catalog.load(TRAPPIST_1_SYSTEM_ID)
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'trappist_1_transit_cinematic_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TransitCinematicAnimator(
            system,
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
            requireBlenderBody=True,
        )
        animator.saveGif(str(outputPath))
    print('TRAPPIST-1 transit cinema completed!')


__all__ = [
    'TransitCinematicAnimator',
    'TransitingPlanet',
    'buildTransitingPlanets',
    'modelFlux',
    'renderTransitCinematicAnimations',
    'transitWeight',
    'warpedTimeByFrame',
]
