"""Solar cycle cinema — the one star whose spots we can count (#102).

The measurement strand has spent three films on stars that are single pixels:
Tabby's Star (#73), TRAPPIST-1 b (#95) and KIC 7944142 (#169). Here the same
grammar turns on the Sun, where the photometry is replaced by something older
and more literal — someone looked at the disk and counted what was on it.

Two observed series drive the edit. SILSO's monthly sunspot number runs from
1749, and the Mandal+2020 composite gives the heliographic latitude, central
meridian distance and area of every group photographed since 1874. So the hero
is not decorated: each disk carries the groups actually recorded on that day, at
their measured positions and in proportion to their measured areas, projected
with the solar tilt for the date. The cycle length, its spread, the peak of the
featured cycle and the equatorward drift are all measured here from those CSVs.
"""

from __future__ import annotations

import calendar
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse

from animate.blender_body_sprites import BlenderBodySpriteAtlas, diskRadiusFraction

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

DEFAULT_SUNSPOT_NUMBER_CSV = 'data/silso_sunspot_number_monthly.csv'
DEFAULT_GROUP_CSV = 'data/sunspot_groups_carrington.csv'
SUN_CATALOG_NAME = 'Sun'

# Spörer's law as usually quoted, for the payoff to be checked against.
PUBLISHED_OPENING_LATITUDE_DEG = 25.0
PUBLISHED_CLOSING_LATITUDE_DEG = 8.0
PUBLISHED_MEAN_CYCLE_YEARS = 11.0

# The 13-month running mean is the conventional smoothing for cycle timing, and
# minima are separated by at least this many years so a double-peaked maximum
# cannot be mistaken for the end of a cycle.
SMOOTHING_MONTHS = 13
MINIMUM_SEPARATION_YEARS = 7.0

BUTTERFLY_LATITUDE_LIMIT_DEG = 45.0
# The share of a cycle averaged at each end to measure the equatorward drift.
DRIFT_PHASE_FRACTION = 0.2

# Groups are drawn at the equivalent-circle radius of their measured area, then
# widened together: a 300 msh group really is about a fiftieth of the disk
# across, which is invisible in a GIF. This is a radius factor, so it costs its
# square in area. Positions are never scaled.
SPOT_RADIUS_EXAGGERATION = 2.5
STAR_DISPLAY_RESOLUTION = 512
STAR_DISK_RADIUS = 0.78
STAR_PANEL_HALF_WIDTH = 0.50

ACT_BOUNDARIES = ((110, 'minimum'), (210, 'maximum'), (290, 'century'), (390, 'butterfly'))
# Each disk is a hard cut to another day's observation, so it has to stay up
# long enough to be read as a face rather than a flicker.
DISK_HOLD_FRAMES = 6
ZOOM_OUT_FRAMES = 70.0
OPEN_FRAMES = 70.0
PAYOFF_FRAMES = 50.0
# The count flattens first and the wings grow into the room it leaves, so the
# two never fight over the middle of the strip.
COLLAPSE_FRAMES = 45.0
SPREAD_DELAY_FRAMES = 25.0
SPREAD_FRAMES = 70.0


@dataclass(frozen=True)
class CycleSolution:
    """Everything the film measures, derived from the committed CSVs."""

    minimaYear: np.ndarray
    cycleCount: int
    meanCycleYears: float
    shortestCycleYears: float
    longestCycleYears: float
    featuredCycleNumber: int
    featuredStartYear: float
    featuredEndYear: float
    featuredPeakYear: float
    featuredPeakNumber: float
    openingLatitudeDeg: float
    closingLatitudeDeg: float


def loadSunspotNumbers(
    csvPath: str | Path = DEFAULT_SUNSPOT_NUMBER_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """SILSO monthly epochs (decimal years) and total sunspot numbers."""
    frame = pd.read_csv(csvPath, comment='#')
    return (
        np.asarray(frame['decimal_year'], dtype=float),
        np.asarray(frame['sunspot_number'], dtype=float),
    )


def loadSunspotGroups(csvPath: str | Path = DEFAULT_GROUP_CSV) -> pd.DataFrame:
    """Group positions and areas, one observed day per Carrington rotation.

    Spotless days are kept as rows with a blank latitude and zero area; the
    ``decimalYear`` column added here is what the strip and butterfly plot
    against.
    """
    frame = pd.read_csv(csvPath, comment='#')
    date = pd.to_datetime(frame['date'])
    frame['decimalYear'] = (
        date.dt.year + (date.dt.dayofyear - 0.5) / np.where(date.dt.is_leap_year, 366.0, 365.0)
    ).astype(float)
    return frame


def smoothMonthly(number: np.ndarray, window: int = SMOOTHING_MONTHS) -> np.ndarray:
    """Centred running mean, the conventional way to time a cycle."""
    kernel = np.ones(window) / window
    padded = np.pad(number, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def findCycleMinima(year: np.ndarray, smoothed: np.ndarray) -> np.ndarray:
    """Epochs of solar minimum: the lowest point of each long quiet stretch."""
    half = int(round(MINIMUM_SEPARATION_YEARS * 12.0 / 2.0))
    minima: list[int] = []
    for index in range(half, len(smoothed) - half):
        window = smoothed[index - half : index + half + 1]
        if smoothed[index] > window.min():
            continue
        if minima and year[index] - year[minima[-1]] <= MINIMUM_SEPARATION_YEARS:
            # A flat minimum can satisfy the test for several months running;
            # keep the deepest of them rather than opening a new cycle.
            if smoothed[index] < smoothed[minima[-1]]:
                minima[-1] = index
            continue
        minima.append(index)
    return year[np.asarray(minima, dtype=int)]


def cycleLatitudeDrift(
    minimaYear: np.ndarray,
    groupYear: np.ndarray,
    groupLatitudeDeg: np.ndarray,
) -> tuple[float, float]:
    """Mean |latitude| in the first and last fifth of a cycle, over all cycles.

    Groups are placed by the phase of the cycle they fall in, so every cycle
    since 1874 contributes to the same average — this is Spörer's law measured
    rather than asserted.
    """
    opening: list[np.ndarray] = []
    closing: list[np.ndarray] = []
    for start, end in zip(minimaYear[:-1], minimaYear[1:], strict=True):
        inside = (groupYear >= start) & (groupYear < end)
        if not inside.any():
            continue
        phase = (groupYear[inside] - start) / (end - start)
        latitude = np.abs(groupLatitudeDeg[inside])
        opening.append(latitude[phase < DRIFT_PHASE_FRACTION])
        closing.append(latitude[phase > 1.0 - DRIFT_PHASE_FRACTION])
    return (
        float(np.mean(np.concatenate(opening))),
        float(np.mean(np.concatenate(closing))),
    )


def diskPositions(
    latitudeDeg: np.ndarray, meridianDistanceDeg: np.ndarray, tiltDeg: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Heliographic position → apparent disk position, in solar radii.

    ``tiltDeg`` is B0, the latitude of the disk centre, which swings +-7.25 deg
    over a year and is why the same group sits at different heights in spring
    and autumn. Returned with the foreshortening factor mu, so a group near the
    limb can be squashed the way it is actually seen.
    """
    latitude = np.radians(np.asarray(latitudeDeg, dtype=float))
    longitude = np.radians(np.asarray(meridianDistanceDeg, dtype=float))
    tilt = np.radians(float(tiltDeg))
    x = np.cos(latitude) * np.sin(longitude)
    y = np.sin(latitude) * np.cos(tilt) - np.cos(latitude) * np.sin(tilt) * np.cos(longitude)
    mu = np.sqrt(np.clip(1.0 - (x**2 + y**2), 0.0, 1.0))
    return x, y, mu


def spotRadius(areaMillionths: np.ndarray) -> np.ndarray:
    """Radius in solar radii of a circle with the group's measured area.

    Areas are millionths of the visible hemisphere, so an area A covers
    2*pi*R^2*A/1e6 and the equivalent circle has radius R*sqrt(2A/1e6).
    """
    return np.sqrt(2.0 * np.asarray(areaMillionths, dtype=float) / 1e6)


def solveSolarCycle(
    year: np.ndarray,
    number: np.ndarray,
    groups: pd.DataFrame,
) -> CycleSolution:
    """Time the cycles, pick the one the disk will follow, measure the drift."""
    smoothed = smoothMonthly(number)
    minima = findCycleMinima(year, smoothed)
    lengths = np.diff(minima)

    # Cycle 1 conventionally starts at the 1755 minimum, which is the first one
    # SILSO's record is long enough to resolve.
    spots = groups[groups['area_millionths'] > 0]
    lastGroupYear = float(spots['decimalYear'].max())
    covered = np.flatnonzero(minima[1:] <= lastGroupYear)
    featuredIndex = int(covered[-1])
    featuredStart = float(minima[featuredIndex])
    featuredEnd = float(minima[featuredIndex + 1])

    inCycle = (year >= featuredStart) & (year < featuredEnd)
    peakIndex = int(np.argmax(np.where(inCycle, number, -np.inf)))

    opening, closing = cycleLatitudeDrift(
        minima,
        np.asarray(spots['decimalYear'], dtype=float),
        np.asarray(spots['latitude_deg'], dtype=float),
    )
    return CycleSolution(
        minimaYear=minima,
        cycleCount=int(len(lengths)),
        meanCycleYears=float(np.mean(lengths)),
        shortestCycleYears=float(np.min(lengths)),
        longestCycleYears=float(np.max(lengths)),
        featuredCycleNumber=featuredIndex + 1,
        featuredStartYear=featuredStart,
        featuredEndYear=featuredEnd,
        featuredPeakYear=float(year[peakIndex]),
        featuredPeakNumber=float(number[peakIndex]),
        openingLatitudeDeg=opening,
        closingLatitudeDeg=closing,
    )


def monthLabel(decimalYear: float) -> str:
    """SILSO's mid-month epoch as a month and year, e.g. 2000.542 → July 2000."""
    year = int(np.floor(decimalYear))
    month = int(np.clip(np.floor((decimalYear - year) * 12.0) + 1, 1, 12))
    return f'{calendar.month_name[month]} {year}'


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


class SolarCycleCinematicAnimator:
    """Quiet disk → crowded disk → every cycle → the butterfly → Spörer's law."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        sunspotNumberCsvPath: str | Path = DEFAULT_SUNSPOT_NUMBER_CSV,
        groupCsvPath: str | Path = DEFAULT_GROUP_CSV,
        *,
        requireBlenderBody: bool = True,
    ):
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES

        self.year, self.number = loadSunspotNumbers(sunspotNumberCsvPath)
        self.groups = loadSunspotGroups(groupCsvPath)
        self.solution = solveSolarCycle(self.year, self.number, self.groups)

        spots = self.groups[self.groups['area_millionths'] > 0]
        self.butterflyYear = np.asarray(spots['decimalYear'], dtype=float)
        self.butterflyLatitude = np.asarray(spots['latitude_deg'], dtype=float)
        self.butterflyArea = np.asarray(spots['area_millionths'], dtype=float)
        self.butterflySpan = (float(self.butterflyYear.min()), float(self.butterflyYear.max()))
        self.recordSpan = (float(self.year.min()), float(self.year.max()))

        # The disk follows one cycle end to end, through the observed days the
        # catalogue has inside it.
        inCycle = self.groups['decimalYear'].between(
            self.solution.featuredStartYear, self.solution.featuredEndYear, inclusive='left'
        )
        self.diskDays = [
            day for _, day in self.groups[inCycle].groupby('carrington_rotation', sort=True)
        ]
        if not self.diskDays:
            raise ValueError('No observed rotations inside the featured cycle')

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
        self.sunAxes = self.figure.add_subplot(grid[0])
        self.stripAxes = self.figure.add_subplot(grid[1])

        self.atlas = BlenderBodySpriteAtlas(self.theme)
        if requireBlenderBody and not self.atlas.hasBody(SUN_CATALOG_NAME):
            raise FileNotFoundError(
                'Missing Blender spin for the Sun. Run:\n'
                '  render.py blender --body "Sun" --spin --theme all'
            )
        self._diskFraction: float | None = None

    # ---- act bookkeeping -------------------------------------------------

    def act(self, frame: int) -> str:
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'payoff'

    def zoomProgress(self, frame: int) -> float:
        """0 while the strip holds one cycle, 1 once it holds the whole record."""
        return smoothStep((frame - ACT_BOUNDARIES[1][0]) / ZOOM_OUT_FRAMES)

    def openProgress(self, frame: int) -> float:
        """0 while the strip counts spots, 1 once it places them by latitude."""
        return smoothStep((frame - ACT_BOUNDARIES[2][0]) / OPEN_FRAMES)

    def countCollapse(self, frame: int) -> float:
        """The count flattening onto the equator line it was counting up from."""
        return smoothStep((frame - ACT_BOUNDARIES[2][0]) / COLLAPSE_FRAMES)

    def wingSpread(self, frame: int) -> float:
        """The wings opening, once the count is most of the way down."""
        start = ACT_BOUNDARIES[2][0] + SPREAD_DELAY_FRAMES
        return smoothStep((frame - start) / SPREAD_FRAMES)

    def payoffProgress(self, frame: int) -> float:
        return smoothStep((frame - ACT_BOUNDARIES[3][0]) / PAYOFF_FRAMES)

    def diskIndex(self, frame: int) -> int:
        """Which observed rotation the hero disk is showing.

        The film has room for fewer cuts than the cycle has rotations, so it
        walks the observed days evenly and holds each one for several frames.
        """
        slots = max((self.animationFrames - 1) // DISK_HOLD_FRAMES, 1)
        slot = min(frame // DISK_HOLD_FRAMES, slots)
        position = slot / slots * (len(self.diskDays) - 1)
        return int(np.clip(round(position), 0, len(self.diskDays) - 1))

    def diskDay(self, frame: int) -> pd.DataFrame:
        return self.diskDays[self.diskIndex(frame)]

    def diskYear(self, frame: int) -> float:
        return float(self.diskDay(frame)['decimalYear'].iloc[0])

    def stripView(self, frame: int) -> tuple[float, float]:
        """The strip's time window: one cycle → the whole record → the butterfly."""
        cycle = (self.solution.featuredStartYear - 0.4, self.solution.featuredEndYear + 0.4)
        record = (self.recordSpan[0] - 2.0, self.recordSpan[1] + 2.0)
        butterfly = (self.butterflySpan[0] - 2.0, self.butterflySpan[1] + 2.0)
        zoom, open_ = self.zoomProgress(frame), self.openProgress(frame)
        low = cycle[0] + (record[0] - cycle[0]) * zoom
        high = cycle[1] + (record[1] - cycle[1]) * zoom
        return low + (butterfly[0] - low) * open_, high + (butterfly[1] - high) * open_

    def countScale(self, frame: int) -> float:
        """Sunspot number that reaches the top of the strip at this frame."""
        low, high = self.stripView(frame)
        window = (self.year >= low) & (self.year <= high)
        return float(self.number[window].max()) if window.any() else float(self.number.max())

    # ---- sun panel -------------------------------------------------------

    def diskFraction(self) -> float:
        """Visible fraction of the sprite's half-width, so spots land on the disk."""
        if self._diskFraction is None:
            sprite = self.atlas.bodyFrame(SUN_CATALOG_NAME, 0, resolution=STAR_DISPLAY_RESOLUTION)
            self._diskFraction = 1.0 if sprite is None else diskRadiusFraction(sprite)
        return self._diskFraction

    def visibleSpots(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Disk coordinates, foreshortening and radius of this day's groups."""
        day = self.diskDay(frame)
        spots = day[day['area_millionths'] > 0]
        if spots.empty:
            empty = np.zeros(0)
            return empty, empty, empty, empty
        x, y, mu = diskPositions(
            np.asarray(spots['latitude_deg'], dtype=float),
            np.asarray(spots['meridian_distance_deg'], dtype=float),
            float(day['tilt_b0_deg'].iloc[0]),
        )
        radius = spotRadius(np.asarray(spots['area_millionths'], dtype=float))
        onDisk = np.hypot(x, y) < 0.995
        return x[onDisk], y[onDisk], mu[onDisk], radius[onDisk]

    def _drawSun(self, frame: int) -> None:
        half = STAR_PANEL_HALF_WIDTH
        self.sunAxes.set_xlim(-half, half)
        self.sunAxes.set_ylim(-half, half)
        self.sunAxes.set_aspect('equal')
        self.sunAxes.axis('off')

        # The sprite is held on one frame: the spots are the only thing allowed
        # to move, and their longitudes are measured from this same face.
        sprite = self.atlas.bodyFrame(SUN_CATALOG_NAME, 0, resolution=STAR_DISPLAY_RESOLUTION)
        diskRadius = STAR_DISK_RADIUS * self.diskFraction()
        if sprite is None:
            self.sunAxes.add_patch(plt.Circle((0, 0), diskRadius, color='#F6C14A', zorder=3))
        else:
            extent = [-STAR_DISK_RADIUS, STAR_DISK_RADIUS, -STAR_DISK_RADIUS, STAR_DISK_RADIUS]
            self.sunAxes.imshow(
                sprite, extent=extent, origin='upper', interpolation='bilinear', zorder=3
            )

        x, y, mu, radius = self.visibleSpots(frame)
        limb = plt.Circle((0.0, 0.0), diskRadius, transform=self.sunAxes.transData)
        for spotX, spotY, spotMu, spotRadius_ in zip(x, y, mu, radius, strict=True):
            scale = diskRadius * SPOT_RADIUS_EXAGGERATION * spotRadius_
            angle = np.degrees(np.arctan2(spotY, spotX))
            # Penumbra then umbra, squashed along the radius by the same
            # foreshortening that flattens a spot near the limb.
            for share, color, alpha in ((1.0, '#3A1E0C', 0.8), (0.55, '#0B0704', 0.95)):
                spot = Ellipse(
                    (spotX * diskRadius, spotY * diskRadius),
                    width=2.0 * share * scale * spotMu,
                    height=2.0 * share * scale,
                    angle=angle,
                    facecolor=color,
                    edgecolor='none',
                    alpha=alpha,
                    zorder=4,
                )
                self.sunAxes.add_patch(spot)
                # A group caught at the limb is half over the edge; the disk
                # cuts it off rather than letting it hang in space.
                spot.set_clip_path(limb)

        self.sunAxes.set_title(
            'The Sun — the only star whose spots we can count',
            color=self.labelColor,
            fontsize=14,
            pad=10,
        )
        day = self.diskDay(frame)
        self.sunAxes.text(
            0.015,
            0.965,
            f'{day["date"].iloc[0]} · CR {int(day["carrington_rotation"].iloc[0])} · '
            f'{len(x)} groups',
            transform=self.sunAxes.transAxes,
            color=self.labelColor,
            fontsize=9,
            va='top',
            alpha=0.75,
        )
        self.sunAxes.text(
            0.985,
            0.965,
            f'{day["observatory"].iloc[0]} · positions as measured · '
            f'spots widened ×{SPOT_RADIUS_EXAGGERATION:g}',
            transform=self.sunAxes.transAxes,
            color=self.labelColor,
            fontsize=8,
            ha='right',
            va='top',
            alpha=0.55,
        )
        self.sunAxes.text(
            0.5,
            0.015,
            self.caption(frame),
            transform=self.sunAxes.transAxes,
            color=self.labelColor,
            fontsize=9,
            ha='center',
            alpha=0.85,
            zorder=6,
        )

    # ---- captions --------------------------------------------------------

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        solution = self.solution
        if act == 'minimum':
            return (
                f'Cycle {solution.featuredCycleNumber} opens at minimum · every disk is a day the '
                'Sun was photographed · the measurement is just: how many groups?'
            )
        if act == 'maximum':
            elapsed = self.diskYear(frame) - solution.featuredStartYear
            return (
                f'The same face, {elapsed:.1f} years on · the monthly count peaks at '
                f'{solution.featuredPeakNumber:.0f} in {monthLabel(solution.featuredPeakYear)}'
            )
        if act == 'century':
            return (
                f'{solution.cycleCount} cycles since 1749 · mean length '
                f'{solution.meanCycleYears:.1f} yr (accepted {PUBLISHED_MEAN_CYCLE_YEARS:.0f}), '
                f'but they run {solution.shortestCycleYears:.1f} to '
                f'{solution.longestCycleYears:.1f}'
            )
        if act == 'butterfly':
            return (
                'The count says how many, never where · every group photographed since '
                f'{self.butterflySpan[0]:.0f}, placed by latitude'
            )
        return (
            f"Spörer's law · cycles open at ±{solution.openingLatitudeDeg:.0f}° and close at "
            f'±{solution.closingLatitudeDeg:.0f}° (quoted ±{PUBLISHED_OPENING_LATITUDE_DEG:.0f} '
            f'to ±{PUBLISHED_CLOSING_LATITUDE_DEG:.0f}) · the clock is magnetic, not counted'
        )

    # ---- strip -----------------------------------------------------------

    def countPositions(self, frame: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Strip coordinates for the sunspot number, and how visible it is.

        The curve collapses onto the equator line as the butterfly opens, which
        is the same y = 0 it was counting up from.
        """
        low, high = self.stripView(frame)
        x = (self.year - low) / (high - low)
        y = self.number / self.countScale(frame) * (1.0 - self.countCollapse(frame))
        visible = (x >= 0.0) & (x <= 1.0)
        # Until the strip zooms out, only the past is drawn: the film is walking
        # through the cycle, not presenting it finished.
        arrived = self.year <= self.diskYear(frame)
        alpha = np.where(arrived | (self.zoomProgress(frame) > 0.0), 1.0, 0.0)
        return x, y, np.where(visible, alpha, 0.0)

    def butterflyPositions(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        low, high = self.stripView(frame)
        x = (self.butterflyYear - low) / (high - low)
        y = self.butterflyLatitude / BUTTERFLY_LATITUDE_LIMIT_DEG * self.wingSpread(frame)
        return x, y

    def _drawStrip(self, frame: int) -> None:
        collapse, spread = self.countCollapse(frame), self.wingSpread(frame)
        self.stripAxes.set_xlim(0.0, 1.0)
        self.stripAxes.set_ylim(-0.08 - 1.0 * spread, 1.08)
        self.stripAxes.axhline(0.0, color=self.labelColor, linewidth=0.8, alpha=0.3, zorder=1)

        x, y, alpha = self.countPositions(frame)
        drawn = alpha > 0.0
        if drawn.any():
            # The filled area helps while one cycle is on screen and turns into
            # a solid floor once 24 of them are, so it thins out as we pull back.
            self.stripAxes.fill_between(
                x[drawn],
                0.0,
                y[drawn],
                color=self.pointColor,
                alpha=0.25 * (1.0 - 0.6 * self.zoomProgress(frame)) * (1.0 - collapse),
                linewidth=0.0,
                zorder=2,
            )
            self.stripAxes.plot(
                x[drawn],
                y[drawn],
                '-',
                color=self.pointColor,
                linewidth=0.9,
                alpha=1.0 - collapse,
                zorder=3,
            )

        if spread > 0.0:
            butterflyX, butterflyY = self.butterflyPositions(frame)
            inView = (butterflyX >= 0.0) & (butterflyX <= 1.0)
            self.stripAxes.scatter(
                butterflyX[inView],
                butterflyY[inView],
                s=np.sqrt(self.butterflyArea[inView]) * 0.5,
                color=self.accentColor,
                alpha=0.5 * spread,
                linewidths=0.0,
                zorder=4,
            )
            if self.payoffProgress(frame) > 0.0:
                self._drawDriftBands(frame)

        self._markFeaturedCycle(frame)
        self._drawPlayhead(frame)
        self._labelStrip(frame)

    def _markFeaturedCycle(self, frame: int) -> None:
        """Show where the cycle on screen sits once the whole record is in view."""
        reveal = self.zoomProgress(frame) * (1.0 - self.wingSpread(frame))
        if reveal <= 0.0:
            return
        low, high = self.stripView(frame)
        start = (self.solution.featuredStartYear - low) / (high - low)
        end = (self.solution.featuredEndYear - low) / (high - low)
        self.stripAxes.axvspan(start, end, color=self.accentColor, alpha=0.16 * reveal, zorder=1)
        self.stripAxes.text(
            (start + end) / 2.0,
            1.02,
            f'cycle {self.solution.featuredCycleNumber}',
            transform=self.stripAxes.get_xaxis_transform(),
            color=self.accentColor,
            fontsize=8,
            ha='center',
            va='bottom',
            alpha=reveal,
            zorder=6,
        )

    def _drawDriftBands(self, frame: int) -> None:
        """The measured opening and closing latitudes, drawn on both wings."""
        reveal = self.payoffProgress(frame)
        for latitude, label in (
            (self.solution.openingLatitudeDeg, 'cycle opens'),
            (self.solution.closingLatitudeDeg, 'cycle closes'),
        ):
            for sign in (1.0, -1.0):
                y = sign * latitude / BUTTERFLY_LATITUDE_LIMIT_DEG
                self.stripAxes.axhline(
                    y,
                    color=self.labelColor,
                    linestyle='--',
                    linewidth=1.0,
                    alpha=0.55 * reveal,
                    zorder=5,
                )
            self.stripAxes.text(
                0.004,
                latitude / BUTTERFLY_LATITUDE_LIMIT_DEG + 0.03,
                f'{label} ±{latitude:.0f}°',
                transform=self.stripAxes.get_yaxis_transform(which='grid'),
                color=self.labelColor,
                fontsize=7,
                alpha=0.9 * reveal,
                zorder=6,
            )

    def _drawPlayhead(self, frame: int) -> None:
        low, high = self.stripView(frame)
        position = (self.diskYear(frame) - low) / (high - low)
        if 0.0 <= position <= 1.0:
            self.stripAxes.axvline(
                position,
                color=self.accentColor,
                linewidth=1.0,
                alpha=0.75 - 0.35 * self.wingSpread(frame),
                zorder=6,
            )

    def _labelStrip(self, frame: int) -> None:
        collapse, spread = self.countCollapse(frame), self.wingSpread(frame)
        low, high = self.stripView(frame)
        ticks = np.linspace(0.0, 1.0, 6)
        self.stripAxes.set_xticks(ticks)
        self.stripAxes.set_xticklabels([f'{low + t * (high - low):.0f}' for t in ticks])
        self.stripAxes.set_xlabel('Year', color=self.labelColor, fontsize=9)

        if spread < 0.35:
            scale = self.countScale(frame)
            values = np.array([0.0, 100.0, 200.0, 300.0])
            values = values[values <= scale]
            self.stripAxes.set_yticks(values / scale * (1.0 - collapse))
            self.stripAxes.set_yticklabels([f'{value:.0f}' for value in values])
            self.stripAxes.set_ylabel(
                'Monthly sunspot number',
                color=self.labelColor,
                fontsize=9,
                alpha=float(np.clip(1.0 - 2.0 * collapse, 0.0, 1.0)),
            )
        else:
            values = np.array([-30.0, -15.0, 0.0, 15.0, 30.0])
            self.stripAxes.set_yticks(values / BUTTERFLY_LATITUDE_LIMIT_DEG * spread)
            self.stripAxes.set_yticklabels([f'{value:+.0f}°' for value in values])
            self.stripAxes.set_ylabel(
                'Heliographic latitude',
                color=self.labelColor,
                fontsize=9,
                alpha=float(np.clip(1.6 * spread - 0.5, 0.0, 1.0)),
            )

        self.stripAxes.tick_params(colors=self.labelColor, labelsize=7)
        self.stripAxes.text(
            1.0,
            -0.145,
            'SILSO v2.0 monthly numbers · Mandal+2020 group positions · cycle length, peak and '
            'latitude drift derived from these CSVs',
            transform=self.stripAxes.transAxes,
            color=self.labelColor,
            fontsize=7,
            ha='right',
            alpha=0.6,
        )
        for spine in self.stripAxes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)

    def update(self, frame: int):
        self.sunAxes.clear()
        self.stripAxes.clear()
        for axes in (self.sunAxes, self.stripAxes):
            axes.set_facecolor(self.panelFace)
            for spine in axes.spines.values():
                spine.set_visible(False)
        self._drawSun(frame)
        self._drawStrip(frame)
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


def renderSolarCycleCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    sunspotNumberCsvPath: str | Path = DEFAULT_SUNSPOT_NUMBER_CSV,
    groupCsvPath: str | Path = DEFAULT_GROUP_CSV,
) -> None:
    outputDirectory = Path('output/animate/sol/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'sol_solar_cycle_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = SolarCycleCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            sunspotNumberCsvPath=sunspotNumberCsvPath,
            groupCsvPath=groupCsvPath,
            requireBlenderBody=True,
        )
        animator.saveGif(str(outputPath))
    print('Solar cycle cinema completed!')


__all__ = [
    'CycleSolution',
    'SolarCycleCinematicAnimator',
    'cycleLatitudeDrift',
    'diskPositions',
    'findCycleMinima',
    'loadSunspotGroups',
    'loadSunspotNumbers',
    'renderSolarCycleCinematicAnimations',
    'smoothMonthly',
    'solveSolarCycle',
    'spotRadius',
]
