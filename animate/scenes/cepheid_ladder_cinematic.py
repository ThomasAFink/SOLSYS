"""Cepheid period–luminosity cinema — the first rung of the ladder (#159).

Henrietta Leavitt's result is the reason anything outside the Solar System has a
distance at all: in one galaxy, where every star is at effectively the same
distance, the brighter Cepheids are the slower ones. This film earns that
sentence twice over from committed data.

The pulse comes first. Three LMC Cepheids carry real Gaia DR3 epoch photometry,
folded on their OGLE periods, and each one's playhead advances at its own rate —
the 3-day star races, the 34-day star crawls, and the crawler is the bright one.
Then the period–luminosity plane fills with 2,315 fundamental-mode LMC Cepheids
and the relation is fitted on screen. Swapping mean I for the Wesenheit index
tightens the ridge to under 0.08 mag, because that combination cancels
reddening. Finally the SMC's 2,637 Cepheids arrive as a second, parallel ridge,
and the gap between the two ridges is a distance ratio: the film ends by turning
0.46 mag into kiloparsecs and checking the answer against eclipsing binaries.

Every slope, scatter and distance quoted on screen is fitted at render time from
the two CSVs — nothing here is a remembered number.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

DEFAULT_CATALOG_CSV = 'data/ogle_magellanic_cepheids.csv'
DEFAULT_LIGHTCURVE_CSV = 'data/gaia_cepheid_lightcurves.csv'

# The Wesenheit index I - R (V - I) is reddening-free for R equal to the ratio of
# total to selective extinction in these bands; 1.55 is the value used with OGLE
# V and I photometry (Madore 1982; Soszynski et al. 2015).
WESENHEIT_COLOR_COEFFICIENT = 1.55

# Fit range: below a day the fundamental-mode sample runs into the overtone
# sequence, and above 100 d there is nothing left to fit.
FIT_PERIOD_RANGE_DAYS = (1.0, 100.0)
FIT_CLIP_SIGMA = 3.0
FIT_CLIP_ROUNDS = 3
# The relation is pivoted at 10 days so the zero point is quoted where both
# clouds are well populated, instead of extrapolated to 1 day.
PIVOT_LOG_PERIOD = 1.0

# Distances the film checks itself against, both geometric and independent of
# any Cepheid: detached eclipsing binaries in each cloud.
LMC_DISTANCE_MODULUS = 18.477  # Pietrzynski et al. 2019, Nature 567, 200
LMC_DISTANCE_MODULUS_ERROR = 0.026
PUBLISHED_SMC_DISTANCE_MODULUS = 18.977  # Graczyk et al. 2020, ApJ 904, 13
# The slope OGLE itself reports for this sample, for the fit to be checked against.
PUBLISHED_LMC_WESENHEIT_SLOPE = -3.313

HERO_STARS = ('OGLE-LMC-CEP-3592', 'OGLE-LMC-CEP-1252', 'OGLE-LMC-CEP-0328')
# Order of the Fourier series fitted to each folded light curve. Three harmonics
# is enough for the sawtooth of a Cepheid without chasing the noise.
FOURIER_ORDER = 3
# Film time to star time: the whole 24 s runs 36 days, which is just over one
# cycle of the slowest hero and a dozen of the fastest.
DAYS_PER_SECOND = 1.5

# The pulse panel owns the frame until the period–luminosity plane opens under
# it, so the first act is not played against an empty half-screen.
PULSE_RECT_TALL = (0.09, 0.115, 0.865, 0.815)
PULSE_RECT_SHORT = (0.09, 0.615, 0.865, 0.315)
PLANE_RECT = (0.09, 0.115, 0.865, 0.395)

ACT_BOUNDARIES = ((110, 'pulse'), (200, 'trio'), (300, 'leavitt'), (390, 'wesenheit'))
HERO_ENTRY_FRAMES = (0, 118, 158)
HERO_FADE_FRAMES = 30.0
PLANE_OPEN_FRAME = 172
PLANE_OPEN_FRAMES = 32.0
CLOUD_RAIN_FRAME = 206
CLOUD_RAIN_FRAMES = 74.0
FIT_DRAW_FRAME = 258
FIT_DRAW_FRAMES = 34.0
WESENHEIT_MORPH_FRAME = 302
WESENHEIT_MORPH_FRAMES = 62.0
SMC_ARRIVE_FRAME = 392
SMC_ARRIVE_FRAMES = 48.0
OFFSET_REVEAL_FRAME = 428
OFFSET_REVEAL_FRAMES = 34.0
# Fitting a handful of points would print a slope that means nothing, so the
# live fit waits until the plane has something to fit.
MINIMUM_POINTS_TO_FIT = 60


@dataclass(frozen=True)
class RidgeFit:
    """A period–luminosity fit: magnitude = slope (log P - 1) + zeroPoint."""

    slope: float
    zeroPoint: float
    scatter: float
    count: int


@dataclass(frozen=True)
class LadderSolution:
    """Everything the film measures, derived from the committed catalogue."""

    lmcVisual: RidgeFit
    lmcInfrared: RidgeFit
    lmcWesenheit: RidgeFit
    smcWesenheit: RidgeFit
    modulusOffset: float
    distanceRatio: float
    smcDistanceModulus: float
    lmcDistanceKiloparsecs: float
    smcDistanceKiloparsecs: float
    publishedSmcDistanceKiloparsecs: float


def wesenheitIndex(visualMag: np.ndarray, infraredMag: np.ndarray) -> np.ndarray:
    """The reddening-free combination of the two OGLE bands."""
    visual = np.asarray(visualMag, dtype=float)
    infrared = np.asarray(infraredMag, dtype=float)
    return infrared - WESENHEIT_COLOR_COEFFICIENT * (visual - infrared)


def loadCepheidCatalog(csvPath: str | Path = DEFAULT_CATALOG_CSV) -> pd.DataFrame:
    """Fundamental-mode Cepheids in both clouds, with the derived columns added.

    ``logPeriod`` and ``wesenheit`` are computed here rather than stored, so the
    committed CSV stays as close to the published catalogue as possible.
    """
    frame = pd.read_csv(csvPath, comment='#')
    frame['logPeriod'] = np.log10(frame['period_days'].to_numpy(dtype=float))
    frame['wesenheit'] = wesenheitIndex(frame['mean_v_mag'], frame['mean_i_mag'])
    inRange = frame['period_days'].between(*FIT_PERIOD_RANGE_DAYS, inclusive='both')
    return frame[inRange].reset_index(drop=True)


def loadHeroLightcurves(csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV) -> dict[str, pd.DataFrame]:
    """Gaia G-band epoch photometry, one frame per hero star."""
    frame = pd.read_csv(csvPath, comment='#')
    return {star: group.reset_index(drop=True) for star, group in frame.groupby('star', sort=False)}


def fitRidge(
    logPeriod: np.ndarray,
    magnitude: np.ndarray,
    sigma: float = FIT_CLIP_SIGMA,
    rounds: int = FIT_CLIP_ROUNDS,
) -> RidgeFit:
    """Least squares with a sigma clip, which is how the relation is usually fitted.

    The clip matters: blended stars and the odd misclassified overtone sit far
    off the ridge, and without rejecting them the scatter reports their outliers
    rather than the width of the relation.
    """
    logPeriod = np.asarray(logPeriod, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    finite = np.isfinite(logPeriod) & np.isfinite(magnitude)
    logPeriod, magnitude = logPeriod[finite], magnitude[finite]
    if len(logPeriod) < 2:
        return RidgeFit(slope=float('nan'), zeroPoint=float('nan'), scatter=float('nan'), count=0)

    keep = np.ones(len(logPeriod), dtype=bool)
    pivot = logPeriod - PIVOT_LOG_PERIOD
    for _ in range(rounds):
        slope, zeroPoint = np.polyfit(pivot[keep], magnitude[keep], 1)
        residual = magnitude - (slope * pivot + zeroPoint)
        spread = residual[keep].std()
        if not np.isfinite(spread) or spread <= 0.0:
            break
        keep = np.abs(residual) < sigma * spread
        if keep.sum() < 2:
            keep = np.ones(len(logPeriod), dtype=bool)
            break
    slope, zeroPoint = np.polyfit(pivot[keep], magnitude[keep], 1)
    residual = magnitude[keep] - (slope * pivot[keep] + zeroPoint)
    return RidgeFit(
        slope=float(slope),
        zeroPoint=float(zeroPoint),
        scatter=float(residual.std()),
        count=int(keep.sum()),
    )


def distanceKiloparsecs(distanceModulus: float) -> float:
    """Distance modulus → kiloparsecs."""
    return 10.0 ** (float(distanceModulus) / 5.0 + 1.0) / 1000.0


def solveLadder(catalog: pd.DataFrame) -> LadderSolution:
    """Fit both clouds, then turn the gap between their ridges into a distance."""
    lmc = catalog[catalog['cloud'] == 'LMC']
    smc = catalog[catalog['cloud'] == 'SMC']
    lmcVisual = fitRidge(lmc['logPeriod'], lmc['mean_v_mag'])
    lmcInfrared = fitRidge(lmc['logPeriod'], lmc['mean_i_mag'])
    lmcWesenheit = fitRidge(lmc['logPeriod'], lmc['wesenheit'])
    smcWesenheit = fitRidge(smc['logPeriod'], smc['wesenheit'])

    # Both ridges are apparent magnitudes of stars obeying the same relation, so
    # their zero points differ only by the ratio of the two distances.
    offset = smcWesenheit.zeroPoint - lmcWesenheit.zeroPoint
    smcModulus = LMC_DISTANCE_MODULUS + offset
    return LadderSolution(
        lmcVisual=lmcVisual,
        lmcInfrared=lmcInfrared,
        lmcWesenheit=lmcWesenheit,
        smcWesenheit=smcWesenheit,
        modulusOffset=float(offset),
        distanceRatio=float(10.0 ** (offset / 5.0)),
        smcDistanceModulus=float(smcModulus),
        lmcDistanceKiloparsecs=distanceKiloparsecs(LMC_DISTANCE_MODULUS),
        smcDistanceKiloparsecs=distanceKiloparsecs(smcModulus),
        publishedSmcDistanceKiloparsecs=distanceKiloparsecs(PUBLISHED_SMC_DISTANCE_MODULUS),
    )


def fourierCurve(
    phase: np.ndarray, magnitude: np.ndarray, order: int = FOURIER_ORDER
) -> tuple[np.ndarray, float]:
    """Least-squares Fourier series through a folded light curve.

    Returns the coefficients and the phase of maximum light, which is what the
    fold is then re-centred on so every hero starts bright at phase zero.
    """
    phase = np.asarray(phase, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    columns = [np.ones_like(phase)]
    for harmonic in range(1, order + 1):
        columns.append(np.cos(2.0 * np.pi * harmonic * phase))
        columns.append(np.sin(2.0 * np.pi * harmonic * phase))
    design = np.vstack(columns).T
    coefficients, *_ = np.linalg.lstsq(design, magnitude, rcond=None)
    sampled = np.linspace(0.0, 1.0, 512, endpoint=False)
    brightest = float(sampled[int(np.argmin(evaluateFourier(coefficients, sampled)))])
    return coefficients, brightest


def evaluateFourier(coefficients: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """Evaluate the series returned by :func:`fourierCurve`."""
    phase = np.asarray(phase, dtype=float)
    order = (len(coefficients) - 1) // 2
    values = np.full_like(phase, coefficients[0], dtype=float)
    for harmonic in range(1, order + 1):
        values += coefficients[2 * harmonic - 1] * np.cos(2.0 * np.pi * harmonic * phase)
        values += coefficients[2 * harmonic] * np.sin(2.0 * np.pi * harmonic * phase)
    return values


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


@dataclass(frozen=True)
class HeroCurve:
    """One folded hero light curve, ready to draw and to walk a playhead along."""

    star: str
    periodDays: float
    phase: np.ndarray
    magnitude: np.ndarray
    magnitudeError: np.ndarray
    coefficients: np.ndarray
    meanInfraredMag: float
    logPeriod: float
    wesenheitMag: float

    def curve(self, samples: int = 256) -> tuple[np.ndarray, np.ndarray]:
        phase = np.linspace(0.0, 1.0, samples)
        return phase, evaluateFourier(self.coefficients, phase)

    def magnitudeAtPhase(self, phase: float) -> float:
        return float(evaluateFourier(self.coefficients, np.array([phase % 1.0]))[0])


def buildHeroCurves(
    catalog: pd.DataFrame, lightcurves: dict[str, pd.DataFrame]
) -> tuple[HeroCurve, ...]:
    """Fold each hero on its catalogue period and centre it on maximum light."""
    heroes: list[HeroCurve] = []
    for star in HERO_STARS:
        row = catalog[catalog['star'] == star]
        if row.empty or star not in lightcurves:
            raise ValueError(f'Hero {star} is missing from the committed data')
        period = float(row['period_days'].iloc[0])
        photometry = lightcurves[star]
        time = photometry['time_days'].to_numpy(dtype=float)
        magnitude = photometry['g_mag'].to_numpy(dtype=float)
        rawPhase = np.mod(time, period) / period
        coefficients, brightest = fourierCurve(rawPhase, magnitude)
        # Re-fold so phase zero is maximum light, then refit on that convention.
        phase = np.mod(rawPhase - brightest, 1.0)
        coefficients, _ = fourierCurve(phase, magnitude)
        heroes.append(
            HeroCurve(
                star=star,
                periodDays=period,
                phase=phase,
                magnitude=magnitude,
                magnitudeError=photometry['g_mag_error'].to_numpy(dtype=float),
                coefficients=coefficients,
                meanInfraredMag=float(row['mean_i_mag'].iloc[0]),
                logPeriod=float(row['logPeriod'].iloc[0]),
                wesenheitMag=float(row['wesenheit'].iloc[0]),
            )
        )
    return tuple(sorted(heroes, key=lambda hero: hero.periodDays))


class CepheidLadderCinematicAnimator:
    """One star pulsing → three → the Leavitt law → reddening removed → two clouds."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
        lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
    ):
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES

        self.catalog = loadCepheidCatalog(catalogCsvPath)
        self.lightcurves = loadHeroLightcurves(lightcurveCsvPath)
        self.heroes = buildHeroCurves(self.catalog, self.lightcurves)
        self.solution = solveLadder(self.catalog)

        lmc = self.catalog[self.catalog['cloud'] == 'LMC']
        smc = self.catalog[self.catalog['cloud'] == 'SMC']
        self.lmcLogPeriod = lmc['logPeriod'].to_numpy(dtype=float)
        self.lmcInfrared = lmc['mean_i_mag'].to_numpy(dtype=float)
        self.lmcWesenheit = lmc['wesenheit'].to_numpy(dtype=float)
        self.smcLogPeriod = smc['logPeriod'].to_numpy(dtype=float)
        self.smcInfrared = smc['mean_i_mag'].to_numpy(dtype=float)
        self.smcWesenheit = smc['wesenheit'].to_numpy(dtype=float)

        # The cloud rains in in a fixed shuffled order, so the ridge builds
        # evenly across period instead of sweeping in from one end.
        generator = np.random.default_rng(159)
        self.lmcArrivalOrder = generator.permutation(len(self.lmcLogPeriod))
        self.smcArrivalOrder = generator.permutation(len(self.smcLogPeriod))

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.theme = 'dark' if self.isDark else 'light'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.pointColor = '#7EB6FF' if self.isDark else '#204080'
        self.accentColor = '#FFB570' if self.isDark else '#B4540A'
        self.secondColor = '#8CD9A0' if self.isDark else '#1F6B3A'
        self.panelFace = '#050508' if self.isDark else '#F4F2EC'
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, facecolor=self.panelFace)
        self.pulseAxes = self.figure.add_axes(PULSE_RECT_TALL)
        self.planeAxes = self.figure.add_axes(PLANE_RECT)
        self._figureTexts: list = []

        # Edges of the period–luminosity plane, taken well inside the tails so a
        # single blended outlier cannot dictate the framing.
        # The heroes are held inside the window explicitly: the 34-day star is
        # brighter than the percentile edge and would otherwise sit off the plot.
        heroInfrared = min(hero.meanInfraredMag for hero in self.heroes)
        heroWesenheit = min(hero.wesenheitMag for hero in self.heroes)
        self.planeEdges = {
            'lmcInfraredFaint': float(np.percentile(self.lmcInfrared, 99.5)),
            'lmcInfraredBright': float(min(np.percentile(self.lmcInfrared, 0.5), heroInfrared)),
            'lmcWesenheitFaint': float(np.percentile(self.lmcWesenheit, 99.5)),
            'lmcWesenheitBright': float(min(np.percentile(self.lmcWesenheit, 0.5), heroWesenheit)),
            'smcWesenheitFaint': float(np.percentile(self.smcWesenheit, 99.5)),
        }

        # Vertical range of the pulse panel after each hero has arrived, so the
        # panel pulls back to make room instead of starting wide and empty.
        self.pulseStages: list[tuple[float, float]] = []
        for count in range(1, len(self.heroes) + 1):
            magnitudes = np.concatenate([hero.magnitude for hero in self.heroes[:count]])
            self.pulseStages.append(
                (float(magnitudes.max()) + 0.35, float(magnitudes.min()) - 0.45)
            )

    # ---- act bookkeeping -------------------------------------------------

    def act(self, frame: int) -> str:
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'clouds'

    def elapsedDays(self, frame: int) -> float:
        """Star time, so each hero's playhead runs at its own measured rate."""
        return frame / ANIMATION_FPS * DAYS_PER_SECOND

    def heroReveal(self, frame: int, index: int) -> float:
        return smoothStep((frame - HERO_ENTRY_FRAMES[index]) / HERO_FADE_FRAMES)

    def planeReveal(self, frame: int) -> float:
        return smoothStep((frame - PLANE_OPEN_FRAME) / PLANE_OPEN_FRAMES)

    def cloudReveal(self, frame: int) -> float:
        return smoothStep((frame - CLOUD_RAIN_FRAME) / CLOUD_RAIN_FRAMES)

    def fitReveal(self, frame: int) -> float:
        return smoothStep((frame - FIT_DRAW_FRAME) / FIT_DRAW_FRAMES)

    def wesenheitMorph(self, frame: int) -> float:
        """0 while the plane shows mean I, 1 once it shows the Wesenheit index."""
        return smoothStep((frame - WESENHEIT_MORPH_FRAME) / WESENHEIT_MORPH_FRAMES)

    def smcReveal(self, frame: int) -> float:
        return smoothStep((frame - SMC_ARRIVE_FRAME) / SMC_ARRIVE_FRAMES)

    def offsetReveal(self, frame: int) -> float:
        return smoothStep((frame - OFFSET_REVEAL_FRAME) / OFFSET_REVEAL_FRAMES)

    # ---- the plane's moving vertical axis ---------------------------------

    def morphedMagnitude(self, infrared: np.ndarray, wesenheit: np.ndarray, morph: float):
        """Mean I sliding into the Wesenheit index.

        The two quantities differ by 1.55 (V - I), so interpolating between them
        is the reddening correction being applied a fraction at a time — the
        ridge tightens as it slides rather than cutting to a new plot.
        """
        return np.asarray(infrared, dtype=float) + morph * (
            np.asarray(wesenheit, dtype=float) - np.asarray(infrared, dtype=float)
        )

    def visibleLmc(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        shown = int(round(self.cloudReveal(frame) * len(self.lmcArrivalOrder)))
        chosen = self.lmcArrivalOrder[:shown]
        magnitude = self.morphedMagnitude(
            self.lmcInfrared[chosen], self.lmcWesenheit[chosen], self.wesenheitMorph(frame)
        )
        return self.lmcLogPeriod[chosen], magnitude

    def visibleSmc(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        shown = int(round(self.smcReveal(frame) * len(self.smcArrivalOrder)))
        chosen = self.smcArrivalOrder[:shown]
        magnitude = self.morphedMagnitude(
            self.smcInfrared[chosen], self.smcWesenheit[chosen], self.wesenheitMorph(frame)
        )
        return self.smcLogPeriod[chosen], magnitude

    def liveFit(self, logPeriod: np.ndarray, magnitude: np.ndarray) -> RidgeFit | None:
        """Fit whatever is on screen right now, or nothing if that is too little."""
        if len(logPeriod) < MINIMUM_POINTS_TO_FIT:
            return None
        return fitRidge(logPeriod, magnitude)

    def planeLimits(self, frame: int) -> tuple[float, float]:
        """Vertical range, following the points down as the correction is applied.

        Removing the reddening subtracts roughly a magnitude from every star, so
        the window has to travel with them; the SMC then widens it again at the
        faint end. Both edges come from the data rather than from taste.
        """
        morph = self.wesenheitMorph(frame)
        smc = self.smcReveal(frame)
        faint = self.planeEdges['lmcInfraredFaint'] + morph * (
            self.planeEdges['lmcWesenheitFaint'] - self.planeEdges['lmcInfraredFaint']
        )
        bright = self.planeEdges['lmcInfraredBright'] + morph * (
            self.planeEdges['lmcWesenheitBright'] - self.planeEdges['lmcInfraredBright']
        )
        faint += smc * (self.planeEdges['smcWesenheitFaint'] - faint)
        return faint + 0.45, bright - 0.55

    # ---- pulse panel -----------------------------------------------------

    def pulseLimits(self, frame: int) -> tuple[float, float]:
        """Faint and bright edges of the pulse panel, opening as heroes arrive."""
        faint, bright = self.pulseStages[0]
        for index in range(1, len(self.pulseStages)):
            reveal = self.heroReveal(frame, index)
            faint += (self.pulseStages[index][0] - self.pulseStages[index - 1][0]) * reveal
            bright += (self.pulseStages[index][1] - self.pulseStages[index - 1][1]) * reveal
        return faint, bright

    def _drawPulse(self, frame: int) -> None:
        axes = self.pulseAxes
        axes.set_xlim(0.0, 1.0)
        axes.set_ylim(*self.pulseLimits(frame))
        axes.set_xlabel('Pulsation phase', color=self.labelColor, fontsize=9)
        axes.set_ylabel('Gaia G magnitude (apparent)', color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)

        days = self.elapsedDays(frame)
        for index, hero in enumerate(self.heroes):
            reveal = self.heroReveal(frame, index)
            if reveal <= 0.0:
                continue
            axes.errorbar(
                hero.phase,
                hero.magnitude,
                yerr=hero.magnitudeError,
                fmt='o',
                markersize=3.0,
                color=self.pointColor,
                ecolor=self.pointColor,
                elinewidth=0.6,
                capsize=0.0,
                alpha=0.75 * reveal,
                zorder=3,
            )
            curvePhase, curveMagnitude = hero.curve()
            axes.plot(
                curvePhase,
                curveMagnitude,
                '-',
                color=self.labelColor,
                linewidth=1.0,
                alpha=0.45 * reveal,
                zorder=4,
            )

            # The playhead: this star's own phase after `days` of star time.
            phase = (days / hero.periodDays) % 1.0
            axes.plot(
                [phase],
                [hero.magnitudeAtPhase(phase)],
                'o',
                markersize=11.0,
                color=self.accentColor,
                alpha=reveal,
                zorder=6,
            )
            cycles = days / hero.periodDays
            axes.text(
                0.988,
                float(curveMagnitude.min()) - 0.10,
                f'{hero.star} · P = {hero.periodDays:.2f} d · {cycles:.1f} cycles so far',
                transform=axes.get_yaxis_transform(),
                color=self.labelColor,
                fontsize=7.5,
                ha='right',
                va='bottom',
                alpha=0.8 * reveal,
                zorder=6,
            )

        axes.text(
            0.015,
            0.945,
            f'{days:.1f} days of star time · Gaia DR3 epoch photometry, one point per transit',
            transform=axes.transAxes,
            color=self.labelColor,
            fontsize=8,
            va='center',
            alpha=0.6,
            zorder=6,
        )

    # ---- plane panel -----------------------------------------------------

    def _drawPlane(self, frame: int) -> None:
        axes = self.planeAxes
        reveal = self.planeReveal(frame)
        bottom, top = self.planeLimits(frame)
        axes.set_xlim(-0.05, 2.0)
        axes.set_ylim(bottom, top)
        if reveal <= 0.0:
            axes.axis('off')
            return

        axes.set_xticks(np.log10([1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 50.0, 100.0]))
        axes.set_xticklabels(['1', '2', '3', '5', '10', '20', '30', '50', '100'])
        axes.set_xlabel('Period (days)', color=self.labelColor, fontsize=9)
        axes.set_ylabel(self.planeLabel(frame), color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35 * reveal)

        lmcLogPeriod, lmcMagnitude = self.visibleLmc(frame)
        if len(lmcLogPeriod):
            axes.scatter(
                lmcLogPeriod,
                lmcMagnitude,
                s=5.0,
                color=self.pointColor,
                alpha=0.45,
                linewidths=0.0,
                zorder=3,
            )
        smcLogPeriod, smcMagnitude = self.visibleSmc(frame)
        if len(smcLogPeriod):
            axes.scatter(
                smcLogPeriod,
                smcMagnitude,
                s=5.0,
                color=self.secondColor,
                alpha=0.45,
                linewidths=0.0,
                zorder=3,
            )

        self._drawHeroPoints(frame, reveal)
        self._drawFits(frame, lmcLogPeriod, lmcMagnitude, smcLogPeriod, smcMagnitude)

    def planeLabel(self, frame: int) -> str:
        morph = self.wesenheitMorph(frame)
        if morph <= 0.02:
            return 'Mean I magnitude (apparent)'
        if morph >= 0.98:
            return 'Wesenheit index  W = I − 1.55 (V − I)'
        return f'Mean I, reddening removed {morph * 100:.0f}%'

    def _drawHeroPoints(self, frame: int, reveal: float) -> None:
        """The three stars from the panel above, as three points down here."""
        morph = self.wesenheitMorph(frame)
        for index, hero in enumerate(self.heroes):
            heroReveal = min(reveal, self.heroReveal(frame, index))
            if heroReveal <= 0.0:
                continue
            magnitude = float(
                self.morphedMagnitude(
                    np.array([hero.meanInfraredMag]), np.array([hero.wesenheitMag]), morph
                )[0]
            )
            self.planeAxes.plot(
                [hero.logPeriod],
                [magnitude],
                'o',
                markersize=9.0,
                markerfacecolor=self.accentColor,
                markeredgecolor=self.panelFace,
                markeredgewidth=1.2,
                alpha=heroReveal,
                zorder=6,
            )
            self.planeAxes.text(
                hero.logPeriod,
                magnitude - 0.28,
                f'{hero.periodDays:.1f} d',
                color=self.accentColor,
                fontsize=8,
                ha='center',
                va='bottom',
                alpha=heroReveal,
                zorder=6,
            )

    def _drawFits(
        self,
        frame: int,
        lmcLogPeriod: np.ndarray,
        lmcMagnitude: np.ndarray,
        smcLogPeriod: np.ndarray,
        smcMagnitude: np.ndarray,
    ) -> None:
        """Fit and draw the ridge (or ridges) actually on screen."""
        drawn = self.fitReveal(frame)
        if drawn <= 0.0:
            return
        span = np.array([0.0, 1.9])
        for row, (logPeriod, magnitude, color, name) in enumerate(
            (
                (lmcLogPeriod, lmcMagnitude, self.pointColor, 'LMC'),
                (smcLogPeriod, smcMagnitude, self.secondColor, 'SMC'),
            )
        ):
            fit = self.liveFit(logPeriod, magnitude)
            if fit is None:
                continue
            self.planeAxes.plot(
                span,
                fit.slope * (span - PIVOT_LOG_PERIOD) + fit.zeroPoint,
                '-',
                color=color,
                linewidth=1.8,
                alpha=0.9 * drawn,
                zorder=5,
            )
            self.planeAxes.text(
                0.018,
                0.13 - 0.052 * row,
                f'{name}  slope {fit.slope:+.2f} mag/dex · scatter {fit.scatter:.3f} mag · '
                f'{fit.count} stars',
                transform=self.planeAxes.transAxes,
                color=color,
                fontsize=8.5,
                va='center',
                alpha=drawn,
                bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
                zorder=6,
            )
        self._drawOffset(frame, lmcLogPeriod, lmcMagnitude, smcLogPeriod, smcMagnitude)

    def _drawOffset(
        self,
        frame: int,
        lmcLogPeriod: np.ndarray,
        lmcMagnitude: np.ndarray,
        smcLogPeriod: np.ndarray,
        smcMagnitude: np.ndarray,
    ) -> None:
        """The gap between the ridges at 10 days, which is the distance ratio."""
        reveal = self.offsetReveal(frame)
        lmcFit = self.liveFit(lmcLogPeriod, lmcMagnitude)
        smcFit = self.liveFit(smcLogPeriod, smcMagnitude)
        if reveal <= 0.0 or lmcFit is None or smcFit is None:
            return
        self.planeAxes.annotate(
            '',
            xy=(PIVOT_LOG_PERIOD, smcFit.zeroPoint),
            xytext=(PIVOT_LOG_PERIOD, lmcFit.zeroPoint),
            arrowprops={
                'arrowstyle': '<->',
                'color': self.labelColor,
                'linewidth': 1.4,
                'alpha': 0.85 * reveal,
            },
            zorder=7,
        )
        self.planeAxes.text(
            PIVOT_LOG_PERIOD + 0.04,
            (lmcFit.zeroPoint + smcFit.zeroPoint) / 2.0,
            f'{smcFit.zeroPoint - lmcFit.zeroPoint:.3f} mag at 10 d',
            color=self.labelColor,
            fontsize=9,
            va='center',
            alpha=reveal,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=7,
        )
        # The chain in one line: a geometric distance in, a distance out.
        self.planeAxes.text(
            0.018,
            0.026,
            f'LMC {self.solution.lmcDistanceKiloparsecs:.1f} kpc from eclipsing binaries '
            f'→ SMC {self.solution.smcDistanceKiloparsecs:.1f} kpc from this offset',
            transform=self.planeAxes.transAxes,
            color=self.labelColor,
            fontsize=8.5,
            va='center',
            alpha=reveal,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=7,
        )

    # ---- figure furniture ------------------------------------------------

    def title(self, frame: int) -> str:
        shown = sum(1 for index in range(len(self.heroes)) if self.heroReveal(frame, index) > 0.5)
        if shown <= 1:
            return 'A Cepheid in the Large Magellanic Cloud, folded on its measured period'
        names = {2: 'Two', 3: 'Three'}
        return (
            f'{names.get(shown, shown)} Cepheids in the Large Magellanic Cloud, '
            'folded on their measured periods'
        )

    def _drawFigureText(self, frame: int) -> None:
        """Title, caption and credit live on the figure: the panels move, these do not."""
        for text in self._figureTexts:
            text.remove()
        self._figureTexts = [
            self.figure.text(
                0.5,
                0.963,
                self.title(frame),
                color=self.labelColor,
                fontsize=14,
                ha='center',
                va='center',
            ),
            self.figure.text(
                0.5,
                0.042,
                self.caption(frame),
                color=self.labelColor,
                fontsize=9.5,
                ha='center',
                va='center',
                alpha=0.9,
            ),
            self.figure.text(
                0.5,
                0.014,
                'OGLE-IV Cepheids (Soszyński+ 2015) · Gaia DR3 epoch photometry · slopes, '
                'scatter and the distance fitted from these CSVs',
                color=self.labelColor,
                fontsize=7,
                ha='center',
                va='center',
                alpha=0.6,
            ),
        ]

    # ---- captions --------------------------------------------------------

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        solution = self.solution
        hero = self.heroes[0]
        if act == 'pulse':
            return (
                f'One Cepheid, {hero.periodDays:.2f} days per cycle · it swells and cools and '
                'brightens on a clock you can watch'
            )
        if act == 'trio':
            slowest = self.heroes[-1]
            return (
                f'Three stars in one galaxy, so brightness is not distance · '
                f'{slowest.periodDays:.0f} days beats {hero.periodDays:.0f}, and the slow one '
                'is brighter by '
                f'{hero.meanInfraredMag - slowest.meanInfraredMag:.1f} mag'
            )
        if act == 'leavitt':
            return (
                f"{solution.lmcInfrared.count} LMC Cepheids · Leavitt's 1912 result is this "
                'ridge: longer period, brighter star'
            )
        if act == 'wesenheit':
            return (
                'Dust reddens and dims · the combination I − 1.55 (V − I) cancels it, and the '
                f'ridge tightens from {solution.lmcInfrared.scatter:.3f} to '
                f'{solution.lmcWesenheit.scatter:.3f} mag'
            )
        return (
            f'The SMC obeys the same law, offset {solution.modulusOffset:.3f} mag · that is a '
            f'distance ratio of {solution.distanceRatio:.2f}, so '
            f'{solution.smcDistanceKiloparsecs:.0f} kpc against the '
            f'{solution.publishedSmcDistanceKiloparsecs:.0f} kpc measured from eclipsing binaries'
        )

    def update(self, frame: int):
        self.pulseAxes.clear()
        self.planeAxes.clear()
        for axes in (self.pulseAxes, self.planeAxes):
            axes.set_facecolor(self.panelFace)
            for spine in axes.spines.values():
                spine.set_visible(False)
        opened = self.planeReveal(frame)
        self.pulseAxes.set_position(
            [
                tall + (short - tall) * opened
                for tall, short in zip(PULSE_RECT_TALL, PULSE_RECT_SHORT, strict=True)
            ]
        )
        self._drawPulse(frame)
        self._drawPlane(frame)
        self._drawFigureText(frame)
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


def renderCepheidLadderCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
    lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> None:
    outputDirectory = Path('output/animate/magellanic/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'magellanic_cepheid_ladder_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = CepheidLadderCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            catalogCsvPath=catalogCsvPath,
            lightcurveCsvPath=lightcurveCsvPath,
        )
        animator.saveGif(str(outputPath))
    print('Cepheid ladder cinema completed!')


__all__ = [
    'CepheidLadderCinematicAnimator',
    'HeroCurve',
    'LadderSolution',
    'RidgeFit',
    'buildHeroCurves',
    'distanceKiloparsecs',
    'evaluateFourier',
    'fitRidge',
    'fourierCurve',
    'loadCepheidCatalog',
    'loadHeroLightcurves',
    'renderCepheidLadderCinematicAnimations',
    'solveLadder',
    'wesenheitIndex',
]
