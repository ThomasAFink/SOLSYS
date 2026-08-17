"""RR Lyrae cinema — the other clock on the first rung (#160).

Cepheids are the famous period–luminosity relation. RR Lyrae are the metal-poor
horizontal-branch version of that sentence: nearly the same lamp in every old
population, two pulsation modes with two light-curve shapes, and a distance
that still needs a metallicity term. This film earns that from committed data.

The pulse comes first. Three LMC RR Lyrae carry real OGLE-IV I-band photometry,
folded on their catalogue periods. The overtone star is a sine and it races;
the two fundamental-mode stars are sawtooths, and Bailey's 1902 result is
visible before any fitting — the longer period has the smaller bump. Then the
Bailey diagram fills with tens of thousands of Magellanic RR Lyrae, RRab and
RRc occupying different loci. Swapping mean I for the Wesenheit index tightens
the RRab ridge because that combination cancels reddening. Finally the SMC
arrives as a second ridge, and the gap is a distance — short of the eclipsing-
binary answer, because this clock still cares about metal.

Every slope, scatter and offset quoted on screen is fitted at render time from
the two CSVs.
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

DEFAULT_CATALOG_CSV = 'data/ogle_magellanic_rrlyrae.csv'
DEFAULT_LIGHTCURVE_CSV = 'data/ogle_rrlyrae_lightcurves.csv'

WESENHEIT_COLOR_COEFFICIENT = 1.55
FIT_CLIP_SIGMA = 3.0
FIT_CLIP_ROUNDS = 3
# Pivot at a typical RRab period so the zero point is quoted in the middle of
# the fundamental-mode sample, not extrapolated to 0.2 d.
PIVOT_PERIOD_DAYS = 0.5
PIVOT_LOG_PERIOD = float(np.log10(PIVOT_PERIOD_DAYS))

LMC_DISTANCE_MODULUS = 18.477  # Pietrzynski et al. 2019, Nature 567, 200
PUBLISHED_SMC_DISTANCE_MODULUS = 18.977  # Graczyk et al. 2020, ApJ 904, 13

HERO_STARS = (
    'OGLE-LMC-RRLYR-03686',
    'OGLE-LMC-RRLYR-16323',
    'OGLE-LMC-RRLYR-12984',
)
FOURIER_ORDER = 3
# Film time to star time: 24 s covers 2.9 days, about four cycles of the slowest
# hero and ten of the overtone.
DAYS_PER_SECOND = 0.12
STACK_GAP = 1.35

PULSE_RECT_TALL = (0.09, 0.115, 0.865, 0.815)
PULSE_RECT_SHORT = (0.09, 0.615, 0.865, 0.315)
PLANE_RECT = (0.09, 0.115, 0.865, 0.395)

ACT_BOUNDARIES = ((110, 'pulse'), (200, 'trio'), (300, 'bailey'), (390, 'candle'))
HERO_ENTRY_FRAMES = (-28, 118, 158)
HERO_FADE_FRAMES = 28.0
PLANE_OPEN_FRAME = 188
PLANE_OPEN_FRAMES = 32.0
BAILEY_RAIN_FRAME = 206
BAILEY_RAIN_FRAMES = 74.0
CANDLE_RAIN_FRAME = 308
CANDLE_RAIN_FRAMES = 70.0
WESENHEIT_MORPH_FRAME = 328
WESENHEIT_MORPH_FRAMES = 50.0
FIT_DRAW_FRAME = 348
FIT_DRAW_FRAMES = 32.0
SMC_ARRIVE_FRAME = 400
SMC_ARRIVE_FRAMES = 44.0
OFFSET_REVEAL_FRAME = 430
OFFSET_REVEAL_FRAMES = 32.0
MINIMUM_POINTS_TO_FIT = 60
# The committed catalogue is 42k stars. Fitting uses all of them; the scatter
# plot rains a seeded subsample so the GIF stays gallery-sized and the Bailey
# loci stay readable instead of becoming a solid blob.
DRAW_LMC_CAP = 4200
DRAW_SMC_CAP = 2200


@dataclass(frozen=True)
class RidgeFit:
    """magnitude = slope (log P - log 0.5 d) + zeroPoint."""

    slope: float
    zeroPoint: float
    scatter: float
    count: int


@dataclass(frozen=True)
class ClockSolution:
    """Everything the film measures from the committed OGLE tables."""

    lmcInfrared: RidgeFit
    lmcWesenheit: RidgeFit
    smcWesenheit: RidgeFit
    modulusOffset: float
    distanceRatio: float
    smcDistanceModulus: float
    lmcDistanceKiloparsecs: float
    smcDistanceKiloparsecs: float
    publishedSmcDistanceKiloparsecs: float
    lmcRRabCount: int
    lmcRRcCount: int
    sampleCount: int
    lmcRRabMedianAmplitude: float
    lmcRRcMedianAmplitude: float


def wesenheitIndex(visualMag: np.ndarray, infraredMag: np.ndarray) -> np.ndarray:
    visual = np.asarray(visualMag, dtype=float)
    infrared = np.asarray(infraredMag, dtype=float)
    return infrared - WESENHEIT_COLOR_COEFFICIENT * (visual - infrared)


def loadRrLyraeCatalog(csvPath: str | Path = DEFAULT_CATALOG_CSV) -> pd.DataFrame:
    """RRab and RRc in both clouds, with derived columns added here not stored."""
    frame = pd.read_csv(csvPath, comment='#')
    frame['type'] = frame['type'].astype(str).str.strip()
    frame['logPeriod'] = np.log10(frame['period_days'].to_numpy(dtype=float))
    frame['wesenheit'] = wesenheitIndex(frame['mean_v_mag'], frame['mean_i_mag'])
    return frame


def loadHeroLightcurves(csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(csvPath, comment='#')
    return {star: group.reset_index(drop=True) for star, group in frame.groupby('star', sort=False)}


def fitRidge(
    logPeriod: np.ndarray,
    magnitude: np.ndarray,
    sigma: float = FIT_CLIP_SIGMA,
    rounds: int = FIT_CLIP_ROUNDS,
) -> RidgeFit:
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
    return 10.0 ** (float(distanceModulus) / 5.0 + 1.0) / 1000.0


def solveClocks(catalog: pd.DataFrame) -> ClockSolution:
    """RRab ridges in both clouds; the gap is a distance still missing metallicity."""
    lmcRRab = catalog[(catalog['cloud'] == 'LMC') & (catalog['type'] == 'RRab')]
    lmcRRc = catalog[(catalog['cloud'] == 'LMC') & (catalog['type'] == 'RRc')]
    smcRRab = catalog[(catalog['cloud'] == 'SMC') & (catalog['type'] == 'RRab')]
    lmcInfrared = fitRidge(lmcRRab['logPeriod'], lmcRRab['mean_i_mag'])
    lmcWesenheit = fitRidge(lmcRRab['logPeriod'], lmcRRab['wesenheit'])
    smcWesenheit = fitRidge(smcRRab['logPeriod'], smcRRab['wesenheit'])
    offset = smcWesenheit.zeroPoint - lmcWesenheit.zeroPoint
    smcModulus = LMC_DISTANCE_MODULUS + offset
    return ClockSolution(
        lmcInfrared=lmcInfrared,
        lmcWesenheit=lmcWesenheit,
        smcWesenheit=smcWesenheit,
        modulusOffset=float(offset),
        distanceRatio=float(10.0 ** (offset / 5.0)),
        smcDistanceModulus=float(smcModulus),
        lmcDistanceKiloparsecs=distanceKiloparsecs(LMC_DISTANCE_MODULUS),
        smcDistanceKiloparsecs=distanceKiloparsecs(smcModulus),
        publishedSmcDistanceKiloparsecs=distanceKiloparsecs(PUBLISHED_SMC_DISTANCE_MODULUS),
        lmcRRabCount=len(lmcRRab),
        lmcRRcCount=len(lmcRRc),
        sampleCount=len(catalog),
        lmcRRabMedianAmplitude=float(lmcRRab['i_amplitude_mag'].median()),
        lmcRRcMedianAmplitude=float(lmcRRc['i_amplitude_mag'].median()),
    )


def fourierCurve(
    phase: np.ndarray, magnitude: np.ndarray, order: int = FOURIER_ORDER
) -> tuple[np.ndarray, float]:
    phase = np.asarray(phase, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    columns = [np.ones_like(phase)]
    for harmonic in range(1, order + 1):
        columns.append(np.cos(2.0 * np.pi * harmonic * phase))
        columns.append(np.sin(2.0 * np.pi * harmonic * phase))
    coefficients, *_ = np.linalg.lstsq(np.vstack(columns).T, magnitude, rcond=None)
    sampled = np.linspace(0.0, 1.0, 512, endpoint=False)
    brightest = float(sampled[int(np.argmin(evaluateFourier(coefficients, sampled)))])
    return coefficients, brightest


def evaluateFourier(coefficients: np.ndarray, phase: np.ndarray) -> np.ndarray:
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
    """One folded RR Lyrae, ready to walk a playhead along."""

    star: str
    subtype: str
    periodDays: float
    iAmplitude: float
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

    def pulseCount(self, elapsedDays: float) -> float:
        return elapsedDays / self.periodDays


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
        time = photometry['time_hjd'].to_numpy(dtype=float)
        magnitude = photometry['i_mag'].to_numpy(dtype=float)
        rawPhase = np.mod(time, period) / period
        coefficients, brightest = fourierCurve(rawPhase, magnitude)
        phase = np.mod(rawPhase - brightest, 1.0)
        coefficients, _ = fourierCurve(phase, magnitude)
        heroes.append(
            HeroCurve(
                star=star,
                subtype=str(row['type'].iloc[0]),
                periodDays=period,
                iAmplitude=float(row['i_amplitude_mag'].iloc[0]),
                phase=phase,
                magnitude=magnitude,
                magnitudeError=photometry['i_mag_error'].to_numpy(dtype=float),
                coefficients=coefficients,
                meanInfraredMag=float(row['mean_i_mag'].iloc[0]),
                logPeriod=float(row['logPeriod'].iloc[0]),
                wesenheitMag=float(row['wesenheit'].iloc[0]),
            )
        )
    return tuple(sorted(heroes, key=lambda hero: hero.periodDays))


class RrLyraeCinematicAnimator:
    """One pulse → two modes → Bailey's diagram → a standard candle that still needs metal."""

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

        self.catalog = loadRrLyraeCatalog(catalogCsvPath)
        self.lightcurves = loadHeroLightcurves(lightcurveCsvPath)
        self.heroes = buildHeroCurves(self.catalog, self.lightcurves)
        self.solution = solveClocks(self.catalog)

        lmc = self.catalog[self.catalog['cloud'] == 'LMC']
        smcRRab = self.catalog[(self.catalog['cloud'] == 'SMC') & (self.catalog['type'] == 'RRab')]
        self.lmcLogPeriod = lmc['logPeriod'].to_numpy(dtype=float)
        self.lmcAmplitude = lmc['i_amplitude_mag'].to_numpy(dtype=float)
        self.lmcIsRRab = (lmc['type'] == 'RRab').to_numpy()
        self.lmcInfrared = lmc['mean_i_mag'].to_numpy(dtype=float)
        self.lmcWesenheit = lmc['wesenheit'].to_numpy(dtype=float)
        self.smcLogPeriod = smcRRab['logPeriod'].to_numpy(dtype=float)
        self.smcInfrared = smcRRab['mean_i_mag'].to_numpy(dtype=float)
        self.smcWesenheit = smcRRab['wesenheit'].to_numpy(dtype=float)

        generator = np.random.default_rng(160)
        lmcCount = len(self.lmcLogPeriod)
        smcCount = len(self.smcLogPeriod)
        self.lmcArrival = generator.choice(
            lmcCount, size=min(DRAW_LMC_CAP, lmcCount), replace=False
        )
        self.smcArrival = generator.choice(
            smcCount, size=min(DRAW_SMC_CAP, smcCount), replace=False
        )
        self.lmcRRabArrival = generator.permutation(np.flatnonzero(self.lmcIsRRab))

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.pointColor = '#7EB6FF' if self.isDark else '#204080'
        self.accentColor = '#FFB570' if self.isDark else '#B4540A'
        self.secondColor = '#8CD9A0' if self.isDark else '#1F6B3A'
        self.heroColors = (self.secondColor, self.accentColor, self.pointColor)
        self.panelFace = '#050508' if self.isDark else '#F4F2EC'
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, facecolor=self.panelFace)
        self.pulseAxes = self.figure.add_axes(PULSE_RECT_TALL)
        self.planeAxes = self.figure.add_axes(PLANE_RECT)
        self._figureTexts: list = []

    def act(self, frame: int) -> str:
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'clouds'

    def elapsedDays(self, frame: int) -> float:
        return frame / ANIMATION_FPS * DAYS_PER_SECOND

    def heroReveal(self, frame: int, index: int) -> float:
        return smoothStep((frame - HERO_ENTRY_FRAMES[index]) / HERO_FADE_FRAMES)

    def planeReveal(self, frame: int) -> float:
        return smoothStep((frame - PLANE_OPEN_FRAME) / PLANE_OPEN_FRAMES)

    def baileyReveal(self, frame: int) -> float:
        return smoothStep((frame - BAILEY_RAIN_FRAME) / BAILEY_RAIN_FRAMES)

    def candleReveal(self, frame: int) -> float:
        return smoothStep((frame - CANDLE_RAIN_FRAME) / CANDLE_RAIN_FRAMES)

    def wesenheitMorph(self, frame: int) -> float:
        return smoothStep((frame - WESENHEIT_MORPH_FRAME) / WESENHEIT_MORPH_FRAMES)

    def fitReveal(self, frame: int) -> float:
        return smoothStep((frame - FIT_DRAW_FRAME) / FIT_DRAW_FRAMES)

    def smcReveal(self, frame: int) -> float:
        return smoothStep((frame - SMC_ARRIVE_FRAME) / SMC_ARRIVE_FRAMES)

    def offsetReveal(self, frame: int) -> float:
        return smoothStep((frame - OFFSET_REVEAL_FRAME) / OFFSET_REVEAL_FRAMES)

    def stackOffset(self, index: int) -> float:
        return STACK_GAP * index

    def morphedMagnitude(self, infrared: np.ndarray, wesenheit: np.ndarray, morph: float):
        return np.asarray(infrared, dtype=float) + morph * (
            np.asarray(wesenheit, dtype=float) - np.asarray(infrared, dtype=float)
        )

    # ---- pulse -----------------------------------------------------------

    def pulseLimits(self, frame: int) -> tuple[float, float]:
        """Faint edge of the stack, opening as the next clock fades in."""
        faint = 1.05
        for index in range(1, len(self.heroes)):
            faint += STACK_GAP * self.heroReveal(frame, index)
        return faint, -0.35

    def _drawPulse(self, frame: int) -> None:
        axes = self.pulseAxes
        axes.set_xlim(-0.02, 1.02)
        axes.set_ylim(*self.pulseLimits(frame))
        axes.set_xlabel('Pulsation phase', color=self.labelColor, fontsize=9)
        axes.set_ylabel(
            'I magnitude, mean-subtracted and stacked', color=self.labelColor, fontsize=9
        )
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)
        days = self.elapsedDays(frame)
        for index, hero in enumerate(self.heroes):
            self._plotHero(axes, hero, index, frame, days)
        axes.text(
            0.015,
            0.955,
            f'{days:.2f} days of star time · OGLE-IV I-band, one point per epoch',
            transform=axes.transAxes,
            color=self.labelColor,
            fontsize=8,
            va='center',
            alpha=0.6,
            zorder=6,
        )

    def _plotHero(self, axes, hero: HeroCurve, index: int, frame: int, days: float) -> None:
        reveal = self.heroReveal(frame, index)
        if reveal <= 0.0:
            return
        color = self.heroColors[index]
        offset = self.stackOffset(index)
        mean = float(hero.coefficients[0])
        delta = hero.magnitude - mean
        axes.plot(
            hero.phase,
            delta + offset,
            'o',
            markersize=2.2,
            color=color,
            alpha=0.35 * reveal,
            zorder=3,
        )
        curvePhase, curveMagnitude = hero.curve()
        axes.plot(
            curvePhase,
            (curveMagnitude - mean) + offset,
            '-',
            color=color,
            linewidth=1.5,
            alpha=0.9 * reveal,
            zorder=4,
        )
        phase = (days / hero.periodDays) % 1.0
        axes.plot(
            [phase],
            [hero.magnitudeAtPhase(phase) - mean + offset],
            'o',
            markersize=10.0,
            color=color,
            markeredgecolor=self.panelFace,
            markeredgewidth=0.8,
            alpha=reveal,
            zorder=6,
        )
        axes.text(
            0.98,
            offset - 0.22,
            (
                f'{hero.subtype}  P = {hero.periodDays:.3f} d  '
                f'A_I = {hero.iAmplitude:.2f} mag  {hero.pulseCount(days):.1f} cycles'
            ),
            color=color,
            fontsize=8,
            ha='right',
            va='center',
            alpha=0.9 * reveal,
            zorder=6,
        )

    # ---- lower panel -----------------------------------------------------

    def _drawPlane(self, frame: int) -> None:
        reveal = self.planeReveal(frame)
        if reveal <= 0.0:
            self.planeAxes.axis('off')
            return
        if self.act(frame) == 'bailey':
            self._drawBailey(frame, reveal)
            return
        self._drawCandle(frame, reveal)

    def _drawBailey(self, frame: int, reveal: float) -> None:
        axes = self.planeAxes
        axes.set_xlim(np.log10(0.18), np.log10(1.05))
        axes.set_ylim(-0.02, 1.05)
        ticks = np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
        axes.set_xticks(np.log10(ticks))
        axes.set_xticklabels([f'{tick:g}' for tick in ticks])
        axes.set_xlabel('Period (days)', color=self.labelColor, fontsize=9)
        axes.set_ylabel('I-band amplitude (mag)', color=self.labelColor, fontsize=9)
        self._stylePlane(axes, reveal)
        shown = int(round(self.baileyReveal(frame) * len(self.lmcArrival)))
        idx = self.lmcArrival[:shown]
        isAb = self.lmcIsRRab[idx]
        if isAb.any():
            axes.scatter(
                self.lmcLogPeriod[idx][isAb],
                self.lmcAmplitude[idx][isAb],
                s=6.0,
                color=self.accentColor,
                alpha=0.28 * reveal,
                linewidths=0.0,
                zorder=3,
            )
        if (~isAb).any():
            axes.scatter(
                self.lmcLogPeriod[idx][~isAb],
                self.lmcAmplitude[idx][~isAb],
                s=6.0,
                color=self.secondColor,
                alpha=0.35 * reveal,
                linewidths=0.0,
                zorder=3,
            )
        for index, hero in enumerate(self.heroes):
            axes.plot(
                [hero.logPeriod],
                [hero.iAmplitude],
                'o',
                markersize=8.5,
                color=self.heroColors[index],
                markeredgecolor=self.panelFace,
                markeredgewidth=1.0,
                alpha=reveal,
                zorder=6,
            )
        if shown >= 40:
            axes.text(
                0.018,
                0.90,
                (
                    f'{self.solution.lmcRRabCount} RRab · {self.solution.lmcRRcCount} RRc · '
                    f'median A_I {self.solution.lmcRRabMedianAmplitude:.2f} vs '
                    f'{self.solution.lmcRRcMedianAmplitude:.2f} mag'
                ),
                transform=axes.transAxes,
                color=self.labelColor,
                fontsize=8.5,
                va='center',
                alpha=reveal,
                bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
                zorder=6,
            )

    def _drawCandle(self, frame: int, reveal: float) -> None:
        axes = self.planeAxes
        morph = self.wesenheitMorph(frame)
        axes.set_xlim(np.log10(0.28), np.log10(1.05))
        faint = 19.55 - morph * 0.85
        bright = 17.55 - morph * 0.85
        axes.set_ylim(faint, bright)
        ticks = np.array([0.3, 0.4, 0.5, 0.6, 0.8, 1.0])
        axes.set_xticks(np.log10(ticks))
        axes.set_xticklabels([f'{tick:g}' for tick in ticks])
        axes.set_xlabel('Period (days)', color=self.labelColor, fontsize=9)
        axes.set_ylabel(self.candleLabel(morph), color=self.labelColor, fontsize=9)
        self._stylePlane(axes, reveal)

        shown = int(round(self.candleReveal(frame) * len(self.lmcArrival)))
        idx = self.lmcArrival[:shown]
        ab = self.lmcIsRRab[idx]
        abIdx = idx[ab]
        if len(abIdx):
            axes.scatter(
                self.lmcLogPeriod[abIdx],
                self.morphedMagnitude(self.lmcInfrared[abIdx], self.lmcWesenheit[abIdx], morph),
                s=6.0,
                color=self.accentColor,
                alpha=0.28 * reveal,
                linewidths=0.0,
                zorder=3,
            )
        smcShown = int(round(self.smcReveal(frame) * len(self.smcArrival)))
        smcIdx = self.smcArrival[:smcShown]
        if len(smcIdx):
            axes.scatter(
                self.smcLogPeriod[smcIdx],
                self.morphedMagnitude(self.smcInfrared[smcIdx], self.smcWesenheit[smcIdx], morph),
                s=6.0,
                color=self.pointColor,
                alpha=0.35 * reveal,
                linewidths=0.0,
                zorder=4,
            )
        fitCount = int(round(self.candleReveal(frame) * len(self.lmcRRabArrival)))
        fitIdx = self.lmcRRabArrival[:fitCount]
        self._drawLiveFit(
            frame,
            self.lmcLogPeriod[fitIdx],
            self.morphedMagnitude(self.lmcInfrared[fitIdx], self.lmcWesenheit[fitIdx], morph),
        )
        self._drawOffset(frame)
        for index, hero in enumerate(self.heroes):
            if hero.subtype != 'RRab':
                continue
            y = hero.meanInfraredMag + morph * (hero.wesenheitMag - hero.meanInfraredMag)
            axes.plot(
                [hero.logPeriod],
                [y],
                'o',
                markersize=8.5,
                color=self.heroColors[index],
                markeredgecolor=self.panelFace,
                markeredgewidth=1.0,
                alpha=reveal,
                zorder=6,
            )

    def _stylePlane(self, axes, reveal: float) -> None:
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35 * reveal)

    def candleLabel(self, morph: float) -> str:
        if morph <= 0.02:
            return 'Mean I magnitude (apparent)'
        if morph >= 0.98:
            return 'Wesenheit  I − 1.55 (V − I)'
        return f'Mean I sliding into Wesenheit ({morph * 100:.0f}%)'

    def _drawLiveFit(self, frame: int, logPeriod: np.ndarray, magnitude: np.ndarray) -> None:
        drawn = self.fitReveal(frame)
        if drawn <= 0.0 or len(logPeriod) < MINIMUM_POINTS_TO_FIT:
            return
        fit = fitRidge(logPeriod, magnitude)
        span = np.array([np.log10(0.32), np.log10(0.90)])
        self.planeAxes.plot(
            span,
            fit.slope * (span - PIVOT_LOG_PERIOD) + fit.zeroPoint,
            '-',
            color=self.labelColor,
            linewidth=1.6,
            alpha=0.85 * drawn,
            zorder=5,
        )
        self.planeAxes.text(
            0.018,
            0.12,
            (
                f'LMC RRab  slope {fit.slope:.2f} mag/dex  scatter {fit.scatter:.3f} mag  '
                f'{fit.count} stars'
            ),
            transform=self.planeAxes.transAxes,
            color=self.labelColor,
            fontsize=8.5,
            va='center',
            alpha=drawn,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=6,
        )

    def _drawOffset(self, frame: int) -> None:
        drawn = self.offsetReveal(frame)
        if drawn <= 0.0:
            return
        solution = self.solution
        self.planeAxes.text(
            0.018,
            0.055,
            (
                f'SMC offset {solution.modulusOffset:.3f} mag → '
                f'{solution.smcDistanceKiloparsecs:.1f} kpc vs '
                f'{solution.publishedSmcDistanceKiloparsecs:.1f} kpc from eclipsing binaries'
            ),
            transform=self.planeAxes.transAxes,
            color=self.pointColor,
            fontsize=8.5,
            va='center',
            alpha=drawn,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=6,
        )

    # ---- furniture -------------------------------------------------------

    def title(self, frame: int) -> str:
        shown = sum(1 for index in range(len(self.heroes)) if self.heroReveal(frame, index) > 0.5)
        act = self.act(frame)
        if shown <= 1:
            return 'An RR Lyrae star — a horizontal-branch clock, folded in I'
        if shown == 2:
            return 'Two modes — overtone sine, fundamental sawtooth'
        if act == 'bailey':
            return "Bailey's diagram — period against amplitude is the classification"
        if act == 'candle':
            return 'A standard candle — almost, once reddening is divided out'
        if act == 'clouds':
            return 'Two clouds — the gap is a distance that still needs metal'
        return 'Three RR Lyrae — the longer fundamental has the smaller bump'

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        solution = self.solution
        rrc, shortAb, longAb = self.heroes
        days = self.elapsedDays(frame)
        if act == 'pulse':
            return (
                f'{rrc.subtype} {rrc.star}  P = {rrc.periodDays:.3f} d · '
                f'A_I = {rrc.iAmplitude:.2f} mag · a sine, not a sawtooth'
            )
        if act == 'trio':
            if self.heroReveal(frame, 2) < 0.45:
                return (
                    f'{shortAb.subtype} {shortAb.star}  P = {shortAb.periodDays:.3f} d · '
                    f'A_I = {shortAb.iAmplitude:.2f} mag, a sawtooth · '
                    f'the overtone has already pulsed {rrc.pulseCount(days):.0f} times'
                )
            return (
                f'in {days:.2f} d the overtone has pulsed {rrc.pulseCount(days):.0f} times, '
                f'the {longAb.periodDays:.3f} d RRab only {longAb.pulseCount(days):.1f} · '
                f'and its bump is {shortAb.iAmplitude - longAb.iAmplitude:.2f} mag smaller '
                '(Bailey 1902)'
            )
        if act == 'bailey':
            return (
                f'{solution.lmcRRabCount} LMC RRab and {solution.lmcRRcCount} RRc · '
                f'median amplitude {solution.lmcRRabMedianAmplitude:.2f} vs '
                f'{solution.lmcRRcMedianAmplitude:.2f} mag · period is the mode'
            )
        if act == 'candle':
            return (
                'Dust reddens and dims · I − 1.55 (V − I) cancels it, and the RRab ridge '
                f'tightens from {solution.lmcInfrared.scatter:.3f} to '
                f'{solution.lmcWesenheit.scatter:.3f} mag'
            )
        return (
            f'The SMC is {solution.modulusOffset:.3f} mag farther on this clock · '
            f'{solution.smcDistanceKiloparsecs:.1f} kpc against '
            f'{solution.publishedSmcDistanceKiloparsecs:.1f} kpc from eclipsing binaries · '
            'RR Lyrae still need a metallicity term'
        )

    def _drawFigureText(self, frame: int) -> None:
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
                'OGLE-IV Collection of Variable Stars (Soszyński+ 2016) · '
                'Magellanic RRab + RRc · slope, scatter and offset fitted from these CSVs',
                color=self.labelColor,
                fontsize=7,
                ha='center',
                va='center',
                alpha=0.6,
            ),
        ]

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


def renderRrLyraeCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
    lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> None:
    outputDirectory = Path('output/animate/magellanic/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'magellanic_rr_lyrae_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = RrLyraeCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            catalogCsvPath=catalogCsvPath,
            lightcurveCsvPath=lightcurveCsvPath,
        )
        animator.saveGif(str(outputPath))
    print('RR Lyrae cinema completed!')


__all__ = [
    'ClockSolution',
    'HeroCurve',
    'RidgeFit',
    'RrLyraeCinematicAnimator',
    'buildHeroCurves',
    'distanceKiloparsecs',
    'evaluateFourier',
    'fitRidge',
    'loadHeroLightcurves',
    'loadRrLyraeCatalog',
    'renderRrLyraeCinematicAnimations',
    'solveClocks',
    'wesenheitIndex',
]
