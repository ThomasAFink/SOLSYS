"""Type Ia standard-candle cinema — the second rung of the ladder (#126).

A Cepheid is a clock. A Type Ia supernova is a bomb whose light curve is the
ruler: the slower it fades, the brighter it was, and once that is divided out
every blast is the same lamp. This film earns that sentence from committed data.

It opens on SN 2011fe, the nearest well-observed Type Ia of the digital era,
walking a real B-band light curve from the Open Supernova Catalog. Two more
arrive — a fast decliner and a slow one — placed at the distance of their
redshift (2011fe sits on the Cepheid rung instead). The slow bomb is brighter,
which is Phillips' relation, and stretching the time axis until the declines
match is the correction that makes them standard. Then the Hubble diagram fills
with 1,543 Pantheon+ supernovae. The slope of magnitude against log redshift is
the inverse-square law; the 43 hosts that already have Cepheids pin the zero
point.

Every decline, slope and scatter quoted on screen is measured at render time
from the two CSVs.
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

DEFAULT_CATALOG_CSV = 'data/pantheonplus_type_ia.csv'
DEFAULT_LIGHTCURVE_CSV = 'data/type_ia_lightcurves.csv'

HERO_NAMES = ('2011fe', '2000cn', '2005eq')
# Hubble-flow window: close enough that dark energy is a perturbation, far
# enough that peculiar velocities do not dominate.
HUBBLE_FLOW_Z = (0.02, 0.15)
FIT_CLIP_SIGMA = 3.0
FIT_CLIP_ROUNDS = 3
# Inverse-square: mu = 5 log10(z) + intercept. The 5 is the whole claim.
INVERSE_SQUARE_SLOPE = 5.0
# Only used to place the two Hubble-flow heroes on an absolute scale. Labelled.
FIDUCIAL_H0_KM_S_MPC = 70.0
SPEED_OF_LIGHT_KM_S = 299792.458
# The SH0ES H0 the intercept would be compared with, if we quoted one. We do
# not: a q0-less cz/H0 is the wrong estimator. The number stays as a check.
PUBLISHED_SH0ES_H0 = 73.04

PULSE_RECT_TALL = (0.09, 0.115, 0.865, 0.815)
PULSE_RECT_SHORT = (0.09, 0.615, 0.865, 0.315)
PLANE_RECT = (0.09, 0.115, 0.865, 0.395)

ACT_BOUNDARIES = ((100, 'pulse'), (190, 'trio'), (290, 'stretch'), (390, 'hubble'))
HERO_ENTRY_FRAMES = (0, 108, 148)
HERO_FADE_FRAMES = 28.0
STRETCH_MORPH_FRAME = 200
STRETCH_MORPH_FRAMES = 70.0
PLANE_OPEN_FRAME = 278
PLANE_OPEN_FRAMES = 32.0
HUBBLE_RAIN_FRAME = 300
HUBBLE_RAIN_FRAMES = 78.0
FIT_DRAW_FRAME = 348
FIT_DRAW_FRAMES = 36.0
CALIBRATOR_REVEAL_FRAME = 400
CALIBRATOR_REVEAL_FRAMES = 36.0
MINIMUM_POINTS_TO_FIT = 40
# Film time walks the light curve: 24 s covers ~48 days around peak.
DAYS_PER_SECOND = 2.0


@dataclass(frozen=True)
class LineFit:
    """y = slope * x + intercept, after a sigma clip."""

    slope: float
    intercept: float
    scatter: float
    count: int


@dataclass(frozen=True)
class CandleSolution:
    """Everything the film measures from the committed CSVs."""

    hubble: LineFit
    phillips: LineFit
    calibratorCount: int
    sampleCount: int
    hubbleFlowCount: int


def loadTypeIaCatalog(csvPath: str | Path = DEFAULT_CATALOG_CSV) -> pd.DataFrame:
    """One row per Pantheon+ supernova, Cepheid modulus blank when absent."""
    frame = pd.read_csv(csvPath, comment='#')
    frame['cepheid_mu'] = pd.to_numeric(frame['cepheid_mu'], errors='coerce')
    return frame


def loadHeroLightcurves(csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV) -> dict[str, pd.DataFrame]:
    """Nightly B-band photometry, one frame per hero."""
    frame = pd.read_csv(csvPath, comment='#')
    return {name: group.reset_index(drop=True) for name, group in frame.groupby('name', sort=False)}


def geometricModulus(redshift: float, cepheidMu: float) -> float:
    """Distance modulus from a Cepheid host, otherwise a Hubble-flow stick.

    The stick uses 70 km/s/Mpc and cz/H0 with no deceleration term. It is a
    placing convention for the two nearby heroes, labelled as such, not a
    measurement of H0.
    """
    if np.isfinite(cepheidMu) and cepheidMu > 0.0:
        return float(cepheidMu)
    return 5.0 * np.log10(SPEED_OF_LIGHT_KM_S * float(redshift) / FIDUCIAL_H0_KM_S_MPC) + 25.0


def declineFifteen(daysFromPeak: np.ndarray, magnitude: np.ndarray) -> float:
    """Δm15(B): how many magnitudes the B light has faded 15 days after peak."""
    days = np.asarray(daysFromPeak, dtype=float)
    magnitude = np.asarray(magnitude, dtype=float)
    order = np.argsort(days)
    days, magnitude = days[order], magnitude[order]
    peakIndex = int(np.argmin(magnitude))
    peakDay = days[peakIndex]
    faded = float(np.interp(peakDay + 15.0, days, magnitude))
    return faded - float(magnitude[peakIndex])


def fitLine(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = FIT_CLIP_SIGMA,
    rounds: int = FIT_CLIP_ROUNDS,
) -> LineFit:
    """Least squares with a sigma clip."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        return LineFit(slope=float('nan'), intercept=float('nan'), scatter=float('nan'), count=0)
    keep = np.ones(len(x), dtype=bool)
    for _ in range(rounds):
        slope, intercept = np.polyfit(x[keep], y[keep], 1)
        residual = y - (slope * x + intercept)
        spread = residual[keep].std()
        if not np.isfinite(spread) or spread <= 0.0:
            break
        keep = np.abs(residual) < sigma * spread
        if keep.sum() < 2:
            keep = np.ones(len(x), dtype=bool)
            break
    slope, intercept = np.polyfit(x[keep], y[keep], 1)
    residual = y[keep] - (slope * x[keep] + intercept)
    return LineFit(
        slope=float(slope),
        intercept=float(intercept),
        scatter=float(residual.std()),
        count=int(keep.sum()),
    )


def solveCandle(catalog: pd.DataFrame) -> CandleSolution:
    """Hubble diagram in the Hubble flow, and Phillips from the same nearby slice."""
    flow = catalog[catalog['z_hd'].between(*HUBBLE_FLOW_Z, inclusive='both')]
    hubble = fitLine(np.log10(flow['z_hd']), flow['mu_sh0es'])
    nearby = catalog[catalog['z_hd'].between(0.01, 0.04, inclusive='both')]
    proxy = nearby['peak_mb'] - 5.0 * np.log10(nearby['z_hd'])
    phillips = fitLine(nearby['stretch_x1'], proxy)
    return CandleSolution(
        hubble=hubble,
        phillips=phillips,
        calibratorCount=int(catalog['is_calibrator'].sum()),
        sampleCount=len(catalog),
        hubbleFlowCount=len(flow),
    )


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


@dataclass(frozen=True)
class HeroCurve:
    """One Type Ia B-band light curve, ready to walk a playhead along."""

    name: str
    redshift: float
    stretchX1: float
    peakMb: float
    modulus: float
    daysFromPeak: np.ndarray
    magnitude: np.ndarray
    magnitudeError: np.ndarray
    decline15: float
    absolutePeak: float

    def magnitudeAtDay(self, day: float) -> float | None:
        if day < float(self.daysFromPeak.min()) or day > float(self.daysFromPeak.max()):
            return None
        return float(np.interp(day, self.daysFromPeak, self.magnitude))


def buildHeroCurves(
    catalog: pd.DataFrame, lightcurves: dict[str, pd.DataFrame]
) -> tuple[HeroCurve, ...]:
    """Three heroes: the Cepheid-calibrated normal, then fast, then slow."""
    heroes: list[HeroCurve] = []
    for name in HERO_NAMES:
        row = catalog[catalog['name'] == name]
        if row.empty or name not in lightcurves:
            raise ValueError(f'Hero {name} is missing from the committed data')
        photometry = lightcurves[name].sort_values('days_from_peak')
        days = photometry['days_from_peak'].to_numpy(dtype=float)
        magnitude = photometry['b_mag'].to_numpy(dtype=float)
        error = np.nan_to_num(
            pd.to_numeric(photometry['b_mag_error'], errors='coerce').to_numpy(dtype=float),
            nan=0.0,
        )
        redshift = float(row['z_hd'].iloc[0])
        modulus = geometricModulus(redshift, float(row['cepheid_mu'].iloc[0]))
        decline = declineFifteen(days, magnitude)
        heroes.append(
            HeroCurve(
                name=name,
                redshift=redshift,
                stretchX1=float(row['stretch_x1'].iloc[0]),
                peakMb=float(row['peak_mb'].iloc[0]),
                modulus=modulus,
                daysFromPeak=days,
                magnitude=magnitude,
                magnitudeError=error,
                decline15=decline,
                absolutePeak=float(np.min(magnitude) - modulus),
            )
        )
    return tuple(heroes)


class TypeIaCandleCinematicAnimator:
    """One blast → three stretches → a Hubble diagram of standardized lamps."""

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

        self.catalog = loadTypeIaCatalog(catalogCsvPath)
        self.lightcurves = loadHeroLightcurves(lightcurveCsvPath)
        self.heroes = buildHeroCurves(self.catalog, self.lightcurves)
        self.solution = solveCandle(self.catalog)
        self.referenceDecline = self.heroes[0].decline15

        flow = self.catalog[self.catalog['z_hd'].between(*HUBBLE_FLOW_Z, inclusive='both')]
        rest = self.catalog[~self.catalog['z_hd'].between(*HUBBLE_FLOW_Z, inclusive='both')]
        self.flowLogZ = np.log10(flow['z_hd'].to_numpy(dtype=float))
        self.flowMu = flow['mu_sh0es'].to_numpy(dtype=float)
        self.restLogZ = np.log10(rest['z_hd'].to_numpy(dtype=float))
        self.restMu = rest['mu_sh0es'].to_numpy(dtype=float)
        calibrators = self.catalog[self.catalog['is_calibrator'] == 1]
        self.calibratorLogZ = np.log10(calibrators['z_hd'].to_numpy(dtype=float))
        self.calibratorMu = calibrators['mu_sh0es'].to_numpy(dtype=float)

        generator = np.random.default_rng(126)
        self.flowArrival = generator.permutation(len(self.flowLogZ))
        self.restArrival = generator.permutation(len(self.restLogZ))

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.pointColor = '#7EB6FF' if self.isDark else '#204080'
        self.accentColor = '#FFB570' if self.isDark else '#B4540A'
        self.secondColor = '#8CD9A0' if self.isDark else '#1F6B3A'
        self.heroColors = (self.accentColor, self.pointColor, self.secondColor)
        self.panelFace = '#050508' if self.isDark else '#F4F2EC'
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, facecolor=self.panelFace)
        self.pulseAxes = self.figure.add_axes(PULSE_RECT_TALL)
        self.planeAxes = self.figure.add_axes(PLANE_RECT)
        self._figureTexts: list = []

    # ---- act bookkeeping -------------------------------------------------

    def act(self, frame: int) -> str:
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'ruler'

    def elapsedDays(self, frame: int) -> float:
        """Star time, centred so peak of 2011fe lands mid-opening act."""
        return frame / ANIMATION_FPS * DAYS_PER_SECOND - 8.0

    def heroReveal(self, frame: int, index: int) -> float:
        return smoothStep((frame - HERO_ENTRY_FRAMES[index]) / HERO_FADE_FRAMES)

    def stretchMorph(self, frame: int) -> float:
        """0 = observed time, 1 = time stretched so every Δm15 matches 2011fe."""
        return smoothStep((frame - STRETCH_MORPH_FRAME) / STRETCH_MORPH_FRAMES)

    def planeReveal(self, frame: int) -> float:
        return smoothStep((frame - PLANE_OPEN_FRAME) / PLANE_OPEN_FRAMES)

    def hubbleReveal(self, frame: int) -> float:
        return smoothStep((frame - HUBBLE_RAIN_FRAME) / HUBBLE_RAIN_FRAMES)

    def fitReveal(self, frame: int) -> float:
        return smoothStep((frame - FIT_DRAW_FRAME) / FIT_DRAW_FRAMES)

    def calibratorReveal(self, frame: int) -> float:
        return smoothStep((frame - CALIBRATOR_REVEAL_FRAME) / CALIBRATOR_REVEAL_FRAMES)

    def stretchedDays(self, hero: HeroCurve, frame: int) -> np.ndarray:
        """Stretch a fast decliner out (and a slow one in) toward 2011fe's width."""
        morph = self.stretchMorph(frame)
        scale = hero.decline15 / self.referenceDecline
        return hero.daysFromPeak * (1.0 + morph * (scale - 1.0))

    def placingModulus(self, hero: HeroCurve) -> float:
        """Hubble-flow stick used only to put the three blasts on one plot.

        2011fe is too nearby for cz/H0 to be a distance, but the same stick on
        all three keeps Phillips as a brightness ranking instead of a zero-point
        offset between the Cepheid rung and H0 = 70.
        """
        return 5.0 * np.log10(SPEED_OF_LIGHT_KM_S * hero.redshift / FIDUCIAL_H0_KM_S_MPC) + 25.0

    def absoluteMagnitude(self, hero: HeroCurve) -> np.ndarray:
        return hero.magnitude - self.placingModulus(hero)

    # ---- pulse panel -----------------------------------------------------

    def _drawPulse(self, frame: int) -> None:
        axes = self.pulseAxes
        morph = self.stretchMorph(frame)
        axes.set_xlim(-18.0, 40.0)
        # Absolute magnitude, inverted. Room for the faint fast decliner.
        axes.set_ylim(-16.6, -20.4)
        axes.set_xlabel(self.pulseXLabel(morph), color=self.labelColor, fontsize=9)
        axes.set_ylabel('Absolute B magnitude', color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)

        day = self.elapsedDays(frame)
        for index, hero in enumerate(self.heroes):
            reveal = self.heroReveal(frame, index)
            if reveal <= 0.0:
                continue
            color = self.heroColors[index]
            stretched = self.stretchedDays(hero, frame)
            absolute = self.absoluteMagnitude(hero)
            axes.errorbar(
                stretched,
                absolute,
                yerr=hero.magnitudeError,
                fmt='o',
                markersize=3.5,
                color=color,
                ecolor=color,
                elinewidth=0.5,
                capsize=0.0,
                alpha=0.8 * reveal,
                zorder=3,
            )
            order = np.argsort(stretched)
            axes.plot(
                stretched[order],
                absolute[order],
                '-',
                color=color,
                linewidth=1.2,
                alpha=0.45 * reveal,
                zorder=4,
            )
            playheadDay = day * (1.0 + morph * (hero.decline15 / self.referenceDecline - 1.0))
            playheadMag = hero.magnitudeAtDay(day)
            if playheadMag is not None:
                axes.plot(
                    [playheadDay],
                    [playheadMag - self.placingModulus(hero)],
                    'o',
                    markersize=11.0,
                    color=color,
                    markeredgecolor=self.panelFace,
                    markeredgewidth=0.8,
                    alpha=reveal,
                    zorder=6,
                )
            axes.text(
                0.985,
                0.18 + 0.10 * index,
                (f'SN {hero.name}  Δm15(B) = {hero.decline15:.2f} mag  x1 = {hero.stretchX1:+.2f}'),
                transform=axes.transAxes,
                color=color,
                fontsize=8,
                ha='right',
                va='center',
                alpha=0.9 * reveal,
                zorder=6,
            )

        axes.text(
            0.015,
            0.955,
            f'{day:+.1f} days from 2011fe peak · B band, nightly median · '
            f'placed at cz/{FIDUCIAL_H0_KM_S_MPC:.0f} km/s/Mpc',
            transform=axes.transAxes,
            color=self.labelColor,
            fontsize=8,
            va='center',
            alpha=0.6,
            zorder=6,
        )

    def pulseXLabel(self, morph: float) -> str:
        if morph <= 0.02:
            return 'Days from peak'
        if morph >= 0.98:
            return 'Days from peak, stretched to SN 2011fe’s decline'
        return f'Days from peak, stretch {morph * 100:.0f}% applied'

    # ---- Hubble plane ----------------------------------------------------

    def visibleHubble(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        shownFlow = int(round(self.hubbleReveal(frame) * len(self.flowArrival)))
        shownRest = int(round(self.hubbleReveal(frame) * 0.35 * len(self.restArrival)))
        flowIdx = self.flowArrival[:shownFlow]
        restIdx = self.restArrival[:shownRest]
        logZ = np.concatenate([self.flowLogZ[flowIdx], self.restLogZ[restIdx]])
        mu = np.concatenate([self.flowMu[flowIdx], self.restMu[restIdx]])
        return logZ, mu

    def liveHubble(self, logZ: np.ndarray, mu: np.ndarray) -> LineFit | None:
        inFlow = (logZ >= np.log10(HUBBLE_FLOW_Z[0])) & (logZ <= np.log10(HUBBLE_FLOW_Z[1]))
        if inFlow.sum() < MINIMUM_POINTS_TO_FIT:
            return None
        return fitLine(logZ[inFlow], mu[inFlow])

    def _drawPlane(self, frame: int) -> None:
        axes = self.planeAxes
        reveal = self.planeReveal(frame)
        axes.set_xlim(np.log10(0.001), np.log10(2.5))
        axes.set_ylim(44.5, 27.5)
        if reveal <= 0.0:
            axes.axis('off')
            return

        ticks = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0])
        axes.set_xticks(np.log10(ticks))
        axes.set_xticklabels([str(tick) if tick < 1 else f'{tick:.0f}' for tick in ticks])
        axes.set_xlabel('Redshift z', color=self.labelColor, fontsize=9)
        axes.set_ylabel('Distance modulus  μ', color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35 * reveal)

        logZ, mu = self.visibleHubble(frame)
        if len(logZ):
            axes.scatter(
                logZ,
                mu,
                s=8.0,
                color=self.pointColor,
                alpha=0.35 * reveal,
                linewidths=0.0,
                zorder=3,
            )

        calibratorAlpha = self.calibratorReveal(frame)
        if calibratorAlpha > 0.0:
            axes.scatter(
                self.calibratorLogZ,
                self.calibratorMu,
                s=28.0,
                color=self.accentColor,
                alpha=0.9 * calibratorAlpha,
                linewidths=0.0,
                zorder=5,
                label='Cepheid hosts',
            )

        self._drawHubbleFit(frame, logZ, mu)
        self._markHeroesOnHubble(frame)

    def _drawHubbleFit(self, frame: int, logZ: np.ndarray, mu: np.ndarray) -> None:
        drawn = self.fitReveal(frame)
        fit = self.liveHubble(logZ, mu)
        if drawn <= 0.0 or fit is None:
            return
        span = np.array([np.log10(HUBBLE_FLOW_Z[0]), np.log10(0.8)])
        self.planeAxes.plot(
            span,
            fit.slope * span + fit.intercept,
            '-',
            color=self.accentColor,
            linewidth=1.8,
            alpha=0.9 * drawn,
            zorder=4,
        )
        # A faint inverse-square line through the same pivot, for the eye.
        pivot = np.log10(0.05)
        expected = INVERSE_SQUARE_SLOPE * (span - pivot) + (fit.slope * pivot + fit.intercept)
        self.planeAxes.plot(
            span,
            expected,
            '--',
            color=self.labelColor,
            linewidth=1.0,
            alpha=0.45 * drawn,
            zorder=4,
        )
        self.planeAxes.text(
            0.018,
            0.12,
            (
                f'Hubble flow  slope {fit.slope:.2f} mag/dex  '
                f'(inverse square {INVERSE_SQUARE_SLOPE:.0f})  '
                f'scatter {fit.scatter:.3f} mag  {fit.count} SN'
            ),
            transform=self.planeAxes.transAxes,
            color=self.labelColor,
            fontsize=8.5,
            va='center',
            alpha=drawn,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=6,
        )
        calibratorAlpha = self.calibratorReveal(frame)
        if calibratorAlpha > 0.0:
            self.planeAxes.text(
                0.018,
                0.055,
                (
                    f'{self.solution.calibratorCount} Cepheid hosts pin the zero point · '
                    'the rung the last film measured'
                ),
                transform=self.planeAxes.transAxes,
                color=self.accentColor,
                fontsize=8.5,
                va='center',
                alpha=calibratorAlpha,
                bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
                zorder=6,
            )

    def _markHeroesOnHubble(self, frame: int) -> None:
        reveal = self.planeReveal(frame)
        if reveal <= 0.0:
            return
        for index, hero in enumerate(self.heroes):
            row = self.catalog[self.catalog['name'] == hero.name].iloc[0]
            self.planeAxes.plot(
                [np.log10(row['z_hd'])],
                [row['mu_sh0es']],
                'o',
                markersize=8.0,
                color=self.heroColors[index],
                markeredgecolor=self.panelFace,
                markeredgewidth=1.0,
                alpha=reveal,
                zorder=6,
            )

    # ---- figure furniture ------------------------------------------------

    def title(self, frame: int) -> str:
        shown = sum(1 for index in range(len(self.heroes)) if self.heroReveal(frame, index) > 0.5)
        if shown <= 1:
            return 'SN 2011fe — a Type Ia supernova in M101, B band as measured'
        if shown == 2:
            return 'Two Type Ia supernovae — the fast one is already fading'
        return 'Three Type Ia supernovae — decline rate is luminosity'

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        solution = self.solution
        hero = self.heroes[0]
        if act == 'pulse':
            return (
                f'SN {hero.name} peaked at B = {hero.peakMb:.2f} · Δm15(B) = '
                f'{hero.decline15:.2f} mag in 15 days · a bomb, not a clock'
            )
        if act == 'trio':
            fast, slow = self.heroes[1], self.heroes[2]
            gap = slow.absolutePeak - fast.absolutePeak
            return (
                f'SN {fast.name} fades {fast.decline15:.2f} mag, SN {slow.name} only '
                f'{slow.decline15:.2f} · the slow one is {abs(gap):.2f} mag brighter '
                '(Phillips 1993)'
            )
        if act == 'stretch':
            return (
                'Stretch the time axis until every decline matches 2011fe · that is the '
                f'correction: SALT2 x1 vs brightness slopes {solution.phillips.slope:+.2f} '
                'mag per unit stretch'
            )
        if act == 'hubble':
            return (
                f'{solution.sampleCount} Type Ia supernovae, standardized · Hubble-flow '
                f'slope {solution.hubble.slope:.2f} mag per dex of redshift against '
                f'{INVERSE_SQUARE_SLOPE:.0f} from the inverse-square law'
            )
        return (
            f'{solution.calibratorCount} of them exploded in Cepheid hosts · the light '
            'curve is the ruler, and the first rung is what sets its length'
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
                'Pantheon+ / SH0ES (Scolnic+ 2022) · Open Supernova Catalog B-band '
                'photometry · slope, scatter and Δm15 fitted from these CSVs',
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


def renderTypeIaCandleCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
    lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> None:
    outputDirectory = Path('output/animate/type_ia/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'type_ia_standard_candle_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = TypeIaCandleCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            catalogCsvPath=catalogCsvPath,
            lightcurveCsvPath=lightcurveCsvPath,
        )
        animator.saveGif(str(outputPath))
    print('Type Ia candle cinema completed!')


__all__ = [
    'CandleSolution',
    'HeroCurve',
    'LineFit',
    'TypeIaCandleCinematicAnimator',
    'buildHeroCurves',
    'declineFifteen',
    'fitLine',
    'geometricModulus',
    'loadHeroLightcurves',
    'loadTypeIaCatalog',
    'renderTypeIaCandleCinematicAnimations',
    'solveCandle',
]
