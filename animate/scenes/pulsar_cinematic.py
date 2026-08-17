"""Pulsar lighthouse cinema — a neutron star as a clock (#103).

A pulsar is a lighthouse: a spinning neutron star whose radio beam sweeps Earth
once per rotation. This film earns that sentence from committed data. It opens
on the Crab's 1.4 GHz folded profile from the EPN, a double spike whose playhead
advances at the star's own 33 ms period. Two more clocks arrive — Vela, then
the slow bright pulsar B0329+54 — and in the same seconds of star time the
Crab races while B0329+54 crawls. A schematic beam, as wide as the measured
pulse, is locked to that playhead. Then the ATNF catalogue fills a period–age
plane and the Crab's characteristic age, P/2Ṗ, is set next to the year the
supernova was seen.

Every period, duty cycle, implied P-dot and age rank quoted on screen is
measured at render time from the two CSVs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Wedge

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

DEFAULT_CATALOG_CSV = 'data/atnf_pulsars.csv'
DEFAULT_PROFILE_CSV = 'data/epn_pulsar_profiles.csv'

HERO_JNAMES = ('J0534+2200', 'J0835-4510', 'J0332+5434')
HERO_DISPLAY_NAMES = ('Crab', 'Vela', 'B0329+54')
# Film time to pulsar time: one film second is 1/7 of a real second, so the
# slow hero completes about two turns while the opening acts play.
SLOWDOWN = 7.0
REAL_SECONDS_PER_FILM_SECOND = 1.0 / SLOWDOWN
SECONDS_PER_JULIAN_YEAR = 365.25 * 86400.0
# The year the film is dated, matching the CSV download stamp. Not datetime.now.
FILM_YEAR = 2026
SN1054_YEAR = 1054
BASELINE_PERCENTILE = 30.0
DUTY_HALF_MAX = 0.5
SECONDARY_MIN_HEIGHT = 0.20
SECONDARY_MIN_SEPARATION = 0.15
STACK_GAP = 1.35

PULSE_RECT_TALL = (0.09, 0.115, 0.865, 0.815)
PULSE_RECT_SHORT = (0.09, 0.615, 0.865, 0.315)
PLANE_RECT = (0.09, 0.115, 0.865, 0.395)

ACT_BOUNDARIES = ((110, 'pulse'), (200, 'trio'), (300, 'beam'), (390, 'ages'))
# Negative so the Crab is fully on at frame 0; the others still fade in.
HERO_ENTRY_FRAMES = (-28, 118, 158)
HERO_FADE_FRAMES = 28.0
PLANE_OPEN_FRAME = 188
PLANE_OPEN_FRAMES = 32.0
AGE_RAIN_FRAME = 308
AGE_RAIN_FRAMES = 72.0
CENSUS_FRAME = 352
CENSUS_FRAMES = 32.0
REMNANT_REVEAL_FRAME = 400
REMNANT_REVEAL_FRAMES = 36.0


@dataclass(frozen=True)
class ClockSolution:
    """Everything the film measures from the committed ATNF table."""

    sampleCount: int
    medianAgeYr: float
    crabAgeYr: float
    remnantAgeYr: float
    youngerThanCrab: int
    crabPdot: float


@dataclass(frozen=True)
class HeroCurve:
    """One pulsar: ATNF clock plus an EPN folded profile, ready to play."""

    jname: str
    bname: str
    displayName: str
    periodS: float
    profilePeriodS: float
    frequencyGhz: float
    ageYr: float
    pdot: float
    phase: np.ndarray
    intensity: np.ndarray
    dutyCycle: float
    peakPhase: float
    secondaryPhase: float | None

    def phaseAt(self, elapsedSeconds: float) -> float:
        return wrapPhase(elapsedSeconds / self.periodS)

    def pulseCount(self, elapsedSeconds: float) -> float:
        return elapsedSeconds / self.periodS

    def intensityAtPhase(self, phase: float) -> float:
        wrapped = wrapPhase(phase)
        extendedPhase = np.concatenate([self.phase - 1.0, self.phase, self.phase + 1.0])
        extendedIntensity = np.concatenate([self.intensity, self.intensity, self.intensity])
        return float(np.interp(wrapped, extendedPhase, extendedIntensity))


def wrapPhase(phase: float) -> float:
    return float(np.mod(phase, 1.0))


def wrappedOffset(phase: float, peak: float) -> float:
    """Phase minus peak, wrapped into [-0.5, 0.5)."""
    return float(np.mod(phase - peak + 0.5, 1.0) - 0.5)


def impliedPeriodDerivative(
    periodS: float | np.ndarray, ageYr: float | np.ndarray
) -> float | np.ndarray:
    """P-dot recovered from the catalogue's own definition Age = P / (2 Pdot)."""
    period = np.asarray(periodS, dtype=float)
    age = np.asarray(ageYr, dtype=float)
    result = period / (2.0 * age * SECONDS_PER_JULIAN_YEAR)
    if result.ndim == 0:
        return float(result)
    return result


def characteristicAgeYears(periodS: float, pdot: float) -> float:
    return float(periodS) / (2.0 * float(pdot) * SECONDS_PER_JULIAN_YEAR)


def remnantAgeYears(filmYear: int = FILM_YEAR, snYear: int = SN1054_YEAR) -> float:
    return float(filmYear - snYear)


def subtractBaseline(intensity: np.ndarray, percentile: float = BASELINE_PERCENTILE) -> np.ndarray:
    values = np.asarray(intensity, dtype=float)
    return values - np.percentile(values, percentile)


def normalizeIntensity(intensity: np.ndarray) -> np.ndarray:
    """Baseline-subtracted Stokes I scaled so the peak is 1."""
    shifted = subtractBaseline(intensity)
    peak = float(np.max(shifted))
    if peak <= 0.0:
        return np.zeros_like(shifted)
    return shifted / peak


def dutyCycle(intensity: np.ndarray, fraction: float = DUTY_HALF_MAX) -> float:
    """Fraction of a rotation the pulse sits above half of (peak − baseline)."""
    shifted = subtractBaseline(intensity)
    height = float(np.max(shifted))
    if height <= 0.0:
        return 0.0
    return float(np.mean(shifted >= fraction * height))


def peakPhase(phase: np.ndarray, intensity: np.ndarray) -> float:
    normalised = normalizeIntensity(intensity)
    return wrapPhase(float(phase[int(np.argmax(normalised))]))


def secondaryPeakPhase(phase: np.ndarray, intensity: np.ndarray) -> float | None:
    """Phase of a second component, or None when the profile is a single spike."""
    normalised = normalizeIntensity(intensity)
    primary = peakPhase(phase, intensity)
    order = np.argsort(phase)
    phase = np.asarray(phase, dtype=float)[order]
    normalised = normalised[order]
    smoothed = np.convolve(normalised, np.ones(5) / 5.0, mode='same')
    delta = np.diff(smoothed)
    candidates: list[tuple[float, float]] = []
    for index in range(len(delta) - 1):
        if delta[index] <= 0.0 or delta[index + 1] > 0.0:
            continue
        peakIndex = index + 1
        height = float(normalised[peakIndex])
        peak = wrapPhase(float(phase[peakIndex]))
        if height < SECONDARY_MIN_HEIGHT:
            continue
        if abs(wrappedOffset(peak, primary)) < SECONDARY_MIN_SEPARATION:
            continue
        candidates.append((height, peak))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def formatPeriod(periodS: float) -> str:
    if periodS < 0.2:
        return f'{periodS * 1000.0:.1f} ms'
    return f'{periodS:.3f} s'


def formatAge(ageYr: float) -> str:
    if ageYr < 10000.0:
        return f'{ageYr:,.0f} yr'
    if ageYr < 1_000_000.0:
        return f'{ageYr / 1000.0:.1f} kyr'
    if ageYr < 1_000_000_000.0:
        return f'{ageYr / 1_000_000.0:.2f} Myr'
    return f'{ageYr / 1_000_000_000.0:.2f} Gyr'


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def loadPulsarCatalog(csvPath: str | Path = DEFAULT_CATALOG_CSV) -> pd.DataFrame:
    """One row per ATNF pulsar with a measured period and characteristic age."""
    frame = pd.read_csv(csvPath, comment='#')
    frame['jname'] = frame['jname'].astype(str).str.strip()
    for column in ('period_s', 'dm_pc_cm3', 'distance_kpc', 'age_yr', 'edot_erg_s'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame['pdot'] = impliedPeriodDerivative(
        frame['period_s'].to_numpy(), frame['age_yr'].to_numpy()
    )
    return frame


def loadHeroProfiles(csvPath: str | Path = DEFAULT_PROFILE_CSV) -> dict[str, pd.DataFrame]:
    """Folded Stokes I, one frame per hero pulsar."""
    frame = pd.read_csv(csvPath, comment='#')
    frame['jname'] = frame['jname'].astype(str).str.strip()
    return {
        name: group.reset_index(drop=True) for name, group in frame.groupby('jname', sort=False)
    }


def buildHeroCurves(
    catalog: pd.DataFrame, profiles: dict[str, pd.DataFrame]
) -> tuple[HeroCurve, ...]:
    """Three clocks: Crab, Vela, B0329+54, fastest first."""
    heroes: list[HeroCurve] = []
    for jname, displayName in zip(HERO_JNAMES, HERO_DISPLAY_NAMES, strict=True):
        row = catalog[catalog['jname'] == jname]
        if row.empty or jname not in profiles:
            raise ValueError(f'Hero {jname} is missing from the committed data')
        photometry = profiles[jname].sort_values('time_s')
        profilePeriod = float(photometry['period_s'].iloc[0])
        phase = photometry['time_s'].to_numpy(dtype=float) / profilePeriod
        intensity = photometry['stokes_i'].to_numpy(dtype=float)
        periodS = float(row['period_s'].iloc[0])
        ageYr = float(row['age_yr'].iloc[0])
        heroes.append(
            HeroCurve(
                jname=jname,
                bname=str(photometry['bname'].iloc[0]),
                displayName=displayName,
                periodS=periodS,
                profilePeriodS=profilePeriod,
                frequencyGhz=float(photometry['frequency_ghz'].iloc[0]),
                ageYr=ageYr,
                pdot=impliedPeriodDerivative(periodS, ageYr),
                phase=phase,
                intensity=normalizeIntensity(intensity),
                dutyCycle=dutyCycle(intensity),
                peakPhase=peakPhase(phase, intensity),
                secondaryPhase=secondaryPeakPhase(phase, intensity),
            )
        )
    return tuple(heroes)


def solveClocks(catalog: pd.DataFrame, crabJname: str = HERO_JNAMES[0]) -> ClockSolution:
    """Census of the committed ATNF table, plus the Crab's implied P-dot."""
    crab = catalog[catalog['jname'] == crabJname]
    if crab.empty:
        raise ValueError(f'{crabJname} is missing from the ATNF table')
    crabAge = float(crab['age_yr'].iloc[0])
    crabPeriod = float(crab['period_s'].iloc[0])
    return ClockSolution(
        sampleCount=len(catalog),
        medianAgeYr=float(catalog['age_yr'].median()),
        crabAgeYr=crabAge,
        remnantAgeYr=remnantAgeYears(),
        youngerThanCrab=int((catalog['age_yr'] < crabAge).sum()),
        crabPdot=impliedPeriodDerivative(crabPeriod, crabAge),
    )


class PulsarCinematicAnimator:
    """One pulse → three clocks → a beam → a catalogue of ages."""

    def __init__(
        self,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
        profileCsvPath: str | Path = DEFAULT_PROFILE_CSV,
    ):
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES

        self.catalog = loadPulsarCatalog(catalogCsvPath)
        self.profiles = loadHeroProfiles(profileCsvPath)
        self.heroes = buildHeroCurves(self.catalog, self.profiles)
        self.solution = solveClocks(self.catalog)

        self.logPeriod = np.log10(self.catalog['period_s'].to_numpy(dtype=float))
        self.logAge = np.log10(self.catalog['age_yr'].to_numpy(dtype=float))
        generator = np.random.default_rng(103)
        self.arrival = generator.permutation(len(self.logPeriod))

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

    def act(self, frame: int) -> str:
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'remnant'

    def elapsedSeconds(self, frame: int) -> float:
        return frame / ANIMATION_FPS * REAL_SECONDS_PER_FILM_SECOND

    def heroReveal(self, frame: int, index: int) -> float:
        return smoothStep((frame - HERO_ENTRY_FRAMES[index]) / HERO_FADE_FRAMES)

    def planeReveal(self, frame: int) -> float:
        return smoothStep((frame - PLANE_OPEN_FRAME) / PLANE_OPEN_FRAMES)

    def agesReveal(self, frame: int) -> float:
        return smoothStep((frame - AGE_RAIN_FRAME) / AGE_RAIN_FRAMES)

    def censusReveal(self, frame: int) -> float:
        return smoothStep((frame - CENSUS_FRAME) / CENSUS_FRAMES)

    def remnantReveal(self, frame: int) -> float:
        return smoothStep((frame - REMNANT_REVEAL_FRAME) / REMNANT_REVEAL_FRAMES)

    def stackOffset(self, index: int) -> float:
        return STACK_GAP * (len(self.heroes) - 1 - index)

    def visibleCensus(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        shown = int(round(self.agesReveal(frame) * len(self.arrival)))
        idx = self.arrival[:shown]
        return self.logPeriod[idx], self.logAge[idx]

    # ---- pulse panel -----------------------------------------------------

    def _drawPulse(self, frame: int) -> None:
        axes = self.pulseAxes
        axes.set_xlim(-0.05, 1.05)
        top = self.stackOffset(0) + 1.15
        axes.set_ylim(-0.18, top)
        axes.set_xlabel('Pulse phase', color=self.labelColor, fontsize=9)
        axes.set_ylabel('Stokes I, normalised and stacked', color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35)
        elapsed = self.elapsedSeconds(frame)
        for index, hero in enumerate(self.heroes):
            self._plotHero(axes, hero, index, frame, elapsed)
        axes.text(
            0.015,
            0.955,
            (
                f'{elapsed * 1000.0:.0f} ms of pulsar time · slowed ×{SLOWDOWN:.0f} · '
                f'EPN folded profiles near 1.4 GHz'
            ),
            transform=axes.transAxes,
            color=self.labelColor,
            fontsize=8,
            va='center',
            alpha=0.6,
            zorder=6,
        )

    def _plotHero(self, axes, hero: HeroCurve, index: int, frame: int, elapsed: float) -> None:
        reveal = self.heroReveal(frame, index)
        if reveal <= 0.0:
            return
        color = self.heroColors[index]
        offset = self.stackOffset(index)
        axes.plot(
            hero.phase,
            hero.intensity + offset,
            '-',
            color=color,
            linewidth=1.6,
            alpha=0.9 * reveal,
            zorder=4,
        )
        playhead = hero.phaseAt(elapsed)
        axes.plot(
            [playhead],
            [hero.intensityAtPhase(playhead) + offset],
            'o',
            markersize=10.0,
            color=color,
            markeredgecolor=self.panelFace,
            markeredgewidth=0.8,
            alpha=reveal,
            zorder=6,
        )
        duty = f'{100.0 * hero.dutyCycle:.1f}%'
        axes.text(
            0.98,
            offset + 0.78,
            (
                f'{hero.displayName}  P = {formatPeriod(hero.periodS)}  '
                f'W50 = {duty}  {hero.pulseCount(elapsed):.1f} pulses'
            ),
            color=color,
            fontsize=8,
            ha='right',
            va='center',
            alpha=0.9 * reveal,
            zorder=6,
        )

    # ---- lower panel: lighthouse, then ages ------------------------------

    def _drawPlane(self, frame: int) -> None:
        reveal = self.planeReveal(frame)
        if reveal <= 0.0:
            self.planeAxes.axis('off')
            return
        if self.agesReveal(frame) > 0.0:
            self._drawAges(frame, reveal)
            return
        self._drawLighthouse(frame, reveal)

    def _drawBeam(
        self,
        axes,
        pointingDeg: float,
        halfWidthDeg: float,
        radius: float,
        color: str,
        alpha: float,
        linewidth: float,
    ) -> None:
        """A ray so the spin is visible, plus a wedge as wide as measured W50."""
        theta = np.deg2rad(pointingDeg)
        axes.plot(
            [0.0, radius * np.cos(theta)],
            [0.0, radius * np.sin(theta)],
            '-',
            color=color,
            linewidth=linewidth,
            alpha=alpha,
            zorder=4,
            solid_capstyle='round',
        )
        axes.add_patch(
            Wedge(
                (0.0, 0.0),
                radius,
                pointingDeg - halfWidthDeg,
                pointingDeg + halfWidthDeg,
                facecolor=color,
                edgecolor='none',
                alpha=min(alpha, 0.45),
            )
        )

    def _drawLighthouse(self, frame: int, alpha: float) -> None:
        axes = self.planeAxes
        axes.set_xlim(-1.55, 2.15)
        axes.set_ylim(-1.25, 1.25)
        axes.set_aspect('equal')
        axes.axis('off')
        if alpha <= 0.0:
            return
        crab = self.heroes[0]
        elapsed = self.elapsedSeconds(frame)
        phase = crab.phaseAt(elapsed)
        pointing = wrappedOffset(phase, crab.peakPhase) * 360.0
        halfWidth = crab.dutyCycle * 360.0 / 2.0
        lit = abs(wrappedOffset(phase, crab.peakPhase)) <= crab.dutyCycle / 2.0
        beamAlpha = (0.55 + 0.35 * float(lit)) * alpha
        ns = Circle(
            (0.0, 0.0), 0.28, facecolor=self.labelColor, edgecolor='none', alpha=0.9 * alpha
        )
        axes.add_patch(ns)
        self._drawBeam(axes, pointing, halfWidth, 1.45, self.accentColor, beamAlpha, 2.2)
        if crab.secondaryPhase is not None:
            second = wrappedOffset(phase, crab.secondaryPhase) * 360.0
            self._drawBeam(axes, second, halfWidth, 1.15, self.pointColor, 0.4 * alpha, 1.4)
        axes.plot(
            [0.32, 1.85], [0.0, 0.0], '--', color=self.labelColor, linewidth=0.8, alpha=0.45 * alpha
        )
        axes.plot(1.85, 0.0, 'o', markersize=7.0, color=self.labelColor, alpha=alpha, zorder=5)
        axes.text(1.85, -0.22, 'Earth', color=self.labelColor, fontsize=8, ha='center', alpha=alpha)
        axes.text(
            0.02,
            0.08,
            (
                f'schematic · beam width = measured W50 ({100.0 * crab.dutyCycle:.1f}% of a turn) · '
                'rotation locked to the Crab playhead'
            ),
            transform=axes.transAxes,
            color=self.labelColor,
            fontsize=8,
            va='center',
            alpha=0.8 * alpha,
        )

    def _drawAges(self, frame: int, reveal: float) -> None:
        axes = self.planeAxes
        axes.set_xlim(np.log10(0.001), np.log10(12.0))
        axes.set_ylim(np.log10(80.0), np.log10(8.0e10))
        ticks = np.array([0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
        axes.set_xticks(np.log10(ticks))
        axes.set_xticklabels([f'{tick:g}' for tick in ticks])
        ageTicks = np.array([1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10])
        axes.set_yticks(np.log10(ageTicks))
        axes.set_yticklabels(
            ['1 kyr', '10 kyr', '100 kyr', '1 Myr', '10 Myr', '100 Myr', '1 Gyr', '10 Gyr']
        )
        axes.set_xlabel('Period P (s)', color=self.labelColor, fontsize=9)
        axes.set_ylabel('Characteristic age  P/2Ṗ  (yr)', color=self.labelColor, fontsize=9)
        axes.tick_params(colors=self.labelColor, labelsize=7)
        for spine in axes.spines.values():
            spine.set_visible(True)
            spine.set_color(self.labelColor)
            spine.set_alpha(0.35 * reveal)
        logP, logAge = self.visibleCensus(frame)
        if len(logP):
            axes.scatter(
                logP,
                logAge,
                s=8.0,
                color=self.pointColor,
                alpha=0.35 * reveal,
                linewidths=0.0,
                zorder=3,
            )
        self._markHeroesOnAges(reveal)
        self._drawCensus(frame, logAge)
        self._drawRemnant(frame)

    def _markHeroesOnAges(self, reveal: float) -> None:
        if reveal <= 0.0:
            return
        for index, hero in enumerate(self.heroes):
            self.planeAxes.plot(
                [np.log10(hero.periodS)],
                [np.log10(hero.ageYr)],
                'o',
                markersize=8.5,
                color=self.heroColors[index],
                markeredgecolor=self.panelFace,
                markeredgewidth=1.0,
                alpha=reveal,
                zorder=6,
            )

    def _drawCensus(self, frame: int, logAge: np.ndarray) -> None:
        drawn = self.censusReveal(frame)
        if drawn <= 0.0 or len(logAge) < 20:
            return
        ages = 10.0**logAge
        median = float(np.median(ages))
        self.planeAxes.text(
            0.018,
            0.12,
            (
                f'{len(logAge)} pulsars on screen · median age {formatAge(median)} · '
                f'Crab Ṗ = {self.solution.crabPdot:.2e} s/s'
            ),
            transform=self.planeAxes.transAxes,
            color=self.labelColor,
            fontsize=8.5,
            va='center',
            alpha=drawn,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=6,
        )

    def _drawRemnant(self, frame: int) -> None:
        drawn = self.remnantReveal(frame)
        if drawn <= 0.0:
            return
        solution = self.solution
        self.planeAxes.axhline(
            np.log10(solution.remnantAgeYr),
            color=self.accentColor,
            linewidth=1.0,
            linestyle='--',
            alpha=0.55 * drawn,
            zorder=4,
        )
        self.planeAxes.text(
            0.018,
            0.055,
            (
                f'SN 1054 was {solution.remnantAgeYr:.0f} yr ago · '
                f'{solution.youngerThanCrab} catalogue objects are younger than the Crab, '
                'none of them a 33 ms radio pulsar'
            ),
            transform=self.planeAxes.transAxes,
            color=self.accentColor,
            fontsize=8.5,
            va='center',
            alpha=drawn,
            bbox={'facecolor': self.panelFace, 'edgecolor': 'none', 'alpha': 0.7, 'pad': 2.0},
            zorder=6,
        )

    # ---- figure furniture ------------------------------------------------

    def title(self, frame: int) -> str:
        shown = sum(1 for index in range(len(self.heroes)) if self.heroReveal(frame, index) > 0.5)
        if shown <= 1:
            return 'The Crab pulsar — a 33 ms radio pulse, folded at 1.4 GHz'
        if shown == 2:
            return 'Two pulsars — each playhead runs on its own period'
        if self.act(frame) in ('beam',):
            return 'A lighthouse — the beam is as wide as the measured pulse'
        if self.act(frame) in ('ages', 'remnant'):
            return 'A catalogue of clocks — period against characteristic age'
        return 'Three pulsars — the fast one races, the slow one crawls'

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        crab, vela, slow = self.heroes
        solution = self.solution
        elapsed = self.elapsedSeconds(frame)
        if act == 'pulse':
            return (
                f'{crab.displayName} {crab.jname}  P = {formatPeriod(crab.periodS)} · '
                f'W50 = {100.0 * crab.dutyCycle:.1f}% of a turn · a neutron star, not a lamp'
            )
        if act == 'trio':
            return (
                f'in {elapsed * 1000.0:.0f} ms of star time the Crab has pulsed '
                f'{crab.pulseCount(elapsed):.0f} times, Vela {vela.pulseCount(elapsed):.0f}, '
                f'{slow.displayName} {slow.pulseCount(elapsed):.1f} · slowed ×{SLOWDOWN:.0f}'
            )
        if act == 'beam':
            return (
                f'the wedge is the measured pulse: W50 = {100.0 * crab.dutyCycle:.1f}% · '
                'when it sweeps Earth the playhead is at the peak'
            )
        if act == 'ages':
            return (
                f'{solution.sampleCount} pulsars with a measured spin-down · '
                f'median age {formatAge(solution.medianAgeYr)} · Crab Ṗ = '
                f'{solution.crabPdot:.2e} s/s from τ = P/2Ṗ'
            )
        return (
            f"the Crab's clock reads {formatAge(solution.crabAgeYr)} (P/2Ṗ) · "
            f'SN 1054 was {solution.remnantAgeYr:.0f} yr ago · characteristic age '
            'assumes it was born spinning infinitely fast'
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
                'EPN folded profiles (Gould & Lyne 1998; Johnston+ 1998) · '
                'ATNF Pulsar Catalogue via VizieR B/psr (Manchester+ 2005) · '
                'W50, Ṗ and age rank measured from these CSVs',
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


def renderPulsarCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    catalogCsvPath: str | Path = DEFAULT_CATALOG_CSV,
    profileCsvPath: str | Path = DEFAULT_PROFILE_CSV,
) -> None:
    outputDirectory = Path('output/animate/pulsar/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'pulsar_lighthouse_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = PulsarCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            catalogCsvPath=catalogCsvPath,
            profileCsvPath=profileCsvPath,
        )
        animator.saveGif(str(outputPath))
    print('Pulsar lighthouse cinema completed!')


__all__ = [
    'ClockSolution',
    'HeroCurve',
    'PulsarCinematicAnimator',
    'buildHeroCurves',
    'characteristicAgeYears',
    'dutyCycle',
    'formatAge',
    'formatPeriod',
    'impliedPeriodDerivative',
    'loadHeroProfiles',
    'loadPulsarCatalog',
    'normalizeIntensity',
    'peakPhase',
    'remnantAgeYears',
    'renderPulsarCinematicAnimations',
    'secondaryPeakPhase',
    'solveClocks',
    'wrapPhase',
]
