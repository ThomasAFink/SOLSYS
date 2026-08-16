"""Asteroseismology cinema — a red giant weighed by its own ringing (#169).

Third film in the measurement strand. Tabby's Star (#73) showed dips you can see
and cannot explain; TRAPPIST-1 b (#95) showed a dip you cannot see until you fold
it. Here the signal is real and visible as a wobble, but what it *is* only exists
in another domain: Fourier transform four years of Kepler photometry and the
wobble resolves into a comb of pure tones, evenly spaced by Dnu and humped around
numax. Those two numbers give the star's radius and mass.

Every flux point is observed: Kepler long cadence for KIC 7944142 (HD 176694).
The spectrum, numax, Dnu, radius and mass are all computed at runtime from that
committed light curve with nothing but numpy.
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

from animate.blender_body_sprites import BlenderBodySpriteAtlas, diskRadiusFraction

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 84
ANIMATION_FPS = 20
ANIMATION_FRAMES = 480

DEFAULT_LIGHTCURVE_CSV = 'data/kic_7944142_kepler_lightcurve.csv'
GIANT_CATALOG_NAME = 'KIC 7944142'
GIANT_COMMON_NAME = 'HD 176694'
SUN_CATALOG_NAME = 'Sun'

# Published seismology for comparison (Yu et al. 2018, VizieR J/ApJS/236/42).
PUBLISHED_NUMAX_MICROHZ = 74.75
PUBLISHED_DNU_MICROHZ = 6.993
PUBLISHED_TEFF_K = 5046.0
PUBLISHED_MASS_SUN = 1.59
PUBLISHED_RADIUS_SUN = 8.38

# Solar reference values for the scaling relations (Huber et al. 2011).
SOLAR_NUMAX_MICROHZ = 3090.0
SOLAR_DNU_MICROHZ = 135.1
SOLAR_TEFF_K = 5777.0

# Granulation and shot noise are fitted either side of the oscillations, so the
# envelope itself never informs its own background.
# Continuum anchor bands, chosen to straddle but never enter the oscillations.
BACKGROUND_BANDS = (
    (2.0, 6.0),
    (6.0, 12.0),
    (12.0, 20.0),
    (20.0, 30.0),
    (30.0, 40.0),
    (115.0, 140.0),
    (140.0, 180.0),
    (180.0, 230.0),
    (230.0, 283.0),
)
ENVELOPE_BAND = (20.0, 160.0)
ENVELOPE_SMOOTH_MICROHZ = 4.0

WOBBLE_WINDOW_DAYS = 18.0
SPECTRUM_VIEW_WIDE = (0.0, 160.0)
SPECTRUM_VIEW_ZOOM = (30.0, 130.0)
FOLD_ORDERS = 4.0
DISPLAY_BIN_MICROHZ = 0.02

# The photosphere is shown wobbling in brightness — the measured quantity — at an
# exaggeration that makes a few hundred ppm visible at all.
BRIGHTNESS_EXAGGERATION = 150.0
STAR_DISPLAY_RESOLUTION = 512
STAR_DISK_RADIUS = 0.76
STAR_PANEL_HALF_WIDTH = 0.50

ACT_BOUNDARIES = ((140, 'wobble'), (200, 'transform'), (280, 'envelope'), (340, 'fold'))


@dataclass(frozen=True)
class SeismicSolution:
    """Everything the film measures, derived from the committed light curve."""

    frequencyMicroHz: np.ndarray
    power: np.ndarray
    background: np.ndarray
    corrected: np.ndarray
    smoothed: np.ndarray
    numaxMicroHz: float
    dnuMicroHz: float
    radiusSun: float
    massSun: float


def loadKeplerLightCurve(
    csvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> tuple[np.ndarray, np.ndarray]:
    """Observed Kepler time (BKJD) and relative flux in ppm."""
    frame = pd.read_csv(csvPath, comment='#')
    return (
        np.asarray(frame['bkjd_day'], dtype=float),
        np.asarray(frame['flux_ppm'], dtype=float),
    )


def powerSpectrum(time: np.ndarray, fluxPpm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Amplitude-squared spectrum in microhertz, from a plain numpy FFT.

    Kepler's quarter gaps are filled with zeros on a regular cadence grid; at a
    duty cycle above 90% that costs a little leakage and saves a dependency.
    """
    cadenceDays = float(np.median(np.diff(time)))
    grid = np.arange(time.min(), time.max() + cadenceDays, cadenceDays)
    filled = np.zeros(len(grid), dtype=float)
    index = np.clip(np.round((time - time.min()) / cadenceDays).astype(int), 0, len(grid) - 1)
    filled[index] = fluxPpm
    spectrum = np.fft.rfft(filled * np.hanning(len(filled)))
    frequency = np.fft.rfftfreq(len(filled), d=cadenceDays * 86400.0) * 1e6
    return frequency, np.abs(spectrum) ** 2


def fitBackground(frequency: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Granulation and shot noise, anchored on medians outside the modes.

    A single power law cannot do this: granulation is flat at low frequency and
    then falls steeply, so a law fitted below the modes stays far too high above
    them. Instead the continuum is interpolated in log-log through the median
    power of bands that skip the oscillation envelope entirely, and the median
    keeps stray peaks inside those bands from lifting it.
    """
    anchorFrequency = []
    anchorPower = []
    for low, high in BACKGROUND_BANDS:
        band = (frequency > low) & (frequency < high)
        if band.any():
            anchorFrequency.append(float(np.median(frequency[band])))
            anchorPower.append(float(np.median(power[band])))
    return 10.0 ** np.interp(
        np.log10(np.maximum(frequency, 1e-6)),
        np.log10(anchorFrequency),
        np.log10(anchorPower),
    )


def smoothSpectrum(
    frequency: np.ndarray, power: np.ndarray, widthMicroHz: float = ENVELOPE_SMOOTH_MICROHZ
) -> np.ndarray:
    resolution = float(frequency[1] - frequency[0])
    window = max(int(round(widthMicroHz / resolution)), 1)
    return np.convolve(power, np.ones(window) / window, mode='same')


def measureNumax(frequency: np.ndarray, smoothed: np.ndarray) -> float:
    """Frequency of maximum oscillation power, from the smoothed envelope."""
    band = (frequency > ENVELOPE_BAND[0]) & (frequency < ENVELOPE_BAND[1])
    return float(frequency[band][np.argmax(smoothed[band])])


def measureDeltaNu(frequency: np.ndarray, corrected: np.ndarray, numaxMicroHz: float) -> float:
    """Mean spacing between overtones, from the spectrum's autocorrelation."""
    envelope = np.abs(frequency - numaxMicroHz) < 3.5 * 0.1 * numaxMicroHz
    segment = corrected[envelope] - corrected[envelope].mean()
    correlation = np.correlate(segment, segment, mode='full')[len(segment) - 1 :]
    lag = np.arange(len(correlation)) * float(frequency[1] - frequency[0])
    # Dnu scales roughly as numax^0.77; this window brackets that without
    # assuming the published answer.
    window = (lag > 0.04 * numaxMicroHz) & (lag < 0.16 * numaxMicroHz)
    return float(lag[window][np.argmax(correlation[window])])


def seismicRadiusAndMass(
    numaxMicroHz: float, dnuMicroHz: float, teffK: float = PUBLISHED_TEFF_K
) -> tuple[float, float]:
    """Scaling relations: two frequencies and a temperature give R and M."""
    numaxRatio = numaxMicroHz / SOLAR_NUMAX_MICROHZ
    dnuRatio = dnuMicroHz / SOLAR_DNU_MICROHZ
    teffRatio = teffK / SOLAR_TEFF_K
    radius = numaxRatio * dnuRatio**-2 * teffRatio**0.5
    mass = numaxRatio**3 * dnuRatio**-4 * teffRatio**1.5
    return float(radius), float(mass)


def solveSeismology(time: np.ndarray, fluxPpm: np.ndarray) -> SeismicSolution:
    """Run the whole measurement chain on an observed light curve."""
    frequency, power = powerSpectrum(time, fluxPpm)
    background = fitBackground(frequency, power)
    corrected = power / background
    smoothed = smoothSpectrum(frequency, corrected)
    numax = measureNumax(frequency, smoothed)
    dnu = measureDeltaNu(frequency, corrected, numax)
    radius, mass = seismicRadiusAndMass(numax, dnu)
    return SeismicSolution(
        frequencyMicroHz=frequency,
        power=power,
        background=background,
        corrected=corrected,
        smoothed=smoothed,
        numaxMicroHz=numax,
        dnuMicroHz=dnu,
        radiusSun=radius,
        massSun=mass,
    )


def smoothStep(value: float) -> float:
    clamped = float(np.clip(value, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


class AsteroseismologyCinematicAnimator:
    """Wobble → transform → envelope → fold → the star's size and mass."""

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

        self.time, self.fluxPpm = loadKeplerLightCurve(lightcurveCsvPath)
        self.solution = solveSeismology(self.time, self.fluxPpm)
        self.rmsPpm = float(np.std(self.fluxPpm))

        # Plot-side downsampling. Taking every nth bin would quietly shave the
        # tallest modes, so each display bin keeps the strongest bin it covers.
        frequency = self.solution.frequencyMicroHz
        display = (frequency > SPECTRUM_VIEW_WIDE[0]) & (frequency < SPECTRUM_VIEW_WIDE[1] + 20.0)
        step = max(int(round(DISPLAY_BIN_MICROHZ / float(frequency[1] - frequency[0]))), 1)
        trimmed = (int(display.sum()) // step) * step
        self.displayFrequency = frequency[display][:trimmed].reshape(-1, step).mean(axis=1)
        self.displayPower = self.solution.corrected[display][:trimmed].reshape(-1, step).max(axis=1)
        self.displaySmoothed = (
            self.solution.smoothed[display][:trimmed].reshape(-1, step).mean(axis=1)
        )

        numax, dnu = self.solution.numaxMicroHz, self.solution.dnuMicroHz
        self.foldable = np.abs(self.displayFrequency - numax) < FOLD_ORDERS * dnu
        # The origin of a fold on Dnu is arbitrary; shift it so the strongest
        # ridge sits inside the panel instead of split across both edges.
        strongest = self.displayFrequency[self.foldable][
            int(np.argmax(self.displayPower[self.foldable]))
        ]
        self.foldPhaseOffset = float((0.62 - (strongest % dnu) / dnu) % 1.0)

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
                for name in (GIANT_CATALOG_NAME, SUN_CATALOG_NAME)
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
        for boundary, name in ACT_BOUNDARIES:
            if frame < boundary:
                return name
        return 'payoff'

    def transformProgress(self, frame: int) -> float:
        """0 while the strip is a time series, 1 once it is a spectrum."""
        return smoothStep((frame - ACT_BOUNDARIES[0][0]) / 60.0)

    def zoomProgress(self, frame: int) -> float:
        return smoothStep((frame - ACT_BOUNDARIES[1][0]) / 80.0)

    def foldProgress(self, frame: int) -> float:
        """0 while the spectrum is linear in frequency, 1 once wrapped on Dnu."""
        return smoothStep((frame - ACT_BOUNDARIES[2][0]) / 60.0)

    def payoffProgress(self, frame: int) -> float:
        return smoothStep((frame - ACT_BOUNDARIES[3][0]) / 60.0)

    def wobbleWindow(self, frame: int) -> tuple[float, float]:
        """Sliding window of real photometry shown during the first act."""
        span = float(self.time.max() - self.time.min()) - WOBBLE_WINDOW_DAYS
        fraction = min(frame, ACT_BOUNDARIES[0][0]) / ACT_BOUNDARIES[0][0]
        # Only creep along the four-year series; the wobble is the subject.
        start = float(self.time.min()) + fraction * min(span, 60.0)
        return start, start + WOBBLE_WINDOW_DAYS

    def spectrumView(self, frame: int) -> tuple[float, float]:
        zoom = self.zoomProgress(frame)
        low = SPECTRUM_VIEW_WIDE[0] + (SPECTRUM_VIEW_ZOOM[0] - SPECTRUM_VIEW_WIDE[0]) * zoom
        high = SPECTRUM_VIEW_WIDE[1] + (SPECTRUM_VIEW_ZOOM[1] - SPECTRUM_VIEW_WIDE[1]) * zoom
        return low, high

    def brightnessFactor(self, frame: int) -> float:
        """Photosphere brightness, driven by the observed flux at this frame."""
        index = int(frame / self.animationFrames * (len(self.fluxPpm) - 1))
        return 1.0 + BRIGHTNESS_EXAGGERATION * self.fluxPpm[index] * 1e-6

    # ---- star panel ------------------------------------------------------

    def _diskFraction(self, catalogName: str) -> float:
        if catalogName not in self._diskFractions:
            sprite = self.atlas.bodyFrame(catalogName, 0, resolution=STAR_DISPLAY_RESOLUTION)
            self._diskFractions[catalogName] = 1.0 if sprite is None else diskRadiusFraction(sprite)
        return self._diskFractions[catalogName]

    def _sunComparison(self, frame: int) -> np.ndarray | None:
        """The Sun at the same scale, once the star has been measured."""
        reveal = self.payoffProgress(frame)
        if reveal <= 0.0:
            return None
        sun = self.atlas.bodyFrame(SUN_CATALOG_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION)
        if sun is None:
            return None
        scaled = sun.copy()
        scaled[..., 3] *= reveal
        return scaled

    def _drawStarPanel(self, frame: int) -> None:
        half = STAR_PANEL_HALF_WIDTH
        self.starAxes.set_xlim(-half, half)
        self.starAxes.set_ylim(-half, half)
        self.starAxes.set_aspect('equal')
        self.starAxes.axis('off')

        star = self.atlas.bodyFrame(GIANT_CATALOG_NAME, frame, resolution=STAR_DISPLAY_RESOLUTION)
        if star is None:
            self.starAxes.add_patch(plt.Circle((0, 0), STAR_DISK_RADIUS, color='#F0A050'))
        else:
            ringing = star.copy()
            ringing[..., :3] = np.clip(ringing[..., :3] * self.brightnessFactor(frame), 0.0, 1.0)
            extent = [-STAR_DISK_RADIUS, STAR_DISK_RADIUS, -STAR_DISK_RADIUS, STAR_DISK_RADIUS]
            self.starAxes.imshow(
                ringing, extent=extent, origin='upper', interpolation='bilinear', zorder=3
            )

        sun = self._sunComparison(frame)
        if sun is not None:
            # True relative size, using the radius this film just measured. Both
            # sprites carry transparent margin, so scale by the visible disks.
            giantVisible = STAR_DISK_RADIUS * self._diskFraction(GIANT_CATALOG_NAME)
            sunVisible = giantVisible / self.solution.radiusSun
            sunExtent = sunVisible / self._diskFraction(SUN_CATALOG_NAME)
            centerX = -half + sunExtent + 0.02
            centerY = -half + sunExtent + 0.02
            self.starAxes.imshow(
                sun,
                extent=[
                    centerX - sunExtent,
                    centerX + sunExtent,
                    centerY - sunExtent,
                    centerY + sunExtent,
                ],
                origin='upper',
                interpolation='bilinear',
                zorder=4,
            )
            self.starAxes.text(
                centerX,
                centerY + sunVisible + 0.02,
                'the Sun, same scale',
                color=self.labelColor,
                fontsize=7,
                ha='center',
                va='bottom',
                alpha=0.75 * self.payoffProgress(frame),
                zorder=5,
            )

        self.starAxes.set_title(
            f'{GIANT_COMMON_NAME} — a red giant weighed by its own ringing',
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
            f'{GIANT_CATALOG_NAME} · brightness wobble ×{BRIGHTNESS_EXAGGERATION:.0f}',
            transform=self.starAxes.transAxes,
            color=self.labelColor,
            fontsize=8,
            ha='right',
            va='top',
            alpha=0.55,
        )

    # ---- captions --------------------------------------------------------

    def caption(self, frame: int) -> str:
        act = self.act(frame)
        if act == 'wobble':
            return (
                f'Kepler long cadence · the surface heaves by {self.rmsPpm:.0f} ppm · '
                'is that noise?'
            )
        if act == 'transform':
            years = (self.time.max() - self.time.min()) / 365.25
            return f'Fourier transform · {years:.1f} years of flux → power at every frequency'
        if act == 'envelope':
            return (
                f'Not noise · a hump of pure tones at νmax = {self.solution.numaxMicroHz:.1f} µHz '
                f'(published {PUBLISHED_NUMAX_MICROHZ:.1f})'
            )
        if act == 'fold':
            return (
                f'Fold the spectrum every Δν = {self.solution.dnuMicroHz:.2f} µHz '
                f'(published {PUBLISHED_DNU_MICROHZ:.2f}) — the same trick that found the planet'
            )
        return (
            f'Two frequencies give the star · R = {self.solution.radiusSun:.1f} R☉, '
            f'M = {self.solution.massSun:.1f} M☉ (published {PUBLISHED_RADIUS_SUN:.1f}, '
            f'{PUBLISHED_MASS_SUN:.1f})'
        )

    # ---- data panel ------------------------------------------------------

    def spectrumPositions(self, frame: int) -> tuple[np.ndarray, np.ndarray]:
        """Normalized x and alpha for every spectrum point at this frame."""
        low, high = self.spectrumView(frame)
        linearX = (self.displayFrequency - low) / (high - low)
        fold = self.foldProgress(frame)
        dnu = self.solution.dnuMicroHz
        foldedX = ((self.displayFrequency % dnu) / dnu + self.foldPhaseOffset) % 1.0
        x = linearX + (foldedX - linearX) * fold
        visible = (linearX >= 0.0) & (linearX <= 1.0)
        alpha = np.where(visible, 0.75, 0.0)
        alpha = np.where(self.foldable, 0.75, alpha * (1.0 - fold))
        return x, alpha

    def _drawWobble(self, frame: int, fade: float) -> None:
        start, end = self.wobbleWindow(frame)
        window = (self.time >= start) & (self.time <= end)
        x = (self.time[window] - start) / (end - start)
        self.lcAxes.plot(
            x,
            self.fluxPpm[window],
            '-',
            color=self.pointColor,
            linewidth=0.9,
            alpha=fade,
            zorder=2,
        )
        self.lcAxes.set_ylim(-2600.0, 2600.0)
        self.lcAxes.set_ylabel('Relative flux (ppm)', color=self.labelColor, fontsize=9)
        ticks = np.linspace(0.0, 1.0, 5)
        self.lcAxes.set_xticks(ticks)
        self.lcAxes.set_xticklabels([f'{start + t * (end - start):.0f}' for t in ticks], alpha=fade)
        self.lcAxes.set_xlabel('BKJD (days)', color=self.labelColor, fontsize=9, alpha=fade)

    def _drawSpectrum(self, frame: int, fade: float) -> None:
        x, alpha = self.spectrumPositions(frame)
        fold = self.foldProgress(frame)
        self.lcAxes.vlines(
            x,
            0.0,
            self.displayPower,
            color=self.pointColor,
            linewidth=0.4,
            alpha=alpha * fade,
            zorder=2,
        )
        headroom = float(self.displayPower[self.foldable].max()) * 1.12
        self.lcAxes.set_ylim(0.0, headroom)
        self.lcAxes.set_ylabel('Power / background', color=self.labelColor, fontsize=9, alpha=fade)

        if fold < 1.0:
            low, high = self.spectrumView(frame)
            smoothX = (self.displayFrequency - low) / (high - low)
            inView = (smoothX >= 0.0) & (smoothX <= 1.0)
            envelopeScale = 0.75 * headroom / max(float(self.displaySmoothed[inView].max()), 1e-9)
            self.lcAxes.plot(
                smoothX,
                self.displaySmoothed * envelopeScale,
                color=self.accentColor,
                linewidth=1.6,
                alpha=fade * self.zoomProgress(frame) * (1.0 - fold),
                zorder=4,
            )
            self.lcAxes.text(
                0.015,
                0.93,
                'smoothed envelope (scaled to fit)',
                transform=self.lcAxes.transAxes,
                color=self.accentColor,
                fontsize=7,
                alpha=fade * self.zoomProgress(frame) * (1.0 - fold),
                zorder=5,
            )
            numaxX = (self.solution.numaxMicroHz - low) / (high - low)
            self.lcAxes.axvline(
                numaxX,
                color=self.labelColor,
                linestyle='--',
                linewidth=1.1,
                alpha=fade * self.zoomProgress(frame) * (1.0 - fold),
                zorder=5,
            )
            ticks = np.linspace(0.0, 1.0, 6)
            self.lcAxes.set_xticks(ticks)
            self.lcAxes.set_xticklabels(
                [f'{low + t * (high - low):.0f}' for t in ticks], alpha=fade * (1.0 - fold)
            )
            self.lcAxes.set_xlabel(
                'Frequency (µHz)', color=self.labelColor, fontsize=9, alpha=fade * (1.0 - fold)
            )
        else:
            dnu = self.solution.dnuMicroHz
            values = np.linspace(0.0, dnu, 5)[:-1]
            positions = (values / dnu + self.foldPhaseOffset) % 1.0
            order = np.argsort(positions)
            self.lcAxes.set_xticks(positions[order])
            self.lcAxes.set_xticklabels([f'{value:.1f}' for value in values[order]])
            self.lcAxes.set_xlabel(
                f'Frequency mod Δν = {self.solution.dnuMicroHz:.2f} µHz',
                color=self.labelColor,
                fontsize=9,
            )

    def _drawDataPanel(self, frame: int) -> None:
        transform = self.transformProgress(frame)
        self.lcAxes.set_xlim(0.0, 1.0)
        # Flux in ppm and power-over-background cannot share a y-axis, so the
        # panel dissolves through empty rather than overlaying the two.
        if transform < 0.5:
            self._drawWobble(frame, float(np.clip(1.0 - 2.4 * transform, 0.0, 1.0)))
        else:
            self._drawSpectrum(frame, float(np.clip(2.4 * transform - 1.4, 0.0, 1.0)))
        self.lcAxes.tick_params(colors=self.labelColor, labelsize=7)
        self.lcAxes.text(
            1.0,
            -0.145,
            'Kepler Q0–Q17 · MAST · KIC 7944142 · spectrum, νmax, Δν, R and M derived from this CSV',
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
        self._drawDataPanel(frame)
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


def renderAsteroseismologyCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    lightcurveCsvPath: str | Path = DEFAULT_LIGHTCURVE_CSV,
) -> None:
    outputDirectory = Path('output/animate/kic_7944142/cinematic')
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = outputDirectory / f'kic_7944142_asteroseismology_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = AsteroseismologyCinematicAnimator(
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            lightcurveCsvPath=lightcurveCsvPath,
            requireBlenderBody=True,
        )
        animator.saveGif(str(outputPath))
    print('Asteroseismology cinema completed!')


__all__ = [
    'AsteroseismologyCinematicAnimator',
    'SeismicSolution',
    'fitBackground',
    'loadKeplerLightCurve',
    'measureDeltaNu',
    'measureNumax',
    'powerSpectrum',
    'renderAsteroseismologyCinematicAnimations',
    'seismicRadiusAndMass',
    'smoothSpectrum',
    'solveSeismology',
]
