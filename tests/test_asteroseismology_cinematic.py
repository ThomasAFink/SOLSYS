"""Tests for the red giant asteroseismology cinema (#169).

The film claims a wobble resolves into evenly spaced tones that measure the
star, so the tests re-derive numax, Dnu, radius and mass from the committed
Kepler light curve and check them against the published values.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.asteroseismology_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_LIGHTCURVE_CSV,
    GIANT_CATALOG_NAME,
    PUBLISHED_DNU_MICROHZ,
    PUBLISHED_MASS_SUN,
    PUBLISHED_NUMAX_MICROHZ,
    PUBLISHED_RADIUS_SUN,
    SUN_CATALOG_NAME,
    AsteroseismologyCinematicAnimator,
    fitBackground,
    loadKeplerLightCurve,
    measureDeltaNu,
    measureNumax,
    powerSpectrum,
    seismicRadiusAndMass,
    smoothSpectrum,
    solveSeismology,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTCURVE = REPO_ROOT / DEFAULT_LIGHTCURVE_CSV


class KeplerLightCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time, self.flux = loadKeplerLightCurve(LIGHTCURVE)

    def test_committed_curve_is_the_full_kepler_baseline(self) -> None:
        self.assertGreater(len(self.time), 60_000)
        self.assertTrue(np.all(np.diff(self.time) > 0.0))
        self.assertGreater(self.time.max() - self.time.min(), 1400.0)
        self.assertAlmostEqual(float(np.median(np.diff(self.time))) * 1440.0, 29.4, delta=0.2)
        self.assertTrue(np.all(np.isfinite(self.flux)))
        # A red giant heaves by hundreds of ppm — far above Kepler's precision
        # for a Kp 7.8 star, which is why the wobble is visible at all.
        self.assertGreater(float(np.std(self.flux)), 300.0)
        self.assertLess(float(np.std(self.flux)), 2000.0)

    def test_provenance_header_is_present(self) -> None:
        header = ' '.join(
            line for line in LIGHTCURVE.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('kepler', 'mast', 'lightkurve', 'pdcsap', 'yu et al'):
            self.assertIn(token, header)


class SeismicMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.time, cls.flux = loadKeplerLightCurve(LIGHTCURVE)
        cls.solution = solveSeismology(cls.time, cls.flux)

    def test_spectrum_covers_the_oscillations_and_resolves_them(self) -> None:
        frequency, power = powerSpectrum(self.time, self.flux)
        self.assertGreater(frequency.max(), 250.0)  # Nyquist above the modes
        self.assertLess(float(frequency[1] - frequency[0]), 0.05)  # resolves Dnu
        self.assertTrue(np.all(power >= 0.0))

    def test_background_is_removed_without_eating_the_modes(self) -> None:
        frequency, power = powerSpectrum(self.time, self.flux)
        corrected = power / fitBackground(frequency, power)
        quiet = ((frequency > 150.0) & (frequency < 200.0)) | (
            (frequency > 25.0) & (frequency < 35.0)
        )
        # Away from the oscillations the ratio sits near one in the median.
        self.assertLess(abs(float(np.median(corrected[quiet])) - 1.0), 0.5)
        envelope = np.abs(frequency - PUBLISHED_NUMAX_MICROHZ) < 10.0
        self.assertGreater(float(corrected[envelope].max()), 20.0)

    def test_numax_matches_the_published_value(self) -> None:
        frequency, power = powerSpectrum(self.time, self.flux)
        corrected = power / fitBackground(frequency, power)
        numax = measureNumax(frequency, smoothSpectrum(frequency, corrected))
        # A boxcar-smoothed envelope peak is coarser than a fitted Gaussian, so
        # a few percent from the published centroid is the honest tolerance.
        self.assertAlmostEqual(numax, PUBLISHED_NUMAX_MICROHZ, delta=0.06 * PUBLISHED_NUMAX_MICROHZ)

    def test_dnu_matches_the_published_spacing(self) -> None:
        frequency, power = powerSpectrum(self.time, self.flux)
        corrected = power / fitBackground(frequency, power)
        dnu = measureDeltaNu(frequency, corrected, self.solution.numaxMicroHz)
        self.assertAlmostEqual(dnu, PUBLISHED_DNU_MICROHZ, delta=0.03 * PUBLISHED_DNU_MICROHZ)

    def test_modes_really_are_evenly_spaced(self) -> None:
        """Folding on Dnu must concentrate the power; a wrong spacing must not."""
        frequency, corrected = self.solution.frequencyMicroHz, self.solution.corrected
        dnu = self.solution.dnuMicroHz
        band = np.abs(frequency - self.solution.numaxMicroHz) < 4.0 * dnu

        def concentration(spacing: float) -> float:
            phase = (frequency[band] % spacing) / spacing
            counts, _ = np.histogram(phase, bins=20, weights=corrected[band])
            return float(counts.max() / counts.mean())

        self.assertGreater(concentration(dnu), 1.6)
        for wrong in (dnu * 0.82, dnu * 1.21):
            self.assertLess(concentration(wrong), concentration(dnu))

    def test_scaling_relations_recover_the_published_star(self) -> None:
        radius, mass = seismicRadiusAndMass(self.solution.numaxMicroHz, self.solution.dnuMicroHz)
        self.assertAlmostEqual(radius, PUBLISHED_RADIUS_SUN, delta=0.10 * PUBLISHED_RADIUS_SUN)
        # Mass goes as numax^3 / Dnu^4, so it amplifies the same frequency error.
        self.assertAlmostEqual(mass, PUBLISHED_MASS_SUN, delta=0.20 * PUBLISHED_MASS_SUN)
        self.assertEqual((radius, mass), (self.solution.radiusSun, self.solution.massSun))

    def test_the_sun_comes_back_as_the_sun(self) -> None:
        """Sanity check on the relations themselves, not on this star."""
        radius, mass = seismicRadiusAndMass(3090.0, 135.1, 5777.0)
        self.assertAlmostEqual(radius, 1.0, places=6)
        self.assertAlmostEqual(mass, 1.0, places=6)


class AsteroseismologyAnimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from animate.blender_body_sprites import spinLoopAvailable

        self.hasSpin = spinLoopAvailable(GIANT_CATALOG_NAME, 'dark') and spinLoopAvailable(
            SUN_CATALOG_NAME, 'dark'
        )

    def _animator(self) -> AsteroseismologyCinematicAnimator:
        return AsteroseismologyCinematicAnimator(
            style='dark_background', lightcurveCsvPath=LIGHTCURVE, requireBlenderBody=True
        )

    def _close(self, animator: AsteroseismologyCinematicAnimator) -> None:
        import matplotlib.pyplot as plt

        plt.close(animator.figure)

    def test_every_act_boundary_draws(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            boundaries = [0, ANIMATION_FRAMES - 1]
            for frame, _ in ACT_BOUNDARIES:
                boundaries.extend((frame - 1, frame))
            for frame in sorted(boundaries):
                animator.update(frame)
                self.assertIsInstance(animator.caption(frame), str)
            self.assertEqual(
                [animator.act(frame) for frame, _ in ACT_BOUNDARIES],
                ['transform', 'envelope', 'fold', 'payoff'],
            )
            self.assertEqual(animator.act(0), 'wobble')
        finally:
            self._close(animator)

    def test_progress_ramps_are_monotonic_and_complete(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            for ramp in (
                animator.transformProgress,
                animator.zoomProgress,
                animator.foldProgress,
                animator.payoffProgress,
            ):
                values = [ramp(frame) for frame in range(ANIMATION_FRAMES)]
                self.assertEqual(values, sorted(values))
                self.assertEqual(values[0], 0.0)
                self.assertEqual(values[-1], 1.0)
        finally:
            self._close(animator)

    def test_fold_collapses_the_spectrum_onto_one_spacing(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            linearX, _ = animator.spectrumPositions(ACT_BOUNDARIES[2][0])
            foldedX, foldedAlpha = animator.spectrumPositions(ANIMATION_FRAMES - 1)
            dnu = animator.solution.dnuMicroHz
            expected = ((animator.displayFrequency % dnu) / dnu + animator.foldPhaseOffset) % 1.0
            np.testing.assert_allclose(foldedX, expected)
            self.assertFalse(np.allclose(linearX, foldedX))
            # Only the oscillation envelope survives the fold.
            np.testing.assert_allclose(foldedAlpha[~animator.foldable], 0.0)
            self.assertTrue(np.all(foldedAlpha[animator.foldable] > 0.0))
        finally:
            self._close(animator)

    def test_display_downsampling_keeps_the_tallest_modes(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            frequency = animator.solution.frequencyMicroHz
            envelope = np.abs(frequency - animator.solution.numaxMicroHz) < 10.0
            fullResolutionPeak = float(animator.solution.corrected[envelope].max())
            shown = np.abs(animator.displayFrequency - animator.solution.numaxMicroHz) < 10.0
            self.assertGreater(len(animator.displayFrequency), 2000)
            self.assertLess(len(animator.displayFrequency), len(frequency) // 2)
            self.assertAlmostEqual(
                float(animator.displayPower[shown].max()),
                fullResolutionPeak,
                delta=0.02 * fullResolutionPeak,
            )
        finally:
            self._close(animator)

    def test_photosphere_brightness_follows_the_observed_flux(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            factors = [animator.brightnessFactor(frame) for frame in range(ANIMATION_FRAMES)]
            self.assertGreater(max(factors), 1.0)
            self.assertLess(min(factors), 1.0)
            # Exaggerated for visibility, but nowhere near a different star.
            self.assertLess(max(factors), 1.6)
            self.assertGreater(min(factors), 0.4)
        finally:
            self._close(animator)

    def test_captions_report_measured_values_next_to_published_ones(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            captions = [animator.caption(frame) for frame in range(ANIMATION_FRAMES)]
            joined = ' '.join(captions).lower()
            for banned in ('simulated', 'model', 'artist'):
                self.assertNotIn(banned, joined)
            self.assertTrue(any('published' in caption for caption in captions))
            self.assertTrue(any('νmax' in caption for caption in captions))
            self.assertTrue(any('Δν' in caption for caption in captions))
            self.assertTrue(any('R☉' in caption and 'M☉' in caption for caption in captions))
        finally:
            self._close(animator)

    def test_framing_is_fixed_across_the_film(self) -> None:
        if not self.hasSpin:
            self.skipTest('KIC 7944142 / Sun Blender spins not present')
        animator = self._animator()
        try:
            animator.update(0)
            start = animator.starAxes.get_xlim()
            animator.update(ANIMATION_FRAMES - 1)
            self.assertEqual(start, animator.starAxes.get_xlim())
            self.assertEqual(animator.lcAxes.get_xlim(), (0.0, 1.0))
        finally:
            self._close(animator)

    def test_missing_packs_name_what_to_render(self) -> None:
        if not self.hasSpin:
            with self.assertRaises(FileNotFoundError):
                self._animator()
            return
        from unittest.mock import patch

        from animate.blender_body_sprites import BlenderBodySpriteAtlas

        with (
            patch.object(
                BlenderBodySpriteAtlas, 'hasBody', lambda _self, name: name != GIANT_CATALOG_NAME
            ),
            self.assertRaises(FileNotFoundError) as raised,
        ):
            self._animator()
        self.assertIn(GIANT_CATALOG_NAME, str(raised.exception))


if __name__ == '__main__':
    unittest.main()
