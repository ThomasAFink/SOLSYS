"""Tests for the TRAPPIST-1 b transit cinema (#95).

The film's claim is that a single TESS transit is invisible and only the fold
reveals it, so the tests check that claim against the committed photometry.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.transit_cinematic import (
    ANIMATION_FRAMES,
    DEFAULT_LIGHTCURVE_CSV,
    FOLD_FRAMES,
    FOLD_HALF_WINDOW_DAYS,
    PUBLISHED_DEPTH,
    PUBLISHED_DURATION_DAYS,
    PUBLISHED_PERIOD_DAYS,
    STREAM_FRAMES,
    TESS_MID_TRANSIT_BTJD,
    TESS_PERIOD_DAYS,
    TRANSITING_PLANET_NAME,
    TRAPPIST_1_CATALOG_NAME,
    TransitCinematicAnimator,
    binSeries,
    diskRadiusFraction,
    foldedProfile,
    loadTessLightCurve,
    observedTransitTimes,
    transitPhase,
    warpedTimeByFrame,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LIGHTCURVE = REPO_ROOT / DEFAULT_LIGHTCURVE_CSV


class TessLightCurveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.time, self.flux = loadTessLightCurve(LIGHTCURVE)

    def test_committed_curve_is_a_real_tess_sector(self) -> None:
        self.assertGreater(len(self.time), 10_000)
        self.assertTrue(np.all(np.diff(self.time) > 0.0))
        self.assertAlmostEqual(self.time.max() - self.time.min(), 23.8, delta=0.5)
        self.assertAlmostEqual(float(np.median(self.flux)), 1.0, places=3)
        self.assertTrue(np.all(np.isfinite(self.flux)))
        # ~2-minute cadence, with the mid-sector downlink gap left in.
        self.assertAlmostEqual(float(np.median(np.diff(self.time))) * 1440.0, 2.0, delta=0.1)
        self.assertGreater(float(np.max(np.diff(self.time))), 1.0)

    def test_provenance_header_is_present(self) -> None:
        header = [line for line in LIGHTCURVE.read_text().splitlines() if line.startswith('#')]
        joined = ' '.join(header).lower()
        for token in ('tess', 'mast', 'lightkurve', 'pdcsap', 'sector'):
            self.assertIn(token, joined)

    def test_scatter_swamps_a_single_transit(self) -> None:
        """The premise of the film: no single transit reaches a detection."""
        scatter = float(np.std(np.diff(self.flux)) / np.sqrt(2.0))
        self.assertGreater(scatter, PUBLISHED_DEPTH)
        significances = []
        for mid in observedTransitTimes(self.time):
            inside = np.abs(self.time - mid) < PUBLISHED_DURATION_DAYS * 0.4
            depth = 1.0 - float(self.flux[inside].mean())
            significances.append(depth / (scatter / np.sqrt(inside.sum())))
        # Individually they are marginal hints at best — none clears 5 sigma.
        self.assertLess(float(np.median(significances)), 3.0)
        self.assertLess(max(significances), 5.0)
        stacked = foldedProfile(self.time, self.flux)
        self.assertGreater(stacked.depth / stacked.depthError, max(significances) + 2.0)

    def test_transit_times_are_periodic_and_covered(self) -> None:
        mids = observedTransitTimes(self.time)
        self.assertGreaterEqual(len(mids), 10)
        spacing = np.diff(np.array(mids))
        # Data gaps drop transits, so spacings are whole multiples of the period.
        multiples = spacing / TESS_PERIOD_DAYS
        np.testing.assert_allclose(multiples, np.round(multiples), atol=1e-9)
        for mid in mids:
            self.assertTrue(np.any(np.abs(self.time - mid) < PUBLISHED_DURATION_DAYS * 0.5))

    def test_measured_period_agrees_with_the_published_one(self) -> None:
        self.assertAlmostEqual(TESS_PERIOD_DAYS, PUBLISHED_PERIOD_DAYS, delta=0.001)

    def test_folding_reveals_the_dip_that_one_transit_hides(self) -> None:
        profile = foldedProfile(self.time, self.flux)
        self.assertGreaterEqual(profile.transitCount, 10)
        self.assertGreater(profile.depth / profile.depthError, 5.0)
        # Deeper than the geometric (Rp/R*)^2 because the star is limb darkened,
        # but not by an implausible factor.
        self.assertGreater(profile.depth, PUBLISHED_DEPTH)
        self.assertLess(profile.depth, PUBLISHED_DEPTH * 2.0)
        deepest = float(profile.phaseHours[np.argmin(profile.flux)])
        self.assertLess(abs(deepest), PUBLISHED_DURATION_DAYS * 24.0 * 0.5)

    def test_no_dip_at_a_control_phase(self) -> None:
        """Folding half a period off must not produce a transit-like dip."""
        offset = foldedProfile(
            self.time, self.flux, midTransitBtjd=TESS_MID_TRANSIT_BTJD + TESS_PERIOD_DAYS * 0.5
        )
        self.assertLess(offset.depth, PUBLISHED_DEPTH)
        real = foldedProfile(self.time, self.flux)
        self.assertGreater(real.depth, offset.depth * 3.0)

    def test_phase_is_signed_distance_to_nearest_mid_transit(self) -> None:
        mid = observedTransitTimes(self.time)[3]
        samples = np.array([mid - 0.01, mid, mid + 0.01])
        np.testing.assert_allclose(transitPhase(samples), [-0.01, 0.0, 0.01], atol=1e-6)
        self.assertTrue(np.all(np.abs(transitPhase(self.time)) <= TESS_PERIOD_DAYS * 0.5 + 1e-9))

    def test_binning_averages_down_the_noise(self) -> None:
        binnedTime, binnedFlux = binSeries(self.time, self.flux, 10.0)
        self.assertLess(len(binnedFlux), len(self.flux))
        self.assertAlmostEqual(float(binnedFlux.mean()), float(self.flux.mean()), places=3)
        rawScatter = float(np.std(np.diff(self.flux)) / np.sqrt(2.0))
        binnedScatter = float(np.std(np.diff(binnedFlux)) / np.sqrt(2.0))
        self.assertLess(binnedScatter, rawScatter)
        # Empty bins across the downlink gap are dropped, not returned as zeros.
        self.assertTrue(np.all(np.isfinite(binnedFlux)))
        self.assertTrue(np.all(np.diff(binnedTime) > 0.0))

    def test_playhead_lingers_on_transits(self) -> None:
        from animate.scenes.transit_cinematic import transitWeight

        mids = observedTransitTimes(self.time)
        warped = warpedTimeByFrame(self.time, transitWeight(self.time, mids), 300)
        self.assertTrue(np.all(np.diff(warped) >= 0.0))
        self.assertAlmostEqual(float(warped[0]), float(self.time.min()), places=6)
        self.assertAlmostEqual(float(warped[-1]), float(self.time.max()), places=6)
        onTransit = sum(
            any(abs(float(time) - mid) < PUBLISHED_DURATION_DAYS * 0.5 for mid in mids)
            for time in warped
        )
        self.assertGreater(onTransit / 300, 0.15)
        self.assertLess(onTransit / 300, 0.75)

    def test_disk_radius_fraction_measures_sprite_margin(self) -> None:
        size = 64
        sprite = np.zeros((size, size, 4), dtype=np.float32)
        yy, xx = np.mgrid[0:size, 0:size]
        center = (size - 1) * 0.5
        sprite[..., 3] = (np.sqrt((xx - center) ** 2 + (yy - center) ** 2) <= 16.0).astype(float)
        self.assertAlmostEqual(diskRadiusFraction(sprite), 0.5, delta=0.02)


class TransitCinematicAnimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from animate.blender_body_sprites import spinLoopAvailable

        self.hasSpin = spinLoopAvailable(TRAPPIST_1_CATALOG_NAME, 'dark') and spinLoopAvailable(
            TRANSITING_PLANET_NAME, 'dark'
        )

    def _animator(self) -> TransitCinematicAnimator:
        return TransitCinematicAnimator(
            style='dark_background',
            lightcurveCsvPath=LIGHTCURVE,
            requireBlenderBody=True,
        )

    def _close(self, animator: TransitCinematicAnimator) -> None:
        import matplotlib.pyplot as plt

        plt.close(animator.figure)

    def test_acts_run_stream_then_fold_then_reveal(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            progress = [animator.foldProgress(frame) for frame in range(ANIMATION_FRAMES)]
            self.assertEqual(progress[STREAM_FRAMES - 1], 0.0)
            self.assertEqual(progress[STREAM_FRAMES + FOLD_FRAMES], 1.0)
            self.assertEqual(progress[-1], 1.0)
            self.assertEqual(progress, sorted(progress))
            # Both acts put the planet on the disk.
            self.assertTrue(any(animator.inTransit(f) for f in range(STREAM_FRAMES)))
            self.assertTrue(
                any(
                    animator.inTransit(f)
                    for f in range(STREAM_FRAMES + FOLD_FRAMES, ANIMATION_FRAMES)
                )
            )
        finally:
            self._close(animator)

    def test_every_frame_including_act_boundaries_draws(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            boundaries = (
                0,
                STREAM_FRAMES - 1,
                STREAM_FRAMES,
                STREAM_FRAMES + FOLD_FRAMES - 1,
                STREAM_FRAMES + FOLD_FRAMES,
                ANIMATION_FRAMES - 1,
            )
            for frame in boundaries:
                animator.update(frame)
                self.assertIsInstance(animator.caption(frame), str)
            self.assertEqual(
                [animator.act(frame) for frame in boundaries],
                ['stream', 'stream', 'fold', 'fold', 'reveal', 'reveal'],
            )
        finally:
            self._close(animator)

    def test_fold_morphs_points_from_time_to_phase(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            streamX, streamAlpha = animator.pointPositions(0)
            foldX, foldAlpha = animator.pointPositions(ANIMATION_FRAMES - 1)
            np.testing.assert_allclose(streamX, animator.streamX)
            np.testing.assert_allclose(foldX, animator.foldX)
            # Everything is visible while streaming; only fold-window points survive.
            self.assertTrue(np.all(streamAlpha > 0.0))
            np.testing.assert_allclose(foldAlpha[~animator.foldable], 0.0)
            self.assertTrue(np.all(foldAlpha[animator.foldable] > 0.0))
            self.assertGreater(animator.foldable.sum(), 200)
            phase = transitPhase(animator.binnedTime)
            np.testing.assert_allclose(animator.foldable, np.abs(phase) <= FOLD_HALF_WINDOW_DAYS)
        finally:
            self._close(animator)

    def test_reveal_sweeps_the_phase_window_and_holds_on_the_dip(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            sweep = animator.revealPhaseByFrame
            self.assertTrue(np.all(np.diff(sweep) >= 0.0))
            self.assertAlmostEqual(float(sweep[0]), -FOLD_HALF_WINDOW_DAYS, places=6)
            self.assertAlmostEqual(float(sweep[-1]), FOLD_HALF_WINDOW_DAYS, places=6)
            inDip = np.mean(np.abs(sweep) <= PUBLISHED_DURATION_DAYS * 0.5)
            # The transit is 0.4% of the phase window; the edit gives it far more.
            self.assertGreater(inDip, 0.3)
        finally:
            self._close(animator)

    def test_transit_frames_darken_the_photosphere(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            transitFrame = next(f for f in range(STREAM_FRAMES) if animator.inTransit(f))
            quietFrame = next(f for f in range(STREAM_FRAMES) if not animator.inTransit(f))
            star = animator.atlas.bodyFrame(TRAPPIST_1_CATALOG_NAME, transitFrame, resolution=512)
            assert star is not None
            onDisk = star[..., 3] > 0.5
            occulted = animator._compositeTransit(star, transitFrame)
            self.assertLess(occulted[onDisk][..., :3].sum(), star[onDisk][..., :3].sum())
            np.testing.assert_allclose(animator._compositeTransit(star, quietFrame), star)
        finally:
            self._close(animator)

    def test_captions_stay_with_observed_data(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            captions = [animator.caption(frame) for frame in range(ANIMATION_FRAMES)]
            joined = ' '.join(captions).lower()
            for banned in ('model', 'dust', 'debris', 'megastructure', 'simulated'):
                self.assertNotIn(banned, joined)
            self.assertTrue(any('observed' in caption for caption in captions))
            self.assertTrue(any('Stack all' in caption for caption in captions))
            self.assertTrue(any('There it is' in caption for caption in captions))
            self.assertTrue(
                any('cannot see it' in caption or 'scatter is' in caption for caption in captions)
            )
        finally:
            self._close(animator)

    def test_framing_is_fixed_across_the_film(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spins not present')
        animator = self._animator()
        try:
            animator.update(0)
            quietLimits = animator.starAxes.get_xlim()
            animator.update(ANIMATION_FRAMES - 1)
            self.assertEqual(quietLimits, animator.starAxes.get_xlim())
            self.assertEqual(animator.lcAxes.get_xlim(), (0.0, 1.0))
        finally:
            self._close(animator)

    def test_missing_packs_name_what_to_render(self) -> None:
        if not self.hasSpin:
            with self.assertRaises(FileNotFoundError):
                self._animator()
            return
        from unittest.mock import patch

        with patch.object(TransitCinematicAnimator, '__init__', TransitCinematicAnimator.__init__):
            from animate.blender_body_sprites import BlenderBodySpriteAtlas

            with (
                patch.object(
                    BlenderBodySpriteAtlas,
                    'hasBody',
                    lambda _self, name: name != TRANSITING_PLANET_NAME,
                ),
                self.assertRaises(FileNotFoundError) as raised,
            ):
                self._animator()
        self.assertIn(TRANSITING_PLANET_NAME, str(raised.exception))


if __name__ == '__main__':
    unittest.main()
