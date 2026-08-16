"""Tests for the TRAPPIST-1 transit lightcurve cinema (#95)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.transit_cinematic import (
    ANIMATION_FRAMES,
    MODEL_SAMPLES,
    TRAPPIST_1_CATALOG_NAME,
    TRAPPIST_1_STAR_RADIUS_KM,
    WINDOW_DAYS,
    TransitCinematicAnimator,
    buildTransitingPlanets,
    diskRadiusFraction,
    modelFlux,
    transitWeight,
    warpedTimeByFrame,
)
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class TransitModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**self.paths).load('trappist_1')
        self.planets = buildTransitingPlanets(self.system)

    def test_every_catalog_planet_transits_in_the_window(self) -> None:
        self.assertEqual(len(self.planets), len(self.system.planets))
        for planet in self.planets:
            self.assertTrue(planet.midTimesDays)
            self.assertTrue(all(0.0 <= mid <= WINDOW_DAYS for mid in planet.midTimesDays))

    def test_dips_are_periodic(self) -> None:
        """The point of the film: repeats are spaced by exactly one period."""
        repeats = [planet for planet in self.planets if len(planet.midTimesDays) > 1]
        self.assertGreaterEqual(len(repeats), 3)
        for planet in repeats:
            spacing = np.diff(np.array(planet.midTimesDays))
            np.testing.assert_allclose(spacing, planet.periodDays, rtol=1e-9)

    def test_depth_is_the_radius_ratio_squared(self) -> None:
        byName = {planet.name: planet for planet in self.planets}
        for catalogPlanet in self.system.planets:
            planet = byName[catalogPlanet.name]
            expected = ((catalogPlanet.diameterKm * 0.5) / TRAPPIST_1_STAR_RADIUS_KM) ** 2
            self.assertAlmostEqual(planet.depth, expected, places=12)
            # TRAPPIST-1 transits are sub-percent — a deeper dip would be wrong.
            self.assertLess(planet.depth, 0.01)
            self.assertGreater(planet.depth, 0.002)

    def test_durations_match_published_scale_and_order(self) -> None:
        byName = {planet.shortName: planet for planet in self.planets}
        # Gillon et al. 2017 durations: b ~36 min through h ~76 min.
        self.assertAlmostEqual(byName['b'].totalDurationDays * 1440.0, 36.0, delta=4.0)
        self.assertAlmostEqual(byName['h'].totalDurationDays * 1440.0, 76.0, delta=8.0)
        durations = [byName[name].totalDurationDays for name in 'bcdefgh']
        self.assertEqual(durations, sorted(durations))
        for planet in self.planets:
            self.assertLess(planet.flatDurationDays, planet.totalDurationDays)

    def test_flux_is_flat_between_transits_and_dips_on_them(self) -> None:
        times = np.linspace(0.0, WINDOW_DAYS, MODEL_SAMPLES)
        flux = modelFlux(self.planets, times)
        self.assertAlmostEqual(float(flux.max()), 1.0, places=12)
        for planet in self.planets:
            mid = planet.midTimesDays[0]
            atMid = float(modelFlux(self.planets, np.array([mid]))[0])
            self.assertLessEqual(atMid, 1.0 - planet.depth + 1e-9)
            justBefore = mid - planet.totalDurationDays
            self.assertAlmostEqual(
                float(modelFlux(self.planets, np.array([justBefore]))[0]), 1.0, places=6
            )

    def test_overlapping_transits_stack_their_depths(self) -> None:
        times = np.linspace(0.0, WINDOW_DAYS, MODEL_SAMPLES)
        flux = modelFlux(self.planets, times)
        deepestTime = float(times[flux.argmin()])
        together = [planet for planet in self.planets if planet.inTransit(deepestTime)]
        self.assertGreaterEqual(len(together), 2)
        self.assertLess(float(flux.min()), 1.0 - max(planet.depth for planet in together))

    def test_playhead_lingers_on_transits(self) -> None:
        times = np.linspace(0.0, WINDOW_DAYS, MODEL_SAMPLES)
        warped = warpedTimeByFrame(times, transitWeight(self.planets, times), ANIMATION_FRAMES)
        self.assertEqual(len(warped), ANIMATION_FRAMES)
        self.assertTrue(np.all(np.diff(warped) >= 0.0))
        self.assertAlmostEqual(float(warped[0]), 0.0, places=6)
        self.assertAlmostEqual(float(warped[-1]), WINDOW_DAYS, places=6)
        onTransit = sum(
            any(planet.inTransit(float(time)) for planet in self.planets) for time in warped
        )
        # Transits are ~5% of the window but must own most of the edit.
        self.assertGreater(onTransit / ANIMATION_FRAMES, 0.5)
        # And the quiet baseline still gets screen time.
        self.assertLess(onTransit / ANIMATION_FRAMES, 0.95)

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

        self.paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**self.paths).load('trappist_1')
        self.hasSpin = spinLoopAvailable(TRAPPIST_1_CATALOG_NAME, 'dark')

    def _animator(self) -> TransitCinematicAnimator:
        return TransitCinematicAnimator(
            self.system,
            style='dark_background',
            starsCsvPath=self.paths['starsCsvPath'],
            requireBlenderBody=True,
        )

    def test_transit_frames_darken_the_photosphere(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spin not present')
        animator = self._animator()
        try:
            transitFrame = next(
                frame for frame in range(ANIMATION_FRAMES) if animator.transitingNow(frame)
            )
            quietFrame = next(
                frame for frame in range(ANIMATION_FRAMES) if not animator.transitingNow(frame)
            )
            star = animator.atlas.bodyFrame(TRAPPIST_1_CATALOG_NAME, transitFrame, resolution=512)
            assert star is not None
            onDisk = star[..., 3] > 0.5
            occulted = animator._compositeTransits(star, transitFrame)
            quiet = animator._compositeTransits(star, quietFrame)
            # A body in front of the star: photosphere pixels lose brightness.
            self.assertLess(occulted[onDisk][..., :3].sum(), star[onDisk][..., :3].sum())
            np.testing.assert_allclose(quiet, star)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_silhouette_tracks_the_chord_across_the_disk(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spin not present')
        animator = self._animator()
        try:
            planet = min(animator.planets, key=lambda item: item.periodDays)
            mid = planet.midTimesDays[0]
            half = planet.totalDurationDays * 0.5
            # Ingress sits on the limb, mid-transit at the chord's impact point.
            self.assertAlmostEqual(planet.phaseOffset(mid), 0.0, places=9)
            self.assertAlmostEqual(planet.phaseOffset(mid - half), -1.0, places=9)
            self.assertTrue(planet.inTransit(mid))
            self.assertFalse(planet.inTransit(mid + half * 1.05))
            self.assertAlmostEqual(
                planet.chordHalfSpan(),
                float(np.sqrt((1.0 + planet.radiusRatio) ** 2 - planet.impactParameter**2)),
                places=12,
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_captions_name_planets_and_never_dust(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spin not present')
        animator = self._animator()
        try:
            captions = [animator._caption(frame) for frame in range(ANIMATION_FRAMES)]
            joined = ' '.join(captions).lower()
            for banned in ('dust', 'debris', 'megastructure', 'comet'):
                self.assertNotIn(banned, joined)
            self.assertTrue(any('in transit' in caption for caption in captions))
            self.assertTrue(any('transiting together' in caption for caption in captions))
            self.assertTrue(any('quiet baseline' in caption.lower() for caption in captions))
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_framing_is_fixed_across_the_film(self) -> None:
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spin not present')
        animator = self._animator()
        try:
            animator.update(0)
            quietLimits = animator.starAxes.get_xlim()
            transitFrame = next(
                frame for frame in range(ANIMATION_FRAMES) if animator.transitingNow(frame)
            )
            animator.update(transitFrame)
            self.assertEqual(quietLimits, animator.starAxes.get_xlim())
            self.assertEqual(animator.lcAxes.get_xlim(), (0.0, WINDOW_DAYS))
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_missing_spin_raises_clear_error(self) -> None:
        if self.hasSpin:
            self.skipTest('spin present — cannot assert missing-pack error')
        with self.assertRaises(FileNotFoundError):
            self._animator()

    def test_every_transiting_planet_pack_is_required(self) -> None:
        """A missing planet pack must fail loudly, not fall back to a plain dot."""
        if not self.hasSpin:
            self.skipTest('TRAPPIST-1 Blender spin not present')
        animator = self._animator()
        try:
            names = [planet.name for planet in animator.planets]
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

        from unittest.mock import patch

        absent = names[-1]
        with (
            patch.object(type(animator.atlas), 'hasBody', lambda _self, name: name != absent),
            self.assertRaises(FileNotFoundError) as raised,
        ):
            self._animator()
        self.assertIn(absent, str(raised.exception))


if __name__ == '__main__':
    unittest.main()
