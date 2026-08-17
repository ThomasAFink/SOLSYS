"""Tests for the solar cycle cinema (#102).

The film claims three things: that the disks carry the groups actually recorded
on those days, that the count has an eleven-year clock in it, and that the spots
march toward the equator as each cycle runs. All three are re-derived here from
the committed CSVs — including the projection, which is checked against the
catalogue's own record of how far each group sat from disk centre.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.solar_cycle_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    BUTTERFLY_LATITUDE_LIMIT_DEG,
    DEFAULT_GROUP_CSV,
    DEFAULT_SUNSPOT_NUMBER_CSV,
    PUBLISHED_MEAN_CYCLE_YEARS,
    SUN_CATALOG_NAME,
    SolarCycleCinematicAnimator,
    cycleLatitudeDrift,
    diskPositions,
    findCycleMinima,
    loadSunspotGroups,
    loadSunspotNumbers,
    monthLabel,
    smoothMonthly,
    solveSolarCycle,
    spotRadius,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SUNSPOT_NUMBERS = REPO_ROOT / DEFAULT_SUNSPOT_NUMBER_CSV
GROUPS = REPO_ROOT / DEFAULT_GROUP_CSV

# Solar minima as timed by SIDC, for the film's own timing to be checked against.
ACCEPTED_MINIMA = (1755.2, 1878.9, 1913.6, 1996.9, 2008.9)


class SunspotRecordTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.year, cls.number = loadSunspotNumbers(SUNSPOT_NUMBERS)
        cls.groups = loadSunspotGroups(GROUPS)

    def test_the_count_is_the_whole_silso_record(self) -> None:
        self.assertLess(self.year.min(), 1749.1)
        self.assertGreater(self.year.max(), 2020.0)
        self.assertTrue(np.all(np.diff(self.year) > 0.0))
        self.assertTrue(np.all(self.number >= 0.0))
        # Version 2 peaks near 400 in 1957 and bottoms out at zero.
        self.assertGreater(self.number.max(), 350.0)
        self.assertEqual(float(self.number.min()), 0.0)
        self.assertAlmostEqual(float(np.median(np.diff(self.year))), 1.0 / 12.0, places=2)

    def test_the_positions_span_the_photographic_era(self) -> None:
        spots = self.groups[self.groups['area_millionths'] > 0]
        self.assertLess(spots['decimalYear'].min(), 1875.0)
        self.assertGreater(spots['decimalYear'].max(), 2019.0)
        self.assertGreater(len(spots), 8000)
        self.assertTrue(np.all(np.abs(spots['latitude_deg']) <= 90.0))
        self.assertTrue(np.all(np.abs(spots['meridian_distance_deg']) <= 180.0))
        # Spotless days are kept, so a quiet Sun is not the same as no record.
        blank = self.groups[self.groups['area_millionths'] <= 0]
        self.assertGreater(len(blank), 100)
        self.assertTrue(blank['latitude_deg'].isna().all())

    def test_provenance_headers_are_present(self) -> None:
        counts = ' '.join(
            line for line in SUNSPOT_NUMBERS.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('silso', 'royal observatory of belgium', 'v2.0', 'clette'):
            self.assertIn(token, counts)
        positions = ' '.join(
            line for line in GROUPS.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('mandal', 'vizier', 'carrington', 'rgo'):
            self.assertIn(token, positions)


class ProjectionTests(unittest.TestCase):
    """The disks are only honest if lat/CMD really land where the catalogue says."""

    @classmethod
    def setUpClass(cls) -> None:
        groups = loadSunspotGroups(GROUPS)
        cls.spots = groups[groups['area_millionths'] > 0]

    def _residuals(self, useTilt: bool) -> np.ndarray:
        residuals = []
        for tilt, day in self.spots.groupby('tilt_b0_deg'):
            x, y, _ = diskPositions(
                np.asarray(day['latitude_deg'], dtype=float),
                np.asarray(day['meridian_distance_deg'], dtype=float),
                float(tilt) if useTilt else 0.0,
            )
            residuals.append(np.hypot(x, y) - np.asarray(day['catalog_disk_fraction'], dtype=float))
        return np.abs(np.concatenate(residuals))

    def test_projection_reproduces_the_catalog_disk_distance(self) -> None:
        residual = self._residuals(useTilt=True)
        self.assertLess(float(np.median(residual)), 0.005)
        self.assertLess(float(np.percentile(residual, 90)), 0.02)

    def test_the_solar_tilt_is_what_makes_it_agree(self) -> None:
        """Dropping B0 is a real error, not a rounding detail."""
        withTilt = float(np.median(self._residuals(useTilt=True)))
        withoutTilt = float(np.median(self._residuals(useTilt=False)))
        self.assertGreater(withoutTilt, 5.0 * withTilt)

    def test_groups_are_foreshortened_toward_the_limb(self) -> None:
        _, _, centre = diskPositions([0.0], [0.0], 0.0)
        _, _, limb = diskPositions([0.0], [80.0], 0.0)
        self.assertAlmostEqual(float(centre[0]), 1.0, places=6)
        self.assertLess(float(limb[0]), 0.2)

    def test_spot_radius_follows_the_millionths_definition(self) -> None:
        # An area of A millionths covers 2*pi*R^2*A/1e6, so its circle has
        # radius R*sqrt(2A/1e6): a 500 msh group is ~3% of a solar radius.
        self.assertAlmostEqual(float(spotRadius(np.array([500.0]))[0]), 0.0316, places=4)
        self.assertAlmostEqual(float(spotRadius(np.array([0.0]))[0]), 0.0)
        doubled = spotRadius(np.array([200.0, 400.0]))
        self.assertAlmostEqual(float(doubled[1] / doubled[0]), np.sqrt(2.0), places=6)


class CycleMeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.year, cls.number = loadSunspotNumbers(SUNSPOT_NUMBERS)
        cls.groups = loadSunspotGroups(GROUPS)
        cls.solution = solveSolarCycle(cls.year, cls.number, cls.groups)

    def test_smoothing_keeps_the_cycle_and_drops_the_month_to_month_noise(self) -> None:
        smoothed = smoothMonthly(self.number)
        self.assertEqual(len(smoothed), len(self.number))
        self.assertLess(float(np.std(np.diff(smoothed))), 0.3 * float(np.std(np.diff(self.number))))
        self.assertAlmostEqual(float(smoothed.mean()), float(self.number.mean()), delta=1.0)

    def test_minima_land_on_the_accepted_epochs(self) -> None:
        minima = findCycleMinima(self.year, smoothMonthly(self.number))
        self.assertGreaterEqual(len(minima), 24)
        for accepted in ACCEPTED_MINIMA:
            self.assertLess(
                float(np.min(np.abs(minima - accepted))),
                1.0,
                f'no measured minimum within a year of {accepted}',
            )
        self.assertTrue(np.all(np.diff(minima) > 7.0))

    def test_the_clock_runs_at_eleven_years_and_keeps_bad_time(self) -> None:
        self.assertAlmostEqual(self.solution.meanCycleYears, PUBLISHED_MEAN_CYCLE_YEARS, delta=0.5)
        # The film's own claim: the mean is eleven, the individual cycles are not.
        self.assertLess(self.solution.shortestCycleYears, 10.0)
        self.assertGreater(self.solution.longestCycleYears, 12.0)

    def test_spots_march_toward_the_equator(self) -> None:
        opening, closing = (
            self.solution.openingLatitudeDeg,
            self.solution.closingLatitudeDeg,
        )
        self.assertGreater(opening, closing + 8.0)
        self.assertTrue(15.0 < opening < 30.0, opening)
        self.assertTrue(5.0 < closing < 15.0, closing)
        self.assertLess(opening, BUTTERFLY_LATITUDE_LIMIT_DEG)

    def test_the_drift_is_not_an_artefact_of_averaging_cycles_together(self) -> None:
        """Shuffling which cycle a group belongs to must destroy the trend."""
        spots = self.groups[self.groups['area_millionths'] > 0]
        latitude = np.asarray(spots['latitude_deg'], dtype=float)
        rotated = np.roll(np.asarray(spots['decimalYear'], dtype=float), len(latitude) // 3)
        opening, closing = cycleLatitudeDrift(self.solution.minimaYear, rotated, latitude)
        self.assertLess(abs(opening - closing), 4.0)

    def test_the_featured_cycle_is_the_last_one_the_positions_cover(self) -> None:
        spots = self.groups[self.groups['area_millionths'] > 0]
        self.assertEqual(self.solution.featuredCycleNumber, 23)
        self.assertLessEqual(self.solution.featuredEndYear, float(spots['decimalYear'].max()))
        self.assertAlmostEqual(self.solution.featuredStartYear, 1996.6, delta=0.5)
        self.assertAlmostEqual(self.solution.featuredPeakNumber, 244.3, delta=0.1)
        self.assertEqual(monthLabel(self.solution.featuredPeakYear), 'July 2000')

    def test_month_labels_read_as_calendar_months(self) -> None:
        self.assertEqual(monthLabel(1749.042), 'January 1749')
        self.assertEqual(monthLabel(2019.958), 'December 2019')


class SolarCycleAnimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from animate.blender_body_sprites import spinLoopAvailable

        self.hasSpin = spinLoopAvailable(SUN_CATALOG_NAME, 'dark')

    def _animator(self) -> SolarCycleCinematicAnimator:
        return SolarCycleCinematicAnimator(
            style='dark_background',
            sunspotNumberCsvPath=SUNSPOT_NUMBERS,
            groupCsvPath=GROUPS,
            requireBlenderBody=True,
        )

    def _close(self, animator: SolarCycleCinematicAnimator) -> None:
        import matplotlib.pyplot as plt

        plt.close(animator.figure)

    def test_every_act_boundary_draws(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
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
                ['maximum', 'century', 'butterfly', 'payoff'],
            )
            self.assertEqual(animator.act(0), 'minimum')
        finally:
            self._close(animator)

    def test_progress_ramps_are_monotonic_and_complete(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            for ramp in (
                animator.zoomProgress,
                animator.openProgress,
                animator.countCollapse,
                animator.wingSpread,
                animator.payoffProgress,
            ):
                values = [ramp(frame) for frame in range(ANIMATION_FRAMES)]
                self.assertEqual(values, sorted(values))
                self.assertEqual(values[0], 0.0)
                self.assertEqual(values[-1], 1.0)
        finally:
            self._close(animator)

    def test_the_count_flattens_before_the_wings_open(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            opening = next(
                frame for frame in range(ANIMATION_FRAMES) if animator.wingSpread(frame) > 0.05
            )
            self.assertGreater(animator.countCollapse(opening), 0.6)
            # And the strip has room for both halves of the butterfly by then.
            animator.update(ANIMATION_FRAMES - 1)
            bottom, top = animator.stripAxes.get_ylim()
            self.assertLess(bottom, -0.9)
            self.assertGreater(top, 0.9)
        finally:
            self._close(animator)

    def test_the_disk_follows_one_cycle_from_minimum_to_minimum(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            years = [animator.diskYear(frame) for frame in range(ANIMATION_FRAMES)]
            self.assertEqual(years, sorted(years))
            self.assertAlmostEqual(years[0], animator.solution.featuredStartYear, delta=0.2)
            self.assertAlmostEqual(years[-1], animator.solution.featuredEndYear, delta=0.2)
            self.assertGreater(len(animator.diskDays), 120)
        finally:
            self._close(animator)

    def test_the_disk_is_crowded_at_maximum_and_bare_at_the_ends(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            counts = [len(animator.visibleSpots(frame)[0]) for frame in range(ANIMATION_FRAMES)]
            peakFrame = int(
                np.argmin(
                    [
                        abs(animator.diskYear(frame) - animator.solution.featuredPeakYear)
                        for frame in range(ANIMATION_FRAMES)
                    ]
                )
            )
            around = slice(max(peakFrame - 30, 0), peakFrame + 30)
            self.assertGreater(np.mean(counts[around]), 2.0 * np.mean(counts[:30]))
            self.assertGreater(np.mean(counts[around]), 2.0 * np.mean(counts[-30:]))
        finally:
            self._close(animator)

    def test_no_group_is_drawn_off_the_disk(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            for frame in range(0, ANIMATION_FRAMES, 7):
                x, y, mu, radius = animator.visibleSpots(frame)
                self.assertTrue(np.all(np.hypot(x, y) < 1.0))
                self.assertTrue(np.all((mu > 0.0) & (mu <= 1.0)))
                self.assertTrue(np.all(radius > 0.0))
        finally:
            self._close(animator)

    def test_the_strip_ends_on_the_butterfly_span(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            start = animator.stripView(0)
            self.assertAlmostEqual(start[0], animator.solution.featuredStartYear - 0.4, places=3)
            widest = animator.stripView(ACT_BOUNDARIES[2][0])
            self.assertLess(widest[0], 1750.0)
            end = animator.stripView(ANIMATION_FRAMES - 1)
            self.assertAlmostEqual(end[0], animator.butterflySpan[0] - 2.0, places=3)
            self.assertAlmostEqual(end[1], animator.butterflySpan[1] + 2.0, places=3)
        finally:
            self._close(animator)

    def test_captions_report_measured_values_next_to_published_ones(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            captions = [animator.caption(frame) for frame in range(ANIMATION_FRAMES)]
            joined = ' '.join(captions).lower()
            for banned in ('simulated', 'model', 'artist'):
                self.assertNotIn(banned, joined)
            self.assertIn(f'{animator.solution.meanCycleYears:.1f} yr', joined)
            self.assertIn('accepted', joined)
            self.assertIn('quoted', joined)
            self.assertIn(f'{animator.solution.openingLatitudeDeg:.0f}°', joined)
            self.assertIn(f'{animator.solution.featuredPeakNumber:.0f}', joined)
        finally:
            self._close(animator)

    def test_the_elapsed_caption_tracks_the_disk_on_screen(self) -> None:
        """The elapsed interval has to come from the date shown, not from prose."""
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            maximumFrames = [
                frame for frame in range(ANIMATION_FRAMES) if animator.act(frame) == 'maximum'
            ]
            for frame in (maximumFrames[0], maximumFrames[-1]):
                elapsed = animator.diskYear(frame) - animator.solution.featuredStartYear
                self.assertIn(f'{elapsed:.1f} years on', animator.caption(frame))
            # The interval really does move across the act, so it cannot be text.
            self.assertNotEqual(
                animator.caption(maximumFrames[0]), animator.caption(maximumFrames[-1])
            )
        finally:
            self._close(animator)

    def test_framing_is_fixed_across_the_film(self) -> None:
        if not self.hasSpin:
            self.skipTest('Sun Blender spin not present')
        animator = self._animator()
        try:
            animator.update(0)
            start = animator.sunAxes.get_xlim()
            animator.update(ANIMATION_FRAMES - 1)
            self.assertEqual(start, animator.sunAxes.get_xlim())
            self.assertEqual(animator.stripAxes.get_xlim(), (0.0, 1.0))
        finally:
            self._close(animator)

    def test_missing_pack_names_what_to_render(self) -> None:
        if not self.hasSpin:
            with self.assertRaises(FileNotFoundError):
                self._animator()
            return
        from unittest.mock import patch

        from animate.blender_body_sprites import BlenderBodySpriteAtlas

        with (
            patch.object(BlenderBodySpriteAtlas, 'hasBody', lambda _self, _name: False),
            self.assertRaises(FileNotFoundError) as raised,
        ):
            self._animator()
        self.assertIn(SUN_CATALOG_NAME, str(raised.exception))


if __name__ == '__main__':
    unittest.main()
