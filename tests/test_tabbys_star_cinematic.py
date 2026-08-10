"""Tests for Tabby's Star lightcurve cinema."""

from __future__ import annotations

import unittest
from pathlib import Path

from animate.scenes.tabbys_star_cinematic import (
    STAR_RADIUS_AU,
    TabbysStarCinematicAnimator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class TabbysStarCinematicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.animator = TabbysStarCinematicAnimator(
            starsCsvPath=str(REPO_ROOT / 'data' / 'nearby_stars_30.csv'),
            lightcurveCsvPath=str(REPO_ROOT / 'data' / 'tabbys_star_lightcurve.csv'),
        )

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.animator.figure)

    def test_loads_kepler_and_clumps(self) -> None:
        self.assertEqual(self.animator.system.systemId, 'tabbys_star')
        self.assertEqual(len(self.animator.fluxByFrame), self.animator.animationFrames)
        self.assertGreaterEqual(len(self.animator.clumps), 2)
        self.assertIsNotNone(self.animator.deepestCrossingFrame)

    def test_no_sol_travel_helpers(self) -> None:
        self.assertFalse(hasattr(self.animator, '_travelProgress'))
        self.assertFalse(hasattr(self.animator, 'hostSolAu'))

    def test_playhead_time_is_monotonic(self) -> None:
        times = [float(self.animator.timeByFrame[frame]) for frame in range(0, 40)]
        self.assertGreater(times[-1], times[0])
        for previous, current in zip(times, times[1:], strict=False):
            self.assertGreaterEqual(current, previous - 1e-9)

    def test_clump_occults_star_on_crossing_frame(self) -> None:
        for clump in self.animator.clumps:
            screenX, _screenY, inFront, occulting = self.animator._clumpScreenState(
                clump, clump.crossingFrame
            )
            self.assertTrue(inFront, msg=clump.crossingFrame)
            self.assertTrue(occulting, msg=clump.crossingFrame)
            self.assertLessEqual(abs(screenX), STAR_RADIUS_AU + clump.sizeAu)

    def test_clump_not_occulting_opposite_side(self) -> None:
        clump = self.animator.clumps[0]
        opposite = (clump.crossingFrame + self.animator.animationFrames // 2) % (
            self.animator.animationFrames
        )
        _x, _y, inFront, occulting = self.animator._clumpScreenState(clump, opposite)
        self.assertFalse(inFront)
        self.assertFalse(occulting)

    def test_sky_pushes_in_on_deepest_dip(self) -> None:
        deep = self.animator.deepestCrossingFrame
        assert deep is not None
        close = self.animator._skyHalfWidth(deep)
        far = (deep + self.animator.animationFrames // 2) % self.animator.animationFrames
        wide = self.animator._skyHalfWidth(far)
        self.assertLess(close, wide)

    def test_caption_mentions_measurement_not_travel(self) -> None:
        title, subtitle = self.animator._caption(0)
        blob = f'{title} {subtitle}'.lower()
        self.assertNotIn('light-year', blob)
        self.assertNotIn('arriving', blob)
        self.assertTrue('kepler' in blob or 'photometry' in blob or 'tabby' in blob)


if __name__ == '__main__':
    unittest.main()
