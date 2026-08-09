"""Tests for Sol → TRAPPIST-1 cinematic."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.sol_centauri_cinematic import (
    BLENDER_PLANET_BODY_SCALE,
    PULLBACK_END,
    SOL_EARTH_CLOSE_HALF_AU,
    SOL_EARTH_HALF_AU,
    SOL_HOLD_END,
)
from animate.scenes.sol_trappist_cinematic import (
    ARRIVAL_TRAPPIST_HOLD_END,
    ARRIVAL_TRAPPIST_INNER_ARRIVE,
    FIELD_STARS_MAX_LY,
    START_HALF_WIDTH_LY,
    TRAPPIST_ARRIVE_HALF_AU,
    TRAPPIST_INNER_HALF_AU,
    TRAPPIST_WIDE_HALF_AU,
    SolTrappistCinematicAnimator,
)
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class SolTrappistCinematicTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = defaultDataPaths(REPO_ROOT)
        self.paths = paths
        self.system = SystemCatalog(**paths).load('trappist_1')
        self.animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
        )

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.animator.figure)

    def test_host_matches_star_catalog_xyz(self) -> None:
        host = self.system.stars[0]
        np.testing.assert_allclose(
            self.animator.hostSolAu,
            [host.positionX, host.positionY, host.positionZ],
            rtol=0,
            atol=0,
        )
        self.assertGreater(self.animator.distanceLy, 40.0)
        self.assertLess(self.animator.distanceLy, 42.0)

    def test_camera_starts_on_earth(self) -> None:
        focus, halfWidth = self.animator._cameraState(0)
        earth = self.animator._planetPositionAu('Earth', 0)
        np.testing.assert_allclose(focus, earth, atol=1e-9)
        self.assertAlmostEqual(halfWidth, SOL_EARTH_HALF_AU, places=6)

    def test_blender_camera_starts_earth_close(self) -> None:
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            focus, halfWidth = animator._cameraState(0)
            earth = animator._planetPositionAu('Earth', 0)
            np.testing.assert_allclose(focus, earth, atol=1e-9)
            self.assertAlmostEqual(halfWidth, SOL_EARTH_CLOSE_HALF_AU, places=6)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_camera_ends_on_trappist_finale(self) -> None:
        last = self.animator.animationFrames - 1
        focus, halfWidth = self.animator._cameraState(last)
        np.testing.assert_allclose(focus, self.animator.hostSolAu, atol=1e-9)
        self.assertAlmostEqual(halfWidth, TRAPPIST_INNER_HALF_AU, places=6)

    def test_arrive_hold_frames_host(self) -> None:
        frame = int(ARRIVAL_TRAPPIST_HOLD_END * (self.animator.animationFrames - 1)) - 2
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            focus, halfWidth = animator._cameraState(frame)
            np.testing.assert_allclose(focus, animator.hostSolAu, atol=1e-6)
            self.assertAlmostEqual(halfWidth, TRAPPIST_ARRIVE_HALF_AU, places=5)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_planets_orbit_host(self) -> None:
        self.assertEqual(len(self.animator.trappistPlanets), 7)
        for planet in self.animator.trappistPlanets:
            position = self.animator._trappistPlanetPositionSol(planet, frame=12)
            offset = position - self.animator.hostSolAu
            self.assertLess(float(np.linalg.norm(offset)), 0.08)
            self.assertAlmostEqual(float(offset[2]), 0.0, places=12)

    def test_field_stars_reach_trappist_distance(self) -> None:
        self.assertGreaterEqual(FIELD_STARS_MAX_LY, 45.0)
        self.assertGreaterEqual(START_HALF_WIDTH_LY, 45.0)
        if not self.animator.fieldStars.empty:
            self.assertFalse((self.animator.fieldStars.get('system_id') == 'trappist_1').any())

    def test_blender_scales_registered(self) -> None:
        for letter in 'bcdefgh':
            name = f'TRAPPIST-1 {letter}'
            self.assertIn(name, BLENDER_PLANET_BODY_SCALE)

    def test_pullback_before_travel(self) -> None:
        frame = int((SOL_HOLD_END + PULLBACK_END) / 2 * (self.animator.animationFrames - 1))
        focus, halfWidth = self.animator._cameraState(frame)
        np.testing.assert_allclose(focus, np.zeros(3), atol=1e-12)
        self.assertGreater(halfWidth, 100.0)

    def test_blender_inner_arrive_uses_wide_then_inner(self) -> None:
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            wideFrame = int(
                ((animator._proximaDiveEnd() + animator._proximaWideHoldEnd()) / 2)
                * (animator.animationFrames - 1)
            )
            _, wideHalf = animator._cameraState(wideFrame)
            self.assertAlmostEqual(wideHalf, TRAPPIST_WIDE_HALF_AU, places=5)

            innerFrame = int(ARRIVAL_TRAPPIST_INNER_ARRIVE * (animator.animationFrames - 1))
            focus, innerHalf = animator._cameraState(innerFrame)
            np.testing.assert_allclose(focus, animator.hostSolAu, atol=1e-9)
            self.assertLessEqual(innerHalf, TRAPPIST_WIDE_HALF_AU + 1e-9)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)


if __name__ == '__main__':
    unittest.main()
