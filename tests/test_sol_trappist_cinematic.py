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
    ARRIVAL_TRAPPIST_CANDIDATE_HOLD_END,
    ARRIVAL_TRAPPIST_HOLD_END,
    ARRIVAL_TRAPPIST_HZ_HOLD_END,
    ARRIVAL_TRAPPIST_INNER_ARRIVE,
    FIELD_STARS_MAX_LY,
    START_HALF_WIDTH_LY,
    TRAPPIST_ARRIVE_HALF_AU,
    TRAPPIST_CANDIDATE_HALF_AU,
    TRAPPIST_HZ_FOCUS_NAMES,
    TRAPPIST_HZ_HALF_AU,
    TRAPPIST_HZ_INNER_AU,
    TRAPPIST_HZ_OUTER_AU,
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
            peri = planet.semiMajorAxisAu * (1.0 - planet.eccentricity)
            apo = planet.semiMajorAxisAu * (1.0 + planet.eccentricity)
            positions = [
                self.animator._trappistPlanetPositionSol(planet, frame=frame)
                for frame in (0, 12, 48)
            ]
            radii = [
                float(np.linalg.norm(position - self.animator.hostSolAu)) for position in positions
            ]
            for radius, position in zip(radii, positions, strict=True):
                self.assertGreaterEqual(radius, peri * 0.99)
                self.assertLessEqual(radius, apo * 1.01)
                self.assertAlmostEqual(
                    float(position[2] - self.animator.hostSolAu[2]), 0.0, places=12
                )
            # Planet must move on its orbit (not a frozen host-centered point).
            self.assertGreater(float(np.linalg.norm(positions[2] - positions[0])), 1e-6)

    def test_field_stars_reach_trappist_distance(self) -> None:
        self.assertGreaterEqual(FIELD_STARS_MAX_LY, 45.0)
        self.assertGreaterEqual(START_HALF_WIDTH_LY, 45.0)
        catalog = SystemCatalog(**self.paths).starCatalog
        withinCut = catalog.starsWithinLightYears(FIELD_STARS_MAX_LY)
        self.assertFalse(withinCut.empty)
        # Extended cut must include the destination host (beyond the old 30 ly field).
        self.assertTrue((withinCut['system_id'] == 'trappist_1').any())
        self.assertGreater(float(withinCut['Distance (ly)'].max()), 30.0)
        # Drawn field stars exclude the host (destination marker owns that point).
        self.assertFalse(self.animator.fieldStars.empty)
        self.assertFalse((self.animator.fieldStars['system_id'] == 'trappist_1').any())

    def test_blender_scales_registered(self) -> None:
        for letter in 'bcdefgh':
            name = f'TRAPPIST-1 {letter}'
            self.assertIn(name, BLENDER_PLANET_BODY_SCALE)

    def test_pullback_before_travel(self) -> None:
        frame = int((SOL_HOLD_END + PULLBACK_END) / 2 * (self.animator.animationFrames - 1))
        focus, halfWidth = self.animator._cameraState(frame)
        np.testing.assert_allclose(focus, np.zeros(3), atol=1e-12)
        self.assertGreater(halfWidth, 100.0)

    def test_dive_half_width_never_zooms_out(self) -> None:
        """Arrive hold is already at TRAPPIST; dive must tighten only (no black gap)."""
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            holdEnd = animator._abHoldEnd()
            diveEnd = animator._proximaDiveEnd()
            frames = animator.animationFrames
            holdFrame = int(holdEnd * (frames - 1))
            _, holdHalf = animator._cameraState(holdFrame)
            previous = holdHalf
            for fraction in np.linspace(holdEnd, diveEnd, 25):
                frame = int(fraction * (frames - 1))
                _, halfWidth = animator._cameraState(frame)
                self.assertLessEqual(halfWidth, holdHalf + 1e-9)
                self.assertLessEqual(halfWidth, previous + 1e-9)
                previous = halfWidth
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_wide_hz_then_full_chain(self) -> None:
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

            hzFrame = int(ARRIVAL_TRAPPIST_HZ_HOLD_END * (animator.animationFrames - 1)) - 2
            focus, hzHalf = animator._cameraState(hzFrame)
            np.testing.assert_allclose(focus, animator.hostSolAu, atol=1e-9)
            self.assertAlmostEqual(hzHalf, TRAPPIST_HZ_HALF_AU, places=5)
            self.assertLess(hzHalf, wideHalf)

            # e and f sit inside the schematic HZ; b is interior; h is exterior.
            byName = {planet.name: planet for planet in animator.trappistPlanets}
            self.assertGreater(byName['TRAPPIST-1 e'].semiMajorAxisAu, TRAPPIST_HZ_INNER_AU)
            self.assertLess(byName['TRAPPIST-1 e'].semiMajorAxisAu, TRAPPIST_HZ_OUTER_AU)
            self.assertGreater(byName['TRAPPIST-1 f'].semiMajorAxisAu, TRAPPIST_HZ_INNER_AU)
            self.assertLess(byName['TRAPPIST-1 f'].semiMajorAxisAu, TRAPPIST_HZ_OUTER_AU)
            self.assertLess(byName['TRAPPIST-1 b'].semiMajorAxisAu, TRAPPIST_HZ_INNER_AU)
            self.assertGreater(byName['TRAPPIST-1 h'].semiMajorAxisAu, TRAPPIST_HZ_OUTER_AU)
            self.assertEqual(TRAPPIST_HZ_FOCUS_NAMES, ('TRAPPIST-1 e', 'TRAPPIST-1 f'))

            # Candidate beat pans onto e/f — not host-centered.
            candFrame = (
                int(ARRIVAL_TRAPPIST_CANDIDATE_HOLD_END * (animator.animationFrames - 1)) - 2
            )
            candFocus, candHalf = animator._cameraState(candFrame)
            expectedFocus = animator._candidateFocusSol(candFrame)
            np.testing.assert_allclose(candFocus, expectedFocus, atol=1e-9)
            self.assertAlmostEqual(candHalf, TRAPPIST_CANDIDATE_HALF_AU, places=5)
            self.assertLess(candHalf, hzHalf)
            self.assertGreater(float(np.linalg.norm(candFocus - animator.hostSolAu)), 0.01)

            innerFrame = int(ARRIVAL_TRAPPIST_INNER_ARRIVE * (animator.animationFrames - 1))
            focus, innerHalf = animator._cameraState(innerFrame)
            np.testing.assert_allclose(focus, animator.hostSolAu, atol=1e-9)
            self.assertAlmostEqual(innerHalf, TRAPPIST_INNER_HALF_AU, places=5)
            # Finale returns to host and pulls wider than the candidate close-up.
            self.assertGreater(innerHalf, candHalf)
            self.assertLess(innerHalf, TRAPPIST_WIDE_HALF_AU)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_trappist_planets_never_flash_catalog_dots(self) -> None:
        """Dive/HZ reveal must queue textured billboards — not scatter markers first."""
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            # Need at least one TRAPPIST spin pack on disk for the no-dot path.
            available = [
                planet.name
                for planet in animator.trappistPlanets
                if animator._blenderBodyAvailable(planet.name)
            ]
            if not available:
                self.skipTest('TRAPPIST Blender spin packs not present')

            scatterHits = 0
            queued: set[str] = set()
            origScatter = animator.axes.scatter
            origQueue = animator._queueBlenderBody

            def scatterSpy(*args, **kwargs):
                nonlocal scatterHits
                scatterHits += 1
                return origScatter(*args, **kwargs)

            def queueSpy(catalogName, position, frame, halfWidthAu, **kwargs):
                ok = origQueue(catalogName, position, frame, halfWidthAu, **kwargs)
                if ok and str(catalogName).startswith('TRAPPIST-1'):
                    queued.add(str(catalogName))
                return ok

            animator.axes.scatter = scatterSpy  # type: ignore[method-assign]
            animator._queueBlenderBody = queueSpy  # type: ignore[method-assign]

            # Sample half-widths from arrive → wide → HZ → candidate (disk gate opens mid-dive).
            halfWidths = [
                TRAPPIST_ARRIVE_HALF_AU,
                TRAPPIST_WIDE_HALF_AU * 2.5,
                TRAPPIST_WIDE_HALF_AU * 2.0,
                TRAPPIST_WIDE_HALF_AU,
                TRAPPIST_HZ_HALF_AU,
                TRAPPIST_CANDIDATE_HALF_AU,
                TRAPPIST_INNER_HALF_AU,
            ]
            for halfWidth in halfWidths:
                animator._viewFocus = animator.hostSolAu.copy()
                animator._viewHalfWidthAu = float(halfWidth)
                animator._pendingBlenderBodies = []
                hzFocus = halfWidth <= TRAPPIST_HZ_HALF_AU * 1.05
                for planet in animator.trappistPlanets:
                    if not animator._blenderBodyAvailable(planet.name):
                        continue
                    before = scatterHits
                    animator._drawOneTrappistPlanet(
                        planet,
                        frame=0,
                        halfWidthAu=float(halfWidth),
                        hzFocus=hzFocus,
                    )
                    self.assertEqual(
                        scatterHits,
                        before,
                        msg=(
                            f'{planet.name} used a catalog scatter at '
                            f'halfWidth={halfWidth:.4f} AU'
                        ),
                    )

            self.assertTrue(queued, 'expected at least one TRAPPIST billboard queue')
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)


if __name__ == '__main__':
    unittest.main()
