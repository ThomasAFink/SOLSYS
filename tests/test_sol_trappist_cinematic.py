"""Tests for Sol → TRAPPIST-1 cinematic."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.sol_centauri_cinematic import (
    BLENDER_PLANET_BODY_SCALE,
    BLENDER_STAR_BODY_SCALE,
    PULLBACK_END,
    SOL_EARTH_CLOSE_HALF_AU,
    SOL_EARTH_HALF_AU,
    SOL_HOLD_END,
)
from animate.scenes.sol_trappist_cinematic import (
    ANIMATION_SPEED_TRAPPIST_PLANETS,
    ANIMATION_SPEED_TRAPPIST_PLANETS_CLOSE,
    ARRIVAL_TRAPPIST_E_ARRIVE,
    ARRIVAL_TRAPPIST_E_HOLD_END,
    ARRIVAL_TRAPPIST_F_ARRIVE,
    ARRIVAL_TRAPPIST_F_HOLD_END,
    ARRIVAL_TRAPPIST_HOLD_END,
    ARRIVAL_TRAPPIST_HZ_ARRIVE,
    ARRIVAL_TRAPPIST_HZ_HOLD_END,
    ARRIVAL_TRAPPIST_INNER_ARRIVE,
    FIELD_STARS_MAX_LY,
    START_HALF_WIDTH_LY,
    TRAPPIST_ARRIVE_HALF_AU,
    TRAPPIST_HZ_FOCUS_NAMES,
    TRAPPIST_HZ_HALF_AU,
    TRAPPIST_HZ_INNER_AU,
    TRAPPIST_HZ_OUTER_AU,
    TRAPPIST_INNER_HALF_AU,
    TRAPPIST_PLANET_HALF_AU,
    TRAPPIST_PLANET_HERO_SCALE,
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

    def test_blender_wide_hz_then_sequential_planet_portraits(self) -> None:
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

            byName = {planet.name: planet for planet in animator.trappistPlanets}
            self.assertGreater(byName['TRAPPIST-1 e'].semiMajorAxisAu, TRAPPIST_HZ_INNER_AU)
            self.assertLess(byName['TRAPPIST-1 e'].semiMajorAxisAu, TRAPPIST_HZ_OUTER_AU)
            self.assertGreater(byName['TRAPPIST-1 f'].semiMajorAxisAu, TRAPPIST_HZ_INNER_AU)
            self.assertLess(byName['TRAPPIST-1 f'].semiMajorAxisAu, TRAPPIST_HZ_OUTER_AU)
            self.assertEqual(TRAPPIST_HZ_FOCUS_NAMES, ('TRAPPIST-1 e', 'TRAPPIST-1 f'))

            frames = animator.animationFrames - 1
            eFrame = int(ARRIVAL_TRAPPIST_E_HOLD_END * frames) - 2
            eFocus, eHalf = animator._cameraState(eFrame)
            expectedE = animator._planetFocusSol('TRAPPIST-1 e', eFrame)
            np.testing.assert_allclose(eFocus, expectedE, atol=1e-9)
            self.assertAlmostEqual(eHalf, TRAPPIST_PLANET_HALF_AU, places=5)
            self.assertLess(eHalf, hzHalf)
            self.assertEqual(animator._portraitHeroName(eFrame), 'TRAPPIST-1 e')

            fFrame = int(ARRIVAL_TRAPPIST_F_HOLD_END * frames) - 2
            fFocus, fHalf = animator._cameraState(fFrame)
            expectedF = animator._planetFocusSol('TRAPPIST-1 f', fFrame)
            np.testing.assert_allclose(fFocus, expectedF, atol=1e-9)
            self.assertAlmostEqual(fHalf, TRAPPIST_PLANET_HALF_AU, places=5)
            self.assertEqual(animator._portraitHeroName(fFrame), 'TRAPPIST-1 f')
            # Sequential: e then f are distinct look-ats.
            self.assertGreater(float(np.linalg.norm(eFocus - fFocus)), 0.005)

            # Portrait disks stay under e↔f orbital spacing (~0.009 AU).
            heroRadius = (
                BLENDER_PLANET_BODY_SCALE['TRAPPIST-1 e']
                * TRAPPIST_PLANET_HERO_SCALE
                * (SOL_EARTH_HALF_AU * 0.18)
            )
            self.assertLess(heroRadius * 2.0, 0.009)

            innerFrame = int(ARRIVAL_TRAPPIST_INNER_ARRIVE * frames)
            focus, innerHalf = animator._cameraState(innerFrame)
            np.testing.assert_allclose(focus, animator.hostSolAu, atol=1e-9)
            self.assertAlmostEqual(innerHalf, TRAPPIST_INNER_HALF_AU, places=5)
            self.assertGreater(innerHalf, eHalf)
            self.assertLess(innerHalf, TRAPPIST_WIDE_HALF_AU)

            # e/f holds each linger.
            self.assertGreaterEqual(
                int((ARRIVAL_TRAPPIST_E_HOLD_END - ARRIVAL_TRAPPIST_E_ARRIVE) * frames), 30
            )
            self.assertGreaterEqual(
                int((ARRIVAL_TRAPPIST_F_HOLD_END - ARRIVAL_TRAPPIST_F_ARRIVE) * frames), 30
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_trappist_orbits_slow_on_final_zooms(self) -> None:
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            frames = animator.animationFrames
            early = int(0.5 * (frames - 1))
            hzHold = int(
                ((ARRIVAL_TRAPPIST_HZ_ARRIVE + ARRIVAL_TRAPPIST_HZ_HOLD_END) / 2) * (frames - 1)
            )
            eHold = int(
                ((ARRIVAL_TRAPPIST_E_ARRIVE + ARRIVAL_TRAPPIST_E_HOLD_END) / 2) * (frames - 1)
            )
            self.assertAlmostEqual(
                animator._trappistPlanetAnimationSpeed(early),
                ANIMATION_SPEED_TRAPPIST_PLANETS,
            )
            self.assertLess(
                animator._trappistPlanetAnimationSpeed(hzHold),
                ANIMATION_SPEED_TRAPPIST_PLANETS,
            )
            self.assertAlmostEqual(
                animator._trappistPlanetAnimationSpeed(eHold),
                ANIMATION_SPEED_TRAPPIST_PLANETS_CLOSE,
            )
            days = [
                animator._trappistMotionDays(frame)
                for frame in range(hzHold - 2, min(frames, eHold + 3))
            ]
            self.assertGreater(days[-1], days[0])
            for previous, current in zip(days, days[1:], strict=False):
                self.assertGreaterEqual(current, previous)

            planet = animator._planetByName('TRAPPIST-1 e')
            senses: list[float] = []
            previousPos = animator._trappistPlanetPositionSol(planet, hzHold - 2)
            for frame in range(hzHold - 1, min(frames, eHold + 3)):
                position = animator._trappistPlanetPositionSol(planet, frame)
                radial = previousPos - animator.hostSolAu
                delta = position - previousPos
                senses.append(float(radial[0] * delta[1] - radial[1] * delta[0]))
                previousPos = position
            nonzero = [value for value in senses if abs(value) > 1e-16]
            self.assertTrue(nonzero)
            self.assertEqual(len({1 if value > 0 else -1 for value in nonzero}), 1)

            # Portrait queues only the hero (not both e and f at once).
            animator._viewFocus = animator._planetFocusSol('TRAPPIST-1 e', eHold)
            animator._viewHalfWidthAu = TRAPPIST_PLANET_HALF_AU
            animator._pendingBlenderBodies = []
            for world in animator.trappistPlanets:
                animator._drawOneTrappistPlanet(
                    world,
                    eHold,
                    TRAPPIST_PLANET_HALF_AU,
                    hzFocus=True,
                    heroName='TRAPPIST-1 e',
                )
            queued = [entry[0] for entry in animator._pendingBlenderBodies]
            self.assertEqual(queued, ['TRAPPIST-1 e'])
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

            halfWidths = [
                TRAPPIST_ARRIVE_HALF_AU,
                TRAPPIST_WIDE_HALF_AU * 2.5,
                TRAPPIST_WIDE_HALF_AU * 2.0,
                TRAPPIST_WIDE_HALF_AU,
                TRAPPIST_HZ_HALF_AU,
                TRAPPIST_PLANET_HALF_AU,
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
                        heroName=None,
                    )
                    self.assertEqual(
                        scatterHits,
                        before,
                        msg=(
                            f'{planet.name} used a catalog scatter at halfWidth={halfWidth:.4f} AU'
                        ),
                    )

            self.assertTrue(queued, 'expected at least one TRAPPIST billboard queue')
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_host_photosphere_billboards_once_the_chain_reveals(self) -> None:
        """Cruise keeps the scatter marker; chain reveal swaps to the M8V pack."""
        animator = SolTrappistCinematicAnimator(
            self.system,
            starsCsvPath=self.paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            if not animator._blenderBodyAvailable('TRAPPIST-1'):
                self.skipTest('TRAPPIST-1 host spin pack not present')

            # Host disk must stay inside the innermost orbit (b, a≈0.0115 AU) or the
            # chain would appear to graze the photosphere.
            hostRadiusAu = BLENDER_STAR_BODY_SCALE['TRAPPIST-1'] * (SOL_EARTH_HALF_AU * 0.18)
            innermost = min(planet.semiMajorAxisAu for planet in animator.trappistPlanets)
            self.assertLess(hostRadiusAu, innermost * 0.5)

            linear = animator._abHoldEnd() + 1e-3
            queuedAt: dict[float, bool] = {}
            for halfWidth in (
                TRAPPIST_ARRIVE_HALF_AU,
                TRAPPIST_WIDE_HALF_AU,
                TRAPPIST_HZ_HALF_AU,
            ):
                animator._viewFocus = animator.hostSolAu.copy()
                animator._viewHalfWidthAu = float(halfWidth)
                animator._pendingBlenderBodies = []
                animator._drawTrappistDestination(
                    0,
                    float(halfWidth),
                    1.0,
                    linear,
                )
                queuedAt[float(halfWidth)] = any(
                    entry[0] == 'TRAPPIST-1' for entry in animator._pendingBlenderBodies
                )

            # Far out the world-fixed disk is a speck — the marker still owns the frame.
            self.assertFalse(queuedAt[TRAPPIST_ARRIVE_HALF_AU])
            self.assertTrue(queuedAt[TRAPPIST_WIDE_HALF_AU])
            self.assertTrue(queuedAt[TRAPPIST_HZ_HALF_AU])
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)


if __name__ == '__main__':
    unittest.main()
