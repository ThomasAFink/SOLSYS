"""Tests for Sol → Centauri cinematic helpers and transform usage."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.animation_styles import ASTEROID_RENDER_STYLES
from animate.scenes.sol_centauri_cinematic import (
    AB_HOLD_END,
    AB_TRAVEL_END,
    ANIMATION_SPEED_AB,
    ANIMATION_SPEED_SOL_NEAR,
    BLENDER_STAR_BILLBOARD_HALF_AU,
    BLENDER_STAR_BODY_SCALE,
    PROXIMA_TRAVEL_END,
    PULLBACK_END,
    SOL_BEAT_BELT_ARRIVE,
    SOL_BEAT_BELT_HOLD_END,
    SOL_BEAT_INNER_ARRIVE,
    SOL_BEAT_INNER_HOLD_END,
    SOL_BEAT_NEAR_SUN_ARRIVE,
    SOL_BEAT_NEAR_SUN_HOLD_END,
    SOL_BEAT_SATURN_ARRIVE,
    SOL_BEAT_SATURN_HOLD_END,
    SOL_BELT_ARRIVE,
    SOL_BELT_HOLD_END,
    SOL_BELT_LINGER_HALF_AU,
    SOL_EARTH_CLOSE_HALF_AU,
    SOL_EARTH_HALF_AU,
    SOL_EARTH_MOON_REVEAL_END,
    SOL_EARTH_SPIN_HOLD_END,
    SOL_HALF_WIDTH_AU,
    SOL_HOLD_END,
    SOL_INNER_HALF_AU,
    SOL_NEAR_SUN_HALF_AU,
    SOL_OUTER_ARRIVE,
    SOL_OUTER_LINGER_HALF_AU,
    SOL_SATURN_LINGER_HALF_AU,
    WIDE_OUT_ARRIVE,
    WIDE_OUT_END,
    SolCentauriCinematicAnimator,
    innerBeltRenderParams,
    kuiperRenderParams,
    parseApparentMagnitude,
    solAnimationSpeed,
    spectralClassColor,
    travelProgress,
)
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths
from solsys.physics.frame_transform import SolCentauriFrameTransform

REPO_ROOT = Path(__file__).resolve().parents[1]


class TravelProgressTests(unittest.TestCase):
    def test_holds_and_midpoint(self) -> None:
        frames = 241
        self.assertEqual(travelProgress(0, frames), 0.0)
        holdFrame = int(0.10 * (frames - 1))
        self.assertEqual(travelProgress(holdFrame, frames), 0.0)
        self.assertEqual(travelProgress(frames - 1, frames), 1.0)
        midTravelFrame = int(0.5 * (PULLBACK_END + AB_TRAVEL_END) * (frames - 1))
        mid = travelProgress(midTravelFrame, frames)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)


class CinematicTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**paths).load('alpha_centauri')
        self.animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
        )

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        plt.close(self.animator.figure)

    def test_uses_shared_frame_transform(self) -> None:
        expected = SolCentauriFrameTransform.fromStarSystem(self.system)
        np.testing.assert_allclose(
            self.animator.transform.originSolAu,
            expected.originSolAu,
            rtol=1e-12,
            atol=1e-12,
        )
        primary = self.animator._starPositionSol(
            self.animator.primaryOrbit, frame=0, speed=ANIMATION_SPEED_AB
        )
        from_transform = expected.toSol(self.animator.transform.toCentauri(primary)[0, :2])[0]
        np.testing.assert_allclose(primary, from_transform, rtol=1e-10, atol=1e-8)

    def test_camera_starts_on_earth_moon(self) -> None:
        focus, halfWidth = self.animator._cameraState(0)
        earth = self.animator._planetPositionAu('Earth', 0)
        np.testing.assert_allclose(focus, earth, atol=1e-9)
        self.assertAlmostEqual(halfWidth, SOL_EARTH_HALF_AU, places=6)

    def test_blender_camera_starts_earth_close_then_reveals_moon(self) -> None:
        from animate.scenes.sol_centauri_cinematic import SOL_EARTH_BLENDER_DWELL_END

        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            focus, halfWidth = animator._cameraState(0)
            earth = animator._planetPositionAu('Earth', 0)
            np.testing.assert_allclose(focus, earth, atol=1e-9)
            self.assertAlmostEqual(halfWidth, SOL_EARTH_CLOSE_HALF_AU, places=6)
            holdFrame = int(SOL_EARTH_SPIN_HOLD_END * (animator.animationFrames - 1)) - 1
            _, holdHalf = animator._cameraState(holdFrame)
            self.assertAlmostEqual(holdHalf, SOL_EARTH_CLOSE_HALF_AU, places=6)
            revealFrame = int(SOL_EARTH_MOON_REVEAL_END * (animator.animationFrames - 1))
            _, revealHalf = animator._cameraState(revealFrame)
            self.assertAlmostEqual(revealHalf, SOL_EARTH_HALF_AU, places=6)
            dwellFrame = int(SOL_EARTH_BLENDER_DWELL_END * (animator.animationFrames - 1)) - 1
            _, dwellHalf = animator._cameraState(dwellFrame)
            self.assertAlmostEqual(dwellHalf, SOL_EARTH_HALF_AU, places=6)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_earth_moon_open_covers_full_lunar_orbit(self) -> None:
        """Earth+Moon plateau should advance Luna by ~one sidereal month."""
        from animate.scenes.sol_centauri_cinematic import (
            LUNAR_OPEN_MOTION_SCALE,
            SOL_EARTH_BLENDER_DWELL_END,
            SOL_MOON_REVEAL_HALF_AU,
        )

        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            moon = animator.moonCatalog.moons['Moon']
            frames = animator.animationFrames
            revealFrame = int(np.ceil(SOL_EARTH_MOON_REVEAL_END * (frames - 1)))
            dwellFrame = int(SOL_EARTH_BLENDER_DWELL_END * (frames - 1)) - 1
            # First frame where Luna is on-screen (half-width past reveal gate).
            visibleFrame = revealFrame
            for frame in range(revealFrame, dwellFrame + 1):
                _, halfWidth = animator._cameraState(frame)
                if halfWidth >= SOL_MOON_REVEAL_HALF_AU - 1e-9:
                    visibleFrame = frame
                    break
            daysStart = animator._lunarMotionDays(
                moon, visibleFrame, SOL_EARTH_HALF_AU
            )
            daysEnd = animator._lunarMotionDays(moon, dwellFrame, SOL_EARTH_HALF_AU)
            orbits = (daysEnd - daysStart) / moon.orbitalPeriodDays
            self.assertGreaterEqual(orbits, 0.95)
            self.assertLessEqual(orbits, 1.25)
            self.assertAlmostEqual(LUNAR_OPEN_MOTION_SCALE, 0.067, places=3)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_sol_beats_hold_near_sun_belt_and_saturn(self) -> None:
        """Staged blender zoom-out (#51) plateaus at Near-Sun, belt, and Saturn scales."""
        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            frames = animator.animationFrames
            nearHold = int(
                0.5 * (SOL_BEAT_NEAR_SUN_ARRIVE + SOL_BEAT_NEAR_SUN_HOLD_END) * (frames - 1)
            )
            focus, halfWidth = animator._cameraState(nearHold)
            np.testing.assert_allclose(focus, np.zeros(3), atol=1e-9)
            self.assertAlmostEqual(halfWidth, SOL_NEAR_SUN_HALF_AU, places=4)
            title, _ = animator._solCaption(halfWidth, nearHold / max(frames - 1, 1))
            self.assertEqual(title, 'The Sun')

            innerMid = int(0.5 * (SOL_BEAT_INNER_ARRIVE + SOL_BEAT_INNER_HOLD_END) * (frames - 1))
            _, innerHalf = animator._cameraState(innerMid)
            self.assertAlmostEqual(innerHalf, SOL_INNER_HALF_AU, places=4)

            beltHold = int(0.5 * (SOL_BEAT_BELT_ARRIVE + SOL_BEAT_BELT_HOLD_END) * (frames - 1))
            _, beltHalf = animator._cameraState(beltHold)
            self.assertAlmostEqual(beltHalf, SOL_BELT_LINGER_HALF_AU, places=4)
            beltTitle, _ = animator._solCaption(beltHalf, beltHold / max(frames - 1, 1))
            self.assertIn('belt', beltTitle.lower())

            saturnHold = int(
                0.5 * (SOL_BEAT_SATURN_ARRIVE + SOL_BEAT_SATURN_HOLD_END) * (frames - 1)
            )
            _, saturnHalf = animator._cameraState(saturnHold)
            self.assertAlmostEqual(saturnHalf, SOL_SATURN_LINGER_HALF_AU, places=4)
            saturnTitle, _ = animator._solCaption(saturnHalf, saturnHold / max(frames - 1, 1))
            self.assertEqual(saturnTitle, 'Saturn')
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_classic_mode_skips_blender_sol_beat_timeline(self) -> None:
        """Dotted mode keeps the pre-#51 belt arrive (no Near-Sun plateau)."""
        frames = self.animator.animationFrames
        # Just before classic belt arrive, half-width is still diving (not held at Near-Sun).
        preBelt = int((SOL_BELT_ARRIVE - 0.01) * (frames - 1))
        _, halfWidth = self.animator._cameraState(preBelt)
        self.assertNotAlmostEqual(halfWidth, SOL_NEAR_SUN_HALF_AU, places=3)
        arrive = int(np.ceil(SOL_BELT_ARRIVE * (frames - 1)))
        _, arriveHalf = self.animator._cameraState(arrive)
        self.assertAlmostEqual(arriveHalf, SOL_BELT_LINGER_HALF_AU, places=4)

    def test_blender_sun_billboard_shrinks_monotonically_through_sol_beats(self) -> None:
        """Sol photosphere follows world scale — no stepped floors that pulse size."""
        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            sunScale = BLENDER_STAR_BODY_SCALE['Sun']
            nearFrac = animator._blenderBillboardFracRadius(
                SOL_NEAR_SUN_HALF_AU, sunScale, catalogName='Sun'
            )
            innerFrac = animator._blenderBillboardFracRadius(
                SOL_INNER_HALF_AU, sunScale, catalogName='Sun'
            )
            beltFrac = animator._blenderBillboardFracRadius(
                SOL_BELT_LINGER_HALF_AU, sunScale, catalogName='Sun'
            )
            saturnFrac = animator._blenderBillboardFracRadius(
                SOL_SATURN_LINGER_HALF_AU, sunScale, catalogName='Sun'
            )
            outerFrac = animator._blenderBillboardFracRadius(
                SOL_OUTER_LINGER_HALF_AU, sunScale, catalogName='Sun'
            )
            fracs = (nearFrac, innerFrac, beltFrac, saturnFrac, outerFrac)
            self.assertTrue(all(f is not None for f in fracs))
            assert all(f is not None for f in fracs)
            for earlier, later in zip(fracs[:-1], fracs[1:], strict=True):
                self.assertGreater(earlier, later)
            # Near-Sun should still read as a clear hero disk.
            self.assertGreater(nearFrac, 0.04)
            starMax = BLENDER_STAR_BILLBOARD_HALF_AU[1]
            self.assertGreater(starMax, SOL_OUTER_LINGER_HALF_AU)
            self.assertIsNone(
                animator._blenderBillboardRadiusAu(
                    starMax + 1.0,
                    openCloseup=True,
                    bodyScale=sunScale,
                    catalogName='Sun',
                )
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_alpha_cen_star_billboards_sized_for_arrival(self) -> None:
        """α Cen A/B readable at AB hold; Proxima capped so the dive does not fill the frame."""
        from animate.scenes.sol_centauri_cinematic import (
            AB_HALF_WIDTH_AU,
            BLENDER_STAR_BODY_SCALE,
            BLENDER_STAR_MAX_FRAC,
            PROXIMA_WIDE_HALF_AU,
        )

        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            aFrac = animator._blenderBillboardFracRadius(
                AB_HALF_WIDTH_AU,
                BLENDER_STAR_BODY_SCALE['Alpha Centauri A'],
                catalogName='Alpha Centauri A',
            )
            bFrac = animator._blenderBillboardFracRadius(
                AB_HALF_WIDTH_AU,
                BLENDER_STAR_BODY_SCALE['Alpha Centauri B'],
                catalogName='Alpha Centauri B',
            )
            proxWide = animator._blenderBillboardFracRadius(
                PROXIMA_WIDE_HALF_AU,
                BLENDER_STAR_BODY_SCALE['Proxima Centauri'],
                catalogName='Proxima Centauri',
            )
            from animate.scenes.sol_centauri_cinematic import PROXIMA_INNER_HALF_AU

            proxInner = animator._blenderBillboardFracRadius(
                PROXIMA_INNER_HALF_AU,
                BLENDER_STAR_BODY_SCALE['Proxima Centauri'],
                catalogName='Proxima Centauri',
            )
            self.assertIsNotNone(aFrac)
            self.assertIsNotNone(bFrac)
            self.assertIsNotNone(proxWide)
            self.assertIsNotNone(proxInner)
            assert aFrac is not None and bFrac is not None
            assert proxWide is not None and proxInner is not None
            self.assertGreater(aFrac, 0.015)
            self.assertGreater(bFrac, 0.012)
            self.assertLess(aFrac, BLENDER_STAR_MAX_FRAC['Alpha Centauri A'])
            # Finale must keep the capped photosphere — not drop to scatter.
            self.assertEqual(proxInner, BLENDER_STAR_MAX_FRAC['Proxima Centauri'])
            self.assertIsNotNone(
                animator._blenderBillboardRadiusAu(
                    PROXIMA_INNER_HALF_AU,
                    openCloseup=True,
                    bodyScale=BLENDER_STAR_BODY_SCALE['Proxima Centauri'],
                    catalogName='Proxima Centauri',
                )
            )
            self.assertIsNone(
                animator._blenderBillboardRadiusAu(
                    5.0,
                    openCloseup=True,
                    bodyScale=BLENDER_STAR_BODY_SCALE['Alpha Centauri A'],
                    catalogName='Alpha Centauri A',
                )
            )
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_blender_sun_is_textured_on_first_on_screen_frame(self) -> None:
        """Leaving Earth must not show a scatter Sol before the photosphere billboard."""
        from animate.scenes.sol_centauri_cinematic import (
            SOL_BEAT_NEAR_SUN_ARRIVE,
            SOL_EARTH_BLENDER_DWELL_END,
            STAR_COLORS,
        )

        paths = defaultDataPaths(REPO_ROOT)
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=paths['starsCsvPath'],
            useBlenderBodies=True,
        )
        try:
            starFrames: list[int] = []
            queueFrames: list[int] = []
            origStar = animator._drawStarMarker
            origQueue = animator._queueBlenderBody

            def starSpy(position, color, size, **kwargs):
                if color == STAR_COLORS['sun']:
                    starFrames.append(int(animator._viewHalfWidthAu * 1000))
                return origStar(position, color, size, **kwargs)

            def queueSpy(catalogName, position, frame, halfWidthAu, **kwargs):
                ok = origQueue(catalogName, position, frame, halfWidthAu, **kwargs)
                if catalogName == 'Sun' and ok:
                    queueFrames.append(frame)
                return ok

            animator._drawStarMarker = starSpy  # type: ignore[method-assign]
            animator._queueBlenderBody = queueSpy  # type: ignore[method-assign]

            frames = animator.animationFrames
            scanStart = int(SOL_EARTH_BLENDER_DWELL_END * (frames - 1)) - 5
            scanEnd = int(SOL_BEAT_NEAR_SUN_ARRIVE * (frames - 1)) + 5
            firstSunFrame: int | None = None
            for frame in range(max(0, scanStart), min(frames, scanEnd)):
                starBefore = len(starFrames)
                queueBefore = len(queueFrames)
                animator.update(frame)
                sawStar = len(starFrames) > starBefore
                sawQueue = len(queueFrames) > queueBefore
                if sawStar or sawQueue:
                    firstSunFrame = frame
                    self.assertTrue(sawQueue, f'Sol first appeared as scatter at frame {frame}')
                    self.assertFalse(sawStar, f'Sol scatter drawn at textured frame {frame}')
                    break
            self.assertIsNotNone(firstSunFrame)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_camera_ends_at_proxima(self) -> None:
        from animate.scenes.sol_centauri_cinematic import PROXIMA_INNER_HALF_AU

        last = self.animator.animationFrames - 1
        focus, halfWidth = self.animator._cameraState(last)
        proxima = self.animator._proximaPositionSol(last)
        np.testing.assert_allclose(focus, proxima, atol=1e-6)
        self.assertAlmostEqual(halfWidth, PROXIMA_INNER_HALF_AU, places=5)

    def test_ab_cruise_keeps_sol_and_centauri_framed(self) -> None:
        frames = self.animator.animationFrames
        # Mid Sol→AB pan should still see both landmarks.
        midFrame = int(0.5 * (PULLBACK_END + AB_TRAVEL_END) * (frames - 1))
        focus, halfWidth = self.animator._cameraState(midFrame)
        distSol = float(np.linalg.norm(focus))
        distAb = float(np.linalg.norm(focus - self.animator.barycenterSolAu))
        self.assertGreaterEqual(halfWidth, distSol)
        self.assertGreaterEqual(halfWidth, distAb)

    def test_sol_orbit_speed_increases_with_zoom(self) -> None:
        # Stay near-baseline through the inner system; ramp only in the outer band.
        self.assertAlmostEqual(solAnimationSpeed(SOL_EARTH_HALF_AU), ANIMATION_SPEED_SOL_NEAR)
        self.assertAlmostEqual(solAnimationSpeed(SOL_INNER_HALF_AU), ANIMATION_SPEED_SOL_NEAR)
        self.assertLess(solAnimationSpeed(22.0), solAnimationSpeed(SOL_HALF_WIDTH_AU))
        self.assertGreater(solAnimationSpeed(SOL_HALF_WIDTH_AU), ANIMATION_SPEED_SOL_NEAR * 20.0)

    def test_belt_linger_holds_jupiter_scale(self) -> None:
        frames = self.animator.animationFrames
        arrive = int(np.ceil(SOL_BELT_ARRIVE * (frames - 1)))
        hold = int(0.5 * (SOL_BELT_ARRIVE + SOL_BELT_HOLD_END) * (frames - 1))
        _, arriveHalf = self.animator._cameraState(arrive)
        _, holdHalf = self.animator._cameraState(hold)
        self.assertAlmostEqual(arriveHalf, SOL_BELT_LINGER_HALF_AU, places=4)
        self.assertAlmostEqual(holdHalf, SOL_BELT_LINGER_HALF_AU, places=4)

    def test_outer_linger_is_tight_and_held(self) -> None:
        frames = self.animator.animationFrames
        arrive = int(np.ceil(SOL_OUTER_ARRIVE * (frames - 1)))
        hold = int(0.5 * (SOL_OUTER_ARRIVE + SOL_HOLD_END) * (frames - 1))
        _, arriveHalf = self.animator._cameraState(arrive)
        _, holdHalf = self.animator._cameraState(hold)
        self.assertAlmostEqual(arriveHalf, SOL_HALF_WIDTH_AU, places=4)
        self.assertAlmostEqual(holdHalf, SOL_HALF_WIDTH_AU, places=4)

    def test_oort_stays_visible_through_pullback(self) -> None:
        """Oort draw must not empty the frame mid-pullback."""
        from animate.scenes.sol_centauri_cinematic import (
            OORT_WORLD_OUTER_AU,
            POPULATION_ANNULUS_INNER_FRAC,
        )

        frames = self.animator.animationFrames
        sawOort = False
        gapAfterSeen = False
        for frame in range(0, frames, 3):
            linear = frame / max(frames - 1, 1)
            if linear < SOL_HOLD_END:
                continue
            if linear > PULLBACK_END:
                break
            _, halfWidth = self.animator._cameraState(frame)
            if halfWidth < 62.0:
                continue
            oortX, oortY, oortZ = self.animator._oortWorldPositions()
            useAnnulus = halfWidth < OORT_WORLD_OUTER_AU / POPULATION_ANNULUS_INNER_FRAC
            if useAnnulus:
                mask = self.animator._populationAnnulusMask(oortX, oortY, oortZ, halfWidth)
            else:
                mask = np.ones(oortX.shape[0], dtype=bool)
            count = int(np.count_nonzero(mask))
            if count > 0:
                sawOort = True
            elif sawOort:
                gapAfterSeen = True
                break
        self.assertTrue(sawOort)
        self.assertFalse(gapAfterSeen)

    def test_population_annulus_clears_solar_core(self) -> None:
        halfWidth = 2000.0
        # One point on Sol, one in the annulus band.
        x = np.array([0.0, halfWidth * 0.5])
        y = np.array([0.0, 0.0])
        z = np.array([0.0, 0.0])
        mask = self.animator._populationAnnulusMask(x, y, z, halfWidth)
        self.assertFalse(bool(mask[0]))
        self.assertTrue(bool(mask[1]))

    def test_triple_view_holds_at_wide_scale(self) -> None:
        frames = self.animator.animationFrames
        arrive = int(np.ceil(WIDE_OUT_ARRIVE * (frames - 1)))
        hold = int(0.5 * (WIDE_OUT_ARRIVE + WIDE_OUT_END) * (frames - 1))
        _, arriveHalf = self.animator._cameraState(arrive)
        _, holdHalf = self.animator._cameraState(hold)
        self.assertAlmostEqual(arriveHalf, self.animator.wideHalfWidthAu, places=2)
        self.assertAlmostEqual(holdHalf, self.animator.wideHalfWidthAu, places=2)

    def test_proxima_dive_does_not_zoom_out_after_triple_pause(self) -> None:
        """After the triple hold, half-width must only tighten toward Proxima."""
        frames = self.animator.animationFrames
        start = int(np.ceil(WIDE_OUT_END * (frames - 1)))
        end = int(PROXIMA_TRAVEL_END * (frames - 1))
        previousHalf = None
        for frame in range(start, end + 1, max(1, (end - start) // 40)):
            _, halfWidth = self.animator._cameraState(frame)
            if previousHalf is not None:
                self.assertLessEqual(halfWidth, previousHalf + 1e-6)
            previousHalf = halfWidth
        self.assertLess(previousHalf, self.animator.wideHalfWidthAu)
        self.assertAlmostEqual(previousHalf, self.animator.proximaWideHalfWidthAu, places=2)

    def test_proxima_dive_focuses_proxima_not_ab_midpoint(self) -> None:
        """Zoom-in after the triple hold must look at Proxima, not AB↔Proxima midpoint."""
        frames = self.animator.animationFrames
        start = int(np.ceil(WIDE_OUT_END * (frames - 1))) + 1
        mid = int(0.5 * (WIDE_OUT_END + PROXIMA_TRAVEL_END) * (frames - 1))
        for frame in (start, mid, frames - 1):
            focus, _ = self.animator._cameraState(frame)
            proxima = self.animator._proximaPositionSol(frame)
            midpoint = 0.5 * (self.animator.barycenterSolAu + proxima)
            np.testing.assert_allclose(focus, proxima, atol=1e-6)
            self.assertGreater(
                float(np.linalg.norm(focus - midpoint)),
                0.25 * float(np.linalg.norm(proxima - self.animator.barycenterSolAu)),
            )

    def test_inner_belt_emphasis_is_continuous_at_peak(self) -> None:
        style = ASTEROID_RENDER_STYLES['dark']
        peak = SOL_BELT_LINGER_HALF_AU
        left = innerBeltRenderParams(peak - 1e-6, style)
        right = innerBeltRenderParams(peak + 1e-6, style)
        self.assertIsNotNone(left)
        self.assertIsNotNone(right)
        for leftValue, rightValue in zip(left, right, strict=True):
            self.assertAlmostEqual(leftValue, rightValue, places=4)

    def test_inner_belts_appear_when_camera_reaches_belt(self) -> None:
        style = ASTEROID_RENDER_STYLES['dark']
        self.assertIsNone(innerBeltRenderParams(1.8, style))
        self.assertIsNotNone(innerBeltRenderParams(2.0, style))

    def test_kuiper_emphasis_is_continuous_at_joins(self) -> None:
        style = ASTEROID_RENDER_STYLES['dark']
        for joinAu in (30.0, 55.0):
            left = kuiperRenderParams(joinAu - 1e-6, style)
            right = kuiperRenderParams(joinAu + 1e-6, style)
            self.assertIsNotNone(left)
            self.assertIsNotNone(right)
            self.assertAlmostEqual(left[0], right[0], places=4)
            self.assertAlmostEqual(left[1], right[1], places=4)

    def test_field_stars_use_spectral_tints_from_catalog(self) -> None:
        self.assertGreater(len(self.animator.fieldStars), 50)
        self.assertIn('fieldColor', self.animator.fieldStars.columns)
        self.assertEqual(spectralClassColor('G2V', fallback='#fff'), '#FFE7A0')
        self.assertEqual(spectralClassColor('M5.5Ve', fallback='#fff'), '#FFB06B')
        self.assertEqual(spectralClassColor('A1V', fallback='#fff'), '#CAD7FF')
        self.assertAlmostEqual(parseApparentMagnitude('\N{MINUS SIGN}1.46'), -1.46)
        self.assertAlmostEqual(parseApparentMagnitude('10.7 J'), 10.7)

    def test_field_stars_use_true_positions_in_neighborhood(self) -> None:
        frames = self.animator.animationFrames
        # End of pullback: Sol neighborhood should include catalog stars at true XYZ.
        pullEnd = int(PULLBACK_END * (frames - 1))
        focus, halfWidth = self.animator._cameraState(pullEnd)
        self.animator._viewFocus = focus
        self.animator._viewHalfWidthAu = halfWidth
        keep = self.animator._fieldStarDrawMask(halfWidth)
        self.assertIsNotNone(keep)
        self.assertGreater(int(np.count_nonzero(keep)), 100)

        # Tight triple hold (~12 kau) must not fake a sky shell of distant stars.
        wideHold = int(0.5 * (WIDE_OUT_ARRIVE + WIDE_OUT_END) * (frames - 1))
        focus, halfWidth = self.animator._cameraState(wideHold)
        self.animator._viewFocus = focus
        self.animator._viewHalfWidthAu = halfWidth
        keep = self.animator._fieldStarDrawMask(halfWidth)
        inView = 0 if keep is None else int(np.count_nonzero(keep))
        self.assertLess(inView, 5)

    def test_famous_asteroids_drawn_at_belt_and_kuiper(self) -> None:
        self.assertGreater(
            self.animator._famousAsteroidAlpha('main_belt', SOL_BELT_LINGER_HALF_AU),
            0.9,
        )
        self.assertGreater(
            self.animator._famousAsteroidAlpha('kuiper', SOL_HALF_WIDTH_AU),
            0.9,
        )
        self.assertLess(
            self.animator._famousAsteroidAlpha('main_belt', SOL_HALF_WIDTH_AU),
            0.2,
        )

    def test_ab_binary_hold_stays_at_close_scale(self) -> None:
        frames = self.animator.animationFrames
        arrive = int(np.ceil(AB_TRAVEL_END * (frames - 1)))
        hold = int(0.5 * (AB_TRAVEL_END + AB_HOLD_END) * (frames - 1))
        _, arriveHalf = self.animator._cameraState(arrive)
        _, holdHalf = self.animator._cameraState(hold)
        self.assertAlmostEqual(arriveHalf, self.animator.abHalfWidthAu, places=2)
        self.assertAlmostEqual(holdHalf, self.animator.abHalfWidthAu, places=2)

    def test_outer_planet_moves_during_sol_opening(self) -> None:
        frames = self.animator.animationFrames
        startFrame = int(np.ceil(SOL_OUTER_ARRIVE * (frames - 1)))
        endFrame = int(SOL_HOLD_END * (frames - 1))
        start = self.animator._planetPositionAu('Neptune', startFrame)
        end = self.animator._planetPositionAu('Neptune', endFrame)
        angle = float(
            np.arccos(
                np.clip(
                    np.dot(start, end) / (np.linalg.norm(start) * np.linalg.norm(end)),
                    -1.0,
                    1.0,
                )
            )
        )
        self.assertGreater(angle, np.radians(25.0))


if __name__ == '__main__':
    unittest.main()
