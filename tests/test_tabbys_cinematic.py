"""Tests for Tabby's Star lightcurve cinema (#73)."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from animate.scenes.tabbys_star import (
    buildOrbitingClumps,
    findDipCrossingFrames,
    loadKeplerLightCurve,
    sampleSeriesToFrames,
)
from animate.scenes.tabbys_star_cinematic import (
    ANIMATION_FRAMES,
    TABBYS_CATALOG_NAME,
    TabbysStarCinematicAnimator,
    _dipStrength,
    _losWeight,
)
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class TabbysLightcurveHelpersTests(unittest.TestCase):
    def test_kepler_csv_loads(self) -> None:
        timeBkjd, flux = loadKeplerLightCurve(REPO_ROOT / 'data' / 'tabbys_star_lightcurve.csv')
        self.assertGreater(len(timeBkjd), 100)
        self.assertEqual(len(timeBkjd), len(flux))
        self.assertTrue(np.all(np.isfinite(flux)))
        self.assertGreater(float(np.nanmax(flux)), 0.95)
        self.assertLess(float(np.nanmin(flux)), 0.95)

    def test_dip_events_drive_clumps(self) -> None:
        _time, flux = loadKeplerLightCurve(REPO_ROOT / 'data' / 'tabbys_star_lightcurve.csv')
        fluxByFrame = sampleSeriesToFrames(flux, ANIMATION_FRAMES, reduceMin=True)
        dips = findDipCrossingFrames(fluxByFrame)
        self.assertGreaterEqual(len(dips), 2)
        clumps = buildOrbitingClumps(dips)
        self.assertEqual(len(clumps), len(dips))
        # Deeper dip → larger/more opaque strength.
        deepest = max(clumps, key=lambda item: item.dipDepth)
        shallowest = min(clumps, key=lambda item: item.dipDepth)
        self.assertGreaterEqual(_dipStrength(deepest), _dipStrength(shallowest))
        # Crossing frames are unique and within the GIF.
        frames = [clump.crossingFrame for clump in clumps]
        self.assertEqual(len(frames), len(set(frames)))
        self.assertTrue(all(0 <= frame < ANIMATION_FRAMES for frame in frames))


class TabbysStarPackTests(unittest.TestCase):
    def test_tabbys_star_pack_is_emissive_f_star(self) -> None:
        from animate.scenes.blender.body_appearance import appearanceForCatalogName
        from animate.scenes.blender.body_scene import buildBodyScene
        from animate.scenes.blender.export_body import bodyOutputDirectory, bodyStem

        for catalogName in ("Tabby's Star", 'KIC 8462852'):
            appearance = appearanceForCatalogName(catalogName)
            self.assertIsNotNone(appearance, catalogName)
            assert appearance is not None
            self.assertEqual(appearance.kind, 'star')
            self.assertEqual(appearance.bodyId, 'tabbys_star')
            self.assertFalse(appearance.atmosphere.enabled)
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps)
            self.assertTrue(maps['color'].is_file())

        scene = buildBodyScene("Tabby's Star", frameCount=8)
        self.assertEqual(scene.body.kind, 'star')
        self.assertEqual(scene.body.name, "Tabby's Star")
        self.assertEqual(scene.body.systemId, 'tabbys_star')
        self.assertEqual(bodyStem("Tabby's Star"), 'tabbys_star')
        self.assertTrue(
            bodyOutputDirectory('star', "Tabby's Star").as_posix().endswith('stars/tabbys_star')
        )


class TabbysCinematicAnimatorTests(unittest.TestCase):
    def setUp(self) -> None:
        from animate.blender_body_sprites import spinLoopAvailable

        paths = defaultDataPaths(REPO_ROOT)
        self.paths = paths
        self.system = SystemCatalog(**paths).load('tabbys_star')
        self.hasSpin = spinLoopAvailable(TABBYS_CATALOG_NAME, 'dark')

    def test_catalog_has_no_planets_and_no_sol_framing(self) -> None:
        self.assertEqual(self.system.systemId, 'tabbys_star')
        self.assertEqual(len(self.system.planets), 0)
        self.assertGreaterEqual(len(self.system.stars), 1)
        host = self.system.stars[0]
        self.assertEqual(host.starName, "Tabby's Star")
        self.assertGreater(host.distanceLy, 1000.0)

    def test_animator_syncs_playhead_and_occultation_windows(self) -> None:
        if not self.hasSpin:
            self.skipTest("Tabby's Star Blender spin not present")
        animator = TabbysStarCinematicAnimator(
            style='dark_background',
            starsCsvPath=self.paths['starsCsvPath'],
            requireBlenderBody=True,
        )
        try:
            self.assertGreaterEqual(len(animator.clumps), 2)
            self.assertEqual(len(animator.fluxByFrame), ANIMATION_FRAMES)
            # Playhead time is monotonic with frame.
            times = [float(animator.timeByFrame[frame]) for frame in range(0, ANIMATION_FRAMES, 40)]
            self.assertEqual(times, sorted(times))
            # At each clump crossing, LOS weight peaks and flux is dipped.
            for clump in animator.clumps:
                weight = _losWeight(clump, clump.crossingFrame, ANIMATION_FRAMES)
                self.assertGreater(weight, 0.95)
                self.assertLess(float(animator.fluxByFrame[clump.crossingFrame]), 0.99)
            # No Sol travel language in the title path — system stays Tabby.
            self.assertEqual(animator.system.systemId, 'tabbys_star')
            # Framing stays fixed (no dip zoom that reads as the star swelling).
            animator.update(0)
            quietLim = animator.starAxes.get_xlim()
            animator.update(animator.deepestClump.crossingFrame)
            deepLim = animator.starAxes.get_xlim()
            self.assertEqual(quietLim, deepLim)
        finally:
            import matplotlib.pyplot as plt

            plt.close(animator.figure)

    def test_missing_spin_raises_clear_error(self) -> None:
        if self.hasSpin:
            self.skipTest('spin present — cannot assert missing-pack error')
        with self.assertRaises(FileNotFoundError):
            TabbysStarCinematicAnimator(
                style='dark_background',
                starsCsvPath=self.paths['starsCsvPath'],
                requireBlenderBody=True,
            )


if __name__ == '__main__':
    unittest.main()
