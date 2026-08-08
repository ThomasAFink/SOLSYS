"""Tests for Blender spin-loop frames in the Sol→Centauri cinematic."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from animate.blender_body_sprites import (
    BlenderBodySpriteAtlas,
    _blendSpinFrames,
    loadSpinLoopFrames,
    spinLoopAvailable,
    tidalLockFrameIndex,
    tidalLockFramePosition,
)
from animate.scenes.blender.flyby_camera import buildSpinCameraPath
from animate.scenes.blender.flyby_scene import buildSpinJob, spinFramesDirectory
from animate.scenes.sol_centauri_cinematic import SolCentauriCinematicAnimator
from PIL import Image
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class SpinCameraTests(unittest.TestCase):
    def test_spin_path_is_fixed_camera_full_turn(self) -> None:
        path = buildSpinCameraPath(0.01, frameCount=8)
        self.assertEqual(len(path), 8)
        self.assertEqual(path[0].cameraAu, path[-1].cameraAu)
        self.assertAlmostEqual(path[0].bodyRotationDeg, 0.0)
        self.assertAlmostEqual(path[4].bodyRotationDeg, 180.0)
        self.assertLess(path[-1].bodyRotationDeg, 360.0)

    def test_spin_job_requests_transparent_film(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            job = buildSpinJob('Earth', theme='dark', framesDirectory=temporary / 'frames')
            self.assertEqual(job['mode'], 'spin')
            self.assertTrue(job['filmTransparent'])
            self.assertEqual(job['frames'][0]['cameraAu'], job['frames'][-1]['cameraAu'])


class TidalLockTests(unittest.TestCase):
    def test_tidal_lock_index_tracks_orbital_phase(self) -> None:
        self.assertEqual(tidalLockFrameIndex(0.0, 48), 0)
        self.assertEqual(tidalLockFrameIndex(np.pi, 48), 24)
        self.assertEqual(tidalLockFrameIndex(2.0 * np.pi * 0.999, 48), 47)
        # Full orbit wraps to the same face.
        self.assertEqual(tidalLockFrameIndex(2.0 * np.pi, 48), 0)
        self.assertAlmostEqual(tidalLockFramePosition(np.pi, 48), 24.0, places=9)

    def test_blend_spin_frames_is_smooth_between_neighbors(self) -> None:
        frames = [
            np.zeros((2, 2, 4), dtype=np.float32),
            np.ones((2, 2, 4), dtype=np.float32),
        ]
        mid = _blendSpinFrames(frames, 0.5)
        np.testing.assert_allclose(mid, 0.5)

    def test_moon_frame_uses_orbital_phase_not_wall_clock(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            moonDir = spinFramesDirectory('Moon', 'dark', outputDirectory=temporary)
            moonDir.mkdir(parents=True)
            for index in range(8):
                Image.new('RGBA', (16, 16), (index * 30, 160, 160, 220)).save(
                    moonDir / f'frame_{index:04d}.png'
                )
            earthDir = spinFramesDirectory('Earth', 'dark', outputDirectory=temporary)
            earthDir.mkdir(parents=True)
            Image.new('RGBA', (16, 16), (20, 90, 180, 230)).save(earthDir / 'frame_0000.png')
            atlas = BlenderBodySpriteAtlas('dark', outputDirectory=temporary)
            a = atlas.moonFrame(0, orbitalPhaseRad=0.0, resolution=16)
            b = atlas.moonFrame(100, orbitalPhaseRad=0.0, resolution=16)
            c = atlas.moonFrame(0, orbitalPhaseRad=np.pi, resolution=16)
            assert a is not None and b is not None and c is not None
            np.testing.assert_allclose(a, b)
            self.assertFalse(np.allclose(a, c))


class SpinAtlasTests(unittest.TestCase):
    def test_load_spin_frames_from_png_loop(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            directory = spinFramesDirectory('Earth', 'dark', outputDirectory=temporary)
            directory.mkdir(parents=True)
            for index in range(4):
                Image.new('RGBA', (32, 32), (10, 80, 180, 200)).save(
                    directory / f'frame_{index:04d}.png'
                )
            frames = loadSpinLoopFrames('Earth', 'dark', outputDirectory=temporary)
            assert frames is not None
            self.assertEqual(len(frames), 4)
            self.assertEqual(frames[0].shape, (32, 32, 4))

            moonDir = spinFramesDirectory('Moon', 'dark', outputDirectory=temporary)
            moonDir.mkdir(parents=True)
            Image.new('RGBA', (32, 32), (160, 160, 160, 220)).save(moonDir / 'frame_0000.png')
            atlas = BlenderBodySpriteAtlas('dark', outputDirectory=temporary)
            self.assertTrue(atlas.hasEarth)
            self.assertTrue(atlas.hasMoon)
            self.assertEqual(atlas.loadedBodyNames(), ())
            frame = atlas.earthFrame(5, resolution=16)
            assert frame is not None
            self.assertEqual(frame.shape, (16, 16, 4))
            self.assertEqual(atlas.loadedBodyNames(), ('Earth',))

    def test_lazy_load_jupiter_and_missing_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            jupiterDir = spinFramesDirectory('Jupiter', 'dark', outputDirectory=temporary)
            jupiterDir.mkdir(parents=True)
            Image.new('RGBA', (16, 16), (200, 160, 90, 230)).save(jupiterDir / 'frame_0000.png')
            atlas = BlenderBodySpriteAtlas('dark', outputDirectory=temporary)
            self.assertTrue(spinLoopAvailable('Jupiter', 'dark', outputDirectory=temporary))
            self.assertTrue(atlas.hasBody('Jupiter'))
            self.assertFalse(atlas.hasBody('Mars'))
            disk = atlas.bodyFrame('Jupiter', 3, resolution=16)
            assert disk is not None
            self.assertEqual(disk.shape, (16, 16, 4))
            self.assertIsNone(atlas.bodyFrame('Mars', 0, resolution=16))


class BlenderBodyOverlayTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**paths).load('alpha_centauri')
        self.starsCsvPath = paths['starsCsvPath']

    def tearDown(self) -> None:
        import matplotlib.pyplot as plt

        for figure in list(plt.get_fignums()):
            plt.close(figure)

    def test_flag_off_skips_atlas(self) -> None:
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=False,
        )
        self.assertIsNone(animator.blenderSprites)

    def test_billboard_radius_is_world_fixed_then_drops(self) -> None:
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=False,
        )
        close = animator._blenderBillboardRadiusAu(0.14, openCloseup=True, bodyScale=1.0)
        mid = animator._blenderBillboardRadiusAu(1.5, openCloseup=False, bodyScale=1.0)
        outerSol = animator._blenderBillboardRadiusAu(42.0, openCloseup=False, bodyScale=1.0)
        far = animator._blenderBillboardRadiusAu(120.0, openCloseup=False, bodyScale=1.0)
        self.assertIsNotNone(close)
        self.assertIsNotNone(mid)
        self.assertIsNotNone(outerSol)
        self.assertIsNone(far)
        assert close is not None and mid is not None and outerSol is not None
        self.assertAlmostEqual(close, mid, places=9)
        self.assertAlmostEqual(close, outerSol, places=9)

    def test_earth_label_pad_shrinks_on_inner_sol_zoom_out(self) -> None:
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=False,
        )
        closeFrac = animator._blenderBillboardFracRadius(0.16, 1.0)
        innerFrac = animator._blenderBillboardFracRadius(6.5, 1.0)
        self.assertIsNotNone(closeFrac)
        self.assertIsNotNone(innerFrac)
        assert closeFrac is not None and innerFrac is not None
        closePad = animator._blenderBodyLabelPad(closeFrac)
        innerPad = animator._blenderBodyLabelPad(innerFrac)
        self.assertAlmostEqual(closePad, 0.03, places=3)
        self.assertLess(innerPad, 0.01)
        self.assertLess(innerFrac + innerPad, closeFrac * 0.35)

    def test_update_paints_overlay_when_spin_assets_present(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            for body in ('Earth', 'Moon', 'Jupiter', 'Saturn'):
                directory = spinFramesDirectory(body, 'light', outputDirectory=temporary)
                directory.mkdir(parents=True)
                Image.new('RGBA', (48, 48), (20, 90, 180, 230)).save(directory / 'frame_0000.png')
            animator = SolCentauriCinematicAnimator(
                self.system,
                style='default',
                starsCsvPath=self.starsCsvPath,
                useBlenderBodies=True,
            )
            animator.blenderSprites = BlenderBodySpriteAtlas('light', outputDirectory=temporary)
            animator.update(0)
            self.assertGreaterEqual(len(animator.bodyOverlay.get_images()), 1)

    def test_missing_planet_pack_falls_back_to_scatter(self) -> None:
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=True,
        )
        with tempfile.TemporaryDirectory() as temporaryName:
            temporary = Path(temporaryName)
            # Only Earth present — Mars must remain a catalog dot path (no queue).
            earthDir = spinFramesDirectory('Earth', 'dark', outputDirectory=temporary)
            earthDir.mkdir(parents=True)
            Image.new('RGBA', (16, 16), (20, 90, 180, 230)).save(earthDir / 'frame_0000.png')
            animator.blenderSprites = BlenderBodySpriteAtlas('dark', outputDirectory=temporary)
            self.assertTrue(animator._blenderBodyAvailable('Earth'))
            self.assertFalse(animator._blenderBodyAvailable('Mars'))
            queued = animator._queueBlenderBody(
                'Mars',
                np.array([1.5, 0.0, 0.0]),
                0,
                6.0,
                openCloseup=False,
                bodyScale=0.58,
                orbitalPhaseRad=None,
                suppressDotFallback=False,
            )
            self.assertFalse(queued)


if __name__ == '__main__':
    unittest.main()
