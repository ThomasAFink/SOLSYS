"""Tests for Blender texture-pack globes in the Sol→Centauri cinematic."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from animate.blender_body_sprites import (
    BlenderBodySpriteAtlas,
    loadBodyGlobePack,
    renderGlobeDisk,
)
from animate.scenes.blender.body_appearance import appearanceForCatalogName
from animate.scenes.sol_centauri_cinematic import SolCentauriCinematicAnimator
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class GlobeHelperTests(unittest.TestCase):
    def test_render_globe_disk_shape_and_edge(self) -> None:
        color = np.zeros((32, 64, 3), dtype=np.float32)
        color[:, :, 2] = 0.8
        disk = renderGlobeDisk(
            color,
            spinDeg=0.0,
            sunDirection=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            resolution=48,
        )
        self.assertEqual(disk.shape, (48, 48, 4))
        self.assertGreater(float(disk[24, 24, 3]), 0.9)
        self.assertLess(float(disk[0, 0, 3]), 0.05)

    def test_load_missing_pack_returns_none(self) -> None:
        with mock.patch(
            'animate.blender_body_sprites.appearanceForCatalogName',
            return_value=None,
        ):
            self.assertIsNone(loadBodyGlobePack('Earth'))


class TexturePackIntegrationTests(unittest.TestCase):
    def test_loads_earth_moon_texture_packs(self) -> None:
        earthAppearance = appearanceForCatalogName('Earth')
        moonAppearance = appearanceForCatalogName('Moon')
        if earthAppearance is None or moonAppearance is None:
            self.skipTest('Earth/Moon appearance registry missing')
        if earthAppearance.textures.color is None or not earthAppearance.textures.color.is_file():
            self.skipTest('Earth color texture not present')
        if moonAppearance.textures.color is None or not moonAppearance.textures.color.is_file():
            self.skipTest('Moon color texture not present')

        atlas = BlenderBodySpriteAtlas('dark')
        self.assertTrue(atlas.hasEarth)
        self.assertTrue(atlas.hasMoon)
        frame0 = atlas.earthFrame(0, resolution=64)
        frame8 = atlas.earthFrame(8, resolution=64)
        assert frame0 is not None and frame8 is not None
        self.assertEqual(frame0.shape, (64, 64, 4))
        # Slow spin should change the disk without jumping to a different camera.
        self.assertFalse(np.allclose(frame0, frame8))


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
        self.assertFalse(animator.useBlenderBodies)
        self.assertIsNone(animator.blenderSprites)

    def test_billboard_radius_closeup_and_far(self) -> None:
        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=False,
        )
        closeHalf = 0.14
        midHalf = 5.0
        close = animator._blenderBillboardRadiusAu(closeHalf, openCloseup=True, bodyScale=1.0)
        mid = animator._blenderBillboardRadiusAu(midHalf, openCloseup=False, bodyScale=1.0)
        far = animator._blenderBillboardRadiusAu(120.0, openCloseup=False, bodyScale=1.0)
        self.assertIsNotNone(close)
        self.assertIsNotNone(mid)
        self.assertIsNone(far)
        assert close is not None and mid is not None
        self.assertGreater(close / closeHalf, mid / midHalf)

    def test_update_paints_overlay_images_when_textures_present(self) -> None:
        earth = appearanceForCatalogName('Earth')
        if earth is None or earth.textures.color is None or not earth.textures.color.is_file():
            self.skipTest('Earth color texture not present')

        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=True,
        )
        self.assertIsNotNone(animator.blenderSprites)
        assert animator.blenderSprites is not None
        self.assertTrue(animator.blenderSprites.hasEarth)
        animator.update(0)
        images = [
            artist
            for artist in animator.bodyOverlay.get_images()
        ]
        self.assertGreaterEqual(len(images), 1)


if __name__ == '__main__':
    unittest.main()
