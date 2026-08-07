"""Tests for Blender flyby → cinematic body sprites."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest import mock

from animate.blender_body_sprites import (
    BlenderBodySpriteAtlas,
    flybyGifPath,
    frameToCircularRgba,
    loadBodySpriteFrames,
)
from animate.scenes.sol_centauri_cinematic import SolCentauriCinematicAnimator
from PIL import Image
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths

REPO_ROOT = Path(__file__).resolve().parents[1]


class SpriteHelperTests(unittest.TestCase):
    def test_flyby_gif_paths(self) -> None:
        earth = flybyGifPath('planet', 'Earth', 'dark')
        moon = flybyGifPath('moon', 'Moon', 'light')
        self.assertTrue(str(earth).endswith('planets/earth/earth_flyby_dark.gif'))
        self.assertTrue(str(moon).endswith('moons/moon/moon_flyby_light.gif'))

    def test_circular_rgba_shape_and_alpha(self) -> None:
        image = Image.new('RGB', (64, 64), (20, 80, 180))
        rgba = frameToCircularRgba(image, size=32)
        self.assertEqual(rgba.shape, (32, 32, 4))
        self.assertGreater(float(rgba[16, 16, 3]), 0.9)
        self.assertLess(float(rgba[0, 0, 3]), 0.05)

    def test_load_missing_returns_none(self) -> None:
        with mock.patch(
            'animate.blender_body_sprites.flybyGifPath',
            return_value=REPO_ROOT / 'does_not_exist.gif',
        ):
            self.assertIsNone(loadBodySpriteFrames('planet', 'Earth', 'dark'))


class SpriteAtlasIntegrationTests(unittest.TestCase):
    def test_loads_checked_in_earth_moon_flybys(self) -> None:
        earthPath = flybyGifPath('planet', 'Earth', 'dark')
        moonPath = flybyGifPath('moon', 'Moon', 'dark')
        if not earthPath.is_file() or not moonPath.is_file():
            self.skipTest('Earth/Moon flyby GIFs not present locally')

        atlas = BlenderBodySpriteAtlas('dark', maxFrames=4)
        self.assertTrue(atlas.hasEarth)
        self.assertTrue(atlas.hasMoon)
        assert atlas.earth is not None
        self.assertEqual(len(atlas.earth), 4)
        frame = atlas.earthFrame(0)
        assert frame is not None
        self.assertEqual(frame.shape[-1], 4)
        self.assertIs(atlas.earthFrame(4), atlas.earthFrame(0))


class BlenderBodyBillboardTests(unittest.TestCase):
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
        closeHalf = 0.01
        midHalf = 5.0
        close = animator._blenderBillboardRadiusAu(closeHalf, openCloseup=True, bodyScale=1.0)
        mid = animator._blenderBillboardRadiusAu(midHalf, openCloseup=False, bodyScale=1.0)
        far = animator._blenderBillboardRadiusAu(120.0, openCloseup=False, bodyScale=1.0)
        self.assertIsNotNone(close)
        self.assertIsNotNone(mid)
        self.assertIsNone(far)
        assert close is not None and mid is not None
        # Close-up uses a large fraction of the half-width; mid zoom is a speck.
        self.assertGreater(close / closeHalf, mid / midHalf)

    def test_draw_update_with_sprites_when_assets_present(self) -> None:
        earthPath = flybyGifPath('planet', 'Earth', 'light')
        if not earthPath.is_file():
            self.skipTest('Earth flyby GIF not present locally')

        animator = SolCentauriCinematicAnimator(
            self.system,
            starsCsvPath=self.starsCsvPath,
            useBlenderBodies=True,
        )
        self.assertIsNotNone(animator.blenderSprites)
        assert animator.blenderSprites is not None
        self.assertTrue(animator.blenderSprites.hasEarth)
        # Opening frame is Earth/Moon close-up — billboards should draw here.
        animator.update(0)
        surfaces = [
            artist
            for artist in animator.axes.get_children()
            if artist.__class__.__name__ == 'Poly3DCollection'
        ]
        self.assertGreaterEqual(len(surfaces), 1)


if __name__ == '__main__':
    unittest.main()
