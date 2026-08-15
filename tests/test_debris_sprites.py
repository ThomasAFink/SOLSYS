"""Tests for soft debris / dust-clump sprites (SOLSYS-78)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from animate.debris_sprites import (
    DEBRIS_CLUMPS,
    debrisCatalogNames,
    debrisSpriteAvailable,
    debrisSpritePath,
    loadDebrisSprite,
    renderDebrisClumpRgba,
    specForCatalogName,
    writeDebrisClumpTextures,
    writeOccultationQa,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class DebrisSpritesTests(unittest.TestCase):
    def test_three_distinct_catalog_clumps(self) -> None:
        names = debrisCatalogNames()
        self.assertEqual(len(names), 3)
        self.assertEqual(len(set(names)), 3)
        for name in names:
            self.assertTrue(name.startswith('Debris Clump '))
            self.assertIsNotNone(specForCatalogName(name))
        # Must not collide with named asteroid catalog strings.
        self.assertIsNone(specForCatalogName('Ceres'))
        self.assertIsNone(specForCatalogName('Vesta'))

    def test_procedural_rgba_has_soft_alpha(self) -> None:
        rgba = renderDebrisClumpRgba(DEBRIS_CLUMPS[0], size=128)
        self.assertEqual(rgba.shape, (128, 128, 4))
        alpha = rgba[..., 3]
        self.assertGreater(float(alpha.max()), 0.55)
        self.assertLess(float(alpha.min()), 0.05)
        # Corners should be mostly empty (soft cloud, not a filled square).
        corner = float(alpha[2, 2] + alpha[2, -3] + alpha[-3, 2] + alpha[-3, -3]) / 4.0
        self.assertLess(corner, 0.08)
        # Dense enough to dim a photosphere, still a soft cloud (not a filled disk).
        self.assertLess(float(alpha.mean()), 0.48)
        # Dark enough to silhouette against a bright star disk.
        opaque = alpha > 0.4
        self.assertTrue(bool(np.any(opaque)))
        meanRgb = float(np.mean(rgba[..., :3][opaque]))
        self.assertLess(meanRgb, 0.22)

    def test_clumps_differ_from_each_other(self) -> None:
        frames = [renderDebrisClumpRgba(spec, size=96) for spec in DEBRIS_CLUMPS]
        for left, right in zip(frames, frames[1:], strict=False):
            delta = float(np.mean(np.abs(left - right)))
            self.assertGreater(delta, 0.02)

    def test_write_and_load_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            written = writeDebrisClumpTextures(root=root, size=64)
            self.assertEqual(len(written), 3)
            for spec in DEBRIS_CLUMPS:
                self.assertTrue(debrisSpriteAvailable(spec.catalogName, root=root))
                path = debrisSpritePath(spec.clumpId, root=root)
                self.assertTrue(path.is_file())
                rgba = loadDebrisSprite(spec.catalogName, root=root)
                self.assertIsNotNone(rgba)
                assert rgba is not None
                self.assertEqual(rgba.shape[2], 4)
                self.assertGreater(float(rgba[..., 3].max()), 0.2)

    def test_repo_textures_present(self) -> None:
        """Committed packs under data/textures/debris/."""
        for spec in DEBRIS_CLUMPS:
            path = debrisSpritePath(spec.clumpId)
            self.assertTrue(path.is_file(), msg=f'missing {path}')
            rgba = loadDebrisSprite(spec.catalogName)
            self.assertIsNotNone(rgba)
            assert rgba is not None
            self.assertGreater(float(rgba[..., 3].max()), 0.3)

    def test_occultation_qa_composites_when_sun_spin_exists(self) -> None:
        sun = (
            REPO_ROOT
            / 'output'
            / 'animate'
            / 'blender'
            / 'stars'
            / 'sun'
            / 'sun_spin_dark'
            / 'frame_0000.png'
        )
        if not sun.is_file():
            self.skipTest('Sol spin frame not present')
        if not debrisSpriteAvailable('Debris Clump A'):
            self.skipTest('debris packs not generated yet')
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'qa.png'
            path = writeOccultationQa(outputPath=out, theme='dark')
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 10_000)


if __name__ == '__main__':
    unittest.main()
