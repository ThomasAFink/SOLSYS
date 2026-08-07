"""Unit tests for the Blender close-up / flyby pipeline."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from animate.scenes.blender.body_appearance import (
    TEXTURE_BODIES_ROOT,
    appearanceForCatalogName,
    registeredCatalogNames,
)
from animate.scenes.blender.body_scene import (
    SCHEMA_ID,
    BodyScene,
    buildBodyScene,
    buildMoonBodyScene,
    buildPlanetBodyScene,
    loadBodyScene,
)
from animate.scenes.blender.export_body import exportBodyScene, exportPlanetBodyScene
from animate.scenes.blender.flyby_camera import buildFlybyCameraPath, flybyCameraLocation
from animate.scenes.blender.flyby_scene import (
    assembleGifFromPngs,
    buildFlybyJob,
    preparePlanetFlybyExport,
    writeFlybyJob,
)
from animate.scenes.blender.load_body import loadPayload, summarizePayload
from animate.scenes.blender.render_flyby import JOB_SCHEMA_ID, loadJob
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
LOAD_BODY_SCRIPT = REPO_ROOT / 'animate' / 'scenes' / 'blender' / 'load_body.py'
RENDER_FLYBY_SCRIPT = REPO_ROOT / 'animate' / 'scenes' / 'blender' / 'render_flyby.py'


class BodySceneBuildTests(unittest.TestCase):
    def test_build_earth_scene(self) -> None:
        scene = buildPlanetBodyScene('Earth', frameCount=24)
        self.assertEqual(scene.schema, SCHEMA_ID)
        self.assertEqual(scene.body.name, 'Earth')
        self.assertEqual(scene.body.systemId, 'sol')
        self.assertEqual(len(scene.keyframes), 24)
        self.assertGreater(scene.body.displayRadiusAu, 0.0)
        self.assertGreater(scene.cameraHintDistanceAu, 0.0)
        radius = sum(component**2 for component in scene.keyframes[0].positionAu) ** 0.5
        self.assertAlmostEqual(radius, 1.0, delta=0.05)

    def test_unknown_planet_raises(self) -> None:
        with self.assertRaises(ValueError):
            buildPlanetBodyScene('Nibiru')

    def test_build_moon_scene(self) -> None:
        scene = buildMoonBodyScene('Moon', frameCount=16)
        self.assertEqual(scene.body.name, 'Moon')
        self.assertEqual(scene.body.kind, 'moon')
        self.assertEqual(scene.body.systemId, 'sol')
        self.assertEqual(len(scene.keyframes), 16)
        self.assertGreater(scene.body.displayRadiusAu, 0.0)
        viaDispatch = buildBodyScene('Moon', frameCount=8)
        self.assertEqual(viaDispatch.body.kind, 'moon')
        self.assertEqual(len(viaDispatch.keyframes), 8)

    def test_unknown_body_raises(self) -> None:
        with self.assertRaises(ValueError):
            buildBodyScene('Nibiru')

    def test_round_trip_json(self) -> None:
        scene = buildPlanetBodyScene('Jupiter', frameCount=8)
        restored = BodyScene.fromDict(json.loads(scene.toJson()))
        self.assertEqual(restored.body.name, 'Jupiter')
        self.assertEqual(len(restored.keyframes), 8)
        self.assertEqual(restored.keyframes[0].positionAu, scene.keyframes[0].positionAu)


class ExportAndLoadTests(unittest.TestCase):
    def test_export_and_host_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            outputDirectory = Path(temporaryDirectory)
            path = exportPlanetBodyScene('Earth', frameCount=12, outputDirectory=outputDirectory)
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, 'earth_body_scene.json')

            scene = loadBodyScene(path)
            self.assertEqual(scene.body.name, 'Earth')

            payload = loadPayload(path)
            summary = summarizePayload(payload)
            self.assertIn('Earth', summary)
            self.assertIn('12 keyframes', summary)

            completed = subprocess.run(
                [sys.executable, str(LOAD_BODY_SCRIPT), str(path)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('Earth', completed.stdout)
            self.assertIn('dry-run', completed.stdout.lower())


class BodyAppearanceTests(unittest.TestCase):
    def test_earth_pack_resolves_color_map(self) -> None:
        appearance = appearanceForCatalogName('Earth')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        self.assertEqual(appearance.bodyId, 'earth')
        self.assertEqual(appearance.kind, 'planet')
        maps = appearance.textures.existingMaps()
        self.assertIn('color', maps)
        self.assertIn('clouds', maps)
        self.assertTrue(maps['color'].is_file())
        self.assertTrue(maps['clouds'].is_file())
        self.assertTrue(str(maps['color']).endswith('earth/color.png'))
        self.assertTrue(str(maps['clouds']).endswith('earth/clouds.png'))
        self.assertTrue(appearance.atmosphere.enabled)
        self.assertGreater(appearance.atmosphere.scale, 1.0)
        jobAppearance = appearance.toJobDict()
        self.assertIn('atmosphere', jobAppearance)
        self.assertTrue(jobAppearance['atmosphere']['enabled'])
        self.assertIn('clouds', jobAppearance['textures'])
        self.assertIn('Earth', registeredCatalogNames())
        self.assertTrue((TEXTURE_BODIES_ROOT / 'earth' / 'color.png').is_file())
        self.assertTrue((TEXTURE_BODIES_ROOT / 'earth' / 'clouds.png').is_file())

    def test_moon_pack_resolves_color_map_without_atmosphere(self) -> None:
        appearance = appearanceForCatalogName('Moon')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        self.assertEqual(appearance.bodyId, 'moon')
        self.assertEqual(appearance.kind, 'moon')
        maps = appearance.textures.existingMaps()
        self.assertIn('color', maps)
        self.assertNotIn('clouds', maps)
        self.assertFalse(appearance.atmosphere.enabled)
        jobAppearance = appearance.toJobDict()
        self.assertNotIn('atmosphere', jobAppearance)
        self.assertIn('color', jobAppearance['textures'])
        self.assertIn('Moon', registeredCatalogNames())
        self.assertTrue((TEXTURE_BODIES_ROOT / 'moon' / 'color.png').is_file())

    def test_unknown_catalog_name_has_no_pack(self) -> None:
        self.assertIsNone(appearanceForCatalogName('Nibiru'))


class FlybyPipelineTests(unittest.TestCase):
    def test_prepare_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            path = preparePlanetFlybyExport(
                'Mars', frameCount=6, outputDirectory=temporaryDirectory
            )
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, 'mars_body_scene.json')

    def test_camera_path_moves_around_body(self) -> None:
        path = buildFlybyCameraPath(0.03, frameCount=12)
        self.assertEqual(len(path), 12)
        first = path[0].cameraAu
        last = path[-1].cameraAu
        self.assertNotAlmostEqual(first[0], last[0], places=4)
        self.assertGreater(path[-1].bodyRotationDeg, path[0].bodyRotationDeg)
        distance = sum(component**2 for component in flybyCameraLocation(0, 12, 0.03)) ** 0.5
        self.assertAlmostEqual(distance, 0.03 * 4.4, delta=1e-9)

    def test_build_and_validate_flyby_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            framesDirectory = Path(temporaryDirectory) / 'frames'
            job = buildFlybyJob(
                'Earth',
                theme='dark',
                frameCount=8,
                framesDirectory=framesDirectory,
            )
            self.assertEqual(job['schema'], JOB_SCHEMA_ID)
            self.assertEqual(job['theme'], 'dark')
            self.assertEqual(len(job['frames']), 8)
            self.assertIn('appearance', job)
            self.assertEqual(job['appearance']['bodyId'], 'earth')
            self.assertIn('color', job['appearance']['textures'])
            self.assertIn('clouds', job['appearance']['textures'])
            self.assertTrue(job['appearance']['atmosphere']['enabled'])
            jobPath = writeFlybyJob(job, Path(temporaryDirectory) / 'earth_flyby_dark_job.json')
            loaded = loadJob(jobPath)
            self.assertEqual(loaded['body']['name'], 'Earth')

            completed = subprocess.run(
                [sys.executable, str(RENDER_FLYBY_SCRIPT), str(jobPath)],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn('dry-run', completed.stdout.lower())

    def test_build_moon_flyby_job(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            path = exportBodyScene('Moon', frameCount=6, outputDirectory=temporaryDirectory)
            self.assertEqual(path.name, 'moon_body_scene.json')
            job = buildFlybyJob(
                'Moon',
                theme='light',
                frameCount=6,
                framesDirectory=Path(temporaryDirectory) / 'frames',
            )
            self.assertEqual(job['body']['name'], 'Moon')
            self.assertEqual(job['body']['kind'], 'moon')
            self.assertEqual(job['appearance']['bodyId'], 'moon')
            self.assertIn('color', job['appearance']['textures'])
            self.assertNotIn('atmosphere', job['appearance'])

    def test_assemble_gif_from_pngs(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            root = Path(temporaryDirectory)
            framePaths: list[Path] = []
            for index in range(3):
                path = root / f'frame_{index:04d}.png'
                Image.new('RGB', (32, 32), color=(index * 40, 80, 160)).save(path)
                framePaths.append(path)
            gifPath = assembleGifFromPngs(framePaths, root / 'earth_flyby_dark.gif', fps=10)
            self.assertTrue(gifPath.is_file())
            with Image.open(gifPath) as gif:
                self.assertEqual(gif.n_frames, 3)


if __name__ == '__main__':
    unittest.main()
