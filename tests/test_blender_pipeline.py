"""Unit tests for the Blender close-up pipeline scaffold."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from animate.scenes.blender.body_scene import (
    SCHEMA_ID,
    BodyScene,
    buildPlanetBodyScene,
    loadBodyScene,
)
from animate.scenes.blender.export_body import exportPlanetBodyScene
from animate.scenes.blender.flyby_scene import (
    FLYBY_EXTENSION_POINT,
    preparePlanetFlybyExport,
    renderPlanetFlyby,
)
from animate.scenes.blender.load_body import loadPayload, summarizePayload

REPO_ROOT = Path(__file__).resolve().parents[1]
LOAD_BODY_SCRIPT = REPO_ROOT / 'animate' / 'scenes' / 'blender' / 'load_body.py'


class BodySceneBuildTests(unittest.TestCase):
    def test_build_earth_scene(self) -> None:
        scene = buildPlanetBodyScene('Earth', frameCount=24)
        self.assertEqual(scene.schema, SCHEMA_ID)
        self.assertEqual(scene.body.name, 'Earth')
        self.assertEqual(scene.body.systemId, 'sol')
        self.assertEqual(len(scene.keyframes), 24)
        self.assertGreater(scene.body.displayRadiusAu, 0.0)
        self.assertGreater(scene.cameraHintDistanceAu, 0.0)
        # First keyframe should sit near 1 AU for a low-eccentricity Earth orbit.
        radius = sum(component**2 for component in scene.keyframes[0].positionAu) ** 0.5
        self.assertAlmostEqual(radius, 1.0, delta=0.05)

    def test_unknown_planet_raises(self) -> None:
        with self.assertRaises(ValueError):
            buildPlanetBodyScene('Nibiru')

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


class FlybyExtensionTests(unittest.TestCase):
    def test_prepare_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            path = preparePlanetFlybyExport('Mars', frameCount=6, outputDirectory=temporaryDirectory)
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, 'mars_body_scene.json')

    def test_render_flyby_is_extension_point(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            with self.assertRaises(NotImplementedError) as context:
                renderPlanetFlyby('Earth', outputDirectory=temporaryDirectory)
            message = str(context.exception)
            self.assertIn(FLYBY_EXTENSION_POINT, message)
            self.assertIn('earth_body_scene.json', message)


if __name__ == '__main__':
    unittest.main()
