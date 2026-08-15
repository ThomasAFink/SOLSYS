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
    buildAsteroidBodyScene,
    buildBodyScene,
    buildMoonBodyScene,
    buildPlanetBodyScene,
    buildSunBodyScene,
    colorRgbaForName,
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

    def test_build_asteroid_and_dwarf_scenes(self) -> None:
        ceres = buildAsteroidBodyScene('Ceres', frameCount=12)
        self.assertEqual(ceres.body.name, 'Ceres')
        self.assertEqual(ceres.body.kind, 'dwarf_planet')
        self.assertEqual(len(ceres.keyframes), 12)
        vesta = buildBodyScene('Vesta', frameCount=8)
        self.assertEqual(vesta.body.kind, 'asteroid')
        self.assertGreater(vesta.body.displayRadiusAu, 0.0)
        # Hex catalog colors resolve (not the gray fallback).
        rgba = colorRgbaForName('#C4A882')
        self.assertAlmostEqual(rgba[0], 196 / 255.0, places=5)
        self.assertAlmostEqual(rgba[1], 168 / 255.0, places=5)

    def test_build_sun_scene(self) -> None:
        scene = buildSunBodyScene(frameCount=12)
        self.assertEqual(scene.body.name, 'Sun')
        self.assertEqual(scene.body.kind, 'star')
        self.assertEqual(scene.body.semiMajorAxisAu, 0.0)
        self.assertEqual(len(scene.keyframes), 12)
        self.assertEqual(scene.keyframes[0].positionAu, (0.0, 0.0, 0.0))
        self.assertEqual(scene.keyframes[-1].positionAu, (0.0, 0.0, 0.0))
        viaDispatch = buildBodyScene('Sun', frameCount=8)
        self.assertEqual(viaDispatch.body.kind, 'star')
        self.assertGreater(viaDispatch.body.displayRadiusAu, 0.0)

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
            self.assertEqual(path.parent.as_posix().endswith('planets/earth'), True)

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
    def test_sun_pack_is_emissive_star_without_atmosphere_shell(self) -> None:
        from animate.scenes.blender.body_appearance import registeredStarCatalogNames
        from animate.scenes.blender.export_body import bodyOutputDirectory

        self.assertEqual(
            registeredStarCatalogNames(),
            (
                'Alpha Centauri A',
                'Alpha Centauri B',
                'KIC 8462852',
                'Proxima Centauri',
                'Sun',
                'TRAPPIST-1',
                "Tabby's Star",
            ),
        )
        appearance = appearanceForCatalogName('Sun')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        self.assertEqual(appearance.kind, 'star')
        # Fresnel atmosphere shell reads as a hard pixelated ring on emissive stars.
        self.assertFalse(appearance.atmosphere.enabled)
        maps = appearance.textures.existingMaps()
        self.assertIn('color', maps)
        self.assertTrue(maps['color'].is_file())
        job = buildFlybyJob('Sun', theme='dark', framesDirectory=Path('/tmp/sun_frames'))
        self.assertEqual(job['body']['kind'], 'star')
        self.assertEqual(job['appearance']['kind'], 'star')
        self.assertNotIn('atmosphere', job['appearance'])
        self.assertTrue((TEXTURE_BODIES_ROOT / 'sun' / 'color.png').is_file())
        self.assertEqual(
            bodyOutputDirectory('star', 'Sun').as_posix().endswith('stars/sun'),
            True,
        )

    def test_proxima_planet_packs_resolve(self) -> None:
        from animate.scenes.blender.body_scene import buildBodyScene
        from animate.scenes.blender.export_body import bodyOutputDirectory

        for catalogName, bodyId in (('Proxima b', 'proxima_b'), ('Proxima d', 'proxima_d')):
            appearance = appearanceForCatalogName(catalogName)
            self.assertIsNotNone(appearance, catalogName)
            assert appearance is not None
            self.assertEqual(appearance.kind, 'planet')
            self.assertEqual(appearance.bodyId, bodyId)
            self.assertTrue(appearance.atmosphere.enabled)
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps)
            self.assertTrue(maps['color'].is_file())
            scene = buildBodyScene(catalogName, frameCount=8)
            self.assertEqual(scene.body.kind, 'planet')
            self.assertEqual(scene.body.systemId, 'alpha_centauri')
            self.assertTrue(
                bodyOutputDirectory('planet', catalogName).as_posix().endswith(f'planets/{bodyId}')
            )

    def test_trappist_1_planet_packs_resolve(self) -> None:
        from animate.scenes.blender.body_scene import buildBodyScene
        from animate.scenes.blender.export_body import bodyOutputDirectory, bodyStem

        letters = ('b', 'c', 'd', 'e', 'f', 'g', 'h')
        for letter in letters:
            catalogName = f'TRAPPIST-1 {letter}'
            bodyId = f'trappist_1_{letter}'
            appearance = appearanceForCatalogName(catalogName)
            self.assertIsNotNone(appearance, catalogName)
            assert appearance is not None
            self.assertEqual(appearance.kind, 'planet')
            self.assertEqual(appearance.bodyId, bodyId)
            self.assertTrue(appearance.atmosphere.enabled)
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps)
            self.assertTrue(maps['color'].is_file())
            scene = buildBodyScene(catalogName, frameCount=8)
            self.assertEqual(scene.body.kind, 'planet')
            self.assertEqual(scene.body.systemId, 'trappist_1')
            self.assertTrue(
                bodyOutputDirectory('planet', catalogName)
                .as_posix()
                .endswith(f'planets/{bodyStem(catalogName)}')
            )

    def test_alpha_centauri_star_packs_are_emissive(self) -> None:
        from animate.scenes.blender.body_scene import buildBodyScene
        from animate.scenes.blender.export_body import bodyOutputDirectory

        for catalogName, bodyId in (
            ('Alpha Centauri A', 'alpha_centauri_a'),
            ('Alpha Centauri B', 'alpha_centauri_b'),
            ('Proxima Centauri', 'proxima_centauri'),
            ("Tabby's Star", 'tabbys_star'),
        ):
            appearance = appearanceForCatalogName(catalogName)
            self.assertIsNotNone(appearance, catalogName)
            assert appearance is not None
            self.assertEqual(appearance.kind, 'star')
            self.assertEqual(appearance.bodyId, bodyId)
            self.assertFalse(appearance.atmosphere.enabled)
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps)
            self.assertTrue(maps['color'].is_file())
            scene = buildBodyScene(catalogName, frameCount=8)
            self.assertEqual(scene.body.kind, 'star')
            self.assertEqual(scene.body.name, catalogName)
            self.assertTrue(
                bodyOutputDirectory('star', catalogName).as_posix().endswith(f'stars/{bodyId}')
            )

    def test_trappist_1_host_pack_is_emissive_ultracool_dwarf(self) -> None:
        from animate.scenes.blender.body_scene import buildBodyScene
        from animate.scenes.blender.export_body import bodyOutputDirectory, bodyStem

        appearance = appearanceForCatalogName('TRAPPIST-1')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        self.assertEqual(appearance.kind, 'star')
        self.assertEqual(appearance.bodyId, 'trappist_1')
        self.assertFalse(appearance.atmosphere.enabled)
        maps = appearance.textures.existingMaps()
        self.assertIn('color', maps)
        self.assertTrue(maps['color'].is_file())
        scene = buildBodyScene('TRAPPIST-1', frameCount=8)
        self.assertEqual(scene.body.kind, 'star')
        self.assertEqual(scene.body.systemId, 'trappist_1')
        # M8V dwarf: an order of magnitude smaller than Sol, smaller than Proxima.
        self.assertLess(scene.body.diameterKm, 250_000.0)
        # Texture folder is underscored; output dirs keep the hyphenated catalog stem.
        self.assertEqual(bodyStem('TRAPPIST-1'), 'trappist-1')
        self.assertTrue(
            bodyOutputDirectory('star', 'TRAPPIST-1').as_posix().endswith('stars/trappist-1')
        )

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

    def test_sol_planet_packs_resolve_color_maps(self) -> None:
        from animate.scenes.blender.body_appearance import registeredPlanetCatalogNames

        expected = (
            'Earth',
            'Jupiter',
            'Mars',
            'Mercury',
            'Neptune',
            'Pluto',
            'Proxima b',
            'Proxima d',
            'Saturn',
            'TRAPPIST-1 b',
            'TRAPPIST-1 c',
            'TRAPPIST-1 d',
            'TRAPPIST-1 e',
            'TRAPPIST-1 f',
            'TRAPPIST-1 g',
            'TRAPPIST-1 h',
            'Uranus',
            'Venus',
        )
        self.assertEqual(registeredPlanetCatalogNames(), expected)
        for name in expected:
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps, name)
            self.assertTrue(maps['color'].is_file(), name)

    def test_saturn_rings_pack_and_job_payload(self) -> None:
        appearance = appearanceForCatalogName('Saturn')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        self.assertTrue(appearance.rings.enabled)
        maps = appearance.textures.existingMaps()
        self.assertIn('rings', maps)
        job = appearance.toJobDict()
        self.assertIn('rings', job)
        self.assertAlmostEqual(job['rings']['tiltDeg'], 26.7)
        self.assertIn('rings', job['textures'])

    def test_job_texture_paths_are_repo_relative(self) -> None:
        """Job JSON is committed with the frames — absolute paths pin it to one machine."""
        from animate.scenes.blender.body_appearance import REPO_ROOT
        from animate.scenes.blender.render_flyby import resolveJobPath

        for name in ('Earth', 'Saturn', 'Sun', 'TRAPPIST-1'):
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            textures = appearance.toJobDict()['textures']
            self.assertTrue(textures, name)
            for key, value in textures.items():
                self.assertFalse(Path(value).is_absolute(), f'{name}.{key} = {value}')
                self.assertTrue(value.startswith('data/textures/'), f'{name}.{key} = {value}')
                # Blender resolves relative job paths against the repo root.
                self.assertEqual(resolveJobPath(value), REPO_ROOT / value)
                self.assertTrue(resolveJobPath(value).is_file(), f'{name}.{key} = {value}')

    def test_committed_flyby_jobs_have_no_machine_paths(self) -> None:
        from animate.scenes.blender.body_appearance import REPO_ROOT

        listing = subprocess.run(
            ['git', 'ls-files', 'output/**/*_job.json'],
            check=False,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )
        jobPaths = [line for line in listing.stdout.split() if line]
        if not jobPaths:
            self.skipTest('no committed flyby jobs')
        for jobPath in jobPaths:
            payload = json.loads((REPO_ROOT / jobPath).read_text(encoding='utf-8'))
            textures = (payload.get('appearance') or {}).get('textures') or {}
            for key, value in textures.items():
                self.assertFalse(Path(value).is_absolute(), f'{jobPath}:{key} = {value}')

    def test_ice_giant_rings_present_but_subtler(self) -> None:
        for name in ('Uranus', 'Neptune'):
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            self.assertTrue(appearance.rings.enabled, name)
            self.assertLess(appearance.rings.opacity, 1.0, name)
            self.assertIn('rings', appearance.textures.existingMaps(), name)

    def test_airless_packs_skip_atmosphere(self) -> None:
        for name in ('Mercury', 'Pluto'):
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            self.assertFalse(appearance.atmosphere.enabled, name)
            self.assertNotIn('atmosphere', appearance.toJobDict())

    def test_major_moon_packs_resolve_color_maps(self) -> None:
        from animate.scenes.blender.body_appearance import registeredMoonCatalogNames

        expected = (
            'Callisto',
            'Charon',
            'Deimos',
            'Enceladus',
            'Europa',
            'Ganymede',
            'Io',
            'Moon',
            'Oberon',
            'Phobos',
            'Rhea',
            'Titan',
            'Titania',
            'Triton',
        )
        self.assertEqual(registeredMoonCatalogNames(), expected)
        for name in expected:
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            self.assertEqual(appearance.kind, 'moon', name)
            maps = appearance.textures.existingMaps()
            self.assertIn('color', maps, name)
            self.assertTrue(maps['color'].is_file(), name)
            self.assertNotIn('clouds', maps, name)

    def test_galilean_and_titan_airless_except_titan_haze(self) -> None:
        for name in ('Io', 'Europa', 'Ganymede', 'Callisto'):
            appearance = appearanceForCatalogName(name)
            assert appearance is not None
            self.assertFalse(appearance.atmosphere.enabled, name)
            self.assertNotIn('atmosphere', appearance.toJobDict())
        titan = appearanceForCatalogName('Titan')
        assert titan is not None
        self.assertTrue(titan.atmosphere.enabled)
        self.assertIn('atmosphere', titan.toJobDict())
        self.assertGreater(titan.atmosphere.scale, 1.0)

    def test_asteroid_and_dwarf_packs_airless(self) -> None:
        from animate.scenes.blender.body_appearance import registeredAsteroidCatalogNames

        expected = (
            'Bennu',
            'Ceres',
            'Eris',
            'Eros',
            'Haumea',
            'Makemake',
            'Pallas',
            'Psyche',
            'Vesta',
        )
        self.assertEqual(registeredAsteroidCatalogNames(), expected)
        for name in expected:
            appearance = appearanceForCatalogName(name)
            self.assertIsNotNone(appearance, name)
            assert appearance is not None
            self.assertIn(appearance.kind, {'asteroid', 'dwarf_planet'}, name)
            self.assertFalse(appearance.atmosphere.enabled, name)
            self.assertNotIn('atmosphere', appearance.toJobDict())
            self.assertNotIn('clouds', appearance.textures.existingMaps())
            self.assertIn('color', appearance.textures.existingMaps())
            self.assertTrue(appearance.textures.existingMaps()['color'].is_file())
        ceres = appearanceForCatalogName('Ceres')
        vesta = appearanceForCatalogName('Vesta')
        assert ceres is not None and vesta is not None
        self.assertEqual(ceres.kind, 'dwarf_planet')
        self.assertEqual(vesta.kind, 'asteroid')


class FlybyPipelineTests(unittest.TestCase):
    def test_prepare_export(self) -> None:
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            path = preparePlanetFlybyExport(
                'Mars', frameCount=6, outputDirectory=temporaryDirectory
            )
            self.assertTrue(path.is_file())
            self.assertEqual(path.name, 'mars_body_scene.json')
            self.assertTrue(path.parent.as_posix().endswith('planets/mars'))

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
            self.assertTrue(path.parent.as_posix().endswith('moons/moon'))
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
