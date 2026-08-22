"""Tests for the K–Pg Earth-impact cinema (#86 / #210).

The film is a camera move, not a chart. Tests re-derive the Yucatán vector, the
true-scale pebble, the inbound clock, the dive and the schematic contact
drawing from the committed CSV and PlanetCatalog Earth diameter.
"""

from __future__ import annotations

import math
import unittest
from pathlib import Path

import numpy as np
from animate.scenes.blender.body_appearance import appearanceForCatalogName
from animate.scenes.kpg_cinematic import (
    ACT_BOUNDARIES,
    ANIMATION_FRAMES,
    DEFAULT_EVENT_CSV,
    EJECTA_ANGLE_DEG,
    IMPACT_FRAME,
    INBOUND_START_RADII,
    actName,
    buildImpactSamples,
    buildKpgJob,
    captionForSample,
    ejectaDirections,
    impactNormal,
    inboundDirection,
    inboundKmAtFrame,
    loadImpactEvent,
    unitFromLatLon,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
EVENT_CSV = REPO_ROOT / DEFAULT_EVENT_CSV


class ImpactCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = loadImpactEvent(EVENT_CSV)

    def test_provenance_header_is_present(self) -> None:
        header = ' '.join(
            line for line in EVENT_CSV.read_text().splitlines() if line.startswith('#')
        ).lower()
        for token in ('chicxulub', 'renne', 'hildebrand', 'schematic', 'blue marble'):
            self.assertIn(token, header)

    def test_the_event_is_chicxulub_on_the_yucatan(self) -> None:
        self.assertEqual(self.event.name, 'Chicxulub')
        self.assertAlmostEqual(self.event.latitudeDeg, 21.4, places=1)
        self.assertLess(self.event.longitudeDeg, 0.0)
        self.assertAlmostEqual(self.event.ageMa, 66.0, places=1)

    def test_earth_diameter_comes_from_the_planet_catalogue(self) -> None:
        self.assertEqual(self.event.earthDiameterKm, 12742)

    def test_the_earth_texture_pack_is_present(self) -> None:
        appearance = appearanceForCatalogName('Earth')
        self.assertIsNotNone(appearance)
        assert appearance is not None
        color = appearance.textures.existingMaps().get('color')
        self.assertIsNotNone(color)
        assert color is not None
        self.assertTrue(color.is_file())


class ImpactGeometryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = loadImpactEvent(EVENT_CSV)

    def test_blue_marble_uv_puts_greenwich_on_plus_x(self) -> None:
        equator = unitFromLatLon(0.0, 0.0)
        np.testing.assert_allclose(equator, (1.0, 0.0, 0.0), atol=1e-9)
        north = unitFromLatLon(90.0, 0.0)
        np.testing.assert_allclose(north, (0.0, 0.0, 1.0), atol=1e-9)

    def test_chicxulub_is_in_the_western_gulf(self) -> None:
        normal = impactNormal(self.event)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=9)
        self.assertGreater(float(normal[2]), 0.0)
        self.assertLess(float(normal[1]), 0.0)

    def test_the_impactor_is_a_speck_at_true_scale(self) -> None:
        self.assertAlmostEqual(self.event.radiusRatio, 10.0 / 12742.0, places=9)
        self.assertLess(self.event.radiusRatio, 0.002)

    def test_inbound_flight_is_oblique_not_radial(self) -> None:
        normal = impactNormal(self.event)
        inbound = inboundDirection(self.event)
        self.assertAlmostEqual(float(np.linalg.norm(inbound)), 1.0, places=9)
        alignment = float(np.dot(inbound, -normal))
        self.assertGreater(alignment, 0.6)
        self.assertLess(alignment, 0.95)

    def test_the_clock_reaches_zero_at_contact(self) -> None:
        self.assertGreater(inboundKmAtFrame(self.event, 0), 6.0 * self.event.earthRadiusKm)
        self.assertEqual(inboundKmAtFrame(self.event, IMPACT_FRAME), 0.0)
        startSeconds = inboundKmAtFrame(self.event, 0) / self.event.speedKmS
        self.assertGreater(startSeconds / 60.0, 20.0)


class ImpactCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.event = loadImpactEvent(EVENT_CSV)
        cls.samples = buildImpactSamples(cls.event)

    def test_every_act_is_reached_in_order(self) -> None:
        seen: list[str] = []
        for frame in range(ANIMATION_FRAMES):
            name = actName(frame)
            if not seen or seen[-1] != name:
                seen.append(name)
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['veil'])

    def test_the_camera_dives_then_pulls_back(self) -> None:
        def distance(index: int) -> float:
            sample = self.samples[index]
            camera = np.array(sample.cameraRadii)
            look = np.array(sample.lookAtRadii)
            return float(np.linalg.norm(camera - look))

        self.assertGreater(distance(0), distance(IMPACT_FRAME))
        self.assertGreater(distance(ANIMATION_FRAMES - 1), distance(IMPACT_FRAME))

    def test_the_camera_stays_in_earth_radii_not_an_odyssey(self) -> None:
        for sample in self.samples:
            radius = float(np.linalg.norm(sample.cameraRadii))
            self.assertLess(radius, 8.0)
            self.assertGreater(radius, 1.05)

    def test_contact_places_the_impactor_on_the_surface(self) -> None:
        sample = self.samples[IMPACT_FRAME]
        radius = float(np.linalg.norm(sample.impactorRadii))
        self.assertAlmostEqual(radius, 1.0 + self.event.radiusRatio, places=6)

    def test_the_opening_inbound_matches_the_published_start(self) -> None:
        opening = self.samples[0]
        expected = INBOUND_START_RADII * self.event.earthRadiusKm
        self.assertAlmostEqual(opening.inboundKm, expected, delta=1.0)

    def test_captions_quote_the_csv_numbers(self) -> None:
        quiet = captionForSample(self.event, self.samples[0])
        self.assertIn(self.event.name, quiet)
        self.assertIn(f'{self.event.craterDiameterKm:.0f}', quiet)
        approach = captionForSample(self.event, self.samples[IMPACT_FRAME - 1])
        self.assertIn(f'{self.event.impactorDiameterKm:.0f}', approach)
        strike = captionForSample(self.event, self.samples[IMPACT_FRAME])
        self.assertIn('schematic', strike)
        self.assertIn('ejecta', strike)
        veil = captionForSample(self.event, self.samples[-1])
        self.assertIn(f'{self.event.ageMa:.0f}', veil)

    def test_contact_drawings_are_zero_until_impact(self) -> None:
        for sample in self.samples[:IMPACT_FRAME]:
            self.assertEqual(sample.fireballScale, 0.0)
            self.assertEqual(sample.ejectaScale, 0.0)
            self.assertEqual(sample.plumeScale, 0.0)
            self.assertEqual(sample.flashScale, 0.0)

    def test_the_fireball_grows_after_the_bang(self) -> None:
        atContact = self.samples[IMPACT_FRAME]
        grown = self.samples[IMPACT_FRAME + 6]
        self.assertAlmostEqual(atContact.flashScale, 1.0, places=6)
        self.assertGreater(atContact.fireballScale, 0.0)
        self.assertGreater(grown.fireballScale, atContact.fireballScale)
        self.assertGreater(grown.ejectaScale, 0.0)
        self.assertGreater(self.samples[-1].plumeScale, 0.5)

    def test_ejecta_curtain_is_forty_five_degrees_from_vertical(self) -> None:
        normal = impactNormal(self.event)
        expected = math.cos(math.radians(EJECTA_ANGLE_DEG))
        for ray in ejectaDirections(normal):
            self.assertAlmostEqual(float(np.dot(ray, normal)), expected, places=6)

    def test_the_job_carries_true_scale_and_the_earth_pack(self) -> None:
        job = buildKpgJob(
            self.event,
            self.samples,
            theme='dark',
            framesDirectory=Path('/tmp/kpg-test-frames'),
        )
        self.assertEqual(job['schema'], 'solsys.blender_kpg_job/v1')
        self.assertAlmostEqual(job['impactor']['radiusScale'], self.event.radiusRatio)
        self.assertIn('color', job['appearance']['textures'])
        self.assertEqual(len(job['frames']), ANIMATION_FRAMES)
        self.assertEqual(job['frames'][IMPACT_FRAME]['flashScale'], 1.0)
        self.assertAlmostEqual(job['contact']['ejectaAngleDeg'], EJECTA_ANGLE_DEG)
        self.assertEqual(len(job['contact']['ejectaDirections']), 16)
        self.assertGreater(job['frames'][IMPACT_FRAME + 8]['ejectaScale'], 0.0)


if __name__ == '__main__':
    unittest.main()
