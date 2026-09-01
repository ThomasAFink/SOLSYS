"""Tests for the K–Pg Earth-impact cinema (#86 / #210).

Tests re-derive the Yucatán vector, the true-scale pebble, the inbound clock
and the scripted act list from the committed CSV and PlanetCatalog Earth diameter.
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
    APPROACH_END,
    CINEMA_ROCK_RADII,
    DEFAULT_EVENT_CSV,
    EJECTA_END,
    FIREBALL_MAX_RADII,
    HOLD_END,
    IMPACT_FRAME,
    INBOUND_START_RADII,
    KPG_EARTH_PACK,
    PROJECTILE_COUNT,
    QUIET_END,
    SOOT_END,
    SPIN_END,
    TSUNAMI_END,
    TWILIGHT_END,
    actName,
    buildImpactSamples,
    buildKpgJob,
    cameraFollowSpinRad,
    captionForSample,
    contactCenterRadii,
    contactPlateAmount,
    debrisRadiusScales,
    earthSpinRad,
    ejectaDirections,
    falloutEnvelope,
    impactNormal,
    inboundDirection,
    inboundKmAtFrame,
    inboundTrailScale,
    loadImpactEvent,
    projectileLaunches,
    projectilePositionRadii,
    projectSiteOnScreen,
    rotateAroundNorth,
    titleForAct,
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
        for token in (
            'chicxulub',
            'renne',
            'hildebrand',
            'reconstruction',
            'late cretaceous',
            'cascade',
        ):
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

    def test_the_kpg_earth_pack_is_present(self) -> None:
        self.assertTrue((KPG_EARTH_PACK / 'color.png').is_file())
        self.assertTrue((KPG_EARTH_PACK / 'clouds.png').is_file())


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
        self.assertEqual(seen, [name for _, name in ACT_BOUNDARIES] + ['recovery'])
        self.assertEqual(ANIMATION_FRAMES, 720)

    def test_the_camera_dives_then_pulls_back(self) -> None:
        def distance(index: int) -> float:
            sample = self.samples[index]
            camera = np.array(sample.cameraRadii)
            look = np.array(sample.lookAtRadii)
            return float(np.linalg.norm(camera - look))

        self.assertGreater(distance(0), distance(IMPACT_FRAME))
        self.assertGreater(distance(ANIMATION_FRAMES - 1), distance(IMPACT_FRAME))
        self.assertLess(distance(HOLD_END), distance(ANIMATION_FRAMES - 1))
        previous = distance(0)
        for index in range(1, IMPACT_FRAME + 1):
            here = distance(index)
            self.assertLess(here, previous)
            previous = here

    def test_the_camera_does_not_ride_the_rock(self) -> None:
        for sample in self.samples:
            self.assertGreater(float(np.linalg.norm(sample.cameraRadii)), 1.2)
            self.assertLess(float(np.linalg.norm(sample.lookAtRadii)), 1.01)

    def test_the_camera_stays_outside_the_atmosphere(self) -> None:
        for sample in self.samples:
            radius = float(np.linalg.norm(sample.cameraRadii))
            self.assertLess(radius, 4.0)
            self.assertGreater(radius, 1.2)
        site = np.array(self.samples[IMPACT_FRAME].lookAtRadii)
        self.assertGreater(float(np.linalg.norm(site)), 0.6)

    def test_the_impact_site_projects_onto_the_frame(self) -> None:
        sample = self.samples[IMPACT_FRAME]
        site = rotateAroundNorth(contactCenterRadii(self.event), cameraFollowSpinRad(IMPACT_FRAME))
        uCoord, vCoord, depth = projectSiteOnScreen(
            sample.cameraRadii,
            sample.lookAtRadii,
            (float(site[0]), float(site[1]), float(site[2])),
            sample.lens,
        )
        self.assertGreater(depth, 0.2)
        self.assertGreater(uCoord, 0.15)
        self.assertLess(uCoord, 0.85)
        self.assertGreater(vCoord, 0.15)
        self.assertLess(vCoord, 0.85)
        self.assertAlmostEqual(
            cameraFollowSpinRad(IMPACT_FRAME), earthSpinRad(IMPACT_FRAME), places=9
        )
        self.assertAlmostEqual(cameraFollowSpinRad(0), earthSpinRad(0), places=9)
        inbound = self.samples[APPROACH_END + 8]
        rock = np.array(inbound.impactorRadii)
        spunRock = rotateAroundNorth(rock, cameraFollowSpinRad(inbound.frame))
        uRock, vRock, rockDepth = projectSiteOnScreen(
            inbound.cameraRadii,
            inbound.lookAtRadii,
            (float(spunRock[0]), float(spunRock[1]), float(spunRock[2])),
            inbound.lens,
        )
        self.assertGreater(rockDepth, 0.0)
        self.assertGreater(uRock, 0.05)
        self.assertLess(uRock, 0.95)

    def test_contact_places_the_impactor_on_the_surface(self) -> None:
        sample = self.samples[IMPACT_FRAME]
        radius = float(np.linalg.norm(sample.impactorRadii))
        self.assertAlmostEqual(radius, 1.0 + self.event.radiusRatio, places=6)

    def test_the_rock_punches_into_the_crust(self) -> None:
        atHit = float(np.linalg.norm(self.samples[IMPACT_FRAME].impactorRadii))
        later = float(np.linalg.norm(self.samples[IMPACT_FRAME + 8].impactorRadii))
        self.assertLess(later, atHit)
        self.assertEqual(self.samples[IMPACT_FRAME].slamScale, 0.0)
        self.assertGreater(self.samples[IMPACT_FRAME - 1].slamScale, 0.9)

    def test_the_opening_inbound_matches_the_published_start(self) -> None:
        opening = self.samples[0]
        expected = INBOUND_START_RADII * self.event.earthRadiusKm
        self.assertAlmostEqual(opening.inboundKm, expected, delta=1.0)

    def test_captions_quote_the_csv_numbers(self) -> None:
        quiet = captionForSample(self.event, self.samples[0])
        self.assertIn(self.event.name, quiet)
        self.assertIn(f'{self.event.craterDiameterKm:.0f}', quiet)
        self.assertIn('reconstruction', quiet)
        approach = captionForSample(self.event, self.samples[QUIET_END + 8])
        self.assertIn(f'{self.event.impactorDiameterKm:.0f}', approach)
        self.assertIn('enlarged', approach)
        contact = captionForSample(self.event, self.samples[IMPACT_FRAME])
        self.assertIn('Yucatán', contact)
        self.assertNotIn('schematic', contact)
        crust = captionForSample(self.event, self.samples[IMPACT_FRAME + 30])
        self.assertIn('hydro', crust)
        ejecta = captionForSample(self.event, self.samples[EJECTA_END - 4])
        self.assertIn('worldwide', ejecta)
        tsunami = captionForSample(self.event, self.samples[TSUNAMI_END - 4])
        self.assertIn('hours compressed', tsunami)
        spin = captionForSample(self.event, self.samples[SPIN_END - 4])
        self.assertIn('glow', spin)
        self.assertIn('compressed', spin)
        soot = captionForSample(self.event, self.samples[SOOT_END - 4])
        self.assertIn('12', soot)
        self.assertIn('hours compressed', soot)
        twilight = captionForSample(self.event, self.samples[TWILIGHT_END - 4])
        self.assertIn('99%', twilight)
        self.assertIn('years compressed', twilight)
        recovery = captionForSample(self.event, self.samples[-1])
        self.assertIn('6 years', recovery)
        self.assertIn('re-green', recovery)
        self.assertIn('years compressed', recovery)
        self.assertEqual(titleForAct('contact'), 'Contact')
        self.assertEqual(titleForAct('ejecta'), 'Ejecta')
        self.assertEqual(titleForAct('spin'), 'The globe turns')
        self.assertEqual(titleForAct('recovery'), 'Recovery')

    def test_contact_drawings_are_zero_until_impact(self) -> None:
        for sample in self.samples[:IMPACT_FRAME]:
            self.assertEqual(sample.fireballScale, 0.0)
            self.assertEqual(sample.ejectaScale, 0.0)
            self.assertEqual(sample.plumeScale, 0.0)
            self.assertEqual(sample.flashScale, 0.0)
            self.assertEqual(sample.falloutScale, 0.0)
            self.assertEqual(sample.crustTear, 0.0)
            self.assertEqual(sample.tsunamiAngle, 0.0)
            self.assertTrue(all(scale == 0.0 for scale in sample.projectileScale))
            self.assertEqual(sample.wildfireAngle, 0.0)
            self.assertEqual(sample.soot, 0.0)
            self.assertEqual(sample.shockAngle, 0.0)
            self.assertEqual(sample.smolder, 0.0)
            self.assertEqual(sample.siteGlow, 0.0)
            self.assertEqual(sample.dieback, 0.0)
            self.assertGreaterEqual(sample.earthSpin, 0.0)
        self.assertEqual(inboundTrailScale(0), 0.0)
        self.assertGreater(inboundTrailScale(IMPACT_FRAME - 1), 0.5)
        self.assertEqual(inboundTrailScale(IMPACT_FRAME), 0.0)
        self.assertGreater(self.samples[IMPACT_FRAME - 1].entryHeat, 0.5)
        self.assertEqual(self.samples[APPROACH_END].entryHeat, 0.0)
        self.assertGreater(self.samples[IMPACT_FRAME].shockAngle, 0.0)
        self.assertLess(self.samples[IMPACT_FRAME].shockAngle, 0.16)

    def test_the_cascade_grows_after_contact(self) -> None:
        atContact = self.samples[IMPACT_FRAME]
        later = self.samples[IMPACT_FRAME + 12]
        self.assertGreater(atContact.flashScale, 0.8)
        self.assertGreater(atContact.fireballScale, 0.35)
        self.assertGreater(later.fireballScale, 0.5)
        self.assertGreater(later.crustTear, 0.1)
        self.assertGreater(self.samples[IMPACT_FRAME + 40].plumeScale, 0.4)
        self.assertGreater(self.samples[-1].shockAngle, 2.5)
        self.assertGreater(self.samples[TWILIGHT_END - 1].soot, 0.9)
        self.assertLess(self.samples[-1].soot, self.samples[TWILIGHT_END - 1].soot)
        self.assertLess(self.samples[SPIN_END].smolder, 0.15)
        self.assertLess(self.samples[SPIN_END].siteGlow, 0.05)
        self.assertGreater(self.samples[IMPACT_FRAME + 10].siteGlow, 0.9)
        self.assertGreater(self.samples[QUIET_END].earthSpin, 0.0)
        self.assertGreater(
            self.samples[IMPACT_FRAME + 80].earthSpin - self.samples[IMPACT_FRAME].earthSpin,
            self.samples[IMPACT_FRAME].earthSpin - self.samples[IMPACT_FRAME - 80].earthSpin,
        )
        self.assertGreater(self.samples[-1].earthSpin, 2.0 * math.pi)
        self.assertEqual(self.samples[IMPACT_FRAME + 10].dieback, 0.0)
        self.assertGreater(self.samples[TWILIGHT_END - 1].dieback, 0.8)
        self.assertGreater(self.samples[TWILIGHT_END + 40].dieback, 0.9)
        self.assertLess(self.samples[-1].dieback, self.samples[TWILIGHT_END - 1].dieback)
        self.assertGreater(self.samples[-1].dieback, 0.75)
        self.assertLess(self.samples[TWILIGHT_END - 1].sunScale, 0.08)
        self.assertGreater(self.samples[-1].sunScale, 0.7)
        self.assertGreater(self.samples[-1].tsunamiAngle, 2.0)
        self.assertGreater(FIREBALL_MAX_RADII, CINEMA_ROCK_RADII)
        self.assertEqual(contactPlateAmount(IMPACT_FRAME), 0.0)

    def test_ejecta_is_a_concentrated_spray(self) -> None:
        normal = impactNormal(self.event)
        inbound = inboundDirection(self.event)
        rays = ejectaDirections(normal, inbound)
        self.assertGreaterEqual(len(rays), 40)
        dots = [float(np.dot(ray, normal)) for ray in rays]
        self.assertTrue(all(0.70 < value < 0.99 for value in dots))
        self.assertGreater(max(dots) - min(dots), 0.12)
        tangent = inbound - normal * float(np.dot(inbound, normal))
        downrange = -tangent / float(np.linalg.norm(tangent))
        along = [float(np.dot(ray, downrange)) for ray in rays]
        self.assertGreater(sum(1 for value in along if value > 0.0), len(along) * 0.55)

    def test_the_job_carries_true_scale_and_the_earth_pack(self) -> None:
        job = buildKpgJob(
            self.event,
            self.samples,
            theme='dark',
            framesDirectory=Path('/tmp/kpg-test-frames'),
        )
        self.assertEqual(job['schema'], 'solsys.blender_kpg_job/v1')
        self.assertAlmostEqual(job['impactor']['radiusScale'], CINEMA_ROCK_RADII)
        self.assertAlmostEqual(job['impactor']['trueRadiusScale'], self.event.radiusRatio)
        self.assertGreater(job['impactor']['radiusScale'], self.event.radiusRatio)
        self.assertLess(job['impactor']['radiusScale'], 0.014)
        self.assertFalse(job['frames'][0]['impactorVisible'])
        self.assertTrue(job['frames'][QUIET_END + 2]['impactorVisible'])
        self.assertTrue(job['frames'][IMPACT_FRAME - 1]['impactorVisible'])
        self.assertFalse(job['frames'][IMPACT_FRAME]['impactorVisible'])
        self.assertIn('earth_kpg', job['appearance']['textures']['color'])
        self.assertIn('bennu', job['impactor']['colorTexture'])
        self.assertEqual(len(job['frames']), ANIMATION_FRAMES)
        self.assertGreater(job['frames'][IMPACT_FRAME]['flashScale'], 0.8)
        self.assertGreater(job['frames'][IMPACT_FRAME]['fireballScale'], 0.35)
        self.assertAlmostEqual(job['contact']['maxFireballRadii'], FIREBALL_MAX_RADII)
        self.assertGreater(job['contact']['maxFireballRadii'], CINEMA_ROCK_RADII)
        self.assertEqual(job['contact']['projectileCount'], PROJECTILE_COUNT)
        self.assertGreaterEqual(job['contact']['rockBurstCount'], 4000)
        self.assertGreaterEqual(job['contact']['emberBurstCount'], 1500)
        self.assertEqual(job['contact']['crustPlateCount'], 6)
        debris = job['contact']['debrisRadiusScale']
        self.assertEqual(debris, list(debrisRadiusScales(self.event.radiusRatio)))
        self.assertTrue(all(scale < self.event.radiusRatio for scale in debris))
        self.assertGreater(job['frames'][IMPACT_FRAME + 24]['crustTear'], 0.3)
        self.assertGreater(job['frames'][TWILIGHT_END - 1]['soot'], 0.9)
        self.assertGreater(job['frames'][-1]['earthSpin'], 2.0 * math.pi)
        self.assertLess(job['frames'][SPIN_END]['siteGlow'], 0.05)
        self.assertGreater(job['frames'][TWILIGHT_END - 1]['dieback'], 0.8)
        self.assertGreater(job['frames'][-1]['dieback'], 0.75)
        self.assertEqual(job['contact']['textures']['explosion'], [])

    def test_ballistic_ejecta_leave_the_yucatan(self) -> None:
        normal = impactNormal(self.event)
        later = self.samples[IMPACT_FRAME + 48]
        self.assertEqual(len(later.projectileRadii), PROJECTILE_COUNT)
        leftSite = 0
        for position in later.projectileRadii:
            vector = np.array(position)
            direction = vector / float(np.linalg.norm(vector))
            if float(np.dot(direction, normal)) < 0.88:
                leftSite += 1
        self.assertGreater(leftSite, 3)
        self.assertGreater(falloutEnvelope(ANIMATION_FRAMES - 1), 0.8)
        self.assertEqual(len(projectileLaunches()), PROJECTILE_COUNT)
        first = projectilePositionRadii(normal, projectileLaunches()[0], IMPACT_FRAME)
        np.testing.assert_allclose(first, normal, atol=1e-6)
        midHeights = [
            float(np.linalg.norm(position))
            for position in self.samples[IMPACT_FRAME + 20].projectileRadii
        ]
        lateHeights = [
            float(np.linalg.norm(position)) for position in self.samples[-1].projectileRadii
        ]
        self.assertGreater(max(midHeights), 1.03)
        self.assertGreater(sum(1 for height in lateHeights if height < 1.04), len(lateHeights) * 0.7)
        sizes = debrisRadiusScales(self.event.radiusRatio)
        self.assertGreater(max(sizes) / max(min(sizes), 1e-9), 2.5)


if __name__ == '__main__':
    unittest.main()
