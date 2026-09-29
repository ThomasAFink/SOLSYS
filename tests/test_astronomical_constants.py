"""Unit tests for shared astronomical constants."""

from __future__ import annotations

import unittest

from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.catalogs.moon_catalog import MoonCatalog


class AstronomicalConstantsTests(unittest.TestCase):
    def test_pluto_apsides_bracket_the_semi_major_axis(self) -> None:
        constants = AstronomicalConstants()
        self.assertAlmostEqual(
            constants.plutoPerihelionAu,
            constants.plutoSemiMajorAxis * (1 - constants.plutoEccentricity),
        )
        self.assertAlmostEqual(
            constants.plutoAphelionAu,
            constants.plutoSemiMajorAxis * (1 + constants.plutoEccentricity),
        )
        self.assertLess(constants.plutoPerihelionAu, constants.plutoSemiMajorAxis)
        self.assertGreater(constants.plutoAphelionAu, constants.plutoSemiMajorAxis)

    def test_julian_year_is_365_25_days(self) -> None:
        constants = AstronomicalConstants()
        self.assertEqual(constants.secondsPerJulianYear, 365.25 * 86400.0)

    def test_moon_orbit_uses_the_shared_astronomical_unit(self) -> None:
        constants = AstronomicalConstants()
        catalog = MoonCatalog()
        moon = catalog.moons['Moon']
        self.assertEqual(catalog.KM_PER_AU, constants.auToKm)
        self.assertAlmostEqual(catalog.semiMajorAxisAu(moon), moon.semiMajorAxisKm / constants.auToKm)
