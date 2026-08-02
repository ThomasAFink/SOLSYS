"""Unit tests for Sol ↔ Alpha Centauri frame transforms."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
from solsys.physics.catalogs.system_catalog import SystemCatalog, defaultDataPaths
from solsys.physics.frame_transform import (
    SolCentauriFrameTransform,
    abBarycenterSolPositionAu,
    orbitalPlaneToSolRotation,
)
from solsys.physics.orbit_calculator import OrbitCalculator

REPO_ROOT = Path(__file__).resolve().parents[1]


class OrbitalPlaneRotationTests(unittest.TestCase):
    def test_matches_elliptical_position_convention(self) -> None:
        inclinationDeg = 79.24
        ascendingNodeDeg = 205.07
        rotation = orbitalPlaneToSolRotation(inclinationDeg, ascendingNodeDeg)
        calculator = OrbitCalculator()

        for angle in (0.0, 0.4, 1.7, -0.9):
            positionX, positionY, positionZ = calculator.ellipticalPosition(
                1.0, 0.0, inclinationDeg, angle, ascendingNodeDeg=ascendingNodeDeg
            )
            local = np.array([np.cos(angle), np.sin(angle), 0.0])
            mapped = rotation @ local
            np.testing.assert_allclose(
                mapped,
                [float(positionX), float(positionY), float(positionZ)],
                rtol=1e-12,
                atol=1e-12,
            )

        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)


class AbBarycenterTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**paths).load('alpha_centauri')

    def test_mass_weighted_between_a_and_b(self) -> None:
        primary = self.system.starByUuid(
            next(o.starUuid for o in self.system.stellarOrbits if o.role == 'primary')
        )
        secondary = self.system.starByUuid(
            next(o.starUuid for o in self.system.stellarOrbits if o.role == 'secondary')
        )
        assert primary is not None and secondary is not None
        barycenter = abBarycenterSolPositionAu(self.system)

        positionA = np.array([primary.positionX, primary.positionY, primary.positionZ], dtype=float)
        positionB = np.array(
            [secondary.positionX, secondary.positionY, secondary.positionZ], dtype=float
        )
        expected = (primary.massSolar * positionA + secondary.massSolar * positionB) / (
            primary.massSolar + secondary.massSolar
        )
        np.testing.assert_allclose(barycenter, expected, rtol=1e-12, atol=1e-12)

        span = positionB - positionA
        spanNormSq = float(np.dot(span, span))
        fraction = float(np.dot(barycenter - positionA, span) / spanNormSq)
        self.assertGreater(fraction, 0.0)
        self.assertLess(fraction, 1.0)


class SolCentauriFrameTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        paths = defaultDataPaths(REPO_ROOT)
        self.system = SystemCatalog(**paths).load('alpha_centauri')
        self.transform = SolCentauriFrameTransform.fromStarSystem(self.system)

    def test_origin_maps_to_barycenter(self) -> None:
        sol = self.transform.toSol([0.0, 0.0])
        np.testing.assert_allclose(sol[0], self.transform.originSolAu, rtol=1e-12, atol=1e-12)

        centauri = self.transform.toCentauri(self.transform.originSolAu)
        np.testing.assert_allclose(centauri[0], [0.0, 0.0, 0.0], atol=1e-9)

    def test_round_trip(self) -> None:
        for point in ([0.0, 0.0], [10.66, 0.0], [0.0, 12.64], [-3.5, 7.25], [1.0, -2.0, 0.5]):
            sol = self.transform.toSol(point)
            back = self.transform.toCentauri(sol[0])
            expected = np.asarray(point, dtype=float)
            if expected.shape[0] == 2:
                expected = np.append(expected, 0.0)
            np.testing.assert_allclose(back[0], expected, rtol=1e-10, atol=1e-9)

    def test_batch_shape(self) -> None:
        points = np.array([[1.0, 0.0], [0.0, 2.0], [-1.0, -1.0]])
        sol = self.transform.toSol(points)
        self.assertEqual(sol.shape, (3, 3))
        back = self.transform.toCentauri(sol)
        np.testing.assert_allclose(back[:, :2], points, rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(back[:, 2], 0.0, atol=1e-9)

    def test_uses_catalog_elements(self) -> None:
        primary = next(o for o in self.system.stellarOrbits if o.role == 'primary')
        self.assertEqual(self.transform.inclinationDeg, primary.inclinationDeg)
        self.assertEqual(
            self.transform.longitudeAscendingNodeDeg, primary.longitudeAscendingNodeDeg
        )
        np.testing.assert_allclose(
            self.transform.originSolAu,
            abBarycenterSolPositionAu(self.system),
            rtol=1e-12,
            atol=1e-12,
        )


if __name__ == '__main__':
    unittest.main()
