"""Random point clouds for asteroid belts and spherical shells."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class BeltPointGenerator:
    """Random point clouds for asteroid belts and spherical shells."""

    @staticmethod
    def ring2d(innerRadiusAu: float, outerRadiusAu: float, numPoints: int) -> Tuple[np.ndarray, np.ndarray]:
        radiusAu = np.random.uniform(innerRadiusAu, outerRadiusAu, numPoints)
        azimuthRad = np.random.uniform(0, 2 * np.pi, numPoints)
        return radiusAu * np.cos(azimuthRad), radiusAu * np.sin(azimuthRad)

    @staticmethod
    def sphericalShell(
        innerRadiusAu: float,
        outerRadiusAu: float,
        shellThicknessAu: float,
        numPoints: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        azimuthRad = np.random.uniform(0, 2 * np.pi, numPoints)
        cosPolarAngle = np.random.uniform(-1, 1, numPoints)
        radiusAu = np.random.uniform(
            innerRadiusAu - shellThicknessAu / 2,
            outerRadiusAu + shellThicknessAu / 2,
            numPoints,
        )
        polarAngleRad = np.arccos(cosPolarAngle)
        positionX = radiusAu * np.sin(polarAngleRad) * np.cos(azimuthRad)
        positionY = radiusAu * np.sin(polarAngleRad) * np.sin(azimuthRad)
        positionZ = radiusAu * np.cos(polarAngleRad)
        return positionX, positionY, positionZ

    @staticmethod
    def jupiterLagrangeCloud2d(
        jupiterAngleRad: float,
        lagrangeOffsetRad: float,
        semiMajorAxisAu: float,
        radialSpreadAu: float,
        angularSpreadRad: float,
        numPoints: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        clusterCenterAngle = jupiterAngleRad + lagrangeOffsetRad
        radiusAu = np.random.uniform(
            semiMajorAxisAu - radialSpreadAu,
            semiMajorAxisAu + radialSpreadAu,
            numPoints,
        )
        azimuthRad = np.linspace(
            clusterCenterAngle - angularSpreadRad / 2,
            clusterCenterAngle + angularSpreadRad / 2,
            numPoints,
        )
        return radiusAu * np.cos(azimuthRad), radiusAu * np.sin(azimuthRad)
