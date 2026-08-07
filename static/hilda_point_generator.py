"""Static Hilda cluster and connecting-band generators."""

from __future__ import annotations

import numpy as np


class HildaPointGenerator:
    @staticmethod
    def clusterPoints(
        jupiterSemiMajorAxisAu: float,
        jupiterEccentricity: float,
        jupiterInclinationDeg: float,
        pointsPerCluster: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(jupiterInclinationDeg)
        radialSpreadAu = 0.8
        verticalSpreadAu = 0.05
        clusterAnglesRad = np.radians([60, 180, 300])
        positionX, positionY, positionZ = [], [], []

        for clusterAngleRad in clusterAnglesRad:
            radiusAu = (
                jupiterSemiMajorAxisAu
                * (1 - jupiterEccentricity**2)
                / (1 + jupiterEccentricity * np.cos(clusterAngleRad))
                - radialSpreadAu / 2
            )
            for _ in range(pointsPerCluster):
                displacementRadiusAu = np.random.uniform(0, radialSpreadAu)
                displacementAngleRad = np.random.uniform(0, 2 * np.pi)
                positionX.append(
                    radiusAu * np.cos(clusterAngleRad)
                    + displacementRadiusAu * np.cos(displacementAngleRad)
                )
                positionY.append(
                    radiusAu * np.sin(clusterAngleRad)
                    + displacementRadiusAu * np.sin(displacementAngleRad)
                )
                positionZ.append(
                    np.sin(inclinationRad) * np.random.uniform(-verticalSpreadAu, verticalSpreadAu)
                )

        return np.array(positionX), np.array(positionY), np.array(positionZ)

    @staticmethod
    def connectingBands(
        clusterCenters: list[tuple[float, float, float]],
        numInterpolationPoints: int,
        spreadRadiusAu: float,
        bowFactor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        interpolatedX, interpolatedY, interpolatedZ = [], [], []
        numClusters = len(clusterCenters)

        for clusterIndex in range(numClusters):
            startCenter = np.array(clusterCenters[clusterIndex])
            endCenter = np.array(clusterCenters[(clusterIndex + 1) % numClusters])
            for stepIndex in range(1, numInterpolationPoints + 1):
                interpolationProgress = stepIndex / (numInterpolationPoints + 1)
                basePoint = (
                    1 - interpolationProgress
                ) * startCenter + interpolationProgress * endCenter
                bowOffsetAu = bowFactor * np.sin(np.pi * interpolationProgress)
                spreadDistanceAu = np.random.uniform(-spreadRadiusAu, spreadRadiusAu) + bowOffsetAu
                directionToSun = -basePoint / np.linalg.norm(basePoint)
                spreadPoint = basePoint + directionToSun * spreadDistanceAu
                interpolatedX.append(spreadPoint[0])
                interpolatedY.append(spreadPoint[1])
                interpolatedZ.append(spreadPoint[2])

        return np.array(interpolatedX), np.array(interpolatedY), np.array(interpolatedZ)
