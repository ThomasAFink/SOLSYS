"""3D interstellar-scale solar system visualizations."""

from __future__ import annotations

import os
import random
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from solsys_core import (
    AstronomicalConstants,
    BeltPointGenerator,
    OrbitCalculator,
    PlanetCatalog,
    PointDensityConfig,
    StarCatalog,
    ViewDefinition,
    ViewRegistry,
)

FIGURE_SIZE_INCHES = (39, 39)
OUTPUT_DPI = 300
SHELL_THICKNESS_AU = 0.05
CAMERA_ELEVATION_DEG = 25
CAMERA_AZIMUTH_DEG = 120
OUMUAMUA_LABEL = "'Oumuamua hyperbolic trajectory"
ZOOMED_OUT_VIEWS = {
    'solar_system_with_kuiper_belt',
    'solar_system_with_oort_cloud',
    'solar_system_with_alpha_centauri',
}


class HildaPointGenerator3D:
    @staticmethod
    def clusterPoints(
        jupiterSemiMajorAxisAu: float,
        jupiterEccentricity: float,
        jupiterInclinationDeg: float,
        pointsPerCluster: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        inclinationRad = np.radians(jupiterInclinationDeg)
        radialSpreadAu = 0.8
        verticalSpreadAu = 0.05
        clusterAnglesRad = np.radians([60, 180, 300])
        positionX, positionY, positionZ = [], [], []

        for clusterAngleRad in clusterAnglesRad:
            radiusAu = (
                jupiterSemiMajorAxisAu * (1 - jupiterEccentricity ** 2)
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
        clusterCenters: List[Tuple[float, float, float]],
        numInterpolationPoints: int,
        spreadRadiusAu: float,
        bowFactor: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        interpolatedX, interpolatedY, interpolatedZ = [], [], []
        numClusters = len(clusterCenters)

        for clusterIndex in range(numClusters):
            startCenter = np.array(clusterCenters[clusterIndex])
            endCenter = np.array(clusterCenters[(clusterIndex + 1) % numClusters])
            for stepIndex in range(1, numInterpolationPoints + 1):
                interpolationProgress = stepIndex / (numInterpolationPoints + 1)
                basePoint = (1 - interpolationProgress) * startCenter + interpolationProgress * endCenter
                bowOffsetAu = bowFactor * np.sin(np.pi * interpolationProgress)
                spreadDistanceAu = np.random.uniform(-spreadRadiusAu, spreadRadiusAu) + bowOffsetAu
                directionToSun = -basePoint / np.linalg.norm(basePoint)
                spreadPoint = basePoint + directionToSun * spreadDistanceAu
                interpolatedX.append(spreadPoint[0])
                interpolatedY.append(spreadPoint[1])
                interpolatedZ.append(spreadPoint[2])

        return np.array(interpolatedX), np.array(interpolatedY), np.array(interpolatedZ)


class SolarSystemVisualizer3D:
    def __init__(self, starsCsvPath: str):
        self.constants = AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self.planetCatalog = PlanetCatalog(self.constants)
        self.orbitCalculator = OrbitCalculator()
        self.beltGenerator = BeltPointGenerator()
        self.hildaGenerator = HildaPointGenerator3D()
        self.labeledStarSystems: set[str] = set()

    def renderAllViews(self, outputDirectory: str = 'output/3d') -> None:
        os.makedirs(outputDirectory, exist_ok=True)
        for viewIndex, viewDefinition in enumerate(ViewRegistry.VIEWS_3D):
            self.labeledStarSystems.clear()
            outputPath = f'{outputDirectory}/{viewIndex}_{viewDefinition.shortName}.jpg'
            print(f'Generating {viewDefinition.title}...')
            self.renderView(viewDefinition, outputPath)
            print(f'Saved to {outputPath}')

    def renderView(self, viewDefinition: ViewDefinition, outputPath: str) -> None:
        figure = plt.figure(figsize=FIGURE_SIZE_INCHES)
        axes = figure.add_subplot(111, projection='3d')
        pointCounts = PointDensityConfig.forView(viewDefinition.shortName)

        axes.scatter([0], [0], [0], color='yellow', s=50)
        axes.view_init(elev=CAMERA_ELEVATION_DEG, azim=CAMERA_AZIMUTH_DEG)

        self._drawAsteroidBelt(axes, pointCounts['asteroidBelt'])
        self._drawJupiterLagrangeClouds(axes, pointCounts['trojansAndGreeks'])
        self._drawHildas(axes, pointCounts['hildas'])
        self._drawKuiperBelt(axes, pointCounts['kuiperBelt'])
        self._drawOortCloud(axes, pointCounts['oortCloud'])
        self._drawPlanets(axes)
        self._drawOumuamua(axes, viewDefinition)
        self._drawStars(axes, viewDefinition)

        axes.set_xlim(viewDefinition.axisMinAu, viewDefinition.axisMaxAu)
        axes.set_ylim(viewDefinition.axisMinAu, viewDefinition.axisMaxAu)
        axes.set_zlim(viewDefinition.axisMinAu, viewDefinition.axisMaxAu)
        axes.set_xlabel('X (AU)')
        axes.set_ylabel('Y (AU)')
        axes.set_zlabel('Z (AU)')
        axes.legend()
        plt.title(viewDefinition.title, fontsize=viewDefinition.titleFontSize, pad=50)
        figure.savefig(outputPath, dpi=OUTPUT_DPI)
        plt.close(figure)

    def _drawAsteroidBelt(self, axes: plt.Axes, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.asteroidBeltInnerAu,
            self.constants.asteroidBeltOuterAu,
            SHELL_THICKNESS_AU,
            numPoints,
        )
        axes.scatter(positionX, positionY, positionZ, color='gray', s=1)

    def _jupiterOrbitalAngleRad(self) -> float:
        jupiter = self.planetCatalog.planets['Jupiter']
        orbitX, orbitY, _ = self.orbitCalculator.ellipticalOrbit3d(
            jupiter.semiMajorAxisAu,
            jupiter.eccentricity,
            jupiter.inclinationDeg,
            numPoints=1000,
        )
        sampleIndex = 50
        return float(np.arctan2(orbitY[sampleIndex], orbitX[sampleIndex]))

    def _drawJupiterLagrangeClouds(self, axes: plt.Axes, numPoints: int) -> None:
        constants = self.constants
        jupiterAngleRad = self._jupiterOrbitalAngleRad()
        inclinationRad = np.radians(constants.jupiterInclinationDeg)
        radialSpreadAu = 0.5

        for lagrangeOffsetRad in (np.deg2rad(60), -np.deg2rad(60)):
            clusterCenterAngle = jupiterAngleRad + lagrangeOffsetRad
            radiusAu = np.random.uniform(
                constants.jupiterSemiMajorAxisAu - radialSpreadAu,
                constants.jupiterSemiMajorAxisAu + radialSpreadAu,
                numPoints,
            )
            azimuthRad = np.linspace(
                clusterCenterAngle - np.pi / 6,
                clusterCenterAngle + np.pi / 6,
                numPoints,
            )
            positionX = radiusAu * np.cos(azimuthRad)
            positionY = radiusAu * np.sin(azimuthRad)
            positionZ = np.sin(inclinationRad) * np.random.uniform(-0.1, 0.1, numPoints)
            axes.scatter(positionX, positionY, positionZ, color='gray', s=1)

    def _drawHildas(self, axes: plt.Axes, numPoints: int) -> None:
        constants = self.constants
        hildaX, hildaY, hildaZ = self.hildaGenerator.clusterPoints(
            constants.jupiterSemiMajorAxisAu,
            constants.jupiterEccentricity,
            constants.jupiterInclinationDeg,
            pointsPerCluster=max(int(numPoints / 4), 1),
        )
        axes.scatter(hildaX, hildaY, hildaZ, color='gray', s=1)

        totalPoints = len(hildaX)
        clusterIndices = [0, totalPoints // 3, 2 * totalPoints // 3]
        clusterCenters = [
            (hildaX[index], hildaY[index], hildaZ[index]) for index in clusterIndices
        ]
        bandX, bandY, bandZ = self.hildaGenerator.connectingBands(
            clusterCenters, numPoints, spreadRadiusAu=0.5, bowFactor=-1.75
        )
        axes.scatter(bandX, bandY, bandZ, color='gray', s=1)

    def _drawKuiperBelt(self, axes: plt.Axes, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.kuiperBeltInnerAu,
            self.constants.kuiperBeltOuterAu,
            SHELL_THICKNESS_AU,
            numPoints,
        )
        axes.scatter(positionX, positionY, positionZ, color='gray', s=1)

    def _drawOortCloud(self, axes: plt.Axes, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.oortCloudInnerAu,
            self.constants.oortCloudOuterAu,
            SHELL_THICKNESS_AU * 5,
            numPoints // 5,
        )
        axes.scatter(positionX, positionY, positionZ, color='gray', s=1)

    def _drawPlanets(self, axes: plt.Axes) -> None:
        for planet in self.planetCatalog.planets.values():
            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=1000,
            )
            axes.plot(orbitX, orbitY, orbitZ, color='black')
            sampleIndex = 50 if planet.name == 'Jupiter' else random.randint(0, len(orbitX) - 1)
            markerSize = int(10 + planet.diameterKm / 2500)
            axes.scatter(
                orbitX[sampleIndex],
                orbitY[sampleIndex],
                orbitZ[sampleIndex],
                color=planet.color,
                s=markerSize,
            )

    def _drawOumuamua(self, axes: plt.Axes, viewDefinition: ViewDefinition) -> None:
        constants = self.constants
        positionX, positionY, positionZ = self.orbitCalculator.hyperbolicOrbit3d(
            constants.oumuamuaPerihelionAu,
            constants.oumuamuaEccentricity,
            constants.oumuamuaInclinationDeg,
            constants.oumuamuaLongitudeAscendingNodeDeg,
            constants.oumuamuaArgumentOfPerihelionDeg,
            numPoints=5000,
        )
        axes.plot(positionX, positionY, positionZ, '--', color='darkred', label=OUMUAMUA_LABEL)

        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        if viewDefinition.shortName in ZOOMED_OUT_VIEWS:
            inboundDirection = np.array([
                positionX[0] / radiusAu[0],
                positionY[0] / radiusAu[0],
                positionZ[0] / radiusAu[0],
            ])
            targetRadiusAu = {
                'solar_system_with_kuiper_belt': 55,
                'solar_system_with_oort_cloud': 45000,
                'solar_system_with_alpha_centauri': 45000,
            }[viewDefinition.shortName]
            labelPoint = inboundDirection * targetRadiusAu
            perpendicular = np.array([-inboundDirection[1], inboundDirection[0], 0.0])
            perpendicularNorm = np.linalg.norm(perpendicular)
            if perpendicularNorm > 0:
                perpendicular /= perpendicularNorm
                if np.dot(perpendicular[:2], labelPoint[:2]) < 0:
                    perpendicular *= -1
                textPoint = labelPoint + perpendicular * (
                    0.10 * (viewDefinition.axisMaxAu - viewDefinition.axisMinAu)
                )
            else:
                textPoint = labelPoint
            axes.text(textPoint[0], textPoint[1], textPoint[2], OUMUAMUA_LABEL, fontsize=10, color='darkred')
        else:
            perihelionIndex = int(np.argmin(radiusAu))
            axes.text(
                positionX[perihelionIndex],
                positionY[perihelionIndex],
                positionZ[perihelionIndex],
                OUMUAMUA_LABEL,
                fontsize=10,
                color='darkred',
            )

    def _drawStars(self, axes: plt.Axes, viewDefinition: ViewDefinition) -> None:
        starsFrame = self.starCatalog.starsDataFrame
        starsInView = starsFrame[
            starsFrame['distanceAu'] <= viewDefinition.axisMaxAu
        ].dropna(subset=['positionX', 'positionY', 'positionZ'])

        for _, starRow in starsInView.iterrows():
            if 'Vega' in starRow['System']:
                axes.scatter(
                    starRow['positionX'],
                    starRow['positionY'],
                    starRow['positionZ'],
                    color='silver',
                    s=500,
                    alpha=0.9,
                )
            else:
                axes.scatter(
                    starRow['positionX'],
                    starRow['positionY'],
                    starRow['positionZ'],
                    color='orange',
                    s=80,
                    alpha=0.9,
                )

        for _, starRow in starsInView.iterrows():
            systemLabel = starRow['System'][:20]
            if systemLabel not in self.labeledStarSystems:
                axes.text(
                    starRow['positionX'],
                    starRow['positionY'],
                    starRow['positionZ'],
                    systemLabel,
                    fontsize=10,
                    ha='left',
                    va='bottom',
                )
                self.labeledStarSystems.add(starRow['System'][:10])


if __name__ == '__main__':
    visualizer = SolarSystemVisualizer3D('data/nearby_stars_30.csv')
    visualizer.renderAllViews()
    print('All 3D visualizations completed!')
