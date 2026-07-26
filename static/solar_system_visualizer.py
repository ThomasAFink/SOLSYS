"""Side-product: static multi-zoom solar-system visualizations."""

from __future__ import annotations

import os
import random
import zlib
from typing import List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from solsys.physics import (
    AstronomicalConstants,
    BeltPointGenerator,
    FamousAsteroidCatalog,
    MoonCatalog,
    OrbitCalculator,
    PlanetCatalog,
    PointDensityConfig,
    StarCatalog,
    ViewDefinition,
    ViewRegistry,
)
from static.hilda_point_generator import HildaPointGenerator
from static.dimension_plotter import DimensionPlotter

FIGURE_SIZE_INCHES = (39, 39)
OUTPUT_DPI = 300
SHELL_THICKNESS_AU = 0.05
CAMERA_ELEVATION_DEG = 25
CAMERA_AZIMUTH_DEG = 120
LABEL_FONT_SIZE = 48
OUMUAMUA_LABEL = "'Oumuamua hyperbolic trajectory"
ZOOMED_OUT_SHORT_NAMES = {
    'solar_system_with_kuiper_belt',
    'solar_system_with_oort_cloud',
    'solar_system_with_alpha_centauri',
}
Dimension = Literal['2d', '3d']
Theme = Literal['light', 'dark']
THEME_STYLES = {
    'light': {
        'mplStyle': 'default',
        'orbitColor': 'black',
        'beltColor': 'gray',
        'annotationFaceColor': 'black',
        'oumuamuaColor': 'darkred',
    },
    'dark': {
        'mplStyle': 'dark_background',
        'orbitColor': '#E0E0E0',
        'beltColor': '#C8C8C8',
        'annotationFaceColor': 'white',
        'oumuamuaColor': '#FF6B6B',
    },
}


class SolarSystemVisualizer:
    """Renders static zoom views in 2D (top-down) or 3D from shared XYZ geometry."""

    def __init__(
        self,
        starsCsvPath: str,
        dimension: Dimension = '2d',
        theme: Theme = 'light',
    ):
        if dimension not in ('2d', '3d'):
            raise ValueError(f"dimension must be '2d' or '3d', got {dimension!r}")
        if theme not in THEME_STYLES:
            raise ValueError(f"theme must be 'light' or 'dark', got {theme!r}")
        self.dimension = dimension
        self.theme = theme
        self.themeColors = THEME_STYLES[theme]
        plt.style.use(self.themeColors['mplStyle'])
        self.constants = AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.famousAsteroidCatalog = FamousAsteroidCatalog()
        self.orbitCalculator = OrbitCalculator()
        self.beltGenerator = BeltPointGenerator()
        self.hildaGenerator = HildaPointGenerator()
        self.labeledStarSystems: set[str] = set()

    @property
    def views(self) -> List[ViewDefinition]:
        return ViewRegistry.VIEWS_3D if self.dimension == '3d' else ViewRegistry.VIEWS_2D

    @property
    def outputDirectoryDefault(self) -> str:
        return f'output/{self.dimension}'

    def renderAllViews(self, outputDirectory: Optional[str] = None) -> None:
        outputDirectory = outputDirectory or self.outputDirectoryDefault
        os.makedirs(outputDirectory, exist_ok=True)

        for viewIndex, viewDefinition in enumerate(self.views):
            self.labeledStarSystems.clear()
            outputPath = (
                f'{outputDirectory}/{viewIndex}_{viewDefinition.shortName}_{self.theme}.jpg'
            )
            print(f'Generating {self.dimension.upper()} {self.theme} {viewDefinition.title}...')
            self.renderView(viewDefinition, outputPath)
            print(f'Saved to {outputPath}')

        if self.dimension == '2d':
            outputPath = f'{outputDirectory}/8_oumuamua_origin_vega_system_25_{self.theme}.jpg'
            print(f"Generating 2D {self.theme} 'Oumuamua and Vega (25 light years)...")
            self._renderOumuamuaVegaView(outputPath)
            print(f'Saved to {outputPath}')

    def renderView(self, viewDefinition: ViewDefinition, outputPath: str) -> None:
        # Stable planet / cloud sample positions so light/dark match across runs.
        seed = zlib.crc32(viewDefinition.shortName.encode('utf-8')) & 0xFFFFFFFF
        random.seed(seed)
        np.random.seed(seed)
        figure = plt.figure(figsize=FIGURE_SIZE_INCHES)
        if self.dimension == '3d':
            axes = figure.add_subplot(111, projection='3d')
        else:
            axes = figure.add_subplot(111)

        plotter = DimensionPlotter(axes, self.dimension)
        pointCounts = PointDensityConfig.forView(viewDefinition.shortName)
        isInnerView = viewDefinition.shortName in {
            'inner_solar_system',
            'inner_solar_system_with_jupiter',
        }

        sunSize = 75 if (self.dimension == '2d' and isInnerView) else (50 if self.dimension == '3d' else 8)
        if self.dimension == '2d':
            axes.plot(0, 0, 'o', markersize=sunSize, color='yellow')
        else:
            plotter.scatter([0], [0], [0], color='yellow', s=sunSize)
            axes.view_init(elev=CAMERA_ELEVATION_DEG, azim=CAMERA_AZIMUTH_DEG)

        self._drawAsteroidBelt(plotter, pointCounts['asteroidBelt'])
        self._drawJupiterLagrangeClouds(plotter, pointCounts['trojansAndGreeks'])
        self._drawHildas(plotter, pointCounts['hildas'])
        self._drawKuiperBelt(plotter, pointCounts['kuiperBelt'])
        self._drawOortCloud(plotter, pointCounts['oortCloud'])
        self._drawPlanets(plotter, viewDefinition, isInnerView)
        self._drawFamousAsteroids(plotter, viewDefinition, isInnerView)
        self._drawOumuamua(plotter, viewDefinition)
        self._drawStars(plotter, viewDefinition)

        if self.dimension == '2d':
            self._drawAnnotations2d(axes, viewDefinition)
            axes.set_xlim(viewDefinition.axisMinAu, viewDefinition.axisMaxAu)
            axes.set_ylim(viewDefinition.axisMinAu, viewDefinition.axisMaxAu)
            axes.set_aspect('equal', 'box')
            axes.axis('off')
            plt.title(viewDefinition.title, fontsize=viewDefinition.titleFontSize, pad=50)
        else:
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

    def _drawAsteroidBelt(self, plotter: DimensionPlotter, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.asteroidBeltInnerAu,
            self.constants.asteroidBeltOuterAu,
            SHELL_THICKNESS_AU,
            numPoints,
        )
        markerSize = 1 if self.dimension == '3d' else 5
        plotter.scatter(
            positionX, positionY, positionZ, color=self.themeColors['beltColor'], s=markerSize
        )

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

    def _drawJupiterLagrangeClouds(self, plotter: DimensionPlotter, numPoints: int) -> None:
        constants = self.constants
        jupiterAngleRad = self._jupiterOrbitalAngleRad()
        inclinationRad = np.radians(constants.jupiterInclinationDeg)
        radialSpreadAu = 0.5
        markerSize = 1 if self.dimension == '3d' else 5

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
            plotter.scatter(positionX, positionY, positionZ, color=self.themeColors['beltColor'], s=markerSize)

    def _drawHildas(self, plotter: DimensionPlotter, numPoints: int) -> None:
        constants = self.constants
        markerSize = 1 if self.dimension == '3d' else 5
        hildaX, hildaY, hildaZ = self.hildaGenerator.clusterPoints(
            constants.jupiterSemiMajorAxisAu,
            constants.jupiterEccentricity,
            constants.jupiterInclinationDeg,
            pointsPerCluster=max(int(numPoints / 4), 1),
        )
        plotter.scatter(hildaX, hildaY, hildaZ, color=self.themeColors['beltColor'], s=markerSize)

        totalPoints = len(hildaX)
        clusterIndices = [0, totalPoints // 3, 2 * totalPoints // 3]
        clusterCenters = [
            (hildaX[index], hildaY[index], hildaZ[index]) for index in clusterIndices
        ]
        bandX, bandY, bandZ = self.hildaGenerator.connectingBands(
            clusterCenters, numPoints, spreadRadiusAu=0.5, bowFactor=-1.75
        )
        plotter.scatter(bandX, bandY, bandZ, color=self.themeColors['beltColor'], s=markerSize)

    def _drawKuiperBelt(self, plotter: DimensionPlotter, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.kuiperBeltInnerAu,
            self.constants.kuiperBeltOuterAu,
            SHELL_THICKNESS_AU,
            numPoints,
        )
        markerSize = 1 if self.dimension == '3d' else 5
        plotter.scatter(positionX, positionY, positionZ, color=self.themeColors['beltColor'], s=markerSize)

    def _drawOortCloud(self, plotter: DimensionPlotter, numPoints: int) -> None:
        positionX, positionY, positionZ = self.beltGenerator.sphericalShell(
            self.constants.oortCloudInnerAu,
            self.constants.oortCloudOuterAu,
            SHELL_THICKNESS_AU * 5,
            numPoints // 5 if self.dimension == '3d' else numPoints,
        )
        markerSize = 1 if self.dimension == '3d' else 5
        plotter.scatter(positionX, positionY, positionZ, color=self.themeColors['beltColor'], s=markerSize)

    def _drawPlanets(
        self, plotter: DimensionPlotter, viewDefinition: ViewDefinition, isInnerView: bool
    ) -> None:
        moonDisplayScale = self.moonCatalog.displayScaleForAxisSpanAu(
            viewDefinition.axisMaxAu - viewDefinition.axisMinAu
        )
        if self.dimension == '2d':
            markerScaleDivisor = 100 if isInnerView else 1000
        else:
            markerScaleDivisor = 2500

        for planet in self.planetCatalog.planets.values():
            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=1000,
            )
            plotter.plot(orbitX, orbitY, orbitZ, color=self.themeColors['orbitColor'])
            sampleIndex = 50 if planet.name == 'Jupiter' else random.randint(0, len(orbitX) - 1)
            markerSize = int(10 + planet.diameterKm / markerScaleDivisor)
            planetX = float(orbitX[sampleIndex])
            planetY = float(orbitY[sampleIndex])
            planetZ = float(orbitZ[sampleIndex])
            plotter.scatter(planetX, planetY, planetZ, color=planet.color, s=markerSize)

            if moonDisplayScale <= 0.0:
                continue

            for moon in self.moonCatalog.forPlanet(planet.name):
                ringAzimuthRad = np.linspace(0, 2 * np.pi, 48)
                orbitRadiusAu = self.moonCatalog.displayOrbitRadiusAu(moon, moonDisplayScale)
                plotter.plot(
                    planetX + orbitRadiusAu * np.cos(ringAzimuthRad),
                    planetY + orbitRadiusAu * np.sin(ringAzimuthRad),
                    np.full(48, planetZ),
                    color=moon.color,
                    alpha=0.25,
                    linewidth=0.6,
                )
                moonMeanAnomalyRad = self.moonCatalog.initialPhaseRad(moon.name)
                moonX, moonY, moonZ = self.moonCatalog.heliocentricPosition(
                    planetX, planetY, planetZ, moon, moonMeanAnomalyRad, moonDisplayScale
                )
                if self.dimension == '2d':
                    moonMarkerSize = self.moonCatalog.markerSize2d(moon, markerScaleDivisor)
                else:
                    moonMarkerSize = self.moonCatalog.markerSize3d(moon)
                plotter.scatter(
                    float(moonX), float(moonY), float(moonZ), color=moon.color, s=moonMarkerSize
                )

    def _drawFamousAsteroids(
        self, plotter: DimensionPlotter, viewDefinition: ViewDefinition, isInnerView: bool
    ) -> None:
        axisSpanAu = viewDefinition.axisMaxAu - viewDefinition.axisMinAu
        if self.dimension == '2d':
            markerScaleDivisor = 80 if isInnerView else 400
            labelFontSize = 14 if isInnerView else 10
        else:
            markerScaleDivisor = 600
            labelFontSize = 8

        for asteroid in self.famousAsteroidCatalog.asteroids.values():
            if not self.famousAsteroidCatalog.visibleForAxisSpanAu(axisSpanAu, asteroid.category):
                continue

            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                asteroid.semiMajorAxisAu,
                asteroid.eccentricity,
                asteroid.inclinationDeg,
                numPoints=300,
            )
            plotter.plot(orbitX, orbitY, orbitZ, color=asteroid.color, alpha=0.45, linewidth=0.8)

            meanAnomalyRad = self.famousAsteroidCatalog.initialPhaseRad(asteroid.name)
            positionX, positionY, positionZ = self.famousAsteroidCatalog.positionAtMeanAnomaly(
                asteroid, meanAnomalyRad, self.orbitCalculator
            )
            if self.dimension == '2d':
                markerSize = self.famousAsteroidCatalog.markerSize2d(asteroid, markerScaleDivisor)
            else:
                markerSize = self.famousAsteroidCatalog.markerSize3d(asteroid)
            plotter.scatter(
                float(positionX),
                float(positionY),
                float(positionZ),
                color=asteroid.color,
                s=markerSize,
            )

            shouldLabel = (
                (self.dimension == '2d' and (isInnerView or asteroid.category == 'kuiper'))
                or (self.dimension == '3d' and (axisSpanAu <= 25 or asteroid.category == 'kuiper'))
            )
            if shouldLabel:
                labelX = float(positionX) + (0.12 if self.dimension == '2d' else 0.0)
                labelY = float(positionY) + (0.12 if self.dimension == '2d' else 0.0)
                plotter.text(
                    labelX,
                    labelY,
                    float(positionZ),
                    asteroid.name,
                    fontsize=labelFontSize,
                    color=asteroid.color,
                )

    def _drawOumuamua(self, plotter: DimensionPlotter, viewDefinition: ViewDefinition) -> None:
        constants = self.constants
        positionX, positionY, positionZ = self.orbitCalculator.hyperbolicOrbit3d(
            constants.oumuamuaPerihelionAu,
            constants.oumuamuaEccentricity,
            constants.oumuamuaInclinationDeg,
            constants.oumuamuaLongitudeAscendingNodeDeg,
            constants.oumuamuaArgumentOfPerihelionDeg,
            numPoints=5000,
        )
        plotter.plot(
            positionX,
            positionY,
            positionZ,
            '--',
            color=self.themeColors['oumuamuaColor'],
            label=OUMUAMUA_LABEL,
        )

        if self.dimension == '2d':
            return

        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        if viewDefinition.shortName in ZOOMED_OUT_SHORT_NAMES:
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
            plotter.text(
                textPoint[0],
                textPoint[1],
                textPoint[2],
                OUMUAMUA_LABEL,
                fontsize=10,
                color=self.themeColors['oumuamuaColor'],
            )
        else:
            perihelionIndex = int(np.argmin(radiusAu))
            plotter.text(
                positionX[perihelionIndex],
                positionY[perihelionIndex],
                positionZ[perihelionIndex],
                OUMUAMUA_LABEL,
                fontsize=10,
                color=self.themeColors['oumuamuaColor'],
            )

    def _drawStars(self, plotter: DimensionPlotter, viewDefinition: ViewDefinition) -> None:
        if self.dimension == '2d':
            viewId = viewDefinition.viewId
            if 'nearest_stars' not in viewId and 'alpha_centauri' not in viewId:
                return
            starsInView = self.starCatalog.starsWithinLightYears(
                ViewRegistry.maxStarDistanceLy(viewId)
            )
            labelOffsetAu = 0.01 * (viewDefinition.axisMaxAu - viewDefinition.axisMinAu)
            for _, starRow in starsInView.iterrows():
                if starRow['System'].startswith('Vega'):
                    plotter.scatter(
                        [starRow['positionX']],
                        [starRow['positionY']],
                        [starRow['positionZ']],
                        color='silver',
                        s=900,
                    )
                else:
                    plotter.scatter(
                        [starRow['positionX']],
                        [starRow['positionY']],
                        [starRow['positionZ']],
                        color='orange',
                        s=225,
                    )
                plotter.text(
                    starRow['positionX'] + labelOffsetAu,
                    starRow['positionY'],
                    starRow['positionZ'],
                    starRow['System'][:20],
                    fontsize=20,
                    ha='left',
                    va='center',
                )
            return

        starsFrame = self.starCatalog.starsDataFrame
        starsInView = starsFrame[
            starsFrame['distanceAu'] <= viewDefinition.axisMaxAu
        ].dropna(subset=['positionX', 'positionY', 'positionZ'])

        for _, starRow in starsInView.iterrows():
            if 'Vega' in starRow['System']:
                plotter.scatter(
                    starRow['positionX'],
                    starRow['positionY'],
                    starRow['positionZ'],
                    color='silver',
                    s=500,
                    alpha=0.9,
                )
            else:
                plotter.scatter(
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
                plotter.text(
                    starRow['positionX'],
                    starRow['positionY'],
                    starRow['positionZ'],
                    systemLabel,
                    fontsize=10,
                    ha='left',
                    va='bottom',
                )
                self.labeledStarSystems.add(starRow['System'][:10])

    def _oumuamuaTrajectory(self, numPoints: int = 5000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.orbitCalculator.hyperbolicOrbit3d(
            self.constants.oumuamuaPerihelionAu,
            self.constants.oumuamuaEccentricity,
            self.constants.oumuamuaInclinationDeg,
            self.constants.oumuamuaLongitudeAscendingNodeDeg,
            self.constants.oumuamuaArgumentOfPerihelionDeg,
            numPoints=numPoints,
        )

    def _oumuamuaInboundDirection(self) -> Tuple[float, float, float]:
        positionX, positionY, positionZ = self._oumuamuaTrajectory()
        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        inboundMagnitude = radiusAu[0]
        if inboundMagnitude == 0:
            return 0.0, -1.0, 0.0
        return (
            float(positionX[0] / inboundMagnitude),
            float(positionY[0] / inboundMagnitude),
            float(positionZ[0] / inboundMagnitude),
        )

    def _perpendicularLabelPosition2d(
        self,
        anchorPosition: Tuple[float, float],
        directionVector: Tuple[float, float],
        axisSpanAu: float,
    ) -> Tuple[float, float]:
        perpendicularX = -directionVector[1]
        perpendicularY = directionVector[0]
        if perpendicularX * anchorPosition[0] + perpendicularY * anchorPosition[1] < 0:
            perpendicularX, perpendicularY = -perpendicularX, -perpendicularY
        labelOffsetAu = 0.10 * axisSpanAu
        return (
            anchorPosition[0] + perpendicularX * labelOffsetAu,
            anchorPosition[1] + perpendicularY * labelOffsetAu,
        )

    def _oumuamuaAnnotation2d(
        self, viewDefinition: ViewDefinition
    ) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
        axisSpanAu = viewDefinition.axisMaxAu - viewDefinition.axisMinAu
        positionX, positionY, _ = self._oumuamuaTrajectory()

        if viewDefinition.shortName in ZOOMED_OUT_SHORT_NAMES:
            inboundDirection = self._oumuamuaInboundDirection()
            targetRadiusAu = {
                'solar_system_with_kuiper_belt': 55,
                'solar_system_with_oort_cloud': 45000,
                'solar_system_with_alpha_centauri': 45000,
            }[viewDefinition.shortName]
            anchorPosition = (
                inboundDirection[0] * targetRadiusAu,
                inboundDirection[1] * targetRadiusAu,
            )
            labelPosition = self._perpendicularLabelPosition2d(
                anchorPosition, inboundDirection[:2], axisSpanAu
            )
            return OUMUAMUA_LABEL, anchorPosition, labelPosition

        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2)
        visibleMask = (
            (positionX >= viewDefinition.axisMinAu)
            & (positionX <= viewDefinition.axisMaxAu)
            & (positionY >= viewDefinition.axisMinAu)
            & (positionY <= viewDefinition.axisMaxAu)
        )
        if visibleMask.any():
            visibleIndices = np.where(visibleMask)[0]
            sampleIndex = int(visibleIndices[len(visibleIndices) // 3])
            anchorPosition = (float(positionX[sampleIndex]), float(positionY[sampleIndex]))
        else:
            sampleIndex = int(np.argmax(radiusAu))
            anchorPosition = (float(positionX[sampleIndex]), float(positionY[sampleIndex]))

        labelOffsetAu = 0.08 * axisSpanAu
        labelPosition = (anchorPosition[0] - labelOffsetAu, anchorPosition[1] - labelOffsetAu)
        return OUMUAMUA_LABEL, anchorPosition, labelPosition

    def _jupiterLagrangeAnnotations2d(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
        jupiterAngleRad = self._jupiterOrbitalAngleRad()
        semiMajorAxisAu = self.constants.jupiterSemiMajorAxisAu
        lagrangeOffsetRad = np.deg2rad(60)
        trojanAnchor = (
            semiMajorAxisAu * np.cos(jupiterAngleRad + lagrangeOffsetRad),
            semiMajorAxisAu * np.sin(jupiterAngleRad + lagrangeOffsetRad),
        )
        greekAnchor = (
            semiMajorAxisAu * np.cos(jupiterAngleRad - lagrangeOffsetRad),
            semiMajorAxisAu * np.sin(jupiterAngleRad - lagrangeOffsetRad),
        )
        return [
            ('Trojans', trojanAnchor, (trojanAnchor[0] + 1.0, trojanAnchor[1] + 0.5)),
            ('Greeks', greekAnchor, (greekAnchor[0] - 1.0, greekAnchor[1] - 2.2)),
        ]

    def _drawAnnotations2d(self, axes: plt.Axes, viewDefinition: ViewDefinition) -> None:
        constants = self.constants
        viewId = viewDefinition.viewId
        annotationConfigs: dict[str, List[Tuple[str, Tuple[float, float], Tuple[float, float]]]] = {
            '0_inner_solar_system': [
                (
                    'Asteroid Belt (2.2-3.2 AU)',
                    (constants.asteroidBeltOuterAu, 0),
                    (constants.asteroidBeltInnerAu + 0.1, 1.5),
                ),
            ],
            '1_inner_solar_system_with_jupiter': [
                (
                    'Asteroid Belt (2.2-3.2 AU)',
                    (constants.asteroidBeltOuterAu, 0),
                    (constants.asteroidBeltInnerAu + 0.1, 2),
                ),
                (
                    'Hildas',
                    (-constants.jupiterSemiMajorAxisAu + 0.5, 0),
                    (-constants.jupiterSemiMajorAxisAu - 1, -2.5),
                ),
            ],
            '2_solar_system_with_kuiper_belt': [
                (
                    f'Kuiper Belt ({constants.kuiperBeltInnerAu}-{constants.kuiperBeltOuterAu} AU)',
                    (constants.kuiperBeltOuterAu, 0),
                    (constants.kuiperBeltOuterAu + 5, 10),
                ),
                (
                    f"Pluto's aphelion ({constants.plutoAphelionAu:.1f} AU)",
                    (-constants.plutoAphelionAu, 0),
                    (-constants.plutoAphelionAu - 25, 10),
                ),
                (
                    f"Pluto's perihelion ({constants.plutoPerihelionAu:.1f} AU)",
                    (constants.plutoPerihelionAu, 0),
                    (constants.plutoPerihelionAu + 10, -10),
                ),
            ],
            '3_solar_system_with_oort_cloud': [
                (
                    f'Kuiper Belt ({constants.kuiperBeltInnerAu}-{constants.kuiperBeltOuterAu} AU)',
                    (constants.kuiperBeltOuterAu - 3500, -4000),
                    (constants.kuiperBeltOuterAu + 80000, 90000),
                ),
                ('Oort Cloud (100000 AU)', (100000, 5), (70000, 25000)),
            ],
            '4_solar_system_with_alpha_centauri': [
                (
                    f'Kuiper Belt ({constants.kuiperBeltInnerAu}-{constants.kuiperBeltOuterAu} AU)',
                    (constants.kuiperBeltOuterAu - 3500, -4000),
                    (constants.kuiperBeltOuterAu + 80000, 110000),
                ),
                ('Oort Cloud (100000 AU)', (-100000, 5), (-180000, 25000)),
            ],
        }

        annotations = list(annotationConfigs.get(viewId, []))
        if viewId == '1_inner_solar_system_with_jupiter':
            annotations.extend(self._jupiterLagrangeAnnotations2d())
        if viewId in {
            '0_inner_solar_system',
            '1_inner_solar_system_with_jupiter',
            '2_solar_system_with_kuiper_belt',
            '3_solar_system_with_oort_cloud',
            '4_solar_system_with_alpha_centauri',
        }:
            annotations.append(self._oumuamuaAnnotation2d(viewDefinition))

        for labelText, arrowTarget, labelPosition in annotations:
            axes.annotate(
                labelText,
                xy=arrowTarget,
                xytext=labelPosition,
                fontsize=LABEL_FONT_SIZE,
                arrowprops=dict(
                    facecolor=self.themeColors['annotationFaceColor'],
                    shrink=0.05,
                ),
            )

    def _oumuamuaVegaSkySeparationDeg(self) -> float:
        positionX, positionY, positionZ = self._oumuamuaTrajectory()
        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        inboundVector = np.array([positionX[0], positionY[0], positionZ[0]]) / radiusAu[0]
        vegaRow = self.starCatalog.vegaRow()
        vegaVector = np.array(
            [vegaRow['positionX'], vegaRow['positionY'], vegaRow['positionZ']], dtype=float
        )
        vegaVector /= np.linalg.norm(vegaVector)
        return float(np.degrees(np.arccos(np.clip(np.dot(inboundVector, vegaVector), -1.0, 1.0))))

    def _renderOumuamuaVegaView(self, outputPath: str) -> None:
        figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        vegaRow = self.starCatalog.vegaRow()
        vegaX, vegaY = float(vegaRow['positionX']), float(vegaRow['positionY'])
        vegaDistanceAu = np.hypot(vegaX, vegaY)
        skySeparationDeg = self._oumuamuaVegaSkySeparationDeg()
        inboundDirection = self._oumuamuaInboundDirection()

        axes.plot(0, 0, 'o', markersize=20, color='yellow')
        oumuamuaX, oumuamuaY, _ = self._oumuamuaTrajectory()
        oumuamuaColor = self.themeColors['oumuamuaColor']
        axes.plot(oumuamuaX, oumuamuaY, '--', color=oumuamuaColor, linewidth=1.5)

        asymptoteLengthAu = vegaDistanceAu * 0.42
        asymptoteEnd = (
            inboundDirection[0] * asymptoteLengthAu,
            inboundDirection[1] * asymptoteLengthAu,
        )
        axes.plot([0, asymptoteEnd[0]], [0, asymptoteEnd[1]], '--', color=oumuamuaColor, linewidth=2)
        axes.plot([0, vegaX], [0, vegaY], ':', color=self.themeColors['beltColor'], linewidth=1.5)

        for _, starRow in self.starCatalog.starsWithinLightYears(25.05).iterrows():
            if starRow['System'].startswith('Vega'):
                axes.plot(starRow['positionX'], starRow['positionY'], 'o', markersize=40, color='silver')
                axes.text(
                    starRow['positionX'] + 15000,
                    starRow['positionY'],
                    'Vega (25 ly)',
                    fontsize=36,
                    ha='left',
                    va='center',
                )
            else:
                axes.plot(
                    starRow['positionX'],
                    starRow['positionY'],
                    'o',
                    markersize=10,
                    color='orange',
                    alpha=0.6,
                )

        asymptoteAnchor = (asymptoteEnd[0] * 0.72, asymptoteEnd[1] * 0.72)
        asymptoteLabelPosition = self._perpendicularLabelPosition2d(
            asymptoteAnchor, inboundDirection[:2], vegaDistanceAu
        )
        axes.annotate(
            "'Oumuamua inbound asymptote",
            xy=asymptoteAnchor,
            xytext=asymptoteLabelPosition,
            fontsize=36,
            arrowprops=dict(
                facecolor=self.themeColors['annotationFaceColor'],
                shrink=0.05,
            ),
        )
        axes.annotate(
            f"Sky direction near Vega (~{skySeparationDeg:.0f}° away) — not from Vega",
            xy=(vegaX * 0.55, vegaY * 0.55),
            xytext=(vegaX * 0.35, vegaY * 0.75),
            fontsize=36,
            arrowprops=dict(
                facecolor=self.themeColors['annotationFaceColor'],
                shrink=0.05,
                linestyle=':',
            ),
        )

        paddingFactor = 0.18
        axes.set_xlim(
            min(0.0, vegaX) - abs(vegaX) * paddingFactor,
            max(0.0, vegaX) + abs(vegaX) * paddingFactor,
        )
        axes.set_ylim(
            min(0.0, vegaY) - abs(vegaY) * paddingFactor,
            max(0.0, vegaY) + abs(vegaY) * paddingFactor,
        )
        axes.set_aspect('equal', 'box')
        axes.axis('off')
        plt.title("'Oumuamua and Vega — Interstellar Context (25 light years)", fontsize=72, pad=50)
        figure.savefig(outputPath, dpi=OUTPUT_DPI)
        plt.close(figure)



def renderAll(dimension: Dimension = '2d', starsCsvPath: str = 'data/nearby_stars_30.csv') -> None:
    for theme in ('light', 'dark'):
        visualizer = SolarSystemVisualizer(starsCsvPath, dimension=dimension, theme=theme)
        visualizer.renderAllViews()
    print(f'All {dimension.upper()} visualizations completed!')
