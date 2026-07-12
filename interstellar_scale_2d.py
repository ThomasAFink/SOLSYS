"""2D interstellar-scale solar system visualizations."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from solsys_core import (
    AstronomicalConstants,
    BeltPointGenerator,
    MoonCatalog,
    OrbitCalculator,
    PlanetCatalog,
    PointDensityConfig,
    StarCatalog,
    ViewRegistry,
)

FIGURE_SIZE_INCHES = (39, 39)
OUTPUT_DPI = 300
LABEL_FONT_SIZE = 48
LAGRANGE_OFFSET_RAD = np.deg2rad(60)
LAGRANGE_ANGULAR_SPREAD_RAD = np.pi / 3
LAGRANGE_RADIAL_SPREAD_AU = 0.5
OUMUAMUA_LABEL = "'Oumuamua hyperbolic trajectory"
ZOOMED_OUT_VIEWS = {
    '2_solar_system_with_kuiper_belt',
    '3_solar_system_with_oort_cloud',
    '4_solar_system_with_alpha_centauri',
}


class SolarSystemVisualizer2D:
    def __init__(self, starsCsvPath: str):
        self.constants = AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.orbitCalculator = OrbitCalculator()
        self.beltGenerator = BeltPointGenerator()
        self.labeledStarSystems: set[str] = set()

    def renderView(self, viewId: str, outputPath: str) -> None:
        if viewId == '8_oumuamua_and_vega_25':
            self._renderOumuamuaVegaView(outputPath)
            return

        figure, axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        pointCounts = PointDensityConfig.forView(viewId)
        self._drawSun(axes, viewId)
        self._drawPlanets(axes, viewId)
        self._drawBeltsAndClouds(axes, pointCounts)
        if 'nearest_stars' in viewId or 'alpha_centauri' in viewId:
            self._drawNearbyStars(axes, viewId)
        self._drawAnnotations(axes, viewId)
        self._applyAxesStyle(axes, viewId)
        figure.savefig(outputPath, dpi=OUTPUT_DPI)
        plt.close(figure)

    def _drawSun(self, axes: plt.Axes, viewId: str) -> None:
        markerSize = 75 if viewId in {
            '0_inner_solar_system', '1_inner_solar_system_with_jupiter'
        } else 8
        axes.plot(0, 0, 'o', markersize=markerSize, color='yellow')

    def _drawPlanets(self, axes: plt.Axes, viewId: str) -> None:
        isInnerView = viewId in {'0_inner_solar_system', '1_inner_solar_system_with_jupiter'}
        markerScaleDivisor = 100 if isInnerView else 1000
        axisMinAu, axisMaxAu = ViewRegistry.axisLimitsForView(viewId)
        moonDisplayScale = self.moonCatalog.displayScaleForAxisSpanAu(axisMaxAu - axisMinAu)

        for planet in self.planetCatalog.planets.values():
            orbitX, orbitY = self.orbitCalculator.ellipticalOrbit2d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=1000,
            )
            axes.plot(orbitX, orbitY, color='black')
            markerSize = int(10 + planet.diameterKm / markerScaleDivisor)
            sampleIndex = 50 if planet.name == 'Jupiter' else random.randint(0, len(orbitX) - 1)
            planetX = float(orbitX[sampleIndex])
            planetY = float(orbitY[sampleIndex])
            axes.scatter(planetX, planetY, color=planet.color, s=markerSize)

            if moonDisplayScale <= 0.0:
                continue

            for moon in self.moonCatalog.forPlanet(planet.name):
                ringX, ringY = self.moonCatalog.moonOrbitRing2d(
                    moon, planetX, planetY, moonDisplayScale
                )
                axes.plot(ringX, ringY, color=moon.color, alpha=0.25, linewidth=0.6, zorder=4)
                moonMeanAnomalyRad = self.moonCatalog.initialPhaseRad(moon.name)
                moonX, moonY, _ = self.moonCatalog.heliocentricPosition(
                    planetX, planetY, 0.0, moon, moonMeanAnomalyRad, moonDisplayScale
                )
                moonMarkerSize = self.moonCatalog.markerSize2d(moon, markerScaleDivisor)
                axes.scatter(float(moonX), float(moonY), color=moon.color, s=moonMarkerSize, zorder=5)

    def _jupiterOrbitalAngleRad(self) -> float:
        jupiter = self.planetCatalog.planets['Jupiter']
        orbitX, orbitY = self.orbitCalculator.ellipticalOrbit2d(
            jupiter.semiMajorAxisAu,
            jupiter.eccentricity,
            jupiter.inclinationDeg,
            numPoints=1000,
        )
        sampleIndex = 50
        return float(np.arctan2(orbitY[sampleIndex], orbitX[sampleIndex]))

    def _drawHildas(self, axes: plt.Axes, numPoints: int) -> None:
        innerRadiusAu = self.constants.asteroidBeltOuterAu + 0.25
        outerRadiusAu = self.constants.jupiterSemiMajorAxisAu - 0.25
        clusterPoints = max(numPoints // 3, 1)
        clusterAnglesRad = np.array([0, 2 * np.pi / 3, 4 * np.pi / 3])

        for clusterAngleRad in clusterAnglesRad:
            radiusAu = np.random.uniform(innerRadiusAu, outerRadiusAu, clusterPoints)
            azimuthRad = clusterAngleRad + np.random.normal(0, np.pi / 12, clusterPoints)
            axes.scatter(
                radiusAu * np.cos(azimuthRad),
                radiusAu * np.sin(azimuthRad),
                color='#AAAAAA',
                s=5,
            )

            for _ in range(clusterPoints):
                segmentProgress = np.random.uniform(0, 1)
                radiusAu = np.random.uniform(innerRadiusAu, outerRadiusAu)
                nextClusterIndex = (np.where(clusterAnglesRad == clusterAngleRad)[0][0] + 1) % 3
                nextClusterAngleRad = clusterAnglesRad[nextClusterIndex]
                azimuthRad = clusterAngleRad + segmentProgress * (nextClusterAngleRad - clusterAngleRad)
                perpendicularOffsetAu = np.random.normal(0, 0.2)
                positionX = radiusAu * np.cos(azimuthRad)
                positionY = radiusAu * np.sin(azimuthRad)
                perpendicularAngleRad = azimuthRad + np.pi / 2
                positionX += perpendicularOffsetAu * np.cos(perpendicularAngleRad)
                positionY += perpendicularOffsetAu * np.sin(perpendicularAngleRad)
                axes.scatter(positionX, positionY, color='#AAAAAA', s=5)

    def _drawBeltsAndClouds(self, axes: plt.Axes, pointCounts: Dict[str, int]) -> None:
        constants = self.constants
        beltGenerator = self.beltGenerator

        beltX, beltY = beltGenerator.ring2d(
            constants.asteroidBeltInnerAu,
            constants.asteroidBeltOuterAu,
            pointCounts['asteroidBelt'],
        )
        axes.scatter(beltX, beltY, color='gray', s=5)

        jupiterAngleRad = self._jupiterOrbitalAngleRad()
        trojanX, trojanY = beltGenerator.jupiterLagrangeCloud2d(
            jupiterAngleRad,
            LAGRANGE_OFFSET_RAD,
            constants.jupiterSemiMajorAxisAu,
            LAGRANGE_RADIAL_SPREAD_AU,
            LAGRANGE_ANGULAR_SPREAD_RAD,
            pointCounts['trojansAndGreeks'],
        )
        greekX, greekY = beltGenerator.jupiterLagrangeCloud2d(
            jupiterAngleRad,
            -LAGRANGE_OFFSET_RAD,
            constants.jupiterSemiMajorAxisAu,
            LAGRANGE_RADIAL_SPREAD_AU,
            LAGRANGE_ANGULAR_SPREAD_RAD,
            pointCounts['trojansAndGreeks'],
        )
        axes.scatter(trojanX, trojanY, color='gray', s=5)
        axes.scatter(greekX, greekY, color='gray', s=5)

        self._drawHildas(axes, pointCounts['hildas'])

        for innerAu, outerAu, pointKey in (
            (constants.kuiperBeltInnerAu, constants.kuiperBeltOuterAu, 'kuiperBelt'),
            (constants.oortCloudInnerAu, constants.oortCloudOuterAu, 'oortCloud'),
        ):
            cloudX, cloudY = beltGenerator.ring2d(innerAu, outerAu, pointCounts[pointKey])
            axes.scatter(cloudX, cloudY, color='gray', s=5)

        oumuamuaX, oumuamuaY = self._oumuamuaTrajectory2d()
        axes.plot(oumuamuaX, oumuamuaY, '--', color='darkred', label=OUMUAMUA_LABEL)

    def _oumuamuaTrajectory2d(self, numPoints: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        positionX, positionY, _ = self.orbitCalculator.hyperbolicOrbit3d(
            self.constants.oumuamuaPerihelionAu,
            self.constants.oumuamuaEccentricity,
            self.constants.oumuamuaInclinationDeg,
            self.constants.oumuamuaLongitudeAscendingNodeDeg,
            self.constants.oumuamuaArgumentOfPerihelionDeg,
            numPoints=numPoints,
        )
        return positionX, positionY

    def _oumuamuaInboundDirection2d(self) -> Tuple[float, float]:
        positionX, positionY = self._oumuamuaTrajectory2d()
        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2)
        inboundMagnitude = radiusAu[0]
        if inboundMagnitude == 0:
            return 0.0, -1.0
        return float(positionX[0] / inboundMagnitude), float(positionY[0] / inboundMagnitude)

    def _perpendicularLabelPosition(
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

    def _oumuamuaAnnotation(self, viewId: str) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
        axisMinAu, axisMaxAu = ViewRegistry.axisLimitsForView(viewId)
        axisSpanAu = axisMaxAu - axisMinAu

        if viewId in ZOOMED_OUT_VIEWS:
            inboundDirection = self._oumuamuaInboundDirection2d()
            targetRadiusAu = {
                '2_solar_system_with_kuiper_belt': 55,
                '3_solar_system_with_oort_cloud': 45000,
                '4_solar_system_with_alpha_centauri': 45000,
            }[viewId]
            anchorPosition = (
                inboundDirection[0] * targetRadiusAu,
                inboundDirection[1] * targetRadiusAu,
            )
            labelPosition = self._perpendicularLabelPosition(
                anchorPosition, inboundDirection, axisSpanAu
            )
            return OUMUAMUA_LABEL, anchorPosition, labelPosition

        positionX, positionY = self._oumuamuaTrajectory2d()
        radiusAu = np.sqrt(positionX ** 2 + positionY ** 2)
        visibleMask = (
            (positionX >= axisMinAu) & (positionX <= axisMaxAu)
            & (positionY >= axisMinAu) & (positionY <= axisMaxAu)
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

    def _jupiterLagrangeAnnotations(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
        jupiterAngleRad = self._jupiterOrbitalAngleRad()
        semiMajorAxisAu = self.constants.jupiterSemiMajorAxisAu
        trojanAnchor = (
            semiMajorAxisAu * np.cos(jupiterAngleRad + LAGRANGE_OFFSET_RAD),
            semiMajorAxisAu * np.sin(jupiterAngleRad + LAGRANGE_OFFSET_RAD),
        )
        greekAnchor = (
            semiMajorAxisAu * np.cos(jupiterAngleRad - LAGRANGE_OFFSET_RAD),
            semiMajorAxisAu * np.sin(jupiterAngleRad - LAGRANGE_OFFSET_RAD),
        )
        return [
            ('Trojans', trojanAnchor, (trojanAnchor[0] + 1.0, trojanAnchor[1] + 0.5)),
            ('Greeks', greekAnchor, (greekAnchor[0] - 1.0, greekAnchor[1] - 2.2)),
        ]

    def _drawAnnotations(self, axes: plt.Axes, viewId: str) -> None:
        constants = self.constants
        annotationConfigs: Dict[str, List[Tuple[str, Tuple[float, float], Tuple[float, float]]]] = {
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

        if viewId == '1_inner_solar_system_with_jupiter':
            annotationConfigs.setdefault(viewId, []).extend(self._jupiterLagrangeAnnotations())

        if viewId in {
            '0_inner_solar_system',
            '1_inner_solar_system_with_jupiter',
            '2_solar_system_with_kuiper_belt',
            '3_solar_system_with_oort_cloud',
            '4_solar_system_with_alpha_centauri',
        }:
            annotationConfigs.setdefault(viewId, []).append(self._oumuamuaAnnotation(viewId))

        for labelText, arrowTarget, labelPosition in annotationConfigs.get(viewId, []):
            axes.annotate(
                labelText,
                xy=arrowTarget,
                xytext=labelPosition,
                fontsize=LABEL_FONT_SIZE,
                arrowprops=dict(facecolor='black', shrink=0.05),
            )

    def _drawNearbyStars(self, axes: plt.Axes, viewId: str) -> None:
        maxDistanceLy = ViewRegistry.maxStarDistanceLy(viewId)
        starsInView = self.starCatalog.starsWithinLightYears(maxDistanceLy)
        axisMinAu, axisMaxAu = ViewRegistry.axisLimitsForView(viewId)
        labelOffsetAu = 0.01 * (axisMaxAu - axisMinAu)

        for _, starRow in starsInView.iterrows():
            if starRow['System'].startswith('Vega'):
                axes.plot(starRow['positionX'], starRow['positionY'], 'o', markersize=30, color='silver')
            else:
                axes.plot(starRow['positionX'], starRow['positionY'], 'o', markersize=15, color='orange')
            axes.text(
                starRow['positionX'] + labelOffsetAu,
                starRow['positionY'],
                starRow['System'][:20],
                fontsize=20,
                ha='left',
                va='center',
            )

    def _applyAxesStyle(self, axes: plt.Axes, viewId: str) -> None:
        axisMinAu, axisMaxAu = ViewRegistry.axisLimitsForView(viewId)
        axes.set_xlim(axisMinAu, axisMaxAu)
        axes.set_ylim(axisMinAu, axisMaxAu)
        axes.set_aspect('equal', 'box')
        axes.axis('off')
        plt.title(ViewRegistry.titleForView(viewId), fontsize=80, pad=50)

    def _vegaPosition2d(self) -> Tuple[float, float]:
        vegaRow = self.starCatalog.vegaRow()
        return float(vegaRow['positionX']), float(vegaRow['positionY'])

    def _oumuamuaVegaSkySeparationDeg(self) -> float:
        positionX, positionY, positionZ = self.orbitCalculator.hyperbolicOrbit3d(
            self.constants.oumuamuaPerihelionAu,
            self.constants.oumuamuaEccentricity,
            self.constants.oumuamuaInclinationDeg,
            self.constants.oumuamuaLongitudeAscendingNodeDeg,
            self.constants.oumuamuaArgumentOfPerihelionDeg,
            numPoints=5000,
        )
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
        vegaX, vegaY = self._vegaPosition2d()
        vegaDistanceAu = np.hypot(vegaX, vegaY)
        skySeparationDeg = self._oumuamuaVegaSkySeparationDeg()
        inboundDirection = self._oumuamuaInboundDirection2d()

        axes.plot(0, 0, 'o', markersize=20, color='yellow')
        oumuamuaX, oumuamuaY = self._oumuamuaTrajectory2d()
        axes.plot(oumuamuaX, oumuamuaY, '--', color='darkred', linewidth=1.5)

        asymptoteLengthAu = vegaDistanceAu * 0.42
        asymptoteEnd = (
            inboundDirection[0] * asymptoteLengthAu,
            inboundDirection[1] * asymptoteLengthAu,
        )
        axes.plot([0, asymptoteEnd[0]], [0, asymptoteEnd[1]], '--', color='darkred', linewidth=2)
        axes.plot([0, vegaX], [0, vegaY], ':', color='gray', linewidth=1.5)

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
        asymptoteLabelPosition = self._perpendicularLabelPosition(
            asymptoteAnchor, inboundDirection, vegaDistanceAu
        )
        axes.annotate(
            "'Oumuamua inbound asymptote",
            xy=asymptoteAnchor,
            xytext=asymptoteLabelPosition,
            fontsize=36,
            arrowprops=dict(facecolor='black', shrink=0.05),
        )
        axes.annotate(
            f"Sky direction near Vega (~{skySeparationDeg:.0f}° away) — not from Vega",
            xy=(vegaX * 0.55, vegaY * 0.55),
            xytext=(vegaX * 0.35, vegaY * 0.75),
            fontsize=36,
            arrowprops=dict(facecolor='black', shrink=0.05, linestyle=':'),
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

    @staticmethod
    def ensureOutputDirectory(outputDirectory: str = 'output/2d') -> None:
        os.makedirs(outputDirectory, exist_ok=True)

    def renderAllViews(self, outputDirectory: str = 'output/2d') -> None:
        self.ensureOutputDirectory(outputDirectory)
        viewDefinitions = [
            ('0_inner_solar_system', 'Inner Solar System (±3.5 AU)'),
            ('1_inner_solar_system_with_jupiter', 'Solar System to Jupiter (±6 AU)'),
            ('2_solar_system_with_kuiper_belt', 'Solar System with Kuiper Belt (±70 AU)'),
            ('3_solar_system_with_oort_cloud', 'Solar System with Oort Cloud (±100,000 AU)'),
            ('4_solar_system_with_alpha_centauri', 'Local Space with Alpha Centauri (±280,000 AU)'),
            ('5_solar_system_with_nearest_stars_10', 'Stars within 10 Light Years (±632,410 AU)'),
            ('6_solar_system_with_nearest_stars_25', 'Stars within 25 Light Years (±1,584,190 AU)'),
            ('7_solar_system_with_nearest_stars_30', 'Stars within 30 Light Years (±1,897,232 AU)'),
            ('8_oumuamua_and_vega_25', "'Oumuamua and Vega (25 light years)"),
        ]

        for viewId, description in viewDefinitions:
            outputPath = (
                f'{outputDirectory}/8_oumuamua_origin_vega_system_25.jpg'
                if viewId == '8_oumuamua_and_vega_25'
                else f'{outputDirectory}/{viewId}.jpg'
            )
            print(f'Generating {description}...')
            self.renderView(viewId, outputPath)
            print(f'Saved to {outputPath}')


if __name__ == '__main__':
    visualizer = SolarSystemVisualizer2D('data/nearby_stars_30.csv')
    visualizer.renderAllViews()
    print('All visualizations completed!')
