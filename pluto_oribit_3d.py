"""3D views of Pluto's orbit and the Kuiper belt."""

from __future__ import annotations

import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from solsys_core import AstronomicalConstants, BeltPointGenerator, MoonCatalog, OrbitCalculator, PlanetCatalog

FIGURE_SIZE_INCHES = (20, 20)
OUTPUT_DPI = 300
ORBIT_SAMPLE_POINTS = 1000
KUIPER_BELT_POINT_COUNT = 20000
CAMERA_ANGLES_DEG: List[Tuple[int, int]] = [
    (90, 0),
    (45, 300),
    (30, 210),
    (20, 120),
]


class PlutoOrbitVisualizer3D:
    def __init__(self):
        self.constants = AstronomicalConstants()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.orbitCalculator = OrbitCalculator()
        self.beltGenerator = BeltPointGenerator()

    def renderAllViews(self, outputDirectory: str = 'output') -> None:
        os.makedirs(outputDirectory, exist_ok=True)
        for elevationDeg, azimuthDeg in CAMERA_ANGLES_DEG:
            outputPath = f'{outputDirectory}/pluto_orbit_3d_view_{elevationDeg}_{azimuthDeg}.jpg'
            print(f'Generating Pluto orbit view (elev={elevationDeg}, azim={azimuthDeg})...')
            self.renderView(elevationDeg, azimuthDeg, outputPath)
            print(f'Saved to {outputPath}')

    def renderView(self, elevationDeg: int, azimuthDeg: int, outputPath: str) -> None:
        constants = self.constants
        pluto = self.planetCatalog.planets['Pluto']

        figure = plt.figure(figsize=FIGURE_SIZE_INCHES)
        axes = figure.add_subplot(111, projection='3d')
        axes.scatter(0, 0, 0, color='yellow', s=100, label='Sun')

        for planet in self.planetCatalog.planets.values():
            if planet.name == 'Pluto':
                continue
            orbitX, orbitY = self.orbitCalculator.ellipticalOrbit2d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=ORBIT_SAMPLE_POINTS,
            )
            axes.plot(orbitX, orbitY, 0, color='black')

        plutoX, plutoY, plutoZ = self.orbitCalculator.ellipticalOrbit3d(
            pluto.semiMajorAxisAu,
            pluto.eccentricity,
            pluto.inclinationDeg,
            numPoints=ORBIT_SAMPLE_POINTS,
        )
        axes.plot(plutoX, plutoY, plutoZ, color='blue', label="Pluto's Orbit")

        plutoSampleIndex = len(plutoX) // 4
        axes.scatter(
            plutoX[plutoSampleIndex],
            plutoY[plutoSampleIndex],
            plutoZ[plutoSampleIndex],
            color=pluto.color,
            s=40,
            label='Pluto',
        )
        for moon in self.moonCatalog.forPlanet('Pluto'):
            moonDisplayScale = self.moonCatalog.DISPLAY_ORBIT_SCALE
            ringAzimuthRad = np.linspace(0, 2 * np.pi, 48)
            orbitRadiusAu = self.moonCatalog.displayOrbitRadiusAu(moon, moonDisplayScale)
            axes.plot(
                plutoX[plutoSampleIndex] + orbitRadiusAu * np.cos(ringAzimuthRad),
                plutoY[plutoSampleIndex] + orbitRadiusAu * np.sin(ringAzimuthRad),
                plutoZ[plutoSampleIndex],
                color=moon.color,
                alpha=0.35,
                linewidth=0.8,
            )
            moonMeanAnomalyRad = self.moonCatalog.initialPhaseRad(moon.name)
            moonX, moonY, moonZ = self.moonCatalog.heliocentricPosition(
                plutoX[plutoSampleIndex],
                plutoY[plutoSampleIndex],
                plutoZ[plutoSampleIndex],
                moon,
                moonMeanAnomalyRad,
                moonDisplayScale,
            )
            axes.scatter(
                float(moonX),
                float(moonY),
                float(moonZ),
                color=moon.color,
                s=self.moonCatalog.markerSize3d(moon, 400),
                label=moon.name,
            )

        kuiperBeltX, kuiperBeltY, kuiperBeltZ = self._kuiperBeltPoints()
        axes.scatter(kuiperBeltX, kuiperBeltY, kuiperBeltZ, color='darkgray', s=1, alpha=0.5)

        axes.text(-constants.plutoAphelionAu, 0, 0, "Pluto's aphelion", color='blue', fontsize=12)
        axes.text(constants.plutoPerihelionAu, 0, 0, "Pluto's perihelion", color='blue', fontsize=12)
        axes.set_xlabel('X (AU)')
        axes.set_ylabel('Y (AU)')
        axes.set_zlabel('Z (AU)')
        axes.view_init(elev=elevationDeg, azim=azimuthDeg)
        axes.set_title('3D Representation of Pluto’s Orbit and the Kuiper Belt', fontsize=20)
        axes.legend()
        plt.axis('off')
        figure.savefig(outputPath, bbox_inches='tight', dpi=OUTPUT_DPI)
        plt.close(figure)

    def _kuiperBeltPoints(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        constants = self.constants
        plutoInclinationRad = np.radians(self.planetCatalog.planets['Pluto'].inclinationDeg)
        radiusAu = np.random.uniform(
            constants.kuiperBeltInnerAu,
            constants.kuiperBeltOuterAu,
            KUIPER_BELT_POINT_COUNT,
        )
        trueAnomalyRad = np.random.uniform(0, 2 * np.pi, KUIPER_BELT_POINT_COUNT)
        polarAngleRad = np.random.uniform(0, 2 * np.pi, KUIPER_BELT_POINT_COUNT)
        positionX = radiusAu * np.cos(trueAnomalyRad)
        positionY = radiusAu * np.sin(trueAnomalyRad) * np.cos(plutoInclinationRad)
        positionZ = (
            radiusAu * np.sin(trueAnomalyRad) * np.sin(plutoInclinationRad) * np.sin(polarAngleRad)
        )
        return positionX, positionY, positionZ


if __name__ == '__main__':
    visualizer = PlutoOrbitVisualizer3D()
    visualizer.renderAllViews()
    print('All Pluto orbit visualizations completed!')
