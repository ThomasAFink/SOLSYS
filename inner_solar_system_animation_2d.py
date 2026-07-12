"""2D inner solar system animation using corrected orbital physics."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from solsys_animation import AnimatedAsteroidPopulation, AsteroidPopulationCounts, planetMeanAnomalyRad
from solsys_core import AstronomicalConstants, FamousAsteroidCatalog, MoonCatalog, OrbitCalculator, PlanetCatalog

FIGURE_SIZE_INCHES = (12, 12)
AXIS_LIMIT_AU = 6.5
ANIMATION_FRAMES = 800
ANIMATION_FPS = 20
ANIMATION_SPEED = 4.0
INNER_PLANET_NAMES = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter')
OUTPUT_DIRECTORY = 'output/animate/2d'


class InnerSolarSystemAnimator2D:
    def __init__(self, style: str = 'default'):
        plt.style.use(style)
        self.figure, self.axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
        self.constants = AstronomicalConstants()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.famousAsteroidCatalog = FamousAsteroidCatalog()
        self.orbitCalculator = OrbitCalculator()
        self.asteroidPopulation = AnimatedAsteroidPopulation(
            self.constants,
            AsteroidPopulationCounts(
                asteroidBelt=800,
                hildas=240,
                trojansAndGreeks=150,
            ),
        )

    def _planetPosition(self, planetName: str, frame: int) -> tuple[float, float]:
        planet = self.planetCatalog.planets[planetName]
        meanAnomalyRad = planetMeanAnomalyRad(planet.orbitalPeriodDays, frame, ANIMATION_SPEED)
        positionX, positionY, _ = self.orbitCalculator.ellipticalPosition(
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.inclinationDeg,
            meanAnomalyRad,
        )
        return float(positionX), float(positionY)

    def _drawMoons(self, frame: int) -> None:
        moonDisplayScale = self.moonCatalog.displayScaleForAxisSpanAu(2 * AXIS_LIMIT_AU)
        if moonDisplayScale <= 0.0:
            return

        for planetName in INNER_PLANET_NAMES:
            positionX, positionY = self._planetPosition(planetName, frame)
            for moon in self.moonCatalog.forPlanet(planetName):
                ringX, ringY = self.moonCatalog.moonOrbitRing2d(
                    moon, positionX, positionY, moonDisplayScale
                )
                self.axes.plot(ringX, ringY, color=moon.color, alpha=0.2, linewidth=0.5, zorder=6)
                moonMeanAnomalyRad = planetMeanAnomalyRad(
                    moon.orbitalPeriodDays, frame, ANIMATION_SPEED
                )
                moonX, moonY, _ = self.moonCatalog.heliocentricPosition(
                    positionX, positionY, 0.0, moon, moonMeanAnomalyRad, moonDisplayScale
                )
                moonMarkerSize = self.moonCatalog.markerSize2d(moon, 1200)
                self.axes.scatter(
                    float(moonX), float(moonY), color=moon.color, s=moonMarkerSize, zorder=8
                )

    def _drawFamousAsteroids(self, frame: int) -> None:
        axisSpanAu = 2 * AXIS_LIMIT_AU
        for asteroid in self.famousAsteroidCatalog.asteroids.values():
            if not self.famousAsteroidCatalog.visibleForAxisSpanAu(axisSpanAu, asteroid.category):
                continue

            orbitX, orbitY = self.orbitCalculator.ellipticalOrbit2d(
                asteroid.semiMajorAxisAu,
                asteroid.eccentricity,
                asteroid.inclinationDeg,
                numPoints=120,
            )
            self.axes.plot(orbitX, orbitY, color=asteroid.color, alpha=0.35, linewidth=0.6, zorder=4)

            meanAnomalyRad = planetMeanAnomalyRad(
                asteroid.orbitalPeriodDays, frame, ANIMATION_SPEED
            )
            positionX, positionY, _ = self.famousAsteroidCatalog.positionAtMeanAnomaly(
                asteroid, meanAnomalyRad, self.orbitCalculator
            )
            markerSize = self.famousAsteroidCatalog.markerSize2d(asteroid, 120)
            self.axes.scatter(
                float(positionX), float(positionY), color=asteroid.color, s=markerSize, zorder=7
            )
            self.axes.text(
                float(positionX) + 0.1,
                float(positionY) + 0.1,
                asteroid.name,
                fontsize=7,
                color=asteroid.color,
                zorder=8,
            )

    def _jupiterMeanAnomalyRad(self, frame: int) -> float:
        jupiter = self.planetCatalog.planets['Jupiter']
        return planetMeanAnomalyRad(jupiter.orbitalPeriodDays, frame, ANIMATION_SPEED)

    def update(self, frame: int) -> None:
        self.axes.clear()

        for planetName in INNER_PLANET_NAMES:
            planet = self.planetCatalog.planets[planetName]
            orbitX, orbitY = self.orbitCalculator.ellipticalOrbit2d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=200,
            )
            self.axes.plot(orbitX, orbitY, color='black', alpha=0.15, linewidth=0.8)

        jupiterMeanAnomalyRad = self._jupiterMeanAnomalyRad(frame)

        beltX, beltY, beltZ = self.asteroidPopulation.asteroidBeltPositions(
            frame, ANIMATION_SPEED, ecliptic2d=True
        )
        hildaX, hildaY, _ = self.asteroidPopulation.hildaPositions(
            frame, jupiterMeanAnomalyRad, ANIMATION_SPEED, ecliptic2d=True
        )
        trojanX, trojanY, _ = self.asteroidPopulation.trojanPositions(frame, jupiterMeanAnomalyRad)
        greekX, greekY, _ = self.asteroidPopulation.greekPositions(frame, jupiterMeanAnomalyRad)

        beltOpacity = 0.3 + 0.2 * (beltZ / (np.max(np.abs(beltZ)) + 1e-6))
        beltOpacity = np.clip(beltOpacity, 0.1, 0.8)

        self.axes.scatter(beltX, beltY, color='gray', s=2, alpha=beltOpacity)
        self.axes.scatter(hildaX, hildaY, color='#888888', s=2, alpha=0.35)
        self.axes.scatter(trojanX, trojanY, color='#666666', s=2, alpha=0.35)
        self.axes.scatter(greekX, greekY, color='#666666', s=2, alpha=0.35)

        self.axes.scatter(0, 0, color='yellow', s=120, zorder=5)
        for planetName in INNER_PLANET_NAMES:
            planet = self.planetCatalog.planets[planetName]
            positionX, positionY = self._planetPosition(planetName, frame)
            markerSize = int(10 + planet.diameterKm / 2500)
            self.axes.scatter(
                positionX, positionY, color=planet.color, s=markerSize, zorder=6
            )
            self.axes.text(positionX + 0.15, positionY + 0.15, planetName, fontsize=8)

        self._drawMoons(frame)
        self._drawFamousAsteroids(frame)

        self.axes.set_aspect('equal')
        self.axes.set_xlim(-AXIS_LIMIT_AU, AXIS_LIMIT_AU)
        self.axes.set_ylim(-AXIS_LIMIT_AU, AXIS_LIMIT_AU)
        self.axes.axis('off')
        self.axes.set_title('Inner Solar System', pad=20)

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=ANIMATION_FRAMES,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        animation.save(outputPath, writer=PillowWriter(fps=ANIMATION_FPS))
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderAllAnimations() -> None:
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        animator = InnerSolarSystemAnimator2D(style=styleName)
        outputPath = f'{OUTPUT_DIRECTORY}/inner_solar_system_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator.saveGif(outputPath)


if __name__ == '__main__':
    renderAllAnimations()
    print('2D animations completed!')
