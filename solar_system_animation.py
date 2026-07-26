"""Solar system animations — 2D is a fixed top-down view; 3D adds camera zoom."""

from __future__ import annotations

import os
from typing import Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from asteroid_motion import AnimatedAsteroidPopulation, AsteroidPopulationCounts, planetMeanAnomalyRad
from solsys_physics import AstronomicalConstants, FamousAsteroidCatalog, MoonCatalog, OrbitCalculator, PlanetCatalog

Dimension = Literal['2d', '3d']

FIGURE_SIZE_INCHES = (12, 12)
ANIMATION_FPS = 20
CAMERA_ELEVATION_DEG = 25
CAMERA_AZIMUTH_DEG = 120

# 2D: fixed inner-system top-down
AXIS_LIMIT_2D_AU = 6.5
ANIMATION_FRAMES_2D = 800
ANIMATION_SPEED_2D = 4.0
INNER_PLANET_NAMES = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter')

# 3D: staged zoom from Oort cloud to inner system
ANIMATION_FRAMES_3D = 500
ANIMATION_SPEED_3D = 3.0
MIN_CAMERA_DISTANCE_AU = 3.2
MAX_CAMERA_DISTANCE_AU = 100000.0
ZOOM_IN_FRAME_FRACTION = 0.80
ZOOM_STAGES = (
    (0.00, 100000.0),
    (0.14, 600.0),
    (0.32, 58.0),
    (0.44, 42.0),
    (0.54, 33.0),
    (0.62, 22.0),
    (0.70, 12.0),
    (0.78, 7.5),
    (0.88, 3.2),
    (1.00, 3.2),
)
KUIPER_VISIBLE_BELOW_AU = 48.0
INNER_BELT_VISIBLE_BELOW_AU = 8.5
OORT_VISIBLE_ABOVE_AU = 1500.0
VISIBILITY_FADE_SPAN_AU = 4.0

ASTEROID_RENDER_STYLES = {
    'light': {
        'beltColor': '#707070',
        'clusterColor': '#808080',
        'kuiperColor': '#757575',
        'oortColor': '#A0A0A0',
        'beltSize': 0.5,
        'clusterSize': 0.5,
        'kuiperSize': 0.375,
        'oortSize': 0.25,
        'beltAlpha': 0.55,
        'clusterAlpha': 0.5,
        'kuiperAlpha': 0.45,
        'oortAlpha': 0.25,
    },
    'dark': {
        'beltColor': '#D8D8D8',
        'clusterColor': '#CFCFCF',
        'kuiperColor': '#BDBDBD',
        'oortColor': '#9A9A9A',
        'beltSize': 0.875,
        'clusterSize': 0.75,
        'kuiperSize': 0.625,
        'oortSize': 0.375,
        'beltAlpha': 0.82,
        'clusterAlpha': 0.78,
        'kuiperAlpha': 0.72,
        'oortAlpha': 0.42,
    },
}


class SolarSystemAnimator:
    """Shared animator: 2D drops Z (top-down); 3D uses full XYZ plus zoom camera."""

    def __init__(self, dimension: Dimension = '2d', style: str = 'default'):
        if dimension not in ('2d', '3d'):
            raise ValueError(f"dimension must be '2d' or '3d', got {dimension!r}")
        self.dimension = dimension
        self.is3d = dimension == '3d'
        plt.style.use(style)
        self.renderStyle = ASTEROID_RENDER_STYLES['dark' if style == 'dark_background' else 'light']
        self.constants = AstronomicalConstants()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.famousAsteroidCatalog = FamousAsteroidCatalog()
        self.orbitCalculator = OrbitCalculator()

        if self.is3d:
            self.figure = plt.figure(figsize=FIGURE_SIZE_INCHES, layout='none')
            self.axes = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), projection='3d')
            self.asteroidPopulation = AnimatedAsteroidPopulation(
                self.constants,
                AsteroidPopulationCounts(
                    asteroidBelt=1000,
                    hildas=250,
                    trojansAndGreeks=100,
                    kuiperBelt=2000,
                    oortCloud=18000,
                ),
                includeKuiperAndOort=True,
                useSphericalShell3d=True,
            )
            self.animationFrames = ANIMATION_FRAMES_3D
            self.baseAnimationSpeed = ANIMATION_SPEED_3D
        else:
            self.figure, self.axes = plt.subplots(figsize=FIGURE_SIZE_INCHES)
            self.asteroidPopulation = AnimatedAsteroidPopulation(
                self.constants,
                AsteroidPopulationCounts(
                    asteroidBelt=800,
                    hildas=240,
                    trojansAndGreeks=150,
                ),
            )
            self.animationFrames = ANIMATION_FRAMES_2D
            self.baseAnimationSpeed = ANIMATION_SPEED_2D

    def _cameraDistanceAu(self, frame: int) -> float:
        if not self.is3d:
            return AXIS_LIMIT_2D_AU

        zoomFrameCount = max(int(self.animationFrames * ZOOM_IN_FRAME_FRACTION), 1)
        if frame >= zoomFrameCount:
            return MIN_CAMERA_DISTANCE_AU

        zoomProgress = frame / zoomFrameCount
        for stageIndex in range(len(ZOOM_STAGES) - 1):
            progressStart, distanceStartAu = ZOOM_STAGES[stageIndex]
            progressEnd, distanceEndAu = ZOOM_STAGES[stageIndex + 1]
            if zoomProgress <= progressEnd:
                segmentSpan = progressEnd - progressStart
                segmentProgress = (
                    (zoomProgress - progressStart) / segmentSpan if segmentSpan > 0 else 1.0
                )
                logDistance = np.log(distanceStartAu) + segmentProgress * (
                    np.log(distanceEndAu) - np.log(distanceStartAu)
                )
                return float(np.exp(logDistance))

        return MIN_CAMERA_DISTANCE_AU

    def _animationSpeedScale(self, cameraDistanceAu: float) -> float:
        if not self.is3d:
            return self.baseAnimationSpeed
        zoomFactor = (cameraDistanceAu / MAX_CAMERA_DISTANCE_AU) ** 0.15
        return max(0.75, zoomFactor) * self.baseAnimationSpeed

    def _jupiterMeanAnomalyRad(self, frame: int, animationSpeed: float) -> float:
        jupiter = self.planetCatalog.planets['Jupiter']
        return planetMeanAnomalyRad(jupiter.orbitalPeriodDays, frame, animationSpeed)

    def _visibilityAlpha(self, positionX, positionY, positionZ, cameraDistanceAu: float) -> np.ndarray:
        distances = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        return np.clip(0.8 * (1 - distances / (cameraDistanceAu * 1.5)), 0.05, 0.8)

    def _groupFadeAlpha(
        self,
        cameraDistanceAu: float,
        showBelowAu: float | None,
        showAboveAu: float | None,
        baseAlpha: float,
    ) -> float:
        if showBelowAu is not None and cameraDistanceAu > showBelowAu + VISIBILITY_FADE_SPAN_AU:
            return 0.0
        if showAboveAu is not None and cameraDistanceAu < showAboveAu - VISIBILITY_FADE_SPAN_AU:
            return 0.0

        fadeAlpha = baseAlpha
        if showBelowAu is not None and cameraDistanceAu > showBelowAu:
            fadeProgress = (cameraDistanceAu - showBelowAu) / VISIBILITY_FADE_SPAN_AU
            fadeAlpha *= max(0.0, 1.0 - fadeProgress)
        if showAboveAu is not None and cameraDistanceAu < showAboveAu:
            fadeProgress = (showAboveAu - cameraDistanceAu) / VISIBILITY_FADE_SPAN_AU
            fadeAlpha *= max(0.0, 1.0 - fadeProgress)
        return fadeAlpha

    def _scatter(
        self,
        positionX,
        positionY,
        positionZ,
        **scatterKwargs,
    ) -> None:
        if self.is3d:
            self.axes.scatter(positionX, positionY, positionZ, depthshade=True, **scatterKwargs)
        else:
            self.axes.scatter(positionX, positionY, **scatterKwargs)

    def _plot(self, positionX, positionY, positionZ, **plotKwargs) -> None:
        if self.is3d:
            self.axes.plot(positionX, positionY, positionZ, **plotKwargs)
        else:
            self.axes.plot(positionX, positionY, **plotKwargs)

    def _scatterGroup(
        self,
        positionX,
        positionY,
        positionZ,
        cameraDistanceAu: float,
        showBelowAu: float | None,
        showAboveAu: float | None,
        baseAlpha: float,
        **scatterKwargs,
    ) -> None:
        if self.is3d:
            fadeAlpha = self._groupFadeAlpha(cameraDistanceAu, showBelowAu, showAboveAu, baseAlpha)
            if fadeAlpha <= 0.0:
                return
            self._scatter(positionX, positionY, positionZ, alpha=fadeAlpha, **scatterKwargs)
            return

        # 2D: top-down of the same XYZ cloud; optional per-point alpha from Z
        alpha = scatterKwargs.pop('alpha', baseAlpha)
        self._scatter(positionX, positionY, positionZ, alpha=alpha, **scatterKwargs)

    def _planetNames(self):
        if self.is3d:
            return self.planetCatalog.planets.keys()
        return INNER_PLANET_NAMES

    def _planetPosition(self, planetName: str, frame: int, animationSpeed: float):
        planet = self.planetCatalog.planets[planetName]
        meanAnomalyRad = planetMeanAnomalyRad(planet.orbitalPeriodDays, frame, animationSpeed)
        return self.orbitCalculator.ellipticalPosition(
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.inclinationDeg,
            meanAnomalyRad,
        )

    def update(self, frame: int) -> None:
        self.axes.clear()
        cameraDistanceAu = self._cameraDistanceAu(frame)
        animationSpeed = self._animationSpeedScale(cameraDistanceAu)
        jupiterMeanAnomalyRad = self._jupiterMeanAnomalyRad(frame, animationSpeed)

        for planetName in self._planetNames():
            planet = self.planetCatalog.planets[planetName]
            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=200,
            )
            if self.is3d:
                orbitAlpha = self._visibilityAlpha(orbitX, orbitY, orbitZ, cameraDistanceAu)
                self._plot(
                    orbitX, orbitY, orbitZ, color='black', alpha=float(np.mean(orbitAlpha)) * 0.3
                )
            else:
                self._plot(orbitX, orbitY, orbitZ, color='black', alpha=0.15, linewidth=0.8)

        beltX, beltY, beltZ = self.asteroidPopulation.asteroidBeltPositions(
            frame, animationSpeed, ecliptic2d=not self.is3d
        )
        hildaX, hildaY, hildaZ = self.asteroidPopulation.hildaPositions(
            frame, jupiterMeanAnomalyRad, animationSpeed, ecliptic2d=not self.is3d
        )
        trojanX, trojanY, trojanZ = self.asteroidPopulation.trojanPositions(
            frame, jupiterMeanAnomalyRad
        )
        greekX, greekY, greekZ = self.asteroidPopulation.greekPositions(
            frame, jupiterMeanAnomalyRad
        )

        renderStyle = self.renderStyle
        if self.is3d:
            kuiperX, kuiperY, kuiperZ = self.asteroidPopulation.kuiperBeltPositions(
                frame, animationSpeed
            )
            oortX, oortY, oortZ = self.asteroidPopulation.oortCloudPositions(frame, animationSpeed)
            self._scatterGroup(
                beltX, beltY, beltZ, cameraDistanceAu,
                showBelowAu=INNER_BELT_VISIBLE_BELOW_AU, showAboveAu=None,
                baseAlpha=renderStyle['beltAlpha'],
                color=renderStyle['beltColor'], s=renderStyle['beltSize'],
            )
            self._scatterGroup(
                hildaX, hildaY, hildaZ, cameraDistanceAu,
                showBelowAu=INNER_BELT_VISIBLE_BELOW_AU, showAboveAu=None,
                baseAlpha=renderStyle['clusterAlpha'],
                color=renderStyle['clusterColor'], s=renderStyle['clusterSize'],
            )
            self._scatterGroup(
                trojanX, trojanY, trojanZ, cameraDistanceAu,
                showBelowAu=INNER_BELT_VISIBLE_BELOW_AU, showAboveAu=None,
                baseAlpha=renderStyle['clusterAlpha'],
                color=renderStyle['clusterColor'], s=renderStyle['clusterSize'],
            )
            self._scatterGroup(
                greekX, greekY, greekZ, cameraDistanceAu,
                showBelowAu=INNER_BELT_VISIBLE_BELOW_AU, showAboveAu=None,
                baseAlpha=renderStyle['clusterAlpha'],
                color=renderStyle['clusterColor'], s=renderStyle['clusterSize'],
            )
            self._scatterGroup(
                kuiperX, kuiperY, kuiperZ, cameraDistanceAu,
                showBelowAu=KUIPER_VISIBLE_BELOW_AU, showAboveAu=None,
                baseAlpha=renderStyle['kuiperAlpha'],
                color=renderStyle['kuiperColor'], s=renderStyle['kuiperSize'],
            )
            self._scatterGroup(
                oortX, oortY, oortZ, cameraDistanceAu,
                showBelowAu=None, showAboveAu=OORT_VISIBLE_ABOVE_AU,
                baseAlpha=renderStyle['oortAlpha'],
                color=renderStyle['oortColor'], s=renderStyle['oortSize'],
            )
        else:
            beltOpacity = 0.3 + 0.2 * (beltZ / (np.max(np.abs(beltZ)) + 1e-6))
            beltOpacity = np.clip(beltOpacity, 0.1, 0.8)
            self._scatter(beltX, beltY, beltZ, color='gray', s=2, alpha=beltOpacity)
            self._scatter(hildaX, hildaY, hildaZ, color='#888888', s=2, alpha=0.35)
            self._scatter(trojanX, trojanY, trojanZ, color='#666666', s=2, alpha=0.35)
            self._scatter(greekX, greekY, greekZ, color='#666666', s=2, alpha=0.35)

        if self.is3d:
            self._scatter([0], [0], [0], color='yellow', s=80)
        else:
            self._scatter([0], [0], [0], color='yellow', s=120, zorder=5)

        for planetName in self._planetNames():
            planet = self.planetCatalog.planets[planetName]
            positionX, positionY, positionZ = self._planetPosition(
                planetName, frame, animationSpeed
            )
            markerSize = int(
                (8 + planet.diameterKm / 3000) if self.is3d else (10 + planet.diameterKm / 2500)
            )
            if self.is3d:
                planetAlpha = float(
                    self._visibilityAlpha(positionX, positionY, positionZ, cameraDistanceAu)
                )
                self._scatter(
                    positionX, positionY, positionZ,
                    color=planet.color, s=markerSize, alpha=planetAlpha,
                )
            else:
                self._scatter(
                    positionX, positionY, positionZ,
                    color=planet.color, s=markerSize, zorder=6,
                )
                self.axes.text(
                    float(positionX) + 0.15, float(positionY) + 0.15, planetName, fontsize=8
                )

            self._drawMoonsForPlanet(
                planet.name, positionX, positionY, positionZ, frame, animationSpeed, cameraDistanceAu
            )

        self._drawFamousAsteroids(frame, animationSpeed, cameraDistanceAu)
        self._applyAxes(cameraDistanceAu)

    def _drawMoonsForPlanet(
        self,
        planetName: str,
        positionX,
        positionY,
        positionZ,
        frame: int,
        animationSpeed: float,
        cameraDistanceAu: float,
    ) -> None:
        if self.is3d:
            moonDisplayScale = self.moonCatalog.displayScaleForCameraAu(cameraDistanceAu)
        else:
            moonDisplayScale = self.moonCatalog.displayScaleForAxisSpanAu(2 * AXIS_LIMIT_2D_AU)
        if moonDisplayScale <= 0.0:
            return

        for moon in self.moonCatalog.forPlanet(planetName):
            ringAzimuthRad = np.linspace(0, 2 * np.pi, 48)
            orbitRadiusAu = self.moonCatalog.displayOrbitRadiusAu(moon, moonDisplayScale)
            ringZ = np.full(48, float(np.asarray(positionZ).reshape(-1)[0]))
            self._plot(
                float(np.asarray(positionX).reshape(-1)[0]) + orbitRadiusAu * np.cos(ringAzimuthRad),
                float(np.asarray(positionY).reshape(-1)[0]) + orbitRadiusAu * np.sin(ringAzimuthRad),
                ringZ,
                color=moon.color,
                alpha=0.2,
                linewidth=0.5,
            )
            moonMeanAnomalyRad = planetMeanAnomalyRad(
                moon.orbitalPeriodDays, frame, animationSpeed
            )
            moonX, moonY, moonZ = self.moonCatalog.heliocentricPosition(
                positionX, positionY, positionZ, moon, moonMeanAnomalyRad, moonDisplayScale
            )
            if self.is3d:
                moonAlpha = float(self._visibilityAlpha(moonX, moonY, moonZ, cameraDistanceAu))
                self._scatter(
                    moonX, moonY, moonZ,
                    color=moon.color,
                    s=self.moonCatalog.markerSize3d(moon, 1200),
                    alpha=moonAlpha,
                )
            else:
                self._scatter(
                    moonX, moonY, moonZ,
                    color=moon.color,
                    s=self.moonCatalog.markerSize2d(moon, 1200),
                    zorder=8,
                )

    def _drawFamousAsteroids(
        self, frame: int, animationSpeed: float, cameraDistanceAu: float
    ) -> None:
        if self.is3d:
            visibilityCheck = lambda category: self.famousAsteroidCatalog.visibleForCameraAu(
                cameraDistanceAu, category
            )
        else:
            axisSpanAu = 2 * AXIS_LIMIT_2D_AU
            visibilityCheck = lambda category: self.famousAsteroidCatalog.visibleForAxisSpanAu(
                axisSpanAu, category
            )

        for asteroid in self.famousAsteroidCatalog.asteroids.values():
            if not visibilityCheck(asteroid.category):
                continue

            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                asteroid.semiMajorAxisAu,
                asteroid.eccentricity,
                asteroid.inclinationDeg,
                numPoints=120,
            )
            if self.is3d:
                orbitAlpha = self._visibilityAlpha(orbitX, orbitY, orbitZ, cameraDistanceAu)
                self._plot(
                    orbitX, orbitY, orbitZ,
                    color=asteroid.color, alpha=float(np.mean(orbitAlpha)) * 0.4,
                )
            else:
                self._plot(
                    orbitX, orbitY, orbitZ,
                    color=asteroid.color, alpha=0.35, linewidth=0.6, zorder=4,
                )

            meanAnomalyRad = planetMeanAnomalyRad(
                asteroid.orbitalPeriodDays, frame, animationSpeed
            )
            positionX, positionY, positionZ = self.famousAsteroidCatalog.positionAtMeanAnomaly(
                asteroid, meanAnomalyRad, self.orbitCalculator
            )
            if self.is3d:
                asteroidAlpha = float(
                    self._visibilityAlpha(positionX, positionY, positionZ, cameraDistanceAu)
                )
                self._scatter(
                    positionX, positionY, positionZ,
                    color=asteroid.color,
                    s=self.famousAsteroidCatalog.markerSize3d(asteroid, 500),
                    alpha=asteroidAlpha,
                )
            else:
                self._scatter(
                    positionX, positionY, positionZ,
                    color=asteroid.color,
                    s=self.famousAsteroidCatalog.markerSize2d(asteroid, 120),
                    zorder=7,
                )
                self.axes.text(
                    float(positionX) + 0.1,
                    float(positionY) + 0.1,
                    asteroid.name,
                    fontsize=7,
                    color=asteroid.color,
                    zorder=8,
                )

    def _applyAxes(self, cameraDistanceAu: float) -> None:
        if self.is3d:
            self.axes.set_xlim(-cameraDistanceAu, cameraDistanceAu)
            self.axes.set_ylim(-cameraDistanceAu, cameraDistanceAu)
            self.axes.set_zlim(-cameraDistanceAu, cameraDistanceAu)
            self.axes.view_init(elev=CAMERA_ELEVATION_DEG, azim=CAMERA_AZIMUTH_DEG)
            self.axes.set_axis_off()
            self.axes.set_title('Solar System Animation', pad=8, y=0.98)
            self.axes.set_position((0.0, 0.0, 1.0, 1.0))
            self.axes.set_box_aspect((1, 1, 1), zoom=1.0)
        else:
            self.axes.set_aspect('equal')
            self.axes.set_xlim(-AXIS_LIMIT_2D_AU, AXIS_LIMIT_2D_AU)
            self.axes.set_ylim(-AXIS_LIMIT_2D_AU, AXIS_LIMIT_2D_AU)
            self.axes.axis('off')
            self.axes.set_title('Inner Solar System', pad=20)

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.animationFrames,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        saveKwargs = {}
        if self.is3d:
            self.figure.set_size_inches(*FIGURE_SIZE_INCHES)
            self.figure.set_dpi(100)
            saveKwargs['savefig_kwargs'] = {
                'pad_inches': 0,
                'facecolor': self.figure.get_facecolor(),
            }
        animation.save(outputPath, writer=PillowWriter(fps=ANIMATION_FPS), **saveKwargs)
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderAllAnimations(dimension: Optional[Dimension] = None) -> None:
    dimensions: tuple[Dimension, ...] = (dimension,) if dimension else ('2d', '3d')
    for selectedDimension in dimensions:
        outputDirectory = f'output/animate/{selectedDimension}'
        for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
            animator = SolarSystemAnimator(dimension=selectedDimension, style=styleName)
            if selectedDimension == '2d':
                outputPath = f'{outputDirectory}/inner_solar_system_{themeName}.gif'
            else:
                outputPath = f'{outputDirectory}/solar_system_{themeName}.gif'
            print(f'Rendering {outputPath}...')
            animator.saveGif(outputPath)
    print('Animations completed!')


if __name__ == '__main__':
    import sys

    if '--2d' in sys.argv:
        renderAllAnimations('2d')
    elif '--3d' in sys.argv:
        renderAllAnimations('3d')
    else:
        renderAllAnimations()
