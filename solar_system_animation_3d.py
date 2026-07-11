"""3D solar system animation with camera zoom using corrected orbital physics."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from solsys_animation import AnimatedAsteroidPopulation, AsteroidPopulationCounts, planetMeanAnomalyRad
from solsys_core import AstronomicalConstants, OrbitCalculator, PlanetCatalog

FIGURE_SIZE_INCHES = (12, 12)
ANIMATION_FRAMES = 500
ANIMATION_FPS = 20
ANIMATION_SPEED = 3.0
MIN_CAMERA_DISTANCE_AU = 3.2
MAX_CAMERA_DISTANCE_AU = 100000.0
ZOOM_IN_FRAME_FRACTION = 0.80
# (zoom progress 0-1, camera limit in AU) — log-interpolated between waypoints
ZOOM_STAGES = (
    (0.00, 100000.0),
    (0.14, 600.0),
    (0.32, 58.0),   # Kuiper belt / Pluto
    (0.44, 42.0),   # Pluto orbit
    (0.54, 33.0),   # Neptune
    (0.62, 22.0),   # Uranus
    (0.70, 12.0),   # Saturn
    (0.78, 7.5),    # Jupiter
    (0.88, 3.2),    # inner solar system
    (1.00, 3.2),
)
# Only draw each population once the camera has zoomed past its scale
KUIPER_VISIBLE_BELOW_AU = 48.0
INNER_BELT_VISIBLE_BELOW_AU = 8.5
OORT_VISIBLE_ABOVE_AU = 1500.0
VISIBILITY_FADE_SPAN_AU = 4.0
CAMERA_ELEVATION_DEG = 25
CAMERA_AZIMUTH_DEG = 120
OUTPUT_DIRECTORY = 'output/animate/3d'

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


class SolarSystemAnimator3D:
    def __init__(self, style: str = 'default'):
        plt.style.use(style)
        self.figure = plt.figure(figsize=FIGURE_SIZE_INCHES, layout='none')
        self.axes = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), projection='3d')
        self.renderStyle = ASTEROID_RENDER_STYLES['dark' if style == 'dark_background' else 'light']
        self.constants = AstronomicalConstants()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.orbitCalculator = OrbitCalculator()
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

    def _cameraDistanceAu(self, frame: int) -> float:
        zoomFrameCount = max(int(ANIMATION_FRAMES * ZOOM_IN_FRAME_FRACTION), 1)
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
        zoomFactor = (cameraDistanceAu / MAX_CAMERA_DISTANCE_AU) ** 0.15
        return max(0.75, zoomFactor) * ANIMATION_SPEED

    def _jupiterMeanAnomalyRad(self, frame: int, animationSpeed: float) -> float:
        jupiter = self.planetCatalog.planets['Jupiter']
        return planetMeanAnomalyRad(jupiter.orbitalPeriodDays, frame, animationSpeed)

    def _visibilityAlpha(self, positionX, positionY, positionZ, cameraDistanceAu: float) -> np.ndarray:
        distances = np.sqrt(positionX ** 2 + positionY ** 2 + positionZ ** 2)
        return np.clip(0.8 * (1 - distances / (cameraDistanceAu * 1.5)), 0.05, 0.8)

    def _groupFadeAlpha(
        self, cameraDistanceAu: float, showBelowAu: float | None, showAboveAu: float | None, baseAlpha: float
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
        fadeAlpha = self._groupFadeAlpha(cameraDistanceAu, showBelowAu, showAboveAu, baseAlpha)
        if fadeAlpha <= 0.0:
            return
        self.axes.scatter(
            positionX, positionY, positionZ,
            alpha=fadeAlpha,
            depthshade=True,
            **scatterKwargs,
        )

    def update(self, frame: int) -> None:
        self.axes.clear()
        cameraDistanceAu = self._cameraDistanceAu(frame)
        animationSpeed = self._animationSpeedScale(cameraDistanceAu)
        jupiterMeanAnomalyRad = self._jupiterMeanAnomalyRad(frame, animationSpeed)

        for planet in self.planetCatalog.planets.values():
            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                numPoints=200,
            )
            orbitAlpha = self._visibilityAlpha(orbitX, orbitY, orbitZ, cameraDistanceAu)
            self.axes.plot(orbitX, orbitY, orbitZ, color='black', alpha=float(np.mean(orbitAlpha)) * 0.3)

        beltX, beltY, beltZ = self.asteroidPopulation.asteroidBeltPositions(frame, animationSpeed)
        hildaX, hildaY, hildaZ = self.asteroidPopulation.hildaPositions(
            frame, jupiterMeanAnomalyRad, animationSpeed
        )
        trojanX, trojanY, trojanZ = self.asteroidPopulation.trojanPositions(frame, jupiterMeanAnomalyRad)
        greekX, greekY, greekZ = self.asteroidPopulation.greekPositions(frame, jupiterMeanAnomalyRad)
        kuiperX, kuiperY, kuiperZ = self.asteroidPopulation.kuiperBeltPositions(frame, animationSpeed)
        oortX, oortY, oortZ = self.asteroidPopulation.oortCloudPositions(frame, animationSpeed)

        renderStyle = self.renderStyle
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

        self.axes.scatter([0], [0], [0], color='yellow', s=80)
        for planet in self.planetCatalog.planets.values():
            meanAnomalyRad = planetMeanAnomalyRad(
                planet.orbitalPeriodDays, frame, animationSpeed
            )
            positionX, positionY, positionZ = self.orbitCalculator.ellipticalPosition(
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.inclinationDeg,
                meanAnomalyRad,
            )
            markerSize = int(8 + planet.diameterKm / 3000)
            planetAlpha = float(self._visibilityAlpha(positionX, positionY, positionZ, cameraDistanceAu))
            self.axes.scatter(
                positionX,
                positionY,
                positionZ,
                color=planet.color,
                s=markerSize,
                alpha=planetAlpha,
            )

        self.axes.set_xlim(-cameraDistanceAu, cameraDistanceAu)
        self.axes.set_ylim(-cameraDistanceAu, cameraDistanceAu)
        self.axes.set_zlim(-cameraDistanceAu, cameraDistanceAu)
        self.axes.view_init(elev=CAMERA_ELEVATION_DEG, azim=CAMERA_AZIMUTH_DEG)
        self.axes.set_axis_off()
        self.axes.set_title('Solar System Animation', pad=8, y=0.98)
        self._applyFigureLayout()

    def _applyFigureLayout(self) -> None:
        """mplot3d draws into a square viewport — match a square figure so it fills the GIF."""
        self.axes.set_position((0.0, 0.0, 1.0, 1.0))
        self.axes.set_box_aspect((1, 1, 1), zoom=1.0)

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath), exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=ANIMATION_FRAMES,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        self.figure.set_size_inches(*FIGURE_SIZE_INCHES)
        self.figure.set_dpi(100)
        animation.save(
            outputPath,
            writer=PillowWriter(fps=ANIMATION_FPS),
            savefig_kwargs={
                'pad_inches': 0,
                'facecolor': self.figure.get_facecolor(),
            },
        )
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderAllAnimations() -> None:
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        animator = SolarSystemAnimator3D(style=styleName)
        outputPath = f'{OUTPUT_DIRECTORY}/solar_system_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator.saveGif(outputPath)


if __name__ == '__main__':
    renderAllAnimations()
    print('3D animations completed!')
