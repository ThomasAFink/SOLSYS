"""Catalog-driven top-down exoplanet system animations (single host star)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from solsys.motion.mean_anomaly import meanAnomalyAtFrame
from solsys.physics import OrbitCalculator
from solsys.physics.catalogs.system_catalog import StarSystem, SystemCatalog, SystemPlanet

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 600
DEFAULT_ANIMATION_SPEED = 0.25
DEFAULT_MIN_AXIS_LIMIT_AU = 0.22
DEFAULT_STAR_COLOR = '#E07060'
SHORT_PERIOD_DAYS_THRESHOLD = 20.0


def trueAnomalyFromMean(meanAnomalyRad: float, eccentricity: float) -> float:
    meanAnomalyRad = float(meanAnomalyRad % (2 * np.pi))
    eccentricAnomaly = meanAnomalyRad
    for _ in range(8):
        eccentricAnomaly = meanAnomalyRad + eccentricity * np.sin(eccentricAnomaly)
    cosE = np.cos(eccentricAnomaly)
    sinE = np.sin(eccentricAnomaly)
    return float(
        np.arctan2(
            np.sqrt(1 - eccentricity**2) * sinE,
            cosE - eccentricity,
        )
    )


def bodyPositionInOrbitalPlane(
    orbitCalculator: OrbitCalculator,
    semiMajorAxisAu: float,
    eccentricity: float,
    periodDays: float,
    argumentPeriapsisDeg: float,
    meanAnomalyDegEpoch: float,
    frame: int,
    speedScale: float,
) -> tuple[float, float]:
    meanAnomalyRad = float(
        meanAnomalyAtFrame(
            np.radians(meanAnomalyDegEpoch),
            periodDays,
            frame,
            speedScale,
        )
    )
    trueAnomalyRad = trueAnomalyFromMean(meanAnomalyRad, eccentricity)
    trueAnomalyRad = trueAnomalyRad + np.radians(argumentPeriapsisDeg)
    positionX, positionY, _ = orbitCalculator.ellipticalPosition(
        semiMajorAxisAu,
        eccentricity,
        0.0,  # orbital-plane / face-on schematic
        trueAnomalyRad,
        ascendingNodeDeg=0.0,
    )
    return float(np.asarray(positionX)), float(np.asarray(positionY))


def orbitPathInOrbitalPlane(
    orbitCalculator: OrbitCalculator,
    semiMajorAxisAu: float,
    eccentricity: float,
    argumentPeriapsisDeg: float,
) -> tuple[np.ndarray, np.ndarray]:
    trueAnomaly = np.linspace(0, 2 * np.pi, 360) + np.radians(argumentPeriapsisDeg)
    positionX, positionY, _ = orbitCalculator.ellipticalPosition(
        semiMajorAxisAu,
        eccentricity,
        0.0,
        trueAnomaly,
        ascendingNodeDeg=0.0,
    )
    return np.asarray(positionX), np.asarray(positionY)


@dataclass(frozen=True)
class ExoplanetSystemSceneConfig:
    """View knobs for a single-host planetary system schematic."""

    hostStarUuid: str | None = None
    hostStarRole: str | None = None
    title: str | None = None
    starLabel: str | None = None
    starColor: str = DEFAULT_STAR_COLOR
    minAxisLimitAu: float = DEFAULT_MIN_AXIS_LIMIT_AU
    axisLimitAu: float | None = None
    planetIdAxisLimits: dict[str, float] = field(default_factory=dict)
    animationSpeed: float = DEFAULT_ANIMATION_SPEED
    shortPeriodSpeedBoost: float = 1.0
    shortPeriodDaysThreshold: float = SHORT_PERIOD_DAYS_THRESHOLD
    footerNote: str = ''
    includeDisputedPlanets: bool = False


def resolveHostStarUuid(system: StarSystem, config: ExoplanetSystemSceneConfig) -> str:
    if config.hostStarUuid:
        return config.hostStarUuid
    if config.hostStarRole:
        for orbit in system.stellarOrbits:
            if orbit.role == config.hostStarRole:
                return orbit.starUuid
        raise KeyError(f'No stellar orbit with role={config.hostStarRole!r} in {system.systemId!r}')

    hostUuids = {planet.hostStarUuid for planet in system.planets}
    if len(hostUuids) == 1:
        return next(iter(hostUuids))
    if len(system.stars) == 1:
        return system.stars[0].starUuid
    raise ValueError(
        f'Ambiguous host star for {system.systemId!r}; set hostStarUuid or hostStarRole'
    )


def selectPlanets(
    system: StarSystem,
    hostStarUuid: str,
    includeDisputedPlanets: bool,
) -> tuple[SystemPlanet, ...]:
    planets = [
        planet
        for planet in system.planetsForHost(hostStarUuid)
        if includeDisputedPlanets or planet.confidence == 'confirmed'
    ]
    if not planets:
        raise ValueError(f'No planets to plot for host {hostStarUuid!r} in {system.systemId!r}')
    return tuple(planets)


def resolveAxisLimitAu(
    planets: tuple[SystemPlanet, ...],
    config: ExoplanetSystemSceneConfig,
) -> float:
    if config.axisLimitAu is not None:
        return float(config.axisLimitAu)
    maxA = max(planet.semiMajorAxisAu for planet in planets)
    axisLimitAu = max(config.minAxisLimitAu, maxA * 1.35)
    for planet in planets:
        extra = config.planetIdAxisLimits.get(planet.planetId)
        if extra is not None:
            axisLimitAu = max(axisLimitAu, extra)
    return float(axisLimitAu)


def resolveStarLabel(
    system: StarSystem, hostStarUuid: str, config: ExoplanetSystemSceneConfig
) -> str:
    if config.starLabel:
        return config.starLabel
    member = system.starByUuid(hostStarUuid)
    if member is None:
        return 'Host'
    name = member.starName.replace('\xa0', ' ')
    return name.split('(')[0].strip() or 'Host'


class ExoplanetSystemAnimator:
    """Top-down Keplerian scene for planets around one host star."""

    def __init__(
        self,
        system: StarSystem,
        config: ExoplanetSystemSceneConfig | None = None,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
    ):
        self.system = system
        self.config = config or ExoplanetSystemSceneConfig()
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.orbitCalculator = OrbitCalculator()
        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'

        self.hostStarUuid = resolveHostStarUuid(system, self.config)
        self.planets = selectPlanets(system, self.hostStarUuid, self.config.includeDisputedPlanets)
        self.axisLimitAu = resolveAxisLimitAu(self.planets, self.config)
        self.starLabel = resolveStarLabel(system, self.hostStarUuid, self.config)
        self.title = self.config.title or f'{system.displayName} planets'
        self.animationFrames = ANIMATION_FRAMES
        self.animationSpeed = self.config.animationSpeed

        self.figure, self.axes = plt.subplots(figsize=figureSizeInches, dpi=dpi)

    def update(self, frame: int):
        self.axes.clear()
        self.axes.set_aspect('equal')
        self.axes.axis('off')
        self.axes.set_xlim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_ylim(-self.axisLimitAu, self.axisLimitAu)
        self.axes.set_title(self.title, color=self.labelColor, pad=16)

        self.axes.scatter([0], [0], s=280, color=self.config.starColor, zorder=5)
        self.axes.text(
            self.axisLimitAu * 0.035,
            self.axisLimitAu * 0.045,
            self.starLabel,
            color=self.labelColor,
            fontsize=9,
        )

        for planet in self.planets:
            alpha = 0.35 if planet.confidence != 'confirmed' else 0.7
            pathX, pathY = orbitPathInOrbitalPlane(
                self.orbitCalculator,
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.argumentPeriapsisDeg,
            )
            self.axes.plot(pathX, pathY, color=planet.color, linewidth=0.8, alpha=alpha)
            speedScale = self.animationSpeed
            if planet.orbitalPeriodDays < self.config.shortPeriodDaysThreshold:
                speedScale *= self.config.shortPeriodSpeedBoost
            positionX, positionY = bodyPositionInOrbitalPlane(
                self.orbitCalculator,
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.orbitalPeriodDays,
                planet.argumentPeriapsisDeg,
                0.0,
                frame,
                speedScale,
            )
            markerSize = max(12, planet.diameterKm / 800)
            self.axes.scatter(
                [positionX],
                [positionY],
                s=markerSize,
                color=planet.color,
                alpha=1.0 if planet.confidence == 'confirmed' else 0.45,
                zorder=6,
            )
            self.axes.text(
                positionX + self.axisLimitAu * 0.03,
                positionY + self.axisLimitAu * 0.03,
                planet.name,
                color=self.labelColor,
                fontsize=8,
                alpha=alpha + 0.2,
            )

        if self.config.footerNote:
            self.axes.text(
                -self.axisLimitAu * 0.95,
                -self.axisLimitAu * 0.92,
                self.config.footerNote,
                color=self.labelColor,
                fontsize=7,
                alpha=0.65,
            )
        return []

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath) or '.', exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.animationFrames,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        self.figure.set_size_inches(*self.figureSizeInches)
        self.figure.set_dpi(self.dpi)
        animation.save(outputPath, writer=PillowWriter(fps=ANIMATION_FPS))
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderExoplanetSystemAnimations(
    systemId: str,
    filenameStem: str,
    outputDirectory: str,
    config: ExoplanetSystemSceneConfig | None = None,
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    catalog = SystemCatalog(starsCsvPath=starsCsvPath)
    system = catalog.load(systemId)
    sceneConfig = config or ExoplanetSystemSceneConfig()
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{outputDirectory}/{filenameStem}_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = ExoplanetSystemAnimator(
            system,
            config=sceneConfig,
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
        )
        animator.saveGif(outputPath)
    print(f'{system.displayName} exoplanet animations completed!')
