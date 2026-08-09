"""Sol → TRAPPIST-1 cinematic — scale odyssey ending on the seven-planet chain.

Reuses Sol opening / pullback / Blender billboard machinery from the Sol→α Cen
cinematic; destination is a single-host system at true Sol XYZ (~40.7 ly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from solsys.motion import AnimatedAsteroidPopulation, AsteroidPopulationCounts
from solsys.physics import (
    AstronomicalConstants,
    FamousAsteroidCatalog,
    OrbitCalculator,
    PlanetCatalog,
)
from solsys.physics.catalogs.moon_catalog import MoonCatalog
from solsys.physics.catalogs.system_catalog import StarSystem, SystemCatalog, SystemPlanet

from animate.animation_styles import ASTEROID_RENDER_STYLES
from animate.blender_body_sprites import BlenderBodySpriteAtlas
from animate.scenes.exoplanet_system import bodyPositionInOrbitalPlane, orbitPathInOrbitalPlane
from animate.scenes.sol_centauri_cinematic import (
    AB_CRUISE_END,
    AB_FOCUS_ARRIVE,
    ANIMATION_FRAMES,
    BLENDER_PLANET_BODY_SCALE,
    CAMERA_ELEVATION_DEG,
    DEFAULT_DPI,
    DEFAULT_FIGURE_SIZE_INCHES,
    EARTH_CLOSE_ELEVATION_DEG,
    EARTH_OPEN_ELEVATION_DEG,
    PULLBACK_END,
    SOL_EARTH_CLOSE_HALF_AU,
    SOL_EARTH_HALF_AU,
    SOL_ELEVATION_DEG,
    SOL_HALF_WIDTH_AU,
    SOL_HOLD_END,
    SOL_NEAR_SUN_HALF_AU,
    SOL_PLANET_NAMES,
    SolCentauriCinematicAnimator,
    lerpAngleDeg,
    logLerp,
    parseApparentMagnitude,
    segmentProgress,
    smootherstep,
    spectralClassColor,
    stagedLogDive,
    timelineProgress,
)
from animate.scenes.trappist_1 import TRAPPIST_1_STAR_COLOR

# Wider than α Cen (~4 ly): TRAPPIST sits at ~40.7 ly.
FIELD_STARS_MAX_LY = 45.0
START_HALF_WIDTH_LY = 45.0
ANIMATION_SPEED_TRAPPIST_PLANETS = 0.20

# Arrival: host readable; planets still tiny. Finale matches exoplanet scene (~0.09 AU).
TRAPPIST_ARRIVE_HALF_AU = 2.0
TRAPPIST_WIDE_HALF_AU = 0.15
TRAPPIST_INNER_HALF_AU = 0.09
TRAPPIST_DIVE_WAYPOINTS_AU = (8000.0, 600.0, 60.0, 6.0)
TRAPPIST_ELEVATION_DEG = 58.0

# Classic dotted timeline (after shared Sol open / pullback).
TRAPPIST_TRAVEL_END = 0.82
TRAPPIST_ARRIVE_HOLD_END = 0.88
TRAPPIST_DIVE_END = 0.93
TRAPPIST_WIDE_HOLD_END = 0.93  # classic: no extra wide hold
TRAPPIST_INNER_ARRIVE = 0.97

# Blender arrival: longer host hold + wide beat before the chain fills the frame.
ARRIVAL_TRAPPIST_TRAVEL_END = 0.80
ARRIVAL_TRAPPIST_HOLD_END = 0.855
ARRIVAL_TRAPPIST_DIVE_END = 0.90
ARRIVAL_TRAPPIST_WIDE_HOLD_END = 0.925
ARRIVAL_TRAPPIST_INNER_ARRIVE = 0.95

OUTPUT_DIRECTORY = 'output/animate/sol_trappist'
BLENDER_OUTPUT_DIRECTORY = 'output/animate/sol_trappist/blender'


class SolTrappistCinematicAnimator(SolCentauriCinematicAnimator):
    """Flight from our solar system to the TRAPPIST-1 planets."""

    def __init__(
        self,
        system: StarSystem,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
        *,
        useBlenderBodies: bool = False,
    ):
        # Parent __init__ hard-requires alpha_centauri and builds AB/Proxima orbits.
        # TRAPPIST reimplements the Sol-shared figure/population setup and overrides
        # destination camera/draw/caption methods instead of calling super().__init__.
        # codeql[py/missing-call-to-init]
        if system.systemId != 'trappist_1':
            raise ValueError(f'Expected trappist_1, got {system.systemId!r}')
        if not system.stars:
            raise ValueError('trappist_1 system has no host star row')

        host = system.stars[0]
        if host.positionX is None or host.positionY is None or host.positionZ is None:
            raise ValueError('TRAPPIST-1 host is missing Sol XYZ in the star catalog')

        self.system = system
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.useBlenderBodies = useBlenderBodies
        self.constants = AstronomicalConstants()
        self.orbitCalculator = OrbitCalculator()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()

        self.hostStar = host
        self.hostSolAu = np.array(
            [host.positionX, host.positionY, host.positionZ],
            dtype=float,
        )
        # Parent camera / path code keys off barycenterSolAu as the destination.
        self.barycenterSolAu = self.hostSolAu.copy()
        self.distanceLy = float(np.linalg.norm(self.hostSolAu) / self.constants.lightYearToAu)
        self.trappistPlanets = tuple(
            sorted(system.planets, key=lambda planet: planet.semiMajorAxisAu)
        )

        # Unused α Cen fields — kept so inherited helpers that touch names do not explode.
        self.transform = None
        self.primaryOrbit = None
        self.secondaryOrbit = None
        self.proximaOrbit = None
        self.proximaPlanets = self.trappistPlanets
        self.primaryOrbitPathSol = np.zeros((0, 3))
        self.secondaryOrbitPathSol = np.zeros((0, 3))
        self.proximaOrbitPathSol = np.zeros((0, 3))

        self.asteroidPopulation = AnimatedAsteroidPopulation(
            self.constants,
            AsteroidPopulationCounts(
                asteroidBelt=480,
                hildas=140,
                trojansAndGreeks=90,
                kuiperBelt=900,
                oortCloud=4500,
            ),
            includeKuiperAndOort=True,
            useSphericalShell3d=True,
        )
        self.famousAsteroidCatalog = FamousAsteroidCatalog()
        self._fixedOortX: np.ndarray | None = None
        self._fixedOortY: np.ndarray | None = None
        self._fixedOortZ: np.ndarray | None = None

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.fieldStarColor = '#E8E8E8' if self.isDark else '#505050'
        self.pathColor = '#9EC9FF' if self.isDark else '#2F6FBF'
        self.hudColor = '#D8EEFF' if self.isDark else '#103050'
        self.orbitColor = '#B0B0B0' if self.isDark else '#606060'
        self.renderStyle = ASTEROID_RENDER_STYLES['dark' if self.isDark else 'light']

        self.solEarthHalfWidthAu = SOL_EARTH_HALF_AU
        self.solHalfWidthAu = SOL_HALF_WIDTH_AU
        self.startHalfWidthAu = START_HALF_WIDTH_LY * self.constants.lightYearToAu
        self.abHalfWidthAu = TRAPPIST_ARRIVE_HALF_AU
        self.wideHalfWidthAu = TRAPPIST_ARRIVE_HALF_AU
        self.proximaWideHalfWidthAu = TRAPPIST_WIDE_HALF_AU
        self.proximaInnerHalfWidthAu = TRAPPIST_INNER_HALF_AU
        self.proximaHalfWidthAu = TRAPPIST_INNER_HALF_AU
        pathAzimuth = float(np.degrees(np.arctan2(self.hostSolAu[1], self.hostSolAu[0])))
        self.travelAzimuthDeg = pathAzimuth + 55.0
        self.solAzimuthDeg = pathAzimuth + 35.0

        self.fieldStars = self._loadFieldStars(starsCsvPath)
        self.solPlanetPaths = {
            name: self._planetOrbitPath(self.planetCatalog.planets[name])
            for name in SOL_PLANET_NAMES
        }
        self.trappistPlanetPathsLocal = {
            planet.planetId: orbitPathInOrbitalPlane(
                self.orbitCalculator,
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.argumentPeriapsisDeg,
            )
            for planet in self.trappistPlanets
        }
        self.proximaPlanetPathsLocal = self.trappistPlanetPathsLocal

        self._viewFocus = np.zeros(3)
        self._viewHalfWidthAu = self.solEarthHalfWidthAu
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, layout='none')
        self.axes = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), projection='3d')
        self.bodyOverlay = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='none', zorder=20)
        self.bodyOverlay.set_axis_off()
        self.bodyOverlay.patch.set_alpha(0.0)
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)
        self._pendingBlenderBodies: list[
            tuple[str, np.ndarray, int, float, bool, float, float | None]
        ] = []
        self._pendingBlenderLabels: list[tuple[str, np.ndarray, float, float]] = []
        self._blenderBodyPaintZorder: dict[tuple[str, float, float, float], int] = {}
        self.blenderSprites: BlenderBodySpriteAtlas | None = None
        if self.useBlenderBodies:
            themeName = 'dark' if self.isDark else 'light'
            self.blenderSprites = BlenderBodySpriteAtlas(themeName)
            print(
                'Blender spin loops: lazy-load by zoom stage '
                f'(theme={themeName}; Earth/Moon probed: '
                f'Earth={"on" if self.blenderSprites.hasEarth else "missing"}, '
                f'Moon={"on" if self.blenderSprites.hasMoon else "missing"})'
            )

    def _abTravelEnd(self) -> float:
        return ARRIVAL_TRAPPIST_TRAVEL_END if self.useBlenderBodies else TRAPPIST_TRAVEL_END

    def _abHoldEnd(self) -> float:
        return ARRIVAL_TRAPPIST_HOLD_END if self.useBlenderBodies else TRAPPIST_ARRIVE_HOLD_END

    def _wideOutArrive(self) -> float:
        # No triple-system zoom-out — hold end feeds straight into the planet dive.
        return self._abHoldEnd()

    def _wideHoldEnd(self) -> float:
        return self._abHoldEnd()

    def _proximaDiveEnd(self) -> float:
        return ARRIVAL_TRAPPIST_DIVE_END if self.useBlenderBodies else TRAPPIST_DIVE_END

    def _proximaWideHoldEnd(self) -> float:
        return ARRIVAL_TRAPPIST_WIDE_HOLD_END if self.useBlenderBodies else TRAPPIST_WIDE_HOLD_END

    def _proximaInnerArrive(self) -> float:
        return ARRIVAL_TRAPPIST_INNER_ARRIVE if self.useBlenderBodies else TRAPPIST_INNER_ARRIVE

    def _loadFieldStars(self, starsCsvPath: str) -> pd.DataFrame:
        catalog = SystemCatalog(starsCsvPath=starsCsvPath).starCatalog
        stars = catalog.starsWithinLightYears(FIELD_STARS_MAX_LY).copy()
        stars = stars[~stars['System'].astype(str).str.contains('Solar System', na=False)]
        if 'system_id' in stars.columns:
            stars = stars[stars['system_id'] != 'trappist_1']
        stars = stars.dropna(subset=['positionX', 'positionY', 'positionZ']).copy()
        if stars.empty:
            return stars

        fallback = self.fieldStarColor
        colors = [
            spectralClassColor(value, fallback=fallback)
            for value in stars.get('Stellarclass', pd.Series(dtype=str))
        ]
        magnitudes = [
            parseApparentMagnitude(value)
            for value in stars.get('Apparent magnitude (v)', pd.Series(dtype=str))
        ]
        sizes: list[float] = []
        alphas: list[float] = []
        for magnitude in magnitudes:
            if magnitude is None:
                sizes.append(14.0)
                alphas.append(0.55)
                continue
            sizes.append(float(np.clip(34.0 - 1.15 * magnitude, 5.0, 48.0)))
            alphas.append(float(np.clip(0.85 - 0.022 * magnitude, 0.22, 0.95)))
        stars['fieldColor'] = colors
        stars['fieldSize'] = sizes
        stars['fieldAlpha'] = alphas
        return stars

    def _proximaPositionSol(self, frame: int) -> np.ndarray:
        del frame
        return self.hostSolAu.copy()

    def _trappistPlanetPositionSol(self, planet: SystemPlanet, frame: int) -> np.ndarray:
        offsetX, offsetY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.orbitalPeriodDays,
            planet.argumentPeriapsisDeg,
            0.0,
            frame,
            ANIMATION_SPEED_TRAPPIST_PLANETS,
        )
        return self.hostSolAu + np.array([offsetX, offsetY, 0.0], dtype=float)

    def _cameraState(self, frame: int) -> tuple[np.ndarray, float]:
        linear = timelineProgress(frame, self.animationFrames)
        travelProgressValue = self._travelProgress(frame)
        diveProgress = self._proximaTravelProgress(frame)
        host = self.hostSolAu
        holdEnd = self._abHoldEnd()
        diveEnd = self._proximaDiveEnd()
        wideHoldEnd = self._proximaWideHoldEnd()
        innerArrive = self._proximaInnerArrive()

        if linear <= SOL_HOLD_END:
            return self._solOpeningCameraState(frame, linear)

        if linear <= PULLBACK_END:
            return np.zeros(3), self._pullbackHalfWidthAu(linear)

        if linear <= holdEnd:
            focus = smootherstep(min(1.0, travelProgressValue / AB_FOCUS_ARRIVE)) * host
            return focus, self._abLegHalfWidthAu(focus, travelProgressValue)

        if linear <= diveEnd:
            return host.copy(), stagedLogDive(
                self.abHalfWidthAu,
                self.proximaWideHalfWidthAu,
                TRAPPIST_DIVE_WAYPOINTS_AU,
                diveProgress,
            )

        if linear <= wideHoldEnd:
            return host.copy(), self.proximaWideHalfWidthAu

        if linear <= innerArrive:
            tighten = segmentProgress(linear, wideHoldEnd, innerArrive) ** 1.35
            return host.copy(), logLerp(
                self.proximaWideHalfWidthAu, self.proximaInnerHalfWidthAu, tighten
            )

        return host.copy(), self.proximaInnerHalfWidthAu

    def update(self, frame: int):
        self.axes.clear()
        self.bodyOverlay.clear()
        self.bodyOverlay.set_axis_off()
        self.bodyOverlay.patch.set_alpha(0.0)
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)
        self._pendingBlenderBodies = []
        self._pendingBlenderLabels = []
        self._blenderBodyPaintZorder = {}
        for textArtist in list(self.figure.texts):
            textArtist.remove()
        focus, halfWidthAu = self._cameraState(frame)
        self._viewFocus = focus
        self._viewHalfWidthAu = halfWidthAu
        travelProgressValue = self._travelProgress(frame)
        diveProgress = self._proximaTravelProgress(frame)
        linear = timelineProgress(frame, self.animationFrames)

        self._drawPath(frame, focus, travelProgressValue, diveProgress, halfWidthAu, linear)
        self._drawSolarSystem(frame, halfWidthAu, travelProgressValue)
        self._drawFieldStars(halfWidthAu)
        self._drawTrappistDestination(frame, halfWidthAu, travelProgressValue, linear)
        self._applyAxes(focus, halfWidthAu, travelProgressValue, diveProgress, linear)
        self._flushBlenderBodyOverlays(halfWidthAu)
        self._flushBlenderBodyLabels(halfWidthAu)
        return []

    def _drawPath(
        self,
        frame: int,
        focus: np.ndarray,
        abProgress: float,
        proximaProgress: float,
        halfWidthAu: float,
        linear: float,
    ) -> None:
        del frame, proximaProgress
        if halfWidthAu < 200.0 and abProgress < 0.05 and linear < self._abHoldEnd():
            return
        if linear >= self._proximaDiveEnd() and halfWidthAu < 40.0:
            return

        path = np.vstack((np.zeros(3), self.hostSolAu))
        self.axes.plot(
            path[:, 0], path[:, 1], path[:, 2], color=self.pathColor, linewidth=1.2, alpha=0.4
        )
        if 0.0 < abProgress < AB_CRUISE_END:
            self._drawTraveler(focus)

    def _drawTrappistDestination(
        self,
        frame: int,
        halfWidthAu: float,
        travelProgressValue: float,
        linear: float,
    ) -> None:
        if halfWidthAu < 100.0 and travelProgressValue < 0.05:
            return

        host = self.hostSolAu
        if linear < self._abHoldEnd():
            # Cruise / arrive: unresolved host marker along the travel path.
            size = np.clip(
                52.0 * (self.startHalfWidthAu / max(halfWidthAu, 1.0)) ** 0.35, 36.0, 190.0
            )
            if travelProgressValue > 0.15:
                size = max(size, 90.0)
            label = 'TRAPPIST-1' if linear >= self._abTravelEnd() else 'TRAPPIST-1 (destination)'
            self._drawStarMarker(
                host,
                TRAPPIST_1_STAR_COLOR,
                size,
                zorder=self._scatterDepthZorder(host, base=5),
            )
            self._label3d(host, f'  {label}', color=self.labelColor, fontsize=9)
            return

        # Dive / finale: host + resonant chain.
        if halfWidthAu > 80.0:
            return

        starSize = 520.0 if halfWidthAu <= TRAPPIST_INNER_HALF_AU * 1.8 else 320.0
        self._drawStarMarker(
            host,
            TRAPPIST_1_STAR_COLOR,
            starSize,
            zorder=self._scatterDepthZorder(host, base=5),
        )
        self._label3d(
            host,
            '  TRAPPIST-1',
            color=self.labelColor,
            fontsize=11.0 if halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.05 else 10.0,
        )

        for planet in self.trappistPlanets:
            self._drawOneTrappistPlanet(planet, frame, halfWidthAu)

    def _drawOneTrappistPlanet(
        self,
        planet: SystemPlanet,
        frame: int,
        halfWidthAu: float,
    ) -> None:
        if planet.semiMajorAxisAu > halfWidthAu * 1.15:
            return

        pathX, pathY = self.trappistPlanetPathsLocal[planet.planetId]
        pathLocal = np.column_stack((pathX, pathY, np.zeros_like(pathX)))
        pathSol = pathLocal + self.hostSolAu
        self.axes.plot(
            pathSol[:, 0],
            pathSol[:, 1],
            pathSol[:, 2],
            color=planet.color,
            linewidth=1.6,
            alpha=0.85,
        )
        position = self._trappistPlanetPositionSol(planet, frame)
        if not self._inView(position, margin=1.15):
            return

        shortName = planet.name.replace('TRAPPIST-1 ', '')
        labelSize = 9.0
        bodyScale = BLENDER_PLANET_BODY_SCALE.get(planet.name)
        queued = False
        if (
            bodyScale is not None
            and self.useBlenderBodies
            and halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.2
        ):
            queued = self._queueBlenderBody(
                planet.name,
                position,
                frame,
                halfWidthAu,
                openCloseup=halfWidthAu <= TRAPPIST_INNER_HALF_AU * 2.5,
                bodyScale=bodyScale,
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
        if queued:
            self._pendingBlenderLabels.append((planet.name, position.copy(), labelSize, bodyScale))
            return

        baseSize = 48.0
        zoomBoost = np.clip(TRAPPIST_WIDE_HALF_AU / max(halfWidthAu, 1e-6), 1.0, 12.0)
        markerSize = baseSize * (zoomBoost**0.65)
        self.axes.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=planet.color,
            s=markerSize,
            alpha=1.0,
            depthshade=False,
            zorder=self._scatterDepthZorder(position, base=4),
        )
        self._label3d(
            position,
            f'  {shortName}',
            color=self.labelColor,
            fontsize=int(labelSize),
            alpha=0.95,
        )

    def _blenderArrivalCaption(
        self, abProgress: float, halfWidthAu: float, linear: float
    ) -> tuple[str, str] | None:
        if linear < PULLBACK_END:
            return None
        remainingLy = (1.0 - abProgress) * self.distanceLy
        if linear < self._abTravelEnd() and abProgress < AB_CRUISE_END:
            return ('Flying toward TRAPPIST-1', f'{remainingLy:.1f} light-years remaining')
        if linear < self._abTravelEnd():
            return ('Arriving at TRAPPIST-1', f'Scale ~{halfWidthAu:.0f} AU across')
        if linear < self._abHoldEnd():
            return (
                'TRAPPIST-1',
                'Ultracool dwarf · seven Earth-sized planets in a resonant chain',
            )
        if linear < self._proximaDiveEnd():
            return (
                'Diving in to TRAPPIST-1',
                'From interstellar cruise down to the inner planets',
            )
        if linear < self._proximaWideHoldEnd():
            return (
                'TRAPPIST-1 system',
                'Seven confirmed worlds · compact orbits inside ~0.06 AU',
            )
        if halfWidthAu > TRAPPIST_INNER_HALF_AU * 1.3:
            return (
                'TRAPPIST-1 system',
                'Closing in on the resonant chain',
            )
        return (
            'TRAPPIST-1 b–h up close',
            'Packed Earth-sized worlds around an ultracool dwarf',
        )

    def _caption(
        self, abProgress: float, proximaProgress: float, halfWidthAu: float, linear: float
    ) -> tuple[str, str]:
        del proximaProgress
        solCaption = self._solCaption(halfWidthAu, linear)
        if solCaption is not None:
            return solCaption
        if self.useBlenderBodies:
            arrival = self._blenderArrivalCaption(abProgress, halfWidthAu, linear)
            if arrival is not None:
                return arrival
        remainingLy = (1.0 - abProgress) * self.distanceLy
        if linear < self._abTravelEnd() and abProgress < AB_CRUISE_END:
            return ('Flying toward TRAPPIST-1', f'{remainingLy:.1f} light-years remaining')
        if linear < self._abHoldEnd():
            if halfWidthAu > 20.0:
                return ('Arriving at TRAPPIST-1', f'Scale ~{halfWidthAu:.0f} AU across')
            return (
                'TRAPPIST-1',
                'Ultracool dwarf · seven Earth-sized planets in a resonant chain',
            )
        if linear < self._proximaDiveEnd():
            return (
                'Diving in to TRAPPIST-1',
                'From interstellar cruise down to the inner planets',
            )
        if halfWidthAu > TRAPPIST_INNER_HALF_AU * 1.3:
            return (
                'TRAPPIST-1 system',
                'Seven confirmed worlds · compact orbits inside ~0.06 AU',
            )
        return (
            'TRAPPIST-1 b–h up close',
            'Packed Earth-sized worlds around an ultracool dwarf',
        )

    def _applyAxes(
        self,
        focus: np.ndarray,
        halfWidthAu: float,
        abProgress: float,
        proximaProgress: float,
        linear: float,
    ) -> None:
        # Reuse Sol / travel tilt logic; swap Proxima close-up elevation for TRAPPIST.
        self.axes.set_xlim(focus[0] - halfWidthAu, focus[0] + halfWidthAu)
        self.axes.set_ylim(focus[1] - halfWidthAu, focus[1] + halfWidthAu)
        self.axes.set_zlim(focus[2] - halfWidthAu, focus[2] + halfWidthAu)

        if linear < PULLBACK_END:
            if halfWidthAu <= SOL_NEAR_SUN_HALF_AU:
                if self.useBlenderBodies and halfWidthAu < self.solEarthHalfWidthAu - 1e-9:
                    closeTilt = smootherstep(
                        (halfWidthAu - SOL_EARTH_CLOSE_HALF_AU)
                        / max(self.solEarthHalfWidthAu - SOL_EARTH_CLOSE_HALF_AU, 1e-6)
                    )
                    elev = EARTH_CLOSE_ELEVATION_DEG + closeTilt * (
                        EARTH_OPEN_ELEVATION_DEG - EARTH_CLOSE_ELEVATION_DEG
                    )
                else:
                    tilt = smootherstep(
                        (halfWidthAu - SOL_EARTH_HALF_AU)
                        / max(SOL_NEAR_SUN_HALF_AU - SOL_EARTH_HALF_AU, 1e-6)
                    )
                    elev = EARTH_OPEN_ELEVATION_DEG + tilt * (
                        SOL_ELEVATION_DEG - EARTH_OPEN_ELEVATION_DEG
                    )
            else:
                elev = SOL_ELEVATION_DEG
            azim = self.solAzimuthDeg
        elif linear < self._abTravelEnd():
            blend = segmentProgress(linear, PULLBACK_END, PULLBACK_END + 0.02)
            elev = SOL_ELEVATION_DEG + blend * (CAMERA_ELEVATION_DEG - SOL_ELEVATION_DEG)
            azim = lerpAngleDeg(self.solAzimuthDeg, self.travelAzimuthDeg, blend)
        elif linear < self._proximaWideHoldEnd():
            elev, azim = CAMERA_ELEVATION_DEG, self.travelAzimuthDeg
        else:
            blend = segmentProgress(linear, self._proximaWideHoldEnd(), self._proximaInnerArrive())
            elev = CAMERA_ELEVATION_DEG + blend * (TRAPPIST_ELEVATION_DEG - CAMERA_ELEVATION_DEG)
            azim = self.travelAzimuthDeg

        self.axes.view_init(elev=elev, azim=azim)
        self.axes.set_axis_off()
        self.axes.set_box_aspect((1, 1, 1), zoom=1.0)
        self.axes.set_position((0.0, 0.0, 1.0, 1.0))

        title, subtitle = self._caption(abProgress, proximaProgress, halfWidthAu, linear)
        self.axes.set_title(title, color=self.labelColor, pad=10, y=0.98, fontsize=13)
        self.figure.text(
            0.5,
            0.035,
            subtitle,
            ha='center',
            color=self.hudColor,
            fontsize=10,
            alpha=0.95,
        )


def renderSolTrappistCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
    *,
    useBlenderBodies: bool = False,
) -> None:
    catalog = SystemCatalog(starsCsvPath=starsCsvPath)
    system = catalog.load('trappist_1')
    if useBlenderBodies:
        outputDirectory = BLENDER_OUTPUT_DIRECTORY
        stem = 'sol_trappist_cinematic_blender'
    else:
        outputDirectory = OUTPUT_DIRECTORY
        stem = 'sol_trappist_cinematic'
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{outputDirectory}/{stem}_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = SolTrappistCinematicAnimator(
            system,
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
            useBlenderBodies=useBlenderBodies,
        )
        animator.saveGif(outputPath)
    print('Sol → TRAPPIST-1 cinematic animations completed!')
