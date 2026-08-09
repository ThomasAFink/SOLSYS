"""Sol → TRAPPIST-1 cinematic — scale odyssey ending on the seven-planet chain.

Reuses Sol opening / pullback / Blender billboard machinery from the Sol→α Cen
cinematic; destination is a single-host system at true Sol XYZ (~40.7 ly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from solsys.physics.catalogs.system_catalog import StarSystem, SystemCatalog, SystemPlanet

from animate.scenes.exoplanet_system import bodyPositionInOrbitalPlane, orbitPathInOrbitalPlane
from animate.scenes.sol_centauri_cinematic import (
    AB_CRUISE_END,
    AB_FOCUS_ARRIVE,
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
    SOL_HOLD_END,
    SOL_NEAR_SUN_HALF_AU,
    SolScaleCinematicAnimator,
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
# HZ linger frames e (~0.029 AU) and f (~0.038 AU); g sits near the outer edge.
TRAPPIST_HZ_HALF_AU = 0.052
TRAPPIST_INNER_HALF_AU = 0.09
# Monotonic tighten from arrive → planet-wide (no Proxima-style zoom-out waypoints).
TRAPPIST_DIVE_WAYPOINTS_AU = (1.0, 0.4)
TRAPPIST_HZ_DIVE_WAYPOINTS_AU = (0.10, 0.07)
TRAPPIST_ELEVATION_DEG = 58.0
# Schematic conservative liquid-water HZ for an ultracool dwarf (not a climate model).
# Spans roughly outside d through e/f; g rides the outer rim.
TRAPPIST_HZ_INNER_AU = 0.024
TRAPPIST_HZ_OUTER_AU = 0.046
TRAPPIST_HZ_FOCUS_NAMES = ('TRAPPIST-1 e', 'TRAPPIST-1 f')
TRAPPIST_HZ_COLOR = '#5CDB8A'

# Classic dotted timeline (after shared Sol open / pullback).
TRAPPIST_TRAVEL_END = 0.82
TRAPPIST_ARRIVE_HOLD_END = 0.87
TRAPPIST_DIVE_END = 0.91
TRAPPIST_WIDE_HOLD_END = 0.93
TRAPPIST_HZ_ARRIVE = 0.95
TRAPPIST_HZ_HOLD_END = 0.97
TRAPPIST_INNER_ARRIVE = 0.985

# Blender arrival: wide chain → HZ (e/f) linger → full-chain finale.
ARRIVAL_TRAPPIST_TRAVEL_END = 0.80
ARRIVAL_TRAPPIST_HOLD_END = 0.845
ARRIVAL_TRAPPIST_DIVE_END = 0.875
ARRIVAL_TRAPPIST_WIDE_HOLD_END = 0.895
ARRIVAL_TRAPPIST_HZ_ARRIVE = 0.92
ARRIVAL_TRAPPIST_HZ_HOLD_END = 0.95
ARRIVAL_TRAPPIST_INNER_ARRIVE = 0.975

OUTPUT_DIRECTORY = 'output/animate/sol_trappist'
BLENDER_OUTPUT_DIRECTORY = 'output/animate/sol_trappist/blender'


class SolTrappistCinematicAnimator(SolScaleCinematicAnimator):
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
        if system.systemId != 'trappist_1':
            raise ValueError(f'Expected trappist_1, got {system.systemId!r}')
        if not system.stars:
            raise ValueError('trappist_1 system has no host star row')

        host = system.stars[0]
        if host.positionX is None or host.positionY is None or host.positionZ is None:
            raise ValueError('TRAPPIST-1 host is missing Sol XYZ in the star catalog')

        self.system = system
        self.hostStar = host
        self.hostSolAu = np.array(
            [host.positionX, host.positionY, host.positionZ],
            dtype=float,
        )
        self.trappistPlanets = tuple(
            sorted(system.planets, key=lambda planet: planet.semiMajorAxisAu)
        )
        super().__init__(
            style=style,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
            useBlenderBodies=useBlenderBodies,
            destinationSolAu=self.hostSolAu,
            startHalfWidthLy=START_HALF_WIDTH_LY,
            arriveHalfWidthAu=TRAPPIST_ARRIVE_HALF_AU,
            wideSystemHalfWidthAu=TRAPPIST_ARRIVE_HALF_AU,
            destinationWideHalfWidthAu=TRAPPIST_WIDE_HALF_AU,
            destinationInnerHalfWidthAu=TRAPPIST_INNER_HALF_AU,
        )
        self.trappistPlanetPathsLocal = {
            planet.planetId: orbitPathInOrbitalPlane(
                self.orbitCalculator,
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.argumentPeriapsisDeg,
            )
            for planet in self.trappistPlanets
        }

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

    def _hzArrive(self) -> float:
        return ARRIVAL_TRAPPIST_HZ_ARRIVE if self.useBlenderBodies else TRAPPIST_HZ_ARRIVE

    def _hzHoldEnd(self) -> float:
        return ARRIVAL_TRAPPIST_HZ_HOLD_END if self.useBlenderBodies else TRAPPIST_HZ_HOLD_END

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
        hzArrive = self._hzArrive()
        hzHoldEnd = self._hzHoldEnd()
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

        if linear <= hzArrive:
            tighten = segmentProgress(linear, wideHoldEnd, hzArrive) ** 1.2
            return host.copy(), stagedLogDive(
                self.proximaWideHalfWidthAu,
                TRAPPIST_HZ_HALF_AU,
                TRAPPIST_HZ_DIVE_WAYPOINTS_AU,
                tighten,
            )

        if linear <= hzHoldEnd:
            return host.copy(), TRAPPIST_HZ_HALF_AU

        if linear <= innerArrive:
            # Ease out from the HZ linger so the full resonant chain re-enters frame.
            pull = segmentProgress(linear, hzHoldEnd, innerArrive) ** 1.15
            return host.copy(), logLerp(TRAPPIST_HZ_HALF_AU, TRAPPIST_INNER_HALF_AU, pull)

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
        # Until the camera is tight enough for the chain, keep a continuous host marker
        # (cruise, arrive hold, and the start of the dive — never blank the destination).
        showChain = linear >= self._abHoldEnd() and halfWidthAu <= TRAPPIST_ARRIVE_HALF_AU * 1.05
        if not showChain:
            size = np.clip(
                52.0 * (self.startHalfWidthAu / max(halfWidthAu, 1.0)) ** 0.35, 36.0, 190.0
            )
            if travelProgressValue > 0.15 or linear >= self._abTravelEnd():
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

        if halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.15:
            self._drawHabitableZoneBand(halfWidthAu)

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

        hzFocus = (
            self._hzArrive() <= linear <= self._hzHoldEnd()
            or abs(halfWidthAu - TRAPPIST_HZ_HALF_AU) < 1e-3
        )
        for planet in self.trappistPlanets:
            self._drawOneTrappistPlanet(planet, frame, halfWidthAu, hzFocus=hzFocus)

    def _drawHabitableZoneBand(self, halfWidthAu: float) -> None:
        """Schematic annular HZ in the face-on orbital plane around the host."""
        del halfWidthAu
        host = self.hostSolAu
        theta = np.linspace(0.0, 2.0 * np.pi, 160)
        cosT = np.cos(theta)
        sinT = np.sin(theta)
        bandAlpha = 0.16 if self.isDark else 0.22
        for radius in np.linspace(TRAPPIST_HZ_INNER_AU, TRAPPIST_HZ_OUTER_AU, 7):
            ring = host + np.column_stack((radius * cosT, radius * sinT, np.zeros_like(theta)))
            self.axes.plot(
                ring[:, 0],
                ring[:, 1],
                ring[:, 2],
                color=TRAPPIST_HZ_COLOR,
                linewidth=2.2,
                alpha=bandAlpha,
                zorder=2,
            )
        for radius, width in (
            (TRAPPIST_HZ_INNER_AU, 1.4),
            (TRAPPIST_HZ_OUTER_AU, 1.4),
        ):
            rim = host + np.column_stack((radius * cosT, radius * sinT, np.zeros_like(theta)))
            self.axes.plot(
                rim[:, 0],
                rim[:, 1],
                rim[:, 2],
                color=TRAPPIST_HZ_COLOR,
                linewidth=width,
                alpha=0.55 if self.isDark else 0.65,
                zorder=3,
            )
        labelPos = host + np.array([TRAPPIST_HZ_OUTER_AU * 0.78, TRAPPIST_HZ_OUTER_AU * 0.55, 0.0])
        self._label3d(
            labelPos,
            '  approx. habitable zone',
            color=TRAPPIST_HZ_COLOR,
            fontsize=8,
            alpha=0.9,
        )

    def _drawOneTrappistPlanet(
        self,
        planet: SystemPlanet,
        frame: int,
        halfWidthAu: float,
        *,
        hzFocus: bool = False,
    ) -> None:
        if planet.semiMajorAxisAu > halfWidthAu * 1.15:
            return

        isHzCandidate = planet.name in TRAPPIST_HZ_FOCUS_NAMES
        pathX, pathY = self.trappistPlanetPathsLocal[planet.planetId]
        pathLocal = np.column_stack((pathX, pathY, np.zeros_like(pathX)))
        pathSol = pathLocal + self.hostSolAu
        orbitAlpha = 0.95 if (hzFocus and isHzCandidate) else (0.45 if hzFocus else 0.85)
        orbitWidth = 2.2 if (hzFocus and isHzCandidate) else 1.6
        self.axes.plot(
            pathSol[:, 0],
            pathSol[:, 1],
            pathSol[:, 2],
            color=planet.color,
            linewidth=orbitWidth,
            alpha=orbitAlpha,
        )
        position = self._trappistPlanetPositionSol(planet, frame)
        if not self._inView(position, margin=1.15):
            return

        shortName = planet.name.replace('TRAPPIST-1 ', '')
        if hzFocus and isHzCandidate:
            labelSize = 11.0
        elif hzFocus:
            labelSize = 7.0
        else:
            labelSize = 9.0
        bodyScale = BLENDER_PLANET_BODY_SCALE.get(planet.name)
        queued = False
        if (
            bodyScale is not None
            and self.useBlenderBodies
            and halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.2
            and (not hzFocus or isHzCandidate or halfWidthAu >= TRAPPIST_INNER_HALF_AU * 0.95)
        ):
            queued = self._queueBlenderBody(
                planet.name,
                position,
                frame,
                halfWidthAu,
                openCloseup=halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.15
                or halfWidthAu <= TRAPPIST_INNER_HALF_AU * 1.05,
                bodyScale=bodyScale * (1.15 if hzFocus and isHzCandidate else 1.0),
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
        if queued:
            self._pendingBlenderLabels.append((planet.name, position.copy(), labelSize, bodyScale))
            return

        baseSize = 64.0 if (hzFocus and isHzCandidate) else 48.0
        zoomBoost = np.clip(TRAPPIST_WIDE_HALF_AU / max(halfWidthAu, 1e-6), 1.0, 12.0)
        markerSize = baseSize * (zoomBoost**0.65)
        self.axes.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=planet.color,
            s=markerSize,
            alpha=1.0 if (not hzFocus or isHzCandidate) else 0.55,
            depthshade=False,
            zorder=self._scatterDepthZorder(position, base=4),
        )
        self._label3d(
            position,
            f'  {shortName}',
            color=self.labelColor,
            fontsize=int(labelSize),
            alpha=0.95 if (not hzFocus or isHzCandidate) else 0.55,
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
                'Seven confirmed worlds · schematic habitable zone highlighted',
            )
        if linear < self._hzArrive():
            return (
                'Closing on the habitable zone',
                'Approximate liquid-water belt around an ultracool dwarf',
            )
        if linear < self._hzHoldEnd():
            return (
                'TRAPPIST-1 e and f',
                'Temperate-zone candidates · e is the strongest HZ case',
            )
        if halfWidthAu > TRAPPIST_INNER_HALF_AU * 0.98 and linear < self._proximaInnerArrive():
            return (
                'TRAPPIST-1 system',
                'Pulling back to the full resonant chain',
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
        if linear < self._hzHoldEnd() and halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.05:
            if halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.1:
                return (
                    'TRAPPIST-1 e and f',
                    'Temperate-zone candidates · e is the strongest HZ case',
                )
            return (
                'TRAPPIST-1 system',
                'Seven confirmed worlds · schematic habitable zone highlighted',
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
        elif linear < self._hzHoldEnd():
            # Steady travel camera through dive, wide hold, and HZ linger.
            elev, azim = CAMERA_ELEVATION_DEG, self.travelAzimuthDeg
        else:
            blend = segmentProgress(linear, self._hzHoldEnd(), self._proximaInnerArrive())
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
