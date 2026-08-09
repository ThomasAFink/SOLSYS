"""Sol → TRAPPIST-1 cinematic — scale odyssey ending on the seven-planet chain.

Reuses Sol opening / pullback / Blender billboard machinery from the Sol→α Cen
cinematic; destination is a single-host system at true Sol XYZ (~40.7 ly).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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
# Final HZ + e/f close-ups: slow the dance so orbits read as motion, not blur.
ANIMATION_SPEED_TRAPPIST_PLANETS_CLOSE = 0.055

# Arrival: host readable; planets still tiny. Finale matches exoplanet scene (~0.09 AU).
TRAPPIST_ARRIVE_HALF_AU = 2.0
TRAPPIST_WIDE_HALF_AU = 0.15
# HZ overview frames the band; then sequential single-planet portraits (e, then f).
TRAPPIST_HZ_HALF_AU = 0.055
TRAPPIST_PLANET_HALF_AU = 0.013
# Back-compat alias used by a few call sites / tests.
TRAPPIST_CANDIDATE_HALF_AU = TRAPPIST_PLANET_HALF_AU
TRAPPIST_INNER_HALF_AU = 0.09
# Modest hero boost — disk stays well under orbital spacing (~0.009 AU e↔f).
TRAPPIST_PLANET_HERO_SCALE = 1.45
# Monotonic tighten from arrive → planet-wide (no Proxima-style zoom-out waypoints).
TRAPPIST_DIVE_WAYPOINTS_AU = (1.0, 0.4)
TRAPPIST_HZ_DIVE_WAYPOINTS_AU = (0.10, 0.07)
TRAPPIST_PLANET_DIVE_WAYPOINTS_AU = (0.032, 0.020)
TRAPPIST_ELEVATION_DEG = 58.0
# Steeper view so the HZ annulus reads as a band, not a foreshortened line.
TRAPPIST_HZ_ELEVATION_DEG = 62.0
TRAPPIST_PLANET_ELEVATION_DEG = 52.0
TRAPPIST_CANDIDATE_ELEVATION_DEG = TRAPPIST_PLANET_ELEVATION_DEG
# Schematic conservative liquid-water HZ for an ultracool dwarf (not a climate model).
# Spans roughly outside d through e/f; g rides the outer rim.
TRAPPIST_HZ_INNER_AU = 0.024
TRAPPIST_HZ_OUTER_AU = 0.046
TRAPPIST_HZ_FOCUS_NAMES = ('TRAPPIST-1 e', 'TRAPPIST-1 f')
TRAPPIST_HZ_COLOR_DARK = '#7DFFB0'
TRAPPIST_HZ_COLOR_LIGHT = '#1B8A4A'

# Classic dotted timeline (after shared Sol open / pullback).
TRAPPIST_TRAVEL_END = 0.80
TRAPPIST_ARRIVE_HOLD_END = 0.85
TRAPPIST_DIVE_END = 0.885
TRAPPIST_WIDE_HOLD_END = 0.90
TRAPPIST_HZ_ARRIVE = 0.915
TRAPPIST_HZ_HOLD_END = 0.932
TRAPPIST_E_ARRIVE = 0.948
TRAPPIST_E_HOLD_END = 0.962
TRAPPIST_F_ARRIVE = 0.975
TRAPPIST_F_HOLD_END = 0.988
TRAPPIST_INNER_ARRIVE = 0.995
# Aliases for older candidate naming in tests / helpers.
TRAPPIST_CANDIDATE_ARRIVE = TRAPPIST_E_ARRIVE
TRAPPIST_CANDIDATE_HOLD_END = TRAPPIST_F_HOLD_END

# Blender: wide chain → HZ → zoom e → pan to f → full-chain finale.
ARRIVAL_TRAPPIST_TRAVEL_END = 0.78
ARRIVAL_TRAPPIST_HOLD_END = 0.825
ARRIVAL_TRAPPIST_DIVE_END = 0.852
ARRIVAL_TRAPPIST_WIDE_HOLD_END = 0.865
ARRIVAL_TRAPPIST_HZ_ARRIVE = 0.882
ARRIVAL_TRAPPIST_HZ_HOLD_END = 0.905
ARRIVAL_TRAPPIST_E_ARRIVE = 0.922
ARRIVAL_TRAPPIST_E_HOLD_END = 0.942
ARRIVAL_TRAPPIST_F_ARRIVE = 0.958
ARRIVAL_TRAPPIST_F_HOLD_END = 0.978
ARRIVAL_TRAPPIST_INNER_ARRIVE = 0.992
ARRIVAL_TRAPPIST_CANDIDATE_ARRIVE = ARRIVAL_TRAPPIST_E_ARRIVE
ARRIVAL_TRAPPIST_CANDIDATE_HOLD_END = ARRIVAL_TRAPPIST_F_HOLD_END

OUTPUT_DIRECTORY = 'output/animate/sol_trappist'
BLENDER_OUTPUT_DIRECTORY = 'output/animate/sol_trappist/blender'


def _hexToRgb(color: str) -> tuple[float, float, float]:
    text = color.lstrip('#')
    return (
        int(text[0:2], 16) / 255.0,
        int(text[2:4], 16) / 255.0,
        int(text[4:6], 16) / 255.0,
    )


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

    def _eArrive(self) -> float:
        return ARRIVAL_TRAPPIST_E_ARRIVE if self.useBlenderBodies else TRAPPIST_E_ARRIVE

    def _eHoldEnd(self) -> float:
        return ARRIVAL_TRAPPIST_E_HOLD_END if self.useBlenderBodies else TRAPPIST_E_HOLD_END

    def _fArrive(self) -> float:
        return ARRIVAL_TRAPPIST_F_ARRIVE if self.useBlenderBodies else TRAPPIST_F_ARRIVE

    def _fHoldEnd(self) -> float:
        return ARRIVAL_TRAPPIST_F_HOLD_END if self.useBlenderBodies else TRAPPIST_F_HOLD_END

    def _candidateArrive(self) -> float:
        """Start of sequential planet portraits (zoom onto e)."""
        return self._eArrive()

    def _candidateHoldEnd(self) -> float:
        """End of sequential planet portraits (after f hold)."""
        return self._fHoldEnd()

    def _proximaInnerArrive(self) -> float:
        return ARRIVAL_TRAPPIST_INNER_ARRIVE if self.useBlenderBodies else TRAPPIST_INNER_ARRIVE

    def _planetByName(self, name: str) -> SystemPlanet:
        for planet in self.trappistPlanets:
            if planet.name == name:
                return planet
        raise KeyError(name)

    def _planetFocusSol(self, name: str, frame: int) -> np.ndarray:
        return self._trappistPlanetPositionSol(self._planetByName(name), frame)

    def _candidateFocusSol(self, frame: int) -> np.ndarray:
        """Active portrait target (e, then f) for the sequential HZ close-ups."""
        linear = timelineProgress(frame, self.animationFrames)
        if linear <= self._eHoldEnd():
            return self._planetFocusSol('TRAPPIST-1 e', frame)
        if linear <= self._fArrive():
            blend = smootherstep(segmentProgress(linear, self._eHoldEnd(), self._fArrive()))
            focusE = self._planetFocusSol('TRAPPIST-1 e', frame)
            focusF = self._planetFocusSol('TRAPPIST-1 f', frame)
            return (1.0 - blend) * focusE + blend * focusF
        return self._planetFocusSol('TRAPPIST-1 f', frame)

    def _portraitHeroName(self, frame: int) -> str | None:
        """Which HZ world is the current portrait subject, if any."""
        linear = timelineProgress(frame, self.animationFrames)
        if self._hzHoldEnd() < linear <= self._eHoldEnd():
            return 'TRAPPIST-1 e'
        if self._eHoldEnd() < linear <= self._fHoldEnd():
            return 'TRAPPIST-1 f'
        return None

    def _trappistPlanetAnimationSpeed(self, frame: int) -> float:
        """Per-frame rate (for the accumulated motion clock only)."""
        linear = timelineProgress(frame, self.animationFrames)
        hzArrive = self._hzArrive()
        if linear < hzArrive:
            return ANIMATION_SPEED_TRAPPIST_PLANETS
        eArrive = self._eArrive()
        if linear >= eArrive:
            return ANIMATION_SPEED_TRAPPIST_PLANETS_CLOSE
        blend = smootherstep(segmentProgress(linear, hzArrive, eArrive))
        return float(
            (1.0 - blend) * ANIMATION_SPEED_TRAPPIST_PLANETS
            + blend * ANIMATION_SPEED_TRAPPIST_PLANETS_CLOSE
        )

    def _ensureTrappistMotionClock(self) -> None:
        """Accumulate zoom-dependent days so slowing orbits never reverse or jump."""
        if getattr(self, '_trappistMotionDaysByFrame', None) is not None:
            return
        days = np.zeros(self.animationFrames, dtype=float)
        accumulated = 0.0
        for frame in range(self.animationFrames):
            accumulated += self._trappistPlanetAnimationSpeed(frame)
            days[frame] = accumulated
        self._trappistMotionDaysByFrame = days

    def _trappistMotionDays(self, frame: int) -> float:
        self._ensureTrappistMotionClock()
        index = int(np.clip(frame, 0, self.animationFrames - 1))
        return float(self._trappistMotionDaysByFrame[index])

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
        # frame=1 + accumulated days ≡ continuous anomaly (see Sol _solMotionDays).
        offsetX, offsetY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.orbitalPeriodDays,
            planet.argumentPeriapsisDeg,
            0.0,
            1,
            self._trappistMotionDays(frame),
        )
        return self.hostSolAu + np.array([offsetX, offsetY, 0.0], dtype=float)

    def _trappistPortraitCameraState(self, frame: int, linear: float) -> tuple[np.ndarray, float]:
        """HZ overview → e portrait → pan to f → pull back to the full chain."""
        host = self.hostSolAu
        hzHoldEnd = self._hzHoldEnd()
        eArrive = self._eArrive()
        eHoldEnd = self._eHoldEnd()
        fArrive = self._fArrive()
        fHoldEnd = self._fHoldEnd()
        innerArrive = self._proximaInnerArrive()
        focusE = self._planetFocusSol('TRAPPIST-1 e', frame)
        focusF = self._planetFocusSol('TRAPPIST-1 f', frame)

        if linear <= eArrive:
            blend = smootherstep(segmentProgress(linear, hzHoldEnd, eArrive))
            focus = (1.0 - blend) * host + blend * focusE
            halfWidth = stagedLogDive(
                TRAPPIST_HZ_HALF_AU,
                TRAPPIST_PLANET_HALF_AU,
                TRAPPIST_PLANET_DIVE_WAYPOINTS_AU,
                blend,
            )
            return focus, halfWidth
        if linear <= eHoldEnd:
            return focusE.copy(), TRAPPIST_PLANET_HALF_AU
        if linear <= fArrive:
            blend = smootherstep(segmentProgress(linear, eHoldEnd, fArrive))
            return (1.0 - blend) * focusE + blend * focusF, TRAPPIST_PLANET_HALF_AU
        if linear <= fHoldEnd:
            return focusF.copy(), TRAPPIST_PLANET_HALF_AU
        if linear <= innerArrive:
            blend = smootherstep(segmentProgress(linear, fHoldEnd, innerArrive))
            focus = (1.0 - blend) * focusF + blend * host
            return focus, logLerp(TRAPPIST_PLANET_HALF_AU, TRAPPIST_INNER_HALF_AU, blend)
        return host.copy(), self.proximaInnerHalfWidthAu

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
        return self._trappistPortraitCameraState(frame, linear)

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

        portraitHero = self._portraitHeroName(frame)
        planetPortrait = portraitHero is not None or halfWidthAu <= TRAPPIST_PLANET_HALF_AU * 1.05
        hzFocus = (
            planetPortrait
            or self._hzArrive() <= linear <= self._hzHoldEnd()
            or abs(halfWidthAu - TRAPPIST_HZ_HALF_AU) < 1e-3
        )
        # HZ band is a host-centered schematic — skip once the camera is on a single world.
        hostCentered = float(np.linalg.norm(getattr(self, '_viewFocus', host) - host)) < (
            halfWidthAu * 0.55
        )
        if halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.15 and hostCentered and not planetPortrait:
            self._drawHabitableZoneBand(halfWidthAu, hzFocus=hzFocus)

        # Screen-fixed host marker: keep it dominant over the smaller planet disks
        # on the first chain reveal (no TRAPPIST star spin pack yet).
        if halfWidthAu <= TRAPPIST_PLANET_HALF_AU * 1.15:
            starSize = 280.0
        elif halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.15:
            starSize = 560.0
        elif halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.15:
            starSize = 720.0
        else:
            starSize = 420.0
        self._drawStarMarker(
            host,
            TRAPPIST_1_STAR_COLOR,
            starSize,
            zorder=self._scatterDepthZorder(host, base=5),
        )
        if not planetPortrait or hostCentered:
            self._label3d(
                host,
                '  TRAPPIST-1',
                color=self.labelColor,
                fontsize=11.0 if halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.05 else 10.0,
            )

        for planet in self.trappistPlanets:
            self._drawOneTrappistPlanet(
                planet,
                frame,
                halfWidthAu,
                hzFocus=hzFocus,
                heroName=portraitHero,
            )

    def _habitableZoneColor(self) -> str:
        return TRAPPIST_HZ_COLOR_DARK if self.isDark else TRAPPIST_HZ_COLOR_LIGHT

    def _drawHabitableZoneBand(self, halfWidthAu: float, *, hzFocus: bool) -> None:
        """Filled schematic HZ annulus — readable at wide + HZ camera scales."""
        host = self.hostSolAu
        color = self._habitableZoneColor()
        theta = np.linspace(0.0, 2.0 * np.pi, 96)
        cosT = np.cos(theta)
        sinT = np.sin(theta)
        outer = host + np.column_stack(
            (TRAPPIST_HZ_OUTER_AU * cosT, TRAPPIST_HZ_OUTER_AU * sinT, np.zeros_like(theta))
        )
        inner = host + np.column_stack(
            (TRAPPIST_HZ_INNER_AU * cosT, TRAPPIST_HZ_INNER_AU * sinT, np.zeros_like(theta))
        )
        # Closed annulus polygon (outer ring, then reverse inner).
        verts = np.vstack((outer, inner[::-1]))
        faceAlpha = 0.34 if hzFocus else (0.26 if self.isDark else 0.30)
        collection = Poly3DCollection(
            [verts],
            facecolors=(*_hexToRgb(color), faceAlpha),
            edgecolors=(*_hexToRgb(color), 0.85 if hzFocus else 0.7),
            linewidths=2.0 if hzFocus else 1.5,
        )
        collection.set_zorder(2)
        self.axes.add_collection3d(collection)

        # Bold rims so the band reads even when 3D foreshortening flattens the fill.
        rimWidth = 3.4 if hzFocus else 2.6
        rimAlpha = 0.95 if hzFocus else 0.8
        for radius in (TRAPPIST_HZ_INNER_AU, TRAPPIST_HZ_OUTER_AU):
            rim = host + np.column_stack((radius * cosT, radius * sinT, np.zeros_like(theta)))
            self.axes.plot(
                rim[:, 0],
                rim[:, 1],
                rim[:, 2],
                color=color,
                linewidth=rimWidth,
                alpha=rimAlpha,
                zorder=3,
                linestyle='--',
            )

        label = (
            'Habitable zone (schematic) · e then f'
            if halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.15
            else 'Habitable zone (schematic)'
        )
        self.bodyOverlay.text(
            0.50,
            0.11,
            label,
            ha='center',
            va='center',
            color=color,
            fontsize=12 if hzFocus else 11,
            fontweight='bold',
            alpha=0.95,
            zorder=25,
            bbox={
                'boxstyle': 'round,pad=0.28',
                'facecolor': '#101010' if self.isDark else '#F7F7F7',
                'edgecolor': color,
                'linewidth': 1.4,
                'alpha': 0.82 if self.isDark else 0.88,
            },
        )

    def _paintTrappistPlanetDisk(
        self,
        planet: SystemPlanet,
        position: np.ndarray,
        frame: int,
        halfWidthAu: float,
        *,
        hzFocus: bool,
        isHero: bool,
        isHzCandidate: bool,
    ) -> None:
        shortName = planet.name.replace('TRAPPIST-1 ', '')
        if isHero:
            labelSize = 12.0
        elif hzFocus and isHzCandidate:
            labelSize = 10.0
        elif hzFocus:
            labelSize = 7.0
        else:
            labelSize = 9.0
        bodyScale = BLENDER_PLANET_BODY_SCALE.get(planet.name)
        if bodyScale is not None and self.useBlenderBodies:
            wantBillboard = (
                isHero
                or not hzFocus
                or isHzCandidate
                or halfWidthAu >= TRAPPIST_INNER_HALF_AU * 0.9
            )
            if not wantBillboard:
                return
            paintScale = bodyScale * (TRAPPIST_PLANET_HERO_SCALE if isHero else 1.0)
            queued = self._queueBlenderBody(
                planet.name,
                position,
                frame,
                halfWidthAu,
                openCloseup=isHero
                or halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.25
                or halfWidthAu <= TRAPPIST_PLANET_HALF_AU * 1.35
                or halfWidthAu <= TRAPPIST_INNER_HALF_AU * 1.05,
                bodyScale=paintScale,
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
            if queued:
                self._pendingBlenderLabels.append(
                    (planet.name, position.copy(), labelSize, paintScale)
                )
                return
            if self._blenderBodyAvailable(planet.name):
                return

        baseSize = 72.0 if isHero else (56.0 if (hzFocus and isHzCandidate) else 48.0)
        zoomBoost = np.clip(TRAPPIST_WIDE_HALF_AU / max(halfWidthAu, 1e-6), 1.0, 12.0)
        markerSize = baseSize * (zoomBoost**0.65)
        emphasize = isHero or not hzFocus or isHzCandidate
        self.axes.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=planet.color,
            s=markerSize,
            alpha=1.0 if emphasize else 0.55,
            depthshade=False,
            zorder=self._scatterDepthZorder(position, base=4),
        )
        self._label3d(
            position,
            f'  {shortName}',
            color=self.labelColor,
            fontsize=int(labelSize),
            alpha=0.95 if emphasize else 0.55,
        )

    def _drawOneTrappistPlanet(
        self,
        planet: SystemPlanet,
        frame: int,
        halfWidthAu: float,
        *,
        hzFocus: bool = False,
        heroName: str | None = None,
    ) -> None:
        isHzCandidate = planet.name in TRAPPIST_HZ_FOCUS_NAMES
        isHero = heroName is not None and planet.name == heroName
        position = self._trappistPlanetPositionSol(planet, frame)
        focus = getattr(self, '_viewFocus', self.hostSolAu)
        distToFocus = float(np.linalg.norm(position - focus))
        inHostFrame = planet.semiMajorAxisAu <= halfWidthAu * 1.15
        nearLookAt = distToFocus <= halfWidthAu * 1.25
        if not inHostFrame and not nearLookAt and not isHero:
            return

        drawOrbit = halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 3.0 and (
            inHostFrame or isHero or nearLookAt
        )
        pathX, pathY = self.trappistPlanetPathsLocal[planet.planetId]
        pathLocal = np.column_stack((pathX, pathY, np.zeros_like(pathX)))
        pathSol = pathLocal + self.hostSolAu
        if drawOrbit:
            orbitAlpha = 0.95 if isHero else (0.35 if heroName else (0.85 if not hzFocus else 0.5))
            self.axes.plot(
                pathSol[:, 0],
                pathSol[:, 1],
                pathSol[:, 2],
                color=planet.color,
                linewidth=2.4 if isHero else 1.5,
                alpha=orbitAlpha,
            )
        if halfWidthAu > TRAPPIST_WIDE_HALF_AU * 2.2:
            return
        if not self._inView(position, margin=1.25):
            return
        # During a single-planet portrait, only texture the hero — companions stay orbits.
        if heroName is not None and not isHero:
            return
        self._paintTrappistPlanetDisk(
            planet,
            position,
            frame,
            halfWidthAu,
            hzFocus=hzFocus,
            isHero=isHero,
            isHzCandidate=isHzCandidate,
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
                'TRAPPIST-1 habitable zone',
                'Schematic temperate belt · next: close-ups of e, then f',
            )
        if linear < self._eHoldEnd():
            return (
                'TRAPPIST-1 e',
                'Temperate-zone candidate · strongest HZ case in the chain',
            )
        if linear < self._fHoldEnd():
            return (
                'TRAPPIST-1 f',
                'Outer temperate-zone world · next further from the dwarf',
            )
        if linear < self._proximaInnerArrive():
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
        if linear < self._fHoldEnd() and halfWidthAu <= TRAPPIST_WIDE_HALF_AU * 1.05:
            if linear >= self._eHoldEnd():
                return (
                    'TRAPPIST-1 f',
                    'Outer temperate-zone world · next further from the dwarf',
                )
            if linear >= self._eArrive():
                return (
                    'TRAPPIST-1 e',
                    'Temperate-zone candidate · strongest HZ case in the chain',
                )
            if halfWidthAu <= TRAPPIST_HZ_HALF_AU * 1.1:
                return (
                    'TRAPPIST-1 habitable zone',
                    'Schematic temperate belt · next: close-ups of e, then f',
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
        elif linear < self._proximaWideHoldEnd():
            elev, azim = CAMERA_ELEVATION_DEG, self.travelAzimuthDeg
        elif linear < self._hzHoldEnd():
            # Tilt up into the HZ linger so the filled annulus stays readable.
            blend = segmentProgress(linear, self._proximaWideHoldEnd(), self._hzArrive())
            elev = CAMERA_ELEVATION_DEG + blend * (TRAPPIST_HZ_ELEVATION_DEG - CAMERA_ELEVATION_DEG)
            azim = self.travelAzimuthDeg
        elif linear < self._fHoldEnd():
            blend = segmentProgress(linear, self._hzHoldEnd(), self._eArrive())
            elev = TRAPPIST_HZ_ELEVATION_DEG + blend * (
                TRAPPIST_PLANET_ELEVATION_DEG - TRAPPIST_HZ_ELEVATION_DEG
            )
            azim = self.travelAzimuthDeg
        else:
            blend = segmentProgress(linear, self._fHoldEnd(), self._proximaInnerArrive())
            elev = TRAPPIST_PLANET_ELEVATION_DEG + blend * (
                TRAPPIST_ELEVATION_DEG - TRAPPIST_PLANET_ELEVATION_DEG
            )
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
