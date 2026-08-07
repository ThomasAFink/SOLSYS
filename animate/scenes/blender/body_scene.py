"""Catalog → Blender body-scene JSON (host-side, no bpy)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.catalogs.moon_catalog import MoonCatalog, MoonOrbit
from solsys.physics.catalogs.planet_catalog import PlanetCatalog, PlanetOrbit
from solsys.physics.orbit_calculator import OrbitCalculator

SCHEMA_ID = 'solsys.blender_body_scene/v1'

# Matplotlib / catalog color names → RGBA for Blender materials.
_COLOR_RGBA: dict[str, tuple[float, float, float, float]] = {
    'gray': (0.55, 0.55, 0.55, 1.0),
    'lightgray': (0.75, 0.75, 0.75, 1.0),
    'darkgray': (0.35, 0.35, 0.35, 1.0),
    'yellow': (0.95, 0.85, 0.2, 1.0),
    'blue': (0.25, 0.45, 0.95, 1.0),
    'red': (0.9, 0.25, 0.2, 1.0),
    'orange': (0.95, 0.55, 0.15, 1.0),
    'gold': (0.9, 0.75, 0.25, 1.0),
    'lightblue': (0.55, 0.8, 0.95, 1.0),
    'brown': (0.55, 0.35, 0.2, 1.0),
    'tan': (0.82, 0.71, 0.55, 1.0),
    'wheat': (0.90, 0.85, 0.70, 1.0),
    'silver': (0.75, 0.75, 0.78, 1.0),
    'whitesmoke': (0.90, 0.90, 0.90, 1.0),
    'white': (0.95, 0.95, 0.95, 1.0),
    'gainsboro': (0.86, 0.86, 0.86, 1.0),
}
# Physical planet radii are ~1e-5 AU; exaggerate so a UV sphere is visible near the body.
_DISPLAY_RADIUS_EXAGGERATION = 800.0
_MIN_DISPLAY_RADIUS_AU = 0.002


@dataclass(frozen=True)
class BodySceneBody:
    name: str
    kind: str
    systemId: str
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    orbitalPeriodDays: float
    diameterKm: int
    color: str
    colorRgba: tuple[float, float, float, float]
    displayRadiusAu: float


@dataclass(frozen=True)
class BodyKeyframe:
    frame: int
    day: float
    positionAu: tuple[float, float, float]


@dataclass(frozen=True)
class BodyScene:
    schema: str
    body: BodySceneBody
    keyframes: tuple[BodyKeyframe, ...]
    cameraHintDistanceAu: float

    def toDict(self) -> dict[str, Any]:
        payload = asdict(self)
        # Nested tuples → lists for JSON.
        payload['body']['colorRgba'] = list(self.body.colorRgba)
        payload['keyframes'] = [
            {
                'frame': keyframe.frame,
                'day': keyframe.day,
                'positionAu': list(keyframe.positionAu),
            }
            for keyframe in self.keyframes
        ]
        return payload

    def toJson(self, *, indent: int = 2) -> str:
        return json.dumps(self.toDict(), indent=indent) + '\n'

    @classmethod
    def fromDict(cls, payload: dict[str, Any]) -> BodyScene:
        if payload.get('schema') != SCHEMA_ID:
            raise ValueError(
                f'Unsupported body scene schema: {payload.get("schema")!r} (expected {SCHEMA_ID!r})'
            )
        bodyPayload = payload['body']
        body = BodySceneBody(
            name=str(bodyPayload['name']),
            kind=str(bodyPayload['kind']),
            systemId=str(bodyPayload['systemId']),
            semiMajorAxisAu=float(bodyPayload['semiMajorAxisAu']),
            eccentricity=float(bodyPayload['eccentricity']),
            inclinationDeg=float(bodyPayload['inclinationDeg']),
            orbitalPeriodDays=float(bodyPayload['orbitalPeriodDays']),
            diameterKm=int(bodyPayload['diameterKm']),
            color=str(bodyPayload['color']),
            colorRgba=tuple(float(channel) for channel in bodyPayload['colorRgba']),  # type: ignore[arg-type]
            displayRadiusAu=float(bodyPayload['displayRadiusAu']),
        )
        keyframes = tuple(
            BodyKeyframe(
                frame=int(item['frame']),
                day=float(item['day']),
                positionAu=(
                    float(item['positionAu'][0]),
                    float(item['positionAu'][1]),
                    float(item['positionAu'][2]),
                ),
            )
            for item in payload['keyframes']
        )
        return cls(
            schema=str(payload['schema']),
            body=body,
            keyframes=keyframes,
            cameraHintDistanceAu=float(payload['cameraHintDistanceAu']),
        )


def loadBodyScene(path: Path | str) -> BodyScene:
    payload = json.loads(Path(path).read_text(encoding='utf-8'))
    return BodyScene.fromDict(payload)


def displayRadiusAu(diameterKm: int, constants: AstronomicalConstants) -> float:
    physicalRadiusAu = (diameterKm / 2.0) / constants.auToKm
    return max(physicalRadiusAu * _DISPLAY_RADIUS_EXAGGERATION, _MIN_DISPLAY_RADIUS_AU)


def colorRgbaForName(colorName: str) -> tuple[float, float, float, float]:
    return _COLOR_RGBA.get(colorName.lower(), (0.7, 0.7, 0.7, 1.0))


def _bodyFromPlanet(planet: PlanetOrbit, constants: AstronomicalConstants) -> BodySceneBody:
    return BodySceneBody(
        name=planet.name,
        kind='planet',
        systemId='sol',
        semiMajorAxisAu=planet.semiMajorAxisAu,
        eccentricity=planet.eccentricity,
        inclinationDeg=planet.inclinationDeg,
        orbitalPeriodDays=planet.orbitalPeriodDays,
        diameterKm=planet.diameterKm,
        color=planet.color,
        colorRgba=colorRgbaForName(planet.color),
        displayRadiusAu=displayRadiusAu(planet.diameterKm, constants),
    )


def _bodyFromMoon(
    moon: MoonOrbit,
    constants: AstronomicalConstants,
    moonCatalog: MoonCatalog,
) -> BodySceneBody:
    return BodySceneBody(
        name=moon.name,
        kind='moon',
        systemId='sol',
        semiMajorAxisAu=moonCatalog.semiMajorAxisAu(moon),
        eccentricity=0.0,
        inclinationDeg=moon.inclinationDeg,
        orbitalPeriodDays=moon.orbitalPeriodDays,
        diameterKm=moon.diameterKm,
        color=moon.color,
        colorRgba=colorRgbaForName(moon.color),
        displayRadiusAu=displayRadiusAu(moon.diameterKm, constants),
    )


def _orbitKeyframes(
    *,
    semiMajorAxisAu: float,
    eccentricity: float,
    inclinationDeg: float,
    orbitalPeriodDays: float,
    frameCount: int,
) -> tuple[BodyKeyframe, ...]:
    calculator = OrbitCalculator()
    trueAnomalies = np.linspace(0.0, 2.0 * np.pi, frameCount, endpoint=False)
    daysPerFrame = orbitalPeriodDays / frameCount
    keyframes: list[BodyKeyframe] = []
    for frame, trueAnomaly in enumerate(trueAnomalies):
        positionX, positionY, positionZ = calculator.ellipticalPosition(
            semiMajorAxisAu,
            eccentricity,
            inclinationDeg,
            float(trueAnomaly),
        )
        keyframes.append(
            BodyKeyframe(
                frame=frame,
                day=frame * daysPerFrame,
                positionAu=(float(positionX), float(positionY), float(positionZ)),
            )
        )
    return tuple(keyframes)


def buildPlanetBodyScene(
    planetName: str = 'Earth',
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible scene for one Sol planet from ``PlanetCatalog``."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')

    constants = constants or AstronomicalConstants()
    catalog = PlanetCatalog(constants)
    try:
        planet = catalog.planets[planetName]
    except KeyError as error:
        known = ', '.join(sorted(catalog.planets))
        raise ValueError(f'Unknown planet {planetName!r}. Known: {known}') from error

    body = _bodyFromPlanet(planet, constants)
    keyframes = _orbitKeyframes(
        semiMajorAxisAu=planet.semiMajorAxisAu,
        eccentricity=planet.eccentricity,
        inclinationDeg=planet.inclinationDeg,
        orbitalPeriodDays=planet.orbitalPeriodDays,
        frameCount=frameCount,
    )
    cameraHintDistanceAu = max(body.displayRadiusAu * 8.0, 0.05)
    return BodyScene(
        schema=SCHEMA_ID,
        body=body,
        keyframes=keyframes,
        cameraHintDistanceAu=cameraHintDistanceAu,
    )


def buildMoonBodyScene(
    moonName: str = 'Moon',
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible scene for one Sol moon from ``MoonCatalog``."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')

    constants = constants or AstronomicalConstants()
    catalog = MoonCatalog()
    try:
        moon = catalog.moons[moonName]
    except KeyError as error:
        known = ', '.join(sorted(catalog.moons))
        raise ValueError(f'Unknown moon {moonName!r}. Known: {known}') from error

    body = _bodyFromMoon(moon, constants, catalog)
    keyframes = _orbitKeyframes(
        semiMajorAxisAu=body.semiMajorAxisAu,
        eccentricity=body.eccentricity,
        inclinationDeg=body.inclinationDeg,
        orbitalPeriodDays=body.orbitalPeriodDays,
        frameCount=frameCount,
    )
    cameraHintDistanceAu = max(body.displayRadiusAu * 8.0, 0.05)
    return BodyScene(
        schema=SCHEMA_ID,
        body=body,
        keyframes=keyframes,
        cameraHintDistanceAu=cameraHintDistanceAu,
    )


def buildBodyScene(
    bodyName: str = 'Earth',
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a body scene from PlanetCatalog or MoonCatalog by name."""
    constants = constants or AstronomicalConstants()
    planets = PlanetCatalog(constants).planets
    moons = MoonCatalog().moons
    if bodyName in planets:
        return buildPlanetBodyScene(bodyName, frameCount=frameCount, constants=constants)
    if bodyName in moons:
        return buildMoonBodyScene(bodyName, frameCount=frameCount, constants=constants)
    knownPlanets = ', '.join(sorted(planets))
    knownMoons = ', '.join(sorted(moons))
    raise ValueError(
        f'Unknown body {bodyName!r}. Known planets: {knownPlanets}. Known moons: {knownMoons}'
    )
