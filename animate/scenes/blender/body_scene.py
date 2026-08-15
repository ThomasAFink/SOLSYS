"""Catalog → Blender body-scene JSON (host-side, no bpy)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.catalogs.famous_asteroid_catalog import (
    FamousAsteroidCatalog,
    FamousAsteroidOrbit,
)
from solsys.physics.catalogs.moon_catalog import MoonCatalog, MoonOrbit
from solsys.physics.catalogs.planet_catalog import PlanetCatalog, PlanetOrbit
from solsys.physics.catalogs.system_catalog import SystemCatalog, SystemPlanet
from solsys.physics.orbit_calculator import OrbitCalculator

# FamousAsteroidCatalog uses hex colors; planets/moons use matplotlib names.
_DWARF_PLANET_NAMES = frozenset({'Ceres', 'Eris', 'Makemake', 'Haumea'})

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
    diameterKm: float
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
            diameterKm=float(bodyPayload['diameterKm']),
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


def displayRadiusAu(diameterKm: float, constants: AstronomicalConstants) -> float:
    physicalRadiusAu = (float(diameterKm) / 2.0) / constants.auToKm
    return max(physicalRadiusAu * _DISPLAY_RADIUS_EXAGGERATION, _MIN_DISPLAY_RADIUS_AU)


def colorRgbaForName(colorName: str) -> tuple[float, float, float, float]:
    """Matplotlib color name or ``#RRGGBB`` / ``#RGB`` hex (asteroid catalog)."""
    named = _COLOR_RGBA.get(colorName.lower())
    if named is not None:
        return named
    text = colorName.strip()
    if text.startswith('#') and len(text) in (4, 7):
        hexDigits = text[1:]
        if len(hexDigits) == 3:
            hexDigits = ''.join(channel * 2 for channel in hexDigits)
        try:
            red = int(hexDigits[0:2], 16) / 255.0
            green = int(hexDigits[2:4], 16) / 255.0
            blue = int(hexDigits[4:6], 16) / 255.0
            return (red, green, blue, 1.0)
        except ValueError:
            pass
    return (0.7, 0.7, 0.7, 1.0)


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


def _bodyFromAsteroid(
    asteroid: FamousAsteroidOrbit,
    constants: AstronomicalConstants,
) -> BodySceneBody:
    kind = 'dwarf_planet' if asteroid.name in _DWARF_PLANET_NAMES else 'asteroid'
    return BodySceneBody(
        name=asteroid.name,
        kind=kind,
        systemId='sol',
        semiMajorAxisAu=asteroid.semiMajorAxisAu,
        eccentricity=asteroid.eccentricity,
        inclinationDeg=asteroid.inclinationDeg,
        orbitalPeriodDays=asteroid.orbitalPeriodDays,
        diameterKm=float(asteroid.diameterKm),
        color=asteroid.color,
        colorRgba=colorRgbaForName(asteroid.color),
        displayRadiusAu=displayRadiusAu(asteroid.diameterKm, constants),
    )


def buildAsteroidBodyScene(
    asteroidName: str = 'Ceres',
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible scene from ``FamousAsteroidCatalog``."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')

    constants = constants or AstronomicalConstants()
    catalog = FamousAsteroidCatalog()
    try:
        asteroid = catalog.asteroids[asteroidName]
    except KeyError as error:
        known = ', '.join(sorted(catalog.asteroids))
        raise ValueError(f'Unknown asteroid {asteroidName!r}. Known: {known}') from error

    body = _bodyFromAsteroid(asteroid, constants)
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


@dataclass(frozen=True)
class _StarSpec:
    """Fixed-origin star for Blender close-ups (not in PlanetCatalog)."""

    name: str
    diameterKm: float
    color: str
    periodDays: float  # approximate equatorial rotation (spin metadata only)
    systemId: str


# Sol + α Cen companions + Tabby's + TRAPPIST-1.
# Diameters ≈ catalog R★ × IAU nominal solar diameter.
_STAR_SPECS: dict[str, _StarSpec] = {
    'Sun': _StarSpec('Sun', 1_392_700.0, 'gold', 25.0, 'sol'),
    'Alpha Centauri A': _StarSpec(
        'Alpha Centauri A', 1_701_000.0, '#F6D56A', 22.0, 'alpha_centauri'
    ),
    'Alpha Centauri B': _StarSpec(
        'Alpha Centauri B', 1_201_000.0, '#E8A050', 41.0, 'alpha_centauri'
    ),
    'Proxima Centauri': _StarSpec('Proxima Centauri', 214_500.0, '#E07060', 83.0, 'alpha_centauri'),
    # F3V · R★ ≈ 1.43 R☉ (nearby_stars_30); ~0.88 d rotation → gallery spin uses 5 d schema.
    "Tabby's Star": _StarSpec("Tabby's Star", 1_991_600.0, '#F8F0D8', 5.0, 'tabbys_star'),
    'KIC 8462852': _StarSpec("Tabby's Star", 1_991_600.0, '#F8F0D8', 5.0, 'tabbys_star'),
    # M8V ultracool dwarf · R★ ≈ 0.119 R☉; ~3.3 d rotation (Spitzer/K2 spot period).
    'TRAPPIST-1': _StarSpec('TRAPPIST-1', 166_000.0, '#FF6B4A', 3.3, 'trappist_1'),
}


def _fixedOriginKeyframes(frameCount: int, *, periodDays: float = 25.0) -> tuple[BodyKeyframe, ...]:
    """Stationary body at the origin; day advances for spin metadata only."""
    daysPerFrame = periodDays / frameCount
    return tuple(
        BodyKeyframe(frame=frame, day=frame * daysPerFrame, positionAu=(0.0, 0.0, 0.0))
        for frame in range(frameCount)
    )


def buildStarBodyScene(
    bodyName: str,
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible star scene (emissive; body-centered at origin)."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')
    spec = _STAR_SPECS.get(bodyName)
    if spec is None:
        known = ', '.join(sorted(_STAR_SPECS))
        raise ValueError(f'Unknown star {bodyName!r}. Known stars: {known}')

    constants = constants or AstronomicalConstants()
    body = BodySceneBody(
        name=spec.name,
        kind='star',
        systemId=spec.systemId,
        semiMajorAxisAu=0.0,
        eccentricity=0.0,
        inclinationDeg=0.0,
        orbitalPeriodDays=spec.periodDays,
        diameterKm=spec.diameterKm,
        color=spec.color,
        colorRgba=colorRgbaForName(spec.color),
        displayRadiusAu=displayRadiusAu(spec.diameterKm, constants),
    )
    keyframes = _fixedOriginKeyframes(frameCount, periodDays=body.orbitalPeriodDays)
    cameraHintDistanceAu = max(body.displayRadiusAu * 8.0, 0.05)
    return BodyScene(
        schema=SCHEMA_ID,
        body=body,
        keyframes=keyframes,
        cameraHintDistanceAu=cameraHintDistanceAu,
    )


def buildSunBodyScene(
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible Sol scene (emissive star; body-centered at origin)."""
    return buildStarBodyScene('Sun', frameCount=frameCount, constants=constants)


def _systemPlanetByName(bodyName: str) -> SystemPlanet | None:
    """Look up an exoplanet by display name across ``SystemCatalog``."""
    catalog = SystemCatalog()
    for systemId in catalog.listSystemIds():
        system = catalog.load(systemId)
        for planet in system.planets:
            if planet.name == bodyName:
                return planet
    return None


def buildSystemPlanetBodyScene(
    bodyName: str,
    *,
    frameCount: int = 120,
    constants: AstronomicalConstants | None = None,
) -> BodyScene:
    """Build a Blender-ingestible scene for a ``SystemCatalog`` planet (e.g. Proxima b)."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')
    planet = _systemPlanetByName(bodyName)
    if planet is None:
        raise ValueError(f'Unknown system planet {bodyName!r}')

    constants = constants or AstronomicalConstants()
    body = BodySceneBody(
        name=planet.name,
        kind='planet',
        systemId=planet.systemId,
        semiMajorAxisAu=planet.semiMajorAxisAu,
        eccentricity=planet.eccentricity,
        inclinationDeg=planet.inclinationDeg,
        orbitalPeriodDays=planet.orbitalPeriodDays,
        diameterKm=planet.diameterKm,
        color=planet.color,
        colorRgba=colorRgbaForName(planet.color),
        displayRadiusAu=displayRadiusAu(planet.diameterKm, constants),
    )
    keyframes = _orbitKeyframes(
        semiMajorAxisAu=max(planet.semiMajorAxisAu, 0.02),
        eccentricity=planet.eccentricity,
        inclinationDeg=planet.inclinationDeg,
        orbitalPeriodDays=max(planet.orbitalPeriodDays, 1.0),
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
    """Build a body scene from planet / moon / asteroid catalogs, stars, or system planets."""
    constants = constants or AstronomicalConstants()
    if bodyName in _STAR_SPECS:
        return buildStarBodyScene(bodyName, frameCount=frameCount, constants=constants)
    planets = PlanetCatalog(constants).planets
    moons = MoonCatalog().moons
    asteroids = FamousAsteroidCatalog().asteroids
    if bodyName in planets:
        return buildPlanetBodyScene(bodyName, frameCount=frameCount, constants=constants)
    if bodyName in moons:
        return buildMoonBodyScene(bodyName, frameCount=frameCount, constants=constants)
    if bodyName in asteroids:
        return buildAsteroidBodyScene(bodyName, frameCount=frameCount, constants=constants)
    systemPlanet = _systemPlanetByName(bodyName)
    if systemPlanet is not None:
        return buildSystemPlanetBodyScene(bodyName, frameCount=frameCount, constants=constants)
    knownStars = ', '.join(sorted(_STAR_SPECS))
    knownPlanets = ', '.join(sorted(planets))
    knownMoons = ', '.join(sorted(moons))
    knownAsteroids = ', '.join(sorted(asteroids))
    raise ValueError(
        f'Unknown body {bodyName!r}. Known stars: {knownStars}. '
        f'Known planets: {knownPlanets}. Known moons: {knownMoons}. '
        f'Known asteroids: {knownAsteroids}. '
        f'Also: SystemCatalog exoplanets (e.g. "Proxima b", "TRAPPIST-1 b").'
    )
