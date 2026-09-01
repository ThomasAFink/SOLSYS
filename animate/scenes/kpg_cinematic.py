"""K–Pg cinema — scripted multi-shot Chicxulub cascade (#86 / #210).

Late Cretaceous Earth, a cinema-scale rock enters, hits the Yucatán, and a
drawn cascade (crust tear, ejecta, tsunami, wildfire, soot) plays out on that
globe. The map is an artist reconstruction, not a palaeomap. The rock is
enlarged; true scale is 10/12742. This is a cinema drawing, not a hydro model.

Geometry is derived at runtime from `data/chicxulub_kpg.csv` and the Earth
diameter in PlanetCatalog.
"""

from __future__ import annotations

import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from animate.scenes.blender.body_appearance import appearanceForCatalogName, jobTexturePath
from animate.scenes.blender.body_scene import buildPlanetBodyScene
from animate.scenes.blender.export_body import DEFAULT_OUTPUT_DIRECTORY, bodyOutputDirectory
from animate.scenes.blender.flyby_scene import assembleGifFromPngs, writeFlybyJob

DEFAULT_EVENT_CSV = 'data/chicxulub_kpg.csv'
JOB_SCHEMA_ID = 'solsys.blender_kpg_job/v1'
RENDER_SCRIPT = Path('animate/scenes/blender/render_kpg.py')
KPG_EARTH_PACK = Path('data/textures/bodies/earth_kpg')

ANIMATION_FPS = 20
ANIMATION_FRAMES = 480
RENDER_RESOLUTION = 960
GALLERY_SIZE = 640

INBOUND_START_RADII = 7.2
OBLIQUITY_FROM_VERTICAL_DEG = 40.0

QUIET_END = 40
APPROACH_END = 120
ENTRY_END = 170
IMPACT_FRAME = 170
CONTACT_END = 190
CRUST_END = 230
EJECTA_END = 290
TSUNAMI_END = 360
HOLD_END = IMPACT_FRAME
SLAM_FRAMES = 16
# Drawn rock is enlarged so the inbound slam reads. True scale stays in the CSV.
CINEMA_ROCK_RADII = 0.010
ASTEROID_TEXTURE = Path('data/textures/bodies/bennu/color.png')
ATMOSPHERE_RADII = 0.10
FIREBALL_MAX_RADII = 0.13
EJECTA_MAX_RADII = 0.36
PLUME_MAX_RADII = 0.48
EJECTA_ANGLE_DEG = 45.0
EJECTA_COUNT = 48
PROJECTILE_COUNT = 320
ROCK_BURST_COUNT = 26000
EMBER_BURST_COUNT = 11200
FALLOUT_SHELL = 0.72
CRUST_PLATE_COUNT = 6
ACT_STILL_FRAMES = (20, 80, 145, 175, 210, 255, 320, 430)

ACT_BOUNDARIES = (
    (QUIET_END, 'quiet'),
    (APPROACH_END, 'approach'),
    (ENTRY_END, 'entry'),
    (CONTACT_END, 'contact'),
    (CRUST_END, 'crust'),
    (EJECTA_END, 'ejecta'),
    (TSUNAMI_END, 'tsunami'),
)
# Leftovers for unused plate helpers. The GIF path does not cut to stills.
EXPLOSION_SEQUENCE = 16
FIRE_CUT_END = CONTACT_END
PLATE_STAMP = 'unused'
KPG_IMPACT_KEYS: tuple[str, ...] = ()
SMOKE_DOMAIN_RADII = 0.78
SMOKE_RESOLUTION = 88
SMOKE_INFLOW_FRAMES = 24


@dataclass(frozen=True)
class ImpactEvent:
    """Published Chicxulub numbers plus Earth from the planet catalogue."""

    name: str
    latitudeDeg: float
    longitudeDeg: float
    ageMa: float
    craterDiameterKm: float
    impactorDiameterKm: float
    speedKmS: float
    earthDiameterKm: float

    @property
    def earthRadiusKm(self) -> float:
        return self.earthDiameterKm / 2.0

    @property
    def impactorRadiusKm(self) -> float:
        return self.impactorDiameterKm / 2.0

    @property
    def radiusRatio(self) -> float:
        return self.impactorDiameterKm / self.earthDiameterKm


def smoothStep(progress: float) -> float:
    clamped = float(np.clip(progress, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def unitFromLatLon(latitudeDeg: float, longitudeDeg: float) -> np.ndarray:
    """Unit vector matching Blue Marble / flyby equirect: +X Greenwich, +Z north."""
    latitude = math.radians(float(latitudeDeg))
    longitude = math.radians(float(longitudeDeg))
    return np.array(
        [
            math.cos(latitude) * math.cos(longitude),
            math.cos(latitude) * math.sin(longitude),
            math.sin(latitude),
        ],
        dtype=float,
    )


def loadImpactEvent(csvPath: str | Path = DEFAULT_EVENT_CSV) -> ImpactEvent:
    frame = pd.read_csv(csvPath, comment='#')
    if frame.empty:
        raise ValueError(f'No impact event in {csvPath}')
    row = frame.iloc[0]
    earth = buildPlanetBodyScene('Earth', frameCount=2)
    return ImpactEvent(
        name=str(row['name']),
        latitudeDeg=float(row['latitude_deg']),
        longitudeDeg=float(row['longitude_deg']),
        ageMa=float(row['age_ma']),
        craterDiameterKm=float(row['crater_diameter_km']),
        impactorDiameterKm=float(row['impactor_diameter_km']),
        speedKmS=float(row['speed_km_s']),
        earthDiameterKm=float(earth.body.diameterKm),
    )


def impactNormal(event: ImpactEvent) -> np.ndarray:
    return unitFromLatLon(event.latitudeDeg, event.longitudeDeg)


def inboundDirection(event: ImpactEvent) -> np.ndarray:
    """Flight direction at contact: mostly inward, 40° off vertical toward local east."""
    normal = impactNormal(event)
    east, _north = tangentBasis(normal)
    tilt = math.radians(OBLIQUITY_FROM_VERTICAL_DEG)
    direction = -normal * math.cos(tilt) + east * math.sin(tilt)
    return direction / float(np.linalg.norm(direction))


def contactCenterRadii(event: ImpactEvent) -> np.ndarray:
    """Impactor centre at first contact, in Earth radii from Earth's centre."""
    return impactNormal(event) * (1.0 + event.radiusRatio)


def tangentBasis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    east = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    if float(np.linalg.norm(east)) < 1e-9:
        east = np.array([0.0, 1.0, 0.0])
    east = east / float(np.linalg.norm(east))
    north = np.cross(normal, east)
    return east, north / float(np.linalg.norm(north))


def ejectaDirections(
    normal: np.ndarray,
    inbound: np.ndarray | None = None,
    count: int = EJECTA_COUNT,
) -> np.ndarray:
    """Irregular spray out of the blast, lofted into space, not a ring."""
    east, north = tangentBasis(normal)
    if inbound is None:
        downrange = east
    else:
        incoming = np.asarray(inbound, dtype=float)
        tangent = incoming - normal * float(np.dot(incoming, normal))
        length = float(np.linalg.norm(tangent))
        downrange = -tangent / length if length > 1e-8 else east
    cross = np.cross(normal, downrange)
    cross = cross / float(np.linalg.norm(cross))
    rays = []
    for index in range(count):
        fan = ((index * 0.6180339887) % 1.0) * 2.0 - 1.0
        fan = math.copysign(abs(fan) ** 0.62, fan)
        azimuth = fan * math.radians(46.0) + 0.22 * math.sin(index * 4.1)
        mix = (index * 0.271 + 0.13) % 1.0
        tilt = math.radians(14.0 + 26.0 * (mix**0.75))
        heading = downrange * math.cos(azimuth) + cross * math.sin(azimuth)
        direction = normal * math.cos(tilt) + heading * math.sin(tilt)
        rays.append(direction / float(np.linalg.norm(direction)))
    return np.asarray(rays, dtype=float)


@dataclass(frozen=True)
class ProjectileLaunch:
    azimuth: float
    tiltDeg: float
    rangeDeg: float
    loftRadii: float
    flightSeconds: float
    delaySeconds: float


def rotateAroundAxis(vector: np.ndarray, axis: np.ndarray, angleRad: float) -> np.ndarray:
    unit = axis / float(np.linalg.norm(axis))
    cosine = math.cos(angleRad)
    return (
        vector * cosine
        + np.cross(unit, vector) * math.sin(angleRad)
        + unit * float(np.dot(unit, vector)) * (1.0 - cosine)
    )


def debrisRadiusScales(asteroidScale: float, count: int = PROJECTILE_COUNT) -> tuple[float, ...]:
    """Fragment radius / Earth radius. Always smaller than the inbound body."""
    return tuple(
        asteroidScale * (0.10 + 0.78 * (((index * 0.41) % 1.0) ** 1.55)) for index in range(count)
    )


def projectileLaunches(count: int = PROJECTILE_COUNT) -> tuple[ProjectileLaunch, ...]:
    """Most fall back nearby; a few go high or worldwide. Not a frozen spray."""
    launches: list[ProjectileLaunch] = []
    for index in range(count):
        spin = (index * 0.6180339887) % 1.0
        rangeMix = (index * 0.271 + 0.07) % 1.0
        rangeDeg = 6.0 + 155.0 * (rangeMix**1.55)
        loftMix = (index * 0.529 + 0.18) % 1.0
        loftRadii = 0.012 + 0.08 * loftMix if loftMix < 0.88 else 0.07 + 0.05 * loftMix
        launches.append(
            ProjectileLaunch(
                azimuth=2.0 * math.pi * spin + 0.85 * math.sin(index * 4.1),
                tiltDeg=22.0 + 48.0 * ((index * 0.415) % 1.0),
                rangeDeg=rangeDeg,
                loftRadii=loftRadii,
                flightSeconds=1.5 + 3.2 * (rangeDeg / 160.0),
                delaySeconds=1.3 * ((index * 0.6180339887 * 11.0) % 1.0),
            )
        )
    return tuple(launches)


def projectilePositionRadii(normal: np.ndarray, launch: ProjectileLaunch, frame: int) -> np.ndarray:
    if frame < IMPACT_FRAME:
        return normal
    age = (frame - IMPACT_FRAME) / ANIMATION_FPS - launch.delaySeconds
    if age <= 0.0:
        return normal
    progress = smoothStep(age / launch.flightSeconds)
    east, north = tangentBasis(normal)
    heading = east * math.cos(launch.azimuth) + north * math.sin(launch.azimuth)
    axis = np.cross(normal, heading)
    along = rotateAroundAxis(normal, axis, math.radians(launch.rangeDeg) * progress)
    height = 1.0 + launch.loftRadii * math.sin(math.pi * progress)
    return along * height


def projectileFlightProgress(launch: ProjectileLaunch, frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    age = (frame - IMPACT_FRAME) / ANIMATION_FPS - launch.delaySeconds
    if age <= 0.0:
        return 0.0
    return age / launch.flightSeconds


def projectileVisibility(launch: ProjectileLaunch, frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    age = (frame - IMPACT_FRAME) / ANIMATION_FPS - launch.delaySeconds
    return 1.0 if age > 0.0 else 0.0


def projectileDirectionRadii(
    normal: np.ndarray, launch: ProjectileLaunch, frame: int
) -> np.ndarray:
    here = projectilePositionRadii(normal, launch, frame)
    previous = projectilePositionRadii(normal, launch, max(frame - 1, 0))
    delta = here - previous
    length = float(np.linalg.norm(delta))
    if length < 1e-8:
        east, north = tangentBasis(normal)
        heading = east * math.cos(launch.azimuth) + north * math.sin(launch.azimuth)
        return heading / float(np.linalg.norm(heading))
    return delta / length


def projectileTrailScale(launch: ProjectileLaunch, frame: int) -> float:
    progress = projectileFlightProgress(launch, frame)
    visible = projectileVisibility(launch, frame)
    if visible < 1e-4 or progress >= 1.0:
        return 0.0
    return visible * (0.45 + 0.55 * math.sin(math.pi * min(progress, 0.999)))


def projectileStrikeScale(launch: ProjectileLaunch, frame: int) -> float:
    progress = projectileFlightProgress(launch, frame)
    if progress < 0.82:
        return 0.0
    return math.exp(-0.42 * abs(progress - 1.0))


def remainingRadiiAt(frame: int) -> float:
    if frame >= IMPACT_FRAME:
        return 0.0
    return INBOUND_START_RADII * (1.0 - smoothStep(frame / IMPACT_FRAME))


def inboundTrailScale(frame: int) -> float:
    if frame >= IMPACT_FRAME:
        return 0.0
    return entryHeatEnvelope(frame)


def entryHeatEnvelope(frame: int) -> float:
    if frame >= IMPACT_FRAME:
        return max(0.0, 1.0 - (frame - IMPACT_FRAME) / 14.0)
    remaining = remainingRadiiAt(frame)
    if remaining > ATMOSPHERE_RADII:
        return 0.0
    return smoothStep((ATMOSPHERE_RADII - remaining) / ATMOSPHERE_RADII)


def smolderEnvelope(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    return min(1.0, 0.62 + 0.38 * smoothStep((frame - IMPACT_FRAME) / 6.0))


def crustTearEnvelope(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    return smoothStep((frame - IMPACT_FRAME) / 8.0)


def tsunamiAngleRad(frame: int) -> float:
    # Hours across a basin, compressed. Keep growing past the far limb so
    # the last ring leaves the shot instead of parking on the globe.
    if frame < IMPACT_FRAME + 16:
        return 0.0
    age = frame - IMPACT_FRAME - 16
    return math.pi * 1.12 * min(1.0, age / 270.0)


def falloutEnvelope(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    return smoothStep((frame - IMPACT_FRAME) / 120.0)


def lightingEnvelope(frame: int, _frameCount: int) -> tuple[float, float]:
    veil = sootEnvelope(frame)
    return 1.0 - 0.55 * veil, veil


def shockAngleRad(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    # Starts around the cinema rock, then grows. Clouds stay put until this kicks.
    aroundRock = 0.09
    return aroundRock + (math.pi * 0.94 - aroundRock) * smoothStep((frame - IMPACT_FRAME) / 70.0)


def wildfireAngleRad(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    near = 0.40 * (0.75 + 0.25 * smoothStep((frame - IMPACT_FRAME) / 5.0))
    far = 1.05 * smoothStep((frame - IMPACT_FRAME) / 70.0)
    return near + far


def sootEnvelope(frame: int) -> float:
    # Days-to-years of soot, compressed. Hold until the site cloud has formed.
    if frame < IMPACT_FRAME + 36:
        return 0.0
    return 0.82 * smoothStep((frame - IMPACT_FRAME - 36) / 200.0)


def contactEnvelope(frame: int, _frameCount: int) -> tuple[float, float, float]:
    """Fireball, ejecta, and plume kick together at contact."""
    if frame < IMPACT_FRAME:
        return 0.0, 0.0, 0.0
    age = frame - IMPACT_FRAME
    fire = float(np.clip(0.92 * math.exp(-0.008 * age) + 0.12, 0.0, 1.0))
    fire *= 1.0 - smoothStep((age - 55.0) / 80.0)
    ejecta = float(np.clip(smoothStep(age / 4.0) * (1.0 - smoothStep((age - 70) / 70.0)), 0, 1))
    plume = float(np.clip(smoothStep(age / 6.0) * (1.0 - smoothStep((age - 36.0) / 70.0)), 0, 1))
    return fire, ejecta, plume


def flashEnvelope(frame: int) -> float:
    if frame < IMPACT_FRAME:
        return 0.0
    return float(np.clip(math.exp(-0.12 * (frame - IMPACT_FRAME)), 0.0, 1.0))


@dataclass(frozen=True)
class ImpactSample:
    frame: int
    cameraRadii: tuple[float, float, float]
    lookAtRadii: tuple[float, float, float]
    impactorRadii: tuple[float, float, float]
    sunScale: float
    flashScale: float
    fireballScale: float
    ejectaScale: float
    plumeScale: float
    falloutScale: float
    projectileRadii: tuple[tuple[float, float, float], ...]
    projectileScale: tuple[float, ...]
    projectileDir: tuple[tuple[float, float, float], ...]
    projectileTrail: tuple[float, ...]
    projectileStrike: tuple[float, ...]
    inboundTrail: float
    veil: float
    shockAngle: float
    wildfireAngle: float
    soot: float
    entryHeat: float
    crustTear: float
    tsunamiAngle: float
    slamScale: float
    rockHeat: float
    smolder: float
    inboundKm: float
    secondsToImpact: float
    lens: float


def actName(frame: int) -> str:
    for boundary, name in ACT_BOUNDARIES:
        if frame < boundary:
            return name
    return 'veil'


def _asTuple(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _cameraPoses() -> tuple[tuple[float, ...], ...]:
    # frame, east, lift, along-normal, look-at along normal, lens mm
    return (
        (0.0, 0.42, 0.95, 2.55, 0.00, 35.0),
        (float(QUIET_END), 0.40, 0.90, 2.48, 0.18, 36.0),
        (float(APPROACH_END), 0.32, 0.70, 2.08, 0.52, 38.0),
        (float(IMPACT_FRAME), 0.22, 0.52, 1.70, 0.72, 40.0),
        (float(CRUST_END), 0.30, 0.66, 2.00, 0.52, 37.0),
        (float(EJECTA_END), 0.38, 0.85, 2.35, 0.22, 36.0),
        (float(TSUNAMI_END), 0.45, 1.00, 2.65, 0.00, 35.0),
        (float(ANIMATION_FRAMES - 1), 0.48, 1.05, 2.80, 0.00, 34.0),
    )


def _blendPose(frame: int) -> tuple[float, float, float, float, float]:
    poses = _cameraPoses()
    if frame <= poses[0][0]:
        return poses[0][1:]
    if frame >= poses[-1][0]:
        return poses[-1][1:]
    for left, right in zip(poses, poses[1:], strict=True):
        if frame <= right[0]:
            span = max(right[0] - left[0], 1.0)
            linear = (frame - left[0]) / span
            # Ease-in-out on the inbound dive reads as a pause at contact.
            mix = linear if right[0] <= IMPACT_FRAME else smoothStep(linear)
            return tuple(a + mix * (b - a) for a, b in zip(left[1:], right[1:], strict=True))
    return poses[-1][1:]


def shotLens(frame: int) -> float:
    return _blendPose(frame)[4]


def _earthCamera(
    frame: int,
    _frameCount: int,
    normal: np.ndarray,
    east: np.ndarray,
    lift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    alongEast, alongLift, alongNormal, look, _lens = _blendPose(frame)
    camera = east * alongEast + lift * alongLift + normal * alongNormal
    lookAt = normal * look
    return camera, lookAt


def _rockHeat(frame: int) -> float:
    return entryHeatEnvelope(frame)


def slamScaleAt(frame: int) -> float:
    if frame < QUIET_END or frame >= IMPACT_FRAME:
        return 0.0
    return 1.0


def inboundKmAtFrame(event: ImpactEvent, frame: int) -> float:
    if frame >= IMPACT_FRAME:
        return 0.0
    progress = smoothStep(frame / IMPACT_FRAME)
    remainingRadii = INBOUND_START_RADII * (1.0 - progress)
    return remainingRadii * event.earthRadiusKm


def buildImpactSamples(
    event: ImpactEvent, frameCount: int = ANIMATION_FRAMES
) -> tuple[ImpactSample, ...]:
    normal = impactNormal(event)
    inbound = inboundDirection(event)
    contact = contactCenterRadii(event)
    east, _north = tangentBasis(normal)
    lift = np.cross(inbound, east)
    lift = lift / float(np.linalg.norm(lift))
    launches = projectileLaunches()

    samples: list[ImpactSample] = []
    for frame in range(frameCount):
        remainingKm = inboundKmAtFrame(event, frame)
        remainingRadii = remainingKm / event.earthRadiusKm
        impactor = contact - inbound * remainingRadii
        if frame > IMPACT_FRAME:
            punched = min((frame - IMPACT_FRAME) / SLAM_FRAMES, 1.0)
            impactor = contact + inbound * (CINEMA_ROCK_RADII * (0.20 + 0.90 * punched))
        camera, lookAt = _earthCamera(frame, frameCount, normal, east, lift)
        fireball, ejecta, plume = contactEnvelope(frame, frameCount)
        flash = flashEnvelope(frame)
        sunScale, veil = lightingEnvelope(frame, frameCount)
        samples.append(
            ImpactSample(
                frame=frame,
                cameraRadii=_asTuple(camera),
                lookAtRadii=_asTuple(lookAt),
                impactorRadii=_asTuple(impactor),
                sunScale=sunScale,
                flashScale=flash,
                fireballScale=fireball,
                ejectaScale=ejecta,
                plumeScale=plume,
                falloutScale=falloutEnvelope(frame),
                projectileRadii=tuple(
                    _asTuple(projectilePositionRadii(normal, launch, frame)) for launch in launches
                ),
                projectileScale=tuple(projectileVisibility(launch, frame) for launch in launches),
                projectileDir=tuple(
                    _asTuple(projectileDirectionRadii(normal, launch, frame)) for launch in launches
                ),
                projectileTrail=tuple(projectileTrailScale(launch, frame) for launch in launches),
                projectileStrike=tuple(projectileStrikeScale(launch, frame) for launch in launches),
                inboundTrail=inboundTrailScale(frame),
                veil=veil,
                shockAngle=shockAngleRad(frame),
                wildfireAngle=wildfireAngleRad(frame),
                soot=sootEnvelope(frame),
                entryHeat=entryHeatEnvelope(frame),
                crustTear=crustTearEnvelope(frame),
                tsunamiAngle=tsunamiAngleRad(frame),
                slamScale=slamScaleAt(frame),
                rockHeat=_rockHeat(frame),
                smolder=smolderEnvelope(frame),
                inboundKm=remainingKm,
                secondsToImpact=remainingKm / event.speedKmS,
                lens=shotLens(frame),
            )
        )
    return tuple(samples)


def titleForAct(act: str) -> str:
    titles = {
        'quiet': '66 million years ago',
        'approach': 'Inbound',
        'entry': 'Atmosphere',
        'contact': 'Contact',
        'crust': 'Crust',
        'ejecta': 'Ejecta',
        'tsunami': 'Tsunami',
        'veil': 'Aftermath',
    }
    return titles.get(act, 'Aftermath')


def captionForSample(event: ImpactEvent, sample: ImpactSample) -> str:
    act = actName(sample.frame)
    if act == 'quiet':
        return (
            f'{event.name}  {event.latitudeDeg:.1f}°N, {abs(event.longitudeDeg):.1f}°W  ·  '
            f'crater {event.craterDiameterKm:.0f} km  ·  artist reconstruction, not a palaeomap'
        )
    if act == 'approach':
        minutes = sample.secondsToImpact / 60.0
        return (
            f'{sample.inboundKm:,.0f} km out  ·  {minutes:.1f} min at {event.speedKmS:.0f} km/s  ·  '
            f'{event.impactorDiameterKm:.0f} km rock enlarged for the shot'
        )
    if act == 'entry':
        return (
            f'entering the atmosphere  ·  {event.impactorDiameterKm:.0f} km body at '
            f'{event.speedKmS:.0f} km/s  ·  rock enlarged'
        )
    if act == 'contact':
        return (
            f'hit at the Yucatán  ·  {event.impactorDiameterKm:.0f} km body at '
            f'{event.speedKmS:.0f} km/s'
        )
    if act == 'crust':
        return 'cinema tear at the site  ·  not a crater hydro model'
    if act == 'ejecta':
        return 'worldwide ejecta is observed  ·  hours compressed  ·  not a debris simulation'
    if act == 'tsunami':
        return 'drawn water rings  ·  hours compressed  ·  not a tsunami height model'
    return f'{event.ageMa:.0f} Ma  ·  worldwide fallout is observed · not a climate model'


def _tileNoise(size: int, seed: int, *, octaves: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    field = np.zeros((size, size), dtype=np.float64)
    weight = 0.0
    amp = 1.0
    cells = 5
    for _ in range(octaves):
        grid = rng.random((cells + 2, cells + 2))
        layer = Image.fromarray((grid * 255.0).astype(np.uint8)).resize(
            (size, size), Image.Resampling.BICUBIC
        )
        field += amp * (np.asarray(layer, dtype=np.float64) / 255.0)
        weight += amp
        amp *= 0.52
        cells *= 2
    return field / max(weight, 1e-9)


def rasterFireTile(size: int = 768) -> np.ndarray:
    """RGBA fire tile: hot ridges, transparent soot holes."""
    heat = np.abs(0.62 * _tileNoise(size, 11) + 0.38 * _tileNoise(size, 29) - 0.5)
    heat = np.clip(1.0 - heat * 2.0, 0.0, 1.0) ** 1.35
    heat *= 0.28 + 0.72 * np.linspace(1.0, 0.42, size)[:, None]
    red = np.clip(0.18 + 1.15 * heat, 0.0, 1.0)
    green = np.clip(0.02 + 0.72 * heat**1.7, 0.0, 1.0)
    blue = np.clip(0.01 + 0.10 * heat**3.2, 0.0, 1.0)
    alpha = np.clip((heat - 0.08) / 0.72, 0.0, 1.0) ** 0.65
    return np.clip(np.stack([red, green, blue, alpha], axis=-1), 0.0, 1.0)


def rasterSmokeTile(size: int = 768) -> np.ndarray:
    """RGBA smoke tile: brown-grey clumps with a few ember flecks."""
    density = (
        np.clip(
            0.55 * _tileNoise(size, 41, octaves=5) + 0.45 * _tileNoise(size, 67, octaves=4),
            0.0,
            1.0,
        )
        ** 0.85
    )
    ember = np.clip((density - 0.72) / 0.28, 0.0, 1.0)
    red = 0.10 + 0.10 * density + 0.55 * ember
    green = 0.07 + 0.07 * density + 0.18 * ember
    blue = 0.05 + 0.05 * density
    alpha = np.clip((density - 0.18) / 0.62, 0.0, 1.0) ** 0.8
    return np.clip(np.stack([red, green, blue, alpha], axis=-1), 0.0, 1.0)


def writeContactTextures(directory: Path | str, size: int = 768) -> dict[str, Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    tiles = (('fire', rasterFireTile(size)), ('smoke', rasterSmokeTile(size)))
    for name, pixels in tiles:
        path = root / f'kpg_{name}.png'
        Image.fromarray((pixels * 255.0).astype(np.uint8), mode='RGBA').save(path)
        written[name] = path
    return written


def rasterExplosionFrame(size: int, progress: float, *, seed: int = 3) -> np.ndarray:
    """One flipbook cell: white-hot core, fingered fire shell, outer haze."""
    time = float(np.clip(progress, 0.0, 1.0))
    axis = np.linspace(-1.0, 1.0, size)
    xx, yy = np.meshgrid(axis, axis)
    yy = yy + 0.14
    radius = np.hypot(xx, yy)
    angle = np.arctan2(yy, xx)
    warp = _tileNoise(size, seed + int(time * 19.0), octaves=4) * 2.0 - 1.0
    fingers = 0.70 + 0.30 * np.sin(angle * 6.0 + seed) + 0.20 * warp
    front = 0.09 + 0.80 * time
    shell = np.exp(-(((radius - front * fingers) / (0.09 + 0.08 * time)) ** 2))
    core = np.exp(-((radius / (0.06 + 0.18 * time)) ** 2))
    haze = np.exp(-((radius / (0.20 + 0.58 * time)) ** 2)) * (0.40 + 0.22 * (1.0 - time))
    heat = np.clip(1.85 * core + 1.20 * shell + 0.50 * haze * (0.5 + 0.5 * warp), 0.0, 1.0)
    heat *= np.clip(1.20 - 0.50 * time, 0.40, 1.0) * np.clip(1.25 - radius, 0.0, 1.0)
    white = np.clip(core * 1.4, 0.0, 1.0)
    red = np.clip(0.12 + 1.25 * heat + 0.70 * white, 0.0, 1.0)
    green = np.clip(0.03 + 0.52 * heat**1.45 + 0.80 * white, 0.0, 1.0)
    blue = np.clip(0.01 + 0.07 * heat**2.6 + 0.62 * white, 0.0, 1.0)
    alpha = np.clip(heat * 1.20, 0.0, 1.0) ** 0.70
    return np.clip(np.stack([red, green, blue, alpha], axis=-1), 0.0, 1.0)


def writeExplosionSequence(
    directory: Path | str, count: int = EXPLOSION_SEQUENCE, size: int = 640
) -> list[Path]:
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for index in range(count):
        path = root / f'kpg_x_{index:02d}.png'
        pixels = rasterExplosionFrame(size, index / max(count - 1, 1), seed=4 + index)
        Image.fromarray((pixels * 255.0).astype(np.uint8), mode='RGBA').save(path)
        written.append(path)
    return written


def _orangeScore(path: Path) -> float:
    sample = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    return float(sample[..., 0].mean() - sample[..., 2].mean())


def _lumaKeyedPlate(pixels: np.ndarray) -> np.ndarray:
    rgb = np.clip(pixels[..., :3], 0.0, 1.0)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    alpha = np.clip((luma - 0.04) / 0.16, 0.0, 1.0)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def _circularFalloff(alpha: np.ndarray) -> np.ndarray:
    height, width = alpha.shape
    ys = (np.arange(height) + 0.5) / height * 2.0 - 1.0
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    xx, yy = np.meshgrid(xs, ys)
    radius = np.hypot(xx, yy)
    return alpha * np.clip(1.22 - radius, 0.0, 1.0) ** 2.2


def _softenPlate(pixels: np.ndarray, radius: float = 1.7) -> np.ndarray:
    from PIL import ImageFilter

    image = Image.fromarray((np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
    image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    return np.asarray(image, dtype=np.float32) / 255.0


def _fitSquareRgba(pixels: np.ndarray, size: int = 1440) -> np.ndarray:
    image = Image.fromarray((np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
    fitted = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(fitted, dtype=np.float32) / 255.0


def _squareCropToFire(pixels: np.ndarray, margin: float = 0.10) -> np.ndarray:
    luma = 0.2126 * pixels[..., 0] + 0.7152 * pixels[..., 1] + 0.0722 * pixels[..., 2]
    mask = luma > 0.08
    if not np.any(mask):
        return pixels
    rows, cols = np.where(mask)
    height, width = luma.shape
    padY = int(height * margin)
    padX = int(width * margin)
    top = max(int(rows.min()) - padY, 0)
    bottom = min(int(rows.max()) + padY + 1, height)
    left = max(int(cols.min()) - padX, 0)
    right = min(int(cols.max()) + padX + 1, width)
    side = max(bottom - top, right - left)
    centerY = (top + bottom) // 2
    centerX = (left + right) // 2
    top = max(centerY - side // 2, 0)
    left = max(centerX - side // 2, 0)
    bottom = min(top + side, height)
    right = min(left + side, width)
    return pixels[top:bottom, left:right]


def _sustainFireSources(source: list[Path], needed: int) -> list[Path]:
    """Play the bake once, then ping-pong the later orange frames. Do not hold one peak."""
    if not source:
        return []
    first = list(source)
    sustain = source[max(len(source) * 2 // 3, 0) :] or source
    bounce = list(sustain) + list(reversed(sustain[1:-1] or sustain))
    rest = [bounce[index % len(bounce)] for index in range(max(needed - len(first), 0))]
    return (first + rest)[:needed]


def kpgEarthPack() -> dict[str, Path]:
    """Late Cretaceous artist maps. Fall back to the modern Earth pack if absent."""
    color = KPG_EARTH_PACK / 'color.png'
    clouds = KPG_EARTH_PACK / 'clouds.png'
    pack: dict[str, Path] = {}
    if color.is_file():
        pack['color'] = color
    if clouds.is_file():
        pack['clouds'] = clouds
    return pack


def kpgAppearanceDict() -> dict:
    appearance = appearanceForCatalogName('Earth')
    if appearance is None or not appearance.textures.existingMaps().get('color'):
        raise RuntimeError('Earth texture pack is required for the K–Pg cinema')
    payload = appearance.toJobDict()
    for name, path in kpgEarthPack().items():
        payload['textures'][name] = jobTexturePath(path)
    return payload


def _photorealImpactKeys() -> list[Path]:
    for folder in ('full', 'fx', 'impact'):
        keys = [KPG_EARTH_PACK / folder / name for name in KPG_IMPACT_KEYS]
        if all(path.is_file() for path in keys):
            return keys
    return []


def projectSiteOnScreen(
    cameraRadii: tuple[float, float, float],
    lookAtRadii: tuple[float, float, float],
    siteRadii: tuple[float, float, float],
    lensMm: float,
) -> tuple[float, float, float]:
    """Project a site in Earth radii into Blender 36mm-sensor screen UV."""
    camera = np.array(cameraRadii, dtype=float)
    lookAt = np.array(lookAtRadii, dtype=float)
    site = np.array(siteRadii, dtype=float)
    forward = lookAt - camera
    length = float(np.linalg.norm(forward))
    if length < 1e-9:
        return 0.5, 0.5, length
    forward = forward / length
    upHint = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, upHint)
    if float(np.linalg.norm(right)) < 1e-6:
        right = np.cross(forward, np.array((0.0, 1.0, 0.0)))
    right = right / float(np.linalg.norm(right))
    up = np.cross(right, forward)
    relative = site - camera
    depth = float(np.dot(relative, forward))
    if depth < 1e-4:
        return 0.5, 0.5, depth
    half = math.tan(math.atan(18.0 / max(lensMm, 1.0)))
    across = float(np.dot(relative, right)) / depth
    rise = float(np.dot(relative, up)) / depth
    return 0.5 + across / (2.0 * half), 0.5 - rise / (2.0 * half), depth


def _explosionRgba(rgb: np.ndarray) -> np.ndarray:
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    teal = np.clip(rgb[..., 2] - rgb[..., 0], 0.0, 1.0)
    veg = np.clip(rgb[..., 1] - rgb[..., 0], 0.0, 1.0)
    earth = np.clip(2.4 * teal + 1.8 * veg, 0.0, 1.0)
    alpha = np.clip((luma - 0.04) / 0.13, 0.0, 1.0) ** 0.70
    alpha = alpha * (1.0 - earth)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def _holdDissolveMixes(
    keyCount: int,
    needed: int,
    *,
    hero: int,
    dissolve: int = 2,
) -> list[tuple[int, int, float]]:
    """Hold each still; short dissolve so contact is not a 61-frame ghost morph."""
    if keyCount <= 1:
        return [(0, 0, 0.0)] * needed
    leftover = needed - dissolve * (keyCount - 1)
    if leftover < keyCount:
        dissolve = 1
        leftover = needed - dissolve * (keyCount - 1)
    holds = [max(leftover // keyCount, 1)] * keyCount
    extra = leftover - sum(holds)
    hero = min(max(hero, 0), keyCount - 1)
    while extra > 0:
        holds[hero] += 1
        extra -= 1
    mixes: list[tuple[int, int, float]] = []
    for index, hold in enumerate(holds):
        mixes.extend((index, index, 0.0) for _ in range(hold))
        if index < keyCount - 1:
            for step in range(dissolve):
                mixes.append((index, index + 1, (step + 1) / (dissolve + 1)))
    last = keyCount - 1
    if len(mixes) < needed:
        mixes.extend((last, last, 0.0) for _ in range(needed - len(mixes)))
    return mixes[:needed]


def _expandPhotorealPlates(keys: list[Path], directory: Path, needed: int) -> list[Path]:
    frames = [
        np.asarray(
            Image.open(path).convert('RGB').resize((1440, 1440), Image.Resampling.LANCZOS),
            dtype=np.float32,
        )
        / 255.0
        for path in keys
    ]
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    mixes = _holdDissolveMixes(len(frames), needed, hero=min(3, max(len(frames) - 1, 0)))
    for index, (left, right, mix) in enumerate(mixes):
        pixels = frames[left] * (1.0 - mix) + frames[right] * mix
        path = directory / f'kpg_fire_{index + 1:04d}.png'
        Image.fromarray((np.clip(pixels, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB').save(path)
        written.append(path)
    (directory / 'from_blender').write_text(f'{PLATE_STAMP}\n', encoding='utf-8')
    return written


def writeImpactPlate(directory: Path | str, frameCount: int = ANIMATION_FRAMES) -> list[Path]:
    """Contact flipbook: photoreal impact stills, or the Cycles fire bake."""
    del frameCount
    root = Path(directory)
    stamp = root / 'from_blender'
    existing = sorted(root.glob('kpg_fire_*.png'))
    needed = FIRE_CUT_END - IMPACT_FRAME + 1
    keys = _photorealImpactKeys()
    stampOk = stamp.is_file() and stamp.read_text(encoding='utf-8').strip() == PLATE_STAMP
    sizeOk = bool(existing) and Image.open(existing[0]).size[0] >= 512
    if keys and stampOk and sizeOk and len(existing) >= needed:
        return existing[:needed]
    if keys:
        for old in existing:
            old.unlink()
        return _expandPhotorealPlates(keys, root, needed)
    stampOk = stamp.is_file() and stamp.read_text(encoding='utf-8').strip() == 'cycles_fullframe_v1'
    if stampOk and sizeOk and len(existing) >= needed:
        return existing[:needed]
    heroes = _ensureHeroExplosion(root / 'cycles_bake')
    source: list[Path] = []
    for path in heroes:
        sample = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
        if float(sample.mean()) >= 0.015 and _orangeScore(path) >= 0.03:
            source.append(path)
    if len(source) < 5:
        raise RuntimeError(f'Cycles fire plate kept only {len(source)} live frames')
    grown = source[8:] if len(source) > 14 else source
    sequence = _sustainFireSources(grown, needed)
    written: list[Path] = []
    root.mkdir(parents=True, exist_ok=True)
    for old in existing:
        old.unlink()
    for index, sourcePath in enumerate(sequence):
        pixels = np.asarray(Image.open(sourcePath).convert('RGB'), dtype=np.float32) / 255.0
        cropped = _squareCropToFire(pixels, margin=0.04)
        fitted = Image.fromarray(
            (np.clip(cropped, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGB'
        ).resize((1440, 1440), Image.Resampling.LANCZOS)
        path = root / f'kpg_fire_{index + 1:04d}.png'
        fitted.save(path)
        written.append(path)
    stamp.write_text('cycles_fullframe_v1\n', encoding='utf-8')
    return written


def buildKpgJob(
    event: ImpactEvent,
    samples: tuple[ImpactSample, ...],
    *,
    theme: str,
    framesDirectory: Path,
    resolution: int = RENDER_RESOLUTION,
    fps: int = ANIMATION_FPS,
    explosionFrames: list[Path] | None = None,
) -> dict:
    earth = buildPlanetBodyScene('Earth', frameCount=max(len(samples), 2))
    appearance = kpgAppearanceDict()
    radius = float(earth.body.displayRadiusAu)
    del explosionFrames
    textures: dict[str, Path] = {}
    explosion: list[Path] = []
    frames = []
    for sample in samples:
        frames.append(
            {
                'frame': sample.frame,
                'cameraAu': [c * radius for c in sample.cameraRadii],
                'lookAtAu': [c * radius for c in sample.lookAtRadii],
                'impactorAu': [c * radius for c in sample.impactorRadii],
                'sunScale': sample.sunScale,
                'flashScale': sample.flashScale,
                'fireballScale': sample.fireballScale,
                'ejectaScale': sample.ejectaScale,
                'plumeScale': sample.plumeScale,
                'falloutScale': sample.falloutScale,
                'projectileAu': [
                    [component * radius for component in position]
                    for position in sample.projectileRadii
                ],
                'projectileScale': list(sample.projectileScale),
                'projectileDir': [list(direction) for direction in sample.projectileDir],
                'projectileTrail': list(sample.projectileTrail),
                'projectileStrike': list(sample.projectileStrike),
                'inboundTrail': sample.inboundTrail,
                'veil': sample.veil,
                'shockAngle': sample.shockAngle,
                'wildfireAngle': sample.wildfireAngle,
                'soot': sample.soot,
                'entryHeat': sample.entryHeat,
                'crustTear': sample.crustTear
                * (
                    1.0
                    if sample.frame <= IMPACT_FRAME + 18
                    else max(0.0, 1.0 - (sample.frame - IMPACT_FRAME - 18) / 22.0)
                ),
                'tsunamiAngle': sample.tsunamiAngle,
                'lens': sample.lens,
                'impactorVisible': sample.slamScale > 1e-4,
                'slamScale': sample.slamScale,
                'rockHeat': sample.rockHeat,
                'smolder': sample.smolder,
            }
        )
    job = {
        'schema': JOB_SCHEMA_ID,
        'theme': theme,
        'body': {
            'name': earth.body.name,
            'kind': earth.body.kind,
            'diameterKm': earth.body.diameterKm,
            'colorRgba': list(earth.body.colorRgba),
            'displayRadiusAu': radius,
        },
        'impactor': {
            'radiusScale': CINEMA_ROCK_RADII,
            'trueRadiusScale': event.radiusRatio,
            'colorRgba': [0.11, 0.09, 0.08, 1.0],
            'colorTexture': jobTexturePath(ASTEROID_TEXTURE),
        },
        'contact': {
            'normal': [float(component) for component in impactNormal(event)],
            'positionRadii': [float(component) for component in contactCenterRadii(event)],
            'maxFireballRadii': FIREBALL_MAX_RADII,
            'maxEjectaRadii': EJECTA_MAX_RADII,
            'maxPlumeRadii': PLUME_MAX_RADII,
            'inbound': [float(component) for component in inboundDirection(event)],
            'ejectaAngleDeg': EJECTA_ANGLE_DEG,
            'ejectaDirections': [
                [float(component) for component in ray]
                for ray in ejectaDirections(impactNormal(event), inboundDirection(event))
            ],
            'projectileCount': PROJECTILE_COUNT,
            'rockBurstCount': ROCK_BURST_COUNT,
            'emberBurstCount': EMBER_BURST_COUNT,
            'debrisRadiusScale': list(debrisRadiusScales(event.radiusRatio)),
            'crustPlateCount': CRUST_PLATE_COUNT,
            'falloutShell': FALLOUT_SHELL,
            'smokeDomainRadii': SMOKE_DOMAIN_RADII,
            'smokeResolution': SMOKE_RESOLUTION,
            'smokeInflowFrames': SMOKE_INFLOW_FRAMES,
            'textures': {
                **{name: str(path) for name, path in textures.items()},
                'explosion': [str(path) for path in explosion],
            },
        },
        'frames': frames,
        'outputDirectory': str(Path(framesDirectory)),
        'resolution': resolution,
        'fps': fps,
        'filmTransparent': False,
        'appearance': appearance,
    }
    return job


def _font(size: int) -> ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont('DejaVu Sans')
        return ImageFont.truetype(path, size=size)
    except (OSError, ImportError, ValueError):
        return ImageFont.load_default()


def _gradeFilmPlate(pixels: np.ndarray) -> np.ndarray:
    graded = np.clip((pixels - 0.02) * 1.08 + 0.02, 0.0, 1.0)
    height, width, _ = graded.shape
    ys = (np.arange(height) + 0.5) / height * 2.0 - 1.0
    xs = (np.arange(width) + 0.5) / width * 2.0 - 1.0
    xx, yy = np.meshgrid(xs, ys)
    vignette = np.clip(1.08 - 0.28 * (xx * xx + yy * yy), 0.72, 1.0)
    graded = graded * vignette[..., None]
    rng = np.random.default_rng(height * 17 + width)
    return np.clip(graded + rng.normal(0.0, 0.004, graded.shape), 0.0, 1.0)


def _letterbox(pixels: np.ndarray, fraction: float = 0.07) -> np.ndarray:
    bar = max(int(pixels.shape[0] * fraction), 1)
    pixels[:bar] = 0.0
    pixels[-bar:] = 0.0
    return pixels


def _compositeBlowout(framePath: Path, platePath: Path | None, amount: float) -> None:
    if amount <= 1e-4 or platePath is None or not platePath.is_file():
        return
    base = Image.open(framePath).convert('RGBA')
    plate = Image.open(platePath).convert('RGBA').resize(base.size, Image.Resampling.LANCZOS)
    pixels = np.asarray(plate, dtype=np.float32)
    pixels[..., 3] = np.clip(pixels[..., 3] * (0.55 + 1.85 * amount), 0.0, 255.0)
    plate = Image.fromarray(pixels.astype(np.uint8), mode='RGBA')
    merged = Image.alpha_composite(base, plate).convert('RGB')
    merged.save(framePath)
    base.close()
    plate.close()
    merged.close()


def _compositeExplosionOnEarth(
    framePath: Path,
    platePath: Path,
    sample: ImpactSample,
    siteRadii: tuple[float, float, float],
    amount: float,
) -> None:
    """Plant the photoreal blast on the Yucatán. Keep the moving globe."""
    base = Image.open(framePath).convert('RGBA')
    width, height = base.size
    plate = Image.open(platePath).convert('RGBA')
    u, v, depth = projectSiteOnScreen(
        sample.cameraRadii, sample.lookAtRadii, siteRadii, sample.lens
    )
    size = int(
        min(width, height)
        * float(np.clip(0.46 / max(depth, 0.28), 0.22, 0.82))
        * (0.62 + 0.38 * amount)
    )
    plate = plate.resize((size, size), Image.Resampling.LANCZOS)
    left = int(u * width - size / 2)
    top = int(v * height - size / 2)
    layer = Image.new('RGBA', base.size, (0, 0, 0, 0))
    layer.paste(plate, (left, top), plate)
    merged = Image.alpha_composite(base, layer).convert('RGB')
    merged.save(framePath)
    base.close()
    plate.close()
    layer.close()
    merged.close()


def _cutToContactShot(framePath: Path, platePath: Path) -> None:
    """The hit is the photoreal frame. Planet and blast stay one picture."""
    base = Image.open(framePath).convert('RGB')
    shot = Image.open(platePath).convert('RGB').resize(base.size, Image.Resampling.LANCZOS)
    shot.save(framePath)
    base.close()
    shot.close()


def contactPlateAmount(frame: int) -> float:
    """No full-frame still cut. The blast stays in the 3D scene."""
    del frame
    return 0.0


def _mixContactShot(framePath: Path, platePath: Path, amount: float) -> None:
    if amount < 1e-4:
        return
    base = Image.open(framePath).convert('RGB')
    shot = Image.open(platePath).convert('RGB').resize(base.size, Image.Resampling.LANCZOS)
    mixed = shot if amount >= 0.995 else Image.blend(base, shot, amount)
    mixed.save(framePath)
    base.close()
    shot.close()
    if mixed is not shot:
        mixed.close()


def overlayCinemaText(
    framePath: Path,
    *,
    title: str,
    caption: str,
    footer: str,
    dark: bool,
    platePath: Path | None = None,
    plateAmount: float = 0.0,
) -> None:
    del platePath, plateAmount
    image = Image.open(framePath).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    fill = (236, 236, 236) if dark else (20, 22, 26)
    titleFont = _font(max(16, height // 48))
    captionFont = _font(max(11, height // 64))
    footerFont = _font(max(9, height // 82))
    draw.text((width / 2, height * 0.045), title, font=titleFont, fill=fill, anchor='mt')
    draw.text((width / 2, height * 0.945), caption, font=captionFont, fill=fill, anchor='ms')
    muted = (168, 168, 168) if dark else (70, 72, 78)
    draw.text((width / 2, height * 0.978), footer, font=footerFont, fill=muted, anchor='ms')
    image.save(framePath)
    image.close()


def _ensureHeroExplosion(heroDirectory: Path) -> list[Path]:
    """Reuse a cached Cycles fire bake, or bake one close-up Mantaflow shot."""
    existing = sorted(heroDirectory.glob('hero_*.png'))
    if len(existing) >= 24:
        return existing
    fallback = heroDirectory.parent.parent / 'kpg_hero_explosion'
    cached = sorted(fallback.glob('hero_*.png'))
    if len(cached) >= 24:
        return cached
    blenderExecutable = shutil.which('blender')
    if blenderExecutable is None:
        raise RuntimeError('blender not found on PATH — cannot bake the hero explosion')
    heroDirectory.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            blenderExecutable,
            '--background',
            '--python',
            str(RENDER_SCRIPT),
            '--',
            '--hero-explosion',
            str(heroDirectory),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f'Hero explosion bake failed with exit code {completed.returncode}')
    written = sorted(heroDirectory.glob('hero_*.png'))
    if len(written) < 12:
        raise RuntimeError(f'Hero explosion produced {len(written)} frames in {heroDirectory}')
    return written


def _runBlenderKpgJob(jobPath: Path) -> None:
    blenderExecutable = shutil.which('blender')
    if blenderExecutable is None:
        raise RuntimeError(
            'blender not found on PATH. Install Blender to render the K–Pg cinema, or run:\n'
            f'  blender --background --python {RENDER_SCRIPT} -- {jobPath}'
        )
    completed = subprocess.run(
        [
            blenderExecutable,
            '--background',
            '--python',
            str(RENDER_SCRIPT),
            '--',
            str(jobPath),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f'Blender K–Pg render failed with exit code {completed.returncode}')


def renderKpgCinematicAnimations(
    *,
    eventCsvPath: str | Path = DEFAULT_EVENT_CSV,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
    themes: tuple[str, ...] = ('light', 'dark'),
) -> tuple[Path, ...]:
    event = loadImpactEvent(eventCsvPath)
    samples = buildImpactSamples(event)
    footer = (
        'Chicxulub · Hildebrand+ 1991 / Renne+ 2013 · Earth pack · '
        'scripted cinema cascade · artist reconstruction'
    )
    outputRoot = Path(outputDirectory)
    bodyDirectory = bodyOutputDirectory('planet', 'Earth', root=outputRoot)
    galleryDirectory = Path('output/animate/earth/cinematic')
    galleryDirectory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for themeName in themes:
        print(f'Rendering K–Pg {themeName}...')
        with tempfile.TemporaryDirectory(prefix=f'solsys_kpg_{themeName}_') as temporary:
            framesDirectory = Path(temporary) / 'frames'
            job = buildKpgJob(
                event,
                samples,
                theme=themeName,
                framesDirectory=framesDirectory,
            )
            jobPath = bodyDirectory / f'earth_kpg_{themeName}_job.json'
            writeFlybyJob(job, jobPath)
            _runBlenderKpgJob(jobPath)
            framePaths = sorted(framesDirectory.glob('frame_*.png'))
            if len(framePaths) != len(samples):
                raise RuntimeError(
                    f'Expected {len(samples)} frames for {themeName}, got {len(framePaths)}'
                )
            for path, sample in zip(framePaths, samples, strict=True):
                overlayCinemaText(
                    path,
                    title=titleForAct(actName(sample.frame)),
                    caption=captionForSample(event, sample),
                    footer=footer,
                    dark=themeName == 'dark',
                )
            gifPath = galleryDirectory / f'earth_kpg_{themeName}.gif'
            assembleGifFromPngs(
                framePaths,
                gifPath,
                fps=ANIMATION_FPS,
                outputSize=GALLERY_SIZE,
                optimize=False,
            )
            written.append(gifPath)
            print(f'Saved {gifPath}')
    print('K–Pg cinema completed!')
    return tuple(written)


__all__ = [
    'ANIMATION_FRAMES',
    'ACT_BOUNDARIES',
    'IMPACT_FRAME',
    'ImpactEvent',
    'ImpactSample',
    'actName',
    'buildImpactSamples',
    'buildKpgJob',
    'captionForSample',
    'inboundDirection',
    'contactEnvelope',
    'ejectaDirections',
    'falloutEnvelope',
    'flashEnvelope',
    'projectileLaunches',
    'projectilePositionRadii',
    'inboundKmAtFrame',
    'impactNormal',
    'loadImpactEvent',
    'renderKpgCinematicAnimations',
    'unitFromLatLon',
    'writeImpactPlate',
]
