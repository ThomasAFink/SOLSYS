"""K–Pg cinema — a camera move into Chicxulub (#86).

Stay on Earth. The playhead is inbound distance, not a chart. A 10 km rock is a
speck against a 12,742 km planet, so the film dives until that speck is
readable, then the flash and the dust veil are labelled schematic. Geography is
the modern Blue Marble map as a stand-in.

Geometry is derived at runtime from `data/chicxulub_kpg.csv` and the Earth
diameter in PlanetCatalog. Blender renders the move; titles are composited after.
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

from animate.scenes.blender.body_appearance import appearanceForCatalogName
from animate.scenes.blender.body_scene import buildPlanetBodyScene
from animate.scenes.blender.export_body import DEFAULT_OUTPUT_DIRECTORY, bodyOutputDirectory
from animate.scenes.blender.flyby_scene import assembleGifFromPngs, writeFlybyJob

DEFAULT_EVENT_CSV = 'data/chicxulub_kpg.csv'
JOB_SCHEMA_ID = 'solsys.blender_kpg_job/v1'
RENDER_SCRIPT = Path('animate/scenes/blender/render_kpg.py')

ANIMATION_FPS = 20
ANIMATION_FRAMES = 200
RENDER_RESOLUTION = 640
# Match the Blender flyby gallery box; photographic frames do not GIF-compress.
GALLERY_SIZE = 640

# Camera and inbound distances in Earth radii.
WIDE_CAMERA_RADII = 4.55
CLOSE_CAMERA_RADII = 1.65
INBOUND_START_RADII = 7.2
OBLIQUITY_FROM_VERTICAL_DEG = 40.0
# 40° off vertical so the rock is not hidden on the camera axis.

QUIET_END = 44
IMPACT_FRAME = 148
STRIKE_END = 164

ACT_BOUNDARIES = (
    (QUIET_END, 'quiet'),
    (IMPACT_FRAME, 'approach'),
    (STRIKE_END, 'strike'),
)


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
    east = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    eastLength = float(np.linalg.norm(east))
    if eastLength < 1e-9:
        east = np.array([0.0, 1.0, 0.0])
    else:
        east = east / eastLength
    tilt = math.radians(OBLIQUITY_FROM_VERTICAL_DEG)
    direction = -normal * math.cos(tilt) + east * math.sin(tilt)
    return direction / float(np.linalg.norm(direction))


def contactCenterRadii(event: ImpactEvent) -> np.ndarray:
    """Impactor centre at first contact, in Earth radii from Earth's centre."""
    return impactNormal(event) * (1.0 + event.radiusRatio)


@dataclass(frozen=True)
class ImpactSample:
    frame: int
    cameraRadii: tuple[float, float, float]
    lookAtRadii: tuple[float, float, float]
    impactorRadii: tuple[float, float, float]
    sunScale: float
    flashScale: float
    veil: float
    inboundKm: float
    secondsToImpact: float


def actName(frame: int) -> str:
    for boundary, name in ACT_BOUNDARIES:
        if frame < boundary:
            return name
    return 'veil'


def _asTuple(vector: np.ndarray) -> tuple[float, float, float]:
    return (float(vector[0]), float(vector[1]), float(vector[2]))


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
    east = np.cross(np.array([0.0, 0.0, 1.0]), normal)
    east = east / float(np.linalg.norm(east))
    lift = np.cross(inbound, east)
    lift = lift / float(np.linalg.norm(lift))

    samples: list[ImpactSample] = []
    for frame in range(frameCount):
        dive = smoothStep((frame - QUIET_END) / (IMPACT_FRAME - QUIET_END))
        pull = smoothStep((frame - STRIKE_END) / (frameCount - 1 - STRIKE_END))
        cameraDistance = WIDE_CAMERA_RADII + (CLOSE_CAMERA_RADII - WIDE_CAMERA_RADII) * dive
        cameraDistance += (WIDE_CAMERA_RADII - CLOSE_CAMERA_RADII) * pull
        # Off-axis so the inbound rock is a silhouette against the Gulf, not hidden.
        camera = contact - inbound * cameraDistance + lift * (0.18 * (1.0 - 0.65 * dive))
        lookAt = contact * dive * (1.0 - 0.7 * pull)
        remainingKm = inboundKmAtFrame(event, frame)
        remainingRadii = remainingKm / event.earthRadiusKm
        impactor = contact - inbound * remainingRadii
        if frame < IMPACT_FRAME:
            flash = 0.0
            veil = 0.0
            sunScale = 1.0
        elif frame < STRIKE_END:
            strike = smoothStep((frame - IMPACT_FRAME) / (STRIKE_END - IMPACT_FRAME))
            flash = math.exp(-3.2 * strike)
            veil = 0.35 * strike
            sunScale = 1.0 - 0.55 * strike
        else:
            recover = smoothStep((frame - STRIKE_END) / max(frameCount - 1 - STRIKE_END, 1))
            flash = 0.0
            veil = 0.85 * (1.0 - 0.55 * recover)
            sunScale = 0.28 + 0.55 * recover
        samples.append(
            ImpactSample(
                frame=frame,
                cameraRadii=_asTuple(camera),
                lookAtRadii=_asTuple(lookAt),
                impactorRadii=_asTuple(impactor),
                sunScale=sunScale,
                flashScale=flash,
                veil=veil,
                inboundKm=remainingKm,
                secondsToImpact=remainingKm / event.speedKmS,
            )
        )
    return tuple(samples)


def titleForAct(act: str) -> str:
    if act == 'quiet':
        return 'Late Cretaceous Earth — 66 million years before now'
    if act == 'approach':
        return 'A 10 km rock, true scale — inbound to the Yucatán'
    if act == 'strike':
        return 'Contact — the flash is schematic'
    return 'Dust veil — schematic, then the light returns'


def captionForSample(event: ImpactEvent, sample: ImpactSample) -> str:
    act = actName(sample.frame)
    if act == 'quiet':
        return (
            f'{event.name}  {event.latitudeDeg:.1f}°N, {abs(event.longitudeDeg):.1f}°W  ·  '
            f'crater {event.craterDiameterKm:.0f} km  ·  modern map as a stand-in'
        )
    if act == 'approach':
        minutes = sample.secondsToImpact / 60.0
        return (
            f'{sample.inboundKm:,.0f} km out  ·  {minutes:.1f} min at {event.speedKmS:.0f} km/s  ·  '
            f'the rock is {event.impactorDiameterKm:.0f}/{event.earthDiameterKm:.0f} of Earth'
        )
    if act == 'strike':
        return (
            f'{event.impactorDiameterKm:.0f} km body at {event.speedKmS:.0f} km/s  ·  '
            'oblique, labelled schematic  ·  not a hydro simulation'
        )
    return (
        f'{event.ageMa:.0f} Ma  ·  the veil is not a climate model  ·  '
        'recovery is the disk brightening, nothing else'
    )


def buildKpgJob(
    event: ImpactEvent,
    samples: tuple[ImpactSample, ...],
    *,
    theme: str,
    framesDirectory: Path,
    resolution: int = RENDER_RESOLUTION,
    fps: int = ANIMATION_FPS,
) -> dict:
    earth = buildPlanetBodyScene('Earth', frameCount=max(len(samples), 2))
    appearance = appearanceForCatalogName('Earth')
    if appearance is None or not appearance.textures.existingMaps().get('color'):
        raise RuntimeError('Earth texture pack is required for the K–Pg cinema')
    radius = float(earth.body.displayRadiusAu)
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
                'veil': sample.veil,
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
            'radiusScale': event.radiusRatio,
            'colorRgba': [0.28, 0.26, 0.24, 1.0],
        },
        'frames': frames,
        'outputDirectory': str(Path(framesDirectory)),
        'resolution': resolution,
        'fps': fps,
        'filmTransparent': False,
        'appearance': appearance.toJobDict(),
    }
    return job


def _font(size: int) -> ImageFont.ImageFont:
    try:
        from matplotlib import font_manager

        path = font_manager.findfont('DejaVu Sans')
        return ImageFont.truetype(path, size=size)
    except (OSError, ImportError, ValueError):
        return ImageFont.load_default()


def overlayCinemaText(
    framePath: Path,
    *,
    title: str,
    caption: str,
    footer: str,
    dark: bool,
) -> None:
    image = Image.open(framePath).convert('RGB')
    draw = ImageDraw.Draw(image)
    width, height = image.size
    fill = (240, 240, 240) if dark else (24, 24, 24)
    titleFont = _font(max(18, height // 42))
    captionFont = _font(max(13, height // 58))
    footerFont = _font(max(10, height // 78))
    draw.text((width / 2, height * 0.035), title, font=titleFont, fill=fill, anchor='mt')
    draw.text((width / 2, height * 0.955), caption, font=captionFont, fill=fill, anchor='ms')
    muted = (180, 180, 180) if dark else (90, 90, 90)
    draw.text((width / 2, height * 0.985), footer, font=footerFont, fill=muted, anchor='ms')
    image.save(framePath)
    image.close()


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
) -> tuple[Path, ...]:
    event = loadImpactEvent(eventCsvPath)
    samples = buildImpactSamples(event)
    footer = (
        'Chicxulub · Hildebrand+ 1991 / Renne+ 2013 · Earth pack · '
        'inbound distance from these numbers · flash and veil schematic'
    )
    outputRoot = Path(outputDirectory)
    bodyDirectory = bodyOutputDirectory('planet', 'Earth', root=outputRoot)
    galleryDirectory = Path('output/animate/earth/cinematic')
    galleryDirectory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for themeName in ('light', 'dark'):
        print(f'Rendering K–Pg {themeName}...')
        with tempfile.TemporaryDirectory(prefix=f'solsys_kpg_{themeName}_') as temporary:
            framesDirectory = Path(temporary) / 'frames'
            job = buildKpgJob(event, samples, theme=themeName, framesDirectory=framesDirectory)
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
            assembleGifFromPngs(framePaths, gifPath, fps=ANIMATION_FPS, outputSize=GALLERY_SIZE)
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
    'inboundKmAtFrame',
    'impactNormal',
    'loadImpactEvent',
    'renderKpgCinematicAnimations',
    'unitFromLatLon',
]
