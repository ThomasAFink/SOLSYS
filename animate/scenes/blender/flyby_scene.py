"""First Blender planet flyby (issue #12): catalog → job JSON → PNG frames → GIF."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from PIL import Image

from animate.scenes.blender.body_appearance import appearanceForCatalogName
from animate.scenes.blender.body_scene import buildBodyScene
from animate.scenes.blender.export_body import (
    DEFAULT_OUTPUT_DIRECTORY,
    bodyOutputDirectory,
    bodyStem,
    exportBodyScene,
)
from animate.scenes.blender.flyby_camera import buildFlybyCameraPath, buildSpinCameraPath
from animate.scenes.blender.render_flyby import JOB_SCHEMA_ID

FLYBY_EXTENSION_POINT = 'animate.scenes.blender.flyby_scene.renderPlanetFlyby'
RENDER_FLYBY_SCRIPT = Path('animate/scenes/blender/render_flyby.py')

Theme = Literal['light', 'dark']
DEFAULT_FLYBY_FRAMES = 72
DEFAULT_FLYBY_RESOLUTION = 640
DEFAULT_FLYBY_FPS = 18
DEFAULT_SPIN_FRAMES = 48
DEFAULT_SPIN_RESOLUTION = 512
DEFAULT_SPIN_FPS = 20


def preparePlanetFlybyExport(
    bodyName: str = 'Earth',
    *,
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Export catalog body-scene JSON used as flyby input metadata."""
    return exportBodyScene(
        bodyName,
        frameCount=frameCount,
        outputDirectory=outputDirectory,
    )


def _bodyJobSkeleton(
    bodyName: str,
    *,
    theme: Theme,
    frameCount: int,
    resolution: int,
    fps: int,
    framesDirectory: Path | str,
    cameraPath,
    filmTransparent: bool = False,
    mode: str = 'flyby',
) -> dict:
    if theme not in ('light', 'dark'):
        raise ValueError(f'theme must be light or dark, got {theme!r}')
    bodyScene = buildBodyScene(bodyName, frameCount=max(frameCount, 2))
    appearance = appearanceForCatalogName(bodyScene.body.name)
    job: dict = {
        'schema': JOB_SCHEMA_ID,
        'theme': theme,
        'mode': mode,
        'body': {
            'name': bodyScene.body.name,
            'kind': bodyScene.body.kind,
            'systemId': bodyScene.body.systemId,
            'diameterKm': bodyScene.body.diameterKm,
            'color': bodyScene.body.color,
            'colorRgba': list(bodyScene.body.colorRgba),
            'displayRadiusAu': bodyScene.body.displayRadiusAu,
        },
        'frames': [
            {
                'frame': sample.frame,
                'cameraAu': list(sample.cameraAu),
                'bodyRotationDeg': sample.bodyRotationDeg,
            }
            for sample in cameraPath
        ],
        'outputDirectory': str(Path(framesDirectory)),
        'resolution': resolution,
        'fps': fps,
        'filmTransparent': filmTransparent,
    }
    if appearance is not None:
        job['appearance'] = appearance.toJobDict()
    return job


def buildFlybyJob(
    bodyName: str = 'Earth',
    *,
    theme: Theme = 'dark',
    frameCount: int = DEFAULT_FLYBY_FRAMES,
    resolution: int = DEFAULT_FLYBY_RESOLUTION,
    fps: int = DEFAULT_FLYBY_FPS,
    framesDirectory: Path | str,
) -> dict:
    """Build a Blender flyby job dict from PlanetCatalog/MoonCatalog + camera path."""
    bodyScene = buildBodyScene(bodyName, frameCount=max(frameCount, 2))
    appearance = appearanceForCatalogName(bodyScene.body.name)
    # Pull back / elevate when rings are present so the annulus opens and fits.
    ringed = appearance is not None and appearance.rings.enabled
    isStar = bodyScene.body.kind == 'star'
    if isStar:
        distanceScale = 5.2
        elevationDeg = 12.0
        # Hot faculae alias hard at 640; render sharper then GIF downscales cleanly.
        if resolution == DEFAULT_FLYBY_RESOLUTION:
            resolution = 960
    elif ringed:
        distanceScale = 5.8
        elevationDeg = 28.0
    else:
        distanceScale = 4.4
        elevationDeg = 16.0
    cameraPath = buildFlybyCameraPath(
        bodyScene.body.displayRadiusAu,
        frameCount=frameCount,
        distanceScale=distanceScale,
        elevationDeg=elevationDeg,
    )
    return _bodyJobSkeleton(
        bodyName,
        theme=theme,
        frameCount=frameCount,
        resolution=resolution,
        fps=fps,
        framesDirectory=framesDirectory,
        cameraPath=cameraPath,
        filmTransparent=False,
        mode='flyby',
    )


def buildSpinJob(
    bodyName: str = 'Earth',
    *,
    theme: Theme = 'dark',
    frameCount: int = DEFAULT_SPIN_FRAMES,
    resolution: int = DEFAULT_SPIN_RESOLUTION,
    fps: int = DEFAULT_SPIN_FPS,
    framesDirectory: Path | str,
) -> dict:
    """Fixed-camera full-rotation job (transparent PNGs for cinematic reuse)."""
    bodyScene = buildBodyScene(bodyName, frameCount=max(frameCount, 2))
    appearance = appearanceForCatalogName(bodyScene.body.name)
    ringed = appearance is not None and appearance.rings.enabled
    isStar = bodyScene.body.kind == 'star'
    if isStar:
        distanceScale = 5.0
        elevationDeg = 14.0
        if resolution == DEFAULT_SPIN_RESOLUTION:
            resolution = 768
    elif ringed:
        distanceScale = 5.8
        elevationDeg = 28.0
    else:
        distanceScale = 4.4
        elevationDeg = 18.0
    cameraPath = buildSpinCameraPath(
        bodyScene.body.displayRadiusAu,
        frameCount=frameCount,
        distanceScale=distanceScale,
        elevationDeg=elevationDeg,
    )
    return _bodyJobSkeleton(
        bodyName,
        theme=theme,
        frameCount=frameCount,
        resolution=resolution,
        fps=fps,
        framesDirectory=framesDirectory,
        cameraPath=cameraPath,
        filmTransparent=True,
        mode='spin',
    )


def spinFramesDirectory(
    bodyName: str,
    theme: Theme,
    *,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Persistent PNG loop: ``…/planets/earth/earth_spin_dark/frame_*.png``."""
    bodyScene = buildBodyScene(bodyName, frameCount=2)
    stem = bodyStem(bodyName)
    return (
        bodyOutputDirectory(bodyScene.body.kind, bodyName, root=outputDirectory)
        / f'{stem}_spin_{theme}'
    )


def writeFlybyJob(job: dict, path: Path | str) -> Path:
    outputPath = Path(path)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    outputPath.write_text(json.dumps(job, indent=2) + '\n', encoding='utf-8')
    return outputPath


def assembleGifFromPngs(
    framePaths: list[Path],
    outputGif: Path,
    *,
    fps: int = DEFAULT_FLYBY_FPS,
    outputSize: int | None = None,
) -> Path:
    if not framePaths:
        raise ValueError('No PNG frames to assemble into a GIF')
    outputGif.parent.mkdir(parents=True, exist_ok=True)
    rgbFrames: list[Image.Image] = []
    for path in framePaths:
        raw = Image.open(path)
        # RGBA spin frames: composite onto black. convert('RGB') would keep bright
        # premultiplied edge texels and look like a solid pixelated atmosphere ring.
        if raw.mode == 'RGBA':
            background = Image.new('RGBA', raw.size, (0, 0, 0, 255))
            frame = Image.alpha_composite(background, raw).convert('RGB')
            raw.close()
        else:
            frame = raw.convert('RGB')
            if frame is not raw:
                raw.close()
        if outputSize is not None and (frame.width != outputSize or frame.height != outputSize):
            resized = frame.resize((outputSize, outputSize), Image.Resampling.LANCZOS)
            frame.close()
            frame = resized
        rgbFrames.append(frame)
    # Shared adaptive palette + dither softens hard faculae banding without flicker.
    palette = rgbFrames[0].quantize(
        colors=256,
        method=Image.Quantize.MEDIANCUT,
        dither=Image.Dither.FLOYDSTEINBERG,
    )
    images = [palette] + [
        frame.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
        for frame in rgbFrames[1:]
    ]
    durationMs = max(int(round(1000 / fps)), 1)
    images[0].save(
        outputGif,
        save_all=True,
        append_images=images[1:],
        duration=durationMs,
        loop=0,
        optimize=True,
    )
    for image in (*images, *rgbFrames):
        image.close()
    return outputGif


def _runBlenderFlybyJob(jobPath: Path) -> None:
    blenderExecutable = shutil.which('blender')
    if blenderExecutable is None:
        raise RuntimeError(
            'blender not found on PATH. Install Blender to render flybys, or run:\n'
            f'  blender --background --python {RENDER_FLYBY_SCRIPT} -- {jobPath}'
        )
    completed = subprocess.run(
        [
            blenderExecutable,
            '--background',
            '--python',
            str(RENDER_FLYBY_SCRIPT),
            '--',
            str(jobPath),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f'Blender flyby render failed with exit code {completed.returncode}')


def renderPlanetFlyby(
    bodyName: str = 'Earth',
    *,
    theme: Theme | Literal['all'] = 'all',
    frameCount: int = DEFAULT_FLYBY_FRAMES,
    resolution: int = DEFAULT_FLYBY_RESOLUTION,
    fps: int = DEFAULT_FLYBY_FPS,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, ...]:
    """Render light/dark body flyby GIFs via Blender; returns written GIF paths."""
    themes: tuple[Theme, ...]
    if theme == 'all':
        themes = ('light', 'dark')
    elif theme in ('light', 'dark'):
        themes = (theme,)
    else:
        raise ValueError(f'theme must be light, dark, or all, got {theme!r}')

    outputRoot = Path(outputDirectory)
    outputRoot.mkdir(parents=True, exist_ok=True)
    # Keep catalog export in the product tree so the #11 pipeline stays exercised.
    # Export lands under blender/{planets|moons}/<body>/ — GIFs/jobs join it.
    bodyDirectory = preparePlanetFlybyExport(bodyName, outputDirectory=outputRoot).parent

    written: list[Path] = []
    stem = bodyStem(bodyName)
    for themeName in themes:
        with tempfile.TemporaryDirectory(prefix=f'solsys_flyby_{stem}_{themeName}_') as temporary:
            framesDirectory = Path(temporary) / 'frames'
            job = buildFlybyJob(
                bodyName,
                theme=themeName,
                frameCount=frameCount,
                resolution=resolution,
                fps=fps,
                framesDirectory=framesDirectory,
            )
            jobPath = bodyDirectory / f'{stem}_flyby_{themeName}_job.json'
            writeFlybyJob(job, jobPath)
            _runBlenderFlybyJob(jobPath)
            framePaths = sorted(framesDirectory.glob('frame_*.png'))
            if not framePaths:
                raise RuntimeError(f'No frames rendered for theme={themeName}')
            gifPath = bodyDirectory / f'{stem}_flyby_{themeName}.gif'
            # Stars render supersampled; Lanczos down to gallery size softens faculae.
            gifSize = (
                DEFAULT_FLYBY_RESOLUTION
                if str(job.get('body', {}).get('kind') or '') == 'star'
                and int(job.get('resolution', 0)) > DEFAULT_FLYBY_RESOLUTION
                else None
            )
            assembleGifFromPngs(framePaths, gifPath, fps=fps, outputSize=gifSize)
            written.append(gifPath)
            print(f'Wrote flyby GIF → {gifPath}')
    return tuple(written)


def renderPlanetSpin(
    bodyName: str = 'Earth',
    *,
    theme: Theme | Literal['all'] = 'all',
    frameCount: int = DEFAULT_SPIN_FRAMES,
    resolution: int = DEFAULT_SPIN_RESOLUTION,
    fps: int = DEFAULT_SPIN_FPS,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, ...]:
    """Render fixed-camera RGBA spin loops for cinematic body billboards."""
    themes: tuple[Theme, ...]
    if theme == 'all':
        themes = ('light', 'dark')
    elif theme in ('light', 'dark'):
        themes = (theme,)
    else:
        raise ValueError(f'theme must be light, dark, or all, got {theme!r}')

    outputRoot = Path(outputDirectory)
    outputRoot.mkdir(parents=True, exist_ok=True)
    bodyDirectory = preparePlanetFlybyExport(bodyName, outputDirectory=outputRoot).parent
    stem = bodyStem(bodyName)
    written: list[Path] = []

    for themeName in themes:
        framesDirectory = spinFramesDirectory(bodyName, themeName, outputDirectory=outputRoot)
        if framesDirectory.exists():
            shutil.rmtree(framesDirectory)
        framesDirectory.mkdir(parents=True, exist_ok=True)
        job = buildSpinJob(
            bodyName,
            theme=themeName,
            frameCount=frameCount,
            resolution=resolution,
            fps=fps,
            framesDirectory=framesDirectory,
        )
        jobPath = bodyDirectory / f'{stem}_spin_{themeName}_job.json'
        writeFlybyJob(job, jobPath)
        _runBlenderFlybyJob(jobPath)
        framePaths = sorted(framesDirectory.glob('frame_*.png'))
        if not framePaths:
            raise RuntimeError(f'No spin frames rendered for theme={themeName}')
        # Preview GIF (RGB) for the gallery; cinematic loads the RGBA PNGs.
        gifPath = bodyDirectory / f'{stem}_spin_{themeName}.gif'
        assembleGifFromPngs(framePaths, gifPath, fps=fps)
        written.append(framesDirectory)
        print(f'Wrote spin loop → {framesDirectory} ({len(framePaths)} frames)')
        print(f'Wrote spin preview GIF → {gifPath}')
    return tuple(written)
