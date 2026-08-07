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
from animate.scenes.blender.body_scene import buildPlanetBodyScene
from animate.scenes.blender.export_body import DEFAULT_OUTPUT_DIRECTORY, exportPlanetBodyScene
from animate.scenes.blender.flyby_camera import buildFlybyCameraPath
from animate.scenes.blender.render_flyby import JOB_SCHEMA_ID

FLYBY_EXTENSION_POINT = 'animate.scenes.blender.flyby_scene.renderPlanetFlyby'
RENDER_FLYBY_SCRIPT = Path('animate/scenes/blender/render_flyby.py')

Theme = Literal['light', 'dark']
DEFAULT_FLYBY_FRAMES = 72
DEFAULT_FLYBY_RESOLUTION = 640
DEFAULT_FLYBY_FPS = 18


def preparePlanetFlybyExport(
    planetName: str = 'Earth',
    *,
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Export catalog body-scene JSON used as flyby input metadata."""
    return exportPlanetBodyScene(
        planetName,
        frameCount=frameCount,
        outputDirectory=outputDirectory,
    )


def buildFlybyJob(
    planetName: str = 'Earth',
    *,
    theme: Theme = 'dark',
    frameCount: int = DEFAULT_FLYBY_FRAMES,
    resolution: int = DEFAULT_FLYBY_RESOLUTION,
    fps: int = DEFAULT_FLYBY_FPS,
    framesDirectory: Path | str,
) -> dict:
    """Build a Blender flyby job dict from PlanetCatalog + camera path."""
    if theme not in ('light', 'dark'):
        raise ValueError(f'theme must be light or dark, got {theme!r}')
    bodyScene = buildPlanetBodyScene(planetName, frameCount=max(frameCount, 2))
    cameraPath = buildFlybyCameraPath(bodyScene.body.displayRadiusAu, frameCount=frameCount)
    appearance = appearanceForCatalogName(bodyScene.body.name)
    job: dict = {
        'schema': JOB_SCHEMA_ID,
        'theme': theme,
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
    }
    if appearance is not None:
        job['appearance'] = appearance.toJobDict()
    return job


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
) -> Path:
    if not framePaths:
        raise ValueError('No PNG frames to assemble into a GIF')
    outputGif.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.open(path).convert('RGB') for path in framePaths]
    durationMs = max(int(round(1000 / fps)), 1)
    images[0].save(
        outputGif,
        save_all=True,
        append_images=images[1:],
        duration=durationMs,
        loop=0,
        optimize=True,
    )
    for image in images:
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
    planetName: str = 'Earth',
    *,
    theme: Theme | Literal['all'] = 'all',
    frameCount: int = DEFAULT_FLYBY_FRAMES,
    resolution: int = DEFAULT_FLYBY_RESOLUTION,
    fps: int = DEFAULT_FLYBY_FPS,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> tuple[Path, ...]:
    """Render light/dark planet flyby GIFs via Blender; returns written GIF paths."""
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
    preparePlanetFlybyExport(planetName, outputDirectory=outputRoot)

    written: list[Path] = []
    stem = planetName.lower().replace(' ', '_')
    for themeName in themes:
        with tempfile.TemporaryDirectory(prefix=f'solsys_flyby_{stem}_{themeName}_') as temporary:
            framesDirectory = Path(temporary) / 'frames'
            job = buildFlybyJob(
                planetName,
                theme=themeName,
                frameCount=frameCount,
                resolution=resolution,
                fps=fps,
                framesDirectory=framesDirectory,
            )
            jobPath = outputRoot / f'{stem}_flyby_{themeName}_job.json'
            writeFlybyJob(job, jobPath)
            _runBlenderFlybyJob(jobPath)
            framePaths = sorted(framesDirectory.glob('frame_*.png'))
            if not framePaths:
                raise RuntimeError(f'No frames rendered for theme={themeName}')
            gifPath = outputRoot / f'{stem}_flyby_{themeName}.gif'
            assembleGifFromPngs(framePaths, gifPath, fps=fps)
            written.append(gifPath)
            print(f'Wrote flyby GIF → {gifPath}')
    return tuple(written)
