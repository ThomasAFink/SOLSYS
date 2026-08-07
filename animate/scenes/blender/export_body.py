"""Write Blender body-scene JSON under output/animate/blender/."""

from __future__ import annotations

from pathlib import Path

from animate.scenes.blender.body_scene import BodyScene, buildPlanetBodyScene

DEFAULT_OUTPUT_DIRECTORY = Path('output/animate/blender')


def exportPlanetBodyScene(
    planetName: str = 'Earth',
    *,
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Export one catalog planet to JSON for Blender ingest. Returns the written path."""
    scene = buildPlanetBodyScene(planetName, frameCount=frameCount)
    return writeBodyScene(scene, outputDirectory=outputDirectory)


def writeBodyScene(
    scene: BodyScene,
    *,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    directory = Path(outputDirectory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = scene.body.name.lower().replace(' ', '_')
    outputPath = directory / f'{stem}_body_scene.json'
    outputPath.write_text(scene.toJson(), encoding='utf-8')
    return outputPath
