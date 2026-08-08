"""Write Blender body-scene JSON under output/animate/blender/{planets,moons,asteroids,…}/…"""

from __future__ import annotations

from pathlib import Path

from animate.scenes.blender.body_scene import BodyScene, buildBodyScene, buildPlanetBodyScene

DEFAULT_OUTPUT_DIRECTORY = Path('output/animate/blender')

_KIND_DIRECTORY: dict[str, str] = {
    'planet': 'planets',
    'moon': 'moons',
    'asteroid': 'asteroids',
    'dwarf_planet': 'dwarf_planets',
}


def bodyStem(bodyName: str) -> str:
    return bodyName.lower().replace(' ', '_')


def bodyOutputDirectory(
    kind: str,
    bodyName: str,
    *,
    root: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """``output/animate/blender/planets/earth`` / ``…/moons/moon`` (etc.)."""
    kindDirectory = _KIND_DIRECTORY.get(kind, 'bodies')
    return Path(root) / kindDirectory / bodyStem(bodyName)


def exportBodyScene(
    bodyName: str = 'Earth',
    *,
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Export one catalog planet, moon, or famous asteroid to JSON for Blender ingest."""
    scene = buildBodyScene(bodyName, frameCount=frameCount)
    return writeBodyScene(scene, outputDirectory=outputDirectory)


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
    """Write ``<stem>_body_scene.json`` under ``{root}/{planets|moons|asteroids|…}/<stem>/``."""
    directory = bodyOutputDirectory(scene.body.kind, scene.body.name, root=outputDirectory)
    directory.mkdir(parents=True, exist_ok=True)
    stem = bodyStem(scene.body.name)
    outputPath = directory / f'{stem}_body_scene.json'
    outputPath.write_text(scene.toJson(), encoding='utf-8')
    return outputPath
