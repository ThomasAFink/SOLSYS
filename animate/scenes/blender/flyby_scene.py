"""Extension point for the first Blender planet flyby (issue #12).

Pipeline stage after catalog export / Blender ingest:

1. ``preparePlanetFlybyExport`` — write body-scene JSON (done here).
2. Blender camera path + shading + light/dark renders — land in #12.
3. Wire polished outputs into the README gallery.
"""

from __future__ import annotations

from pathlib import Path

from animate.scenes.blender.export_body import DEFAULT_OUTPUT_DIRECTORY, exportPlanetBodyScene

FLYBY_EXTENSION_POINT = 'animate.scenes.blender.flyby_scene.renderPlanetFlyby'


def preparePlanetFlybyExport(
    planetName: str = 'Earth',
    *,
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Export catalog state for a future flyby render. Does not invoke Blender."""
    return exportPlanetBodyScene(
        planetName,
        frameCount=frameCount,
        outputDirectory=outputDirectory,
    )


def renderPlanetFlyby(
    planetName: str = 'Earth',
    *,
    theme: str = 'dark',
    frameCount: int = 120,
    outputDirectory: Path | str = DEFAULT_OUTPUT_DIRECTORY,
) -> Path:
    """Reserved for issue #12 — first polished light/dark flyby.

    Currently only prepares the body-scene export so the CLI hook and gallery
    wiring have a stable call site.
    """
    _ = theme  # light/dark renders land with the flyby implementation.
    exportPath = preparePlanetFlybyExport(
        planetName,
        frameCount=frameCount,
        outputDirectory=outputDirectory,
    )
    raise NotImplementedError(
        f'{FLYBY_EXTENSION_POINT} is the issue #12 extension point. '
        f'Body scene prepared at {exportPath}. '
        'Implement Blender camera path + light/dark renders next.'
    )
