"""Single CLI entry point for all SOLSYS visualizations."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from animate import renderAllAnimations
from animate.scenes.alpha_centauri import renderAlphaCentauriAnimations
from animate.scenes.barnards_star import renderBarnardsStarAnimations
from animate.scenes.blender.export_body import exportPlanetBodyScene
from animate.scenes.interstellar_objects import renderInterstellarObjectAnimations
from animate.scenes.sol_centauri_cinematic import renderSolCentauriCinematicAnimations
from animate.scenes.tabbys_star import renderTabbysStarAnimations
from animate.scenes.trappist_1 import renderTrappist1Animations
from static import renderAll as renderStatic
from static import renderNeighborhood

BLENDER_LOAD_SCRIPT = Path('animate/scenes/blender/load_body.py')

Dimension = Literal['2d', '3d']

# Output geometry lives here so README-friendly sizing is one place to tune.
# Pixel size ≈ inches × dpi.
ANIMATE_FIGURE_SIZE_INCHES = (12.0, 12.0)
ANIMATE_DPI_2D = 100  # original size
ANIMATE_DPI_3D = 50  # half of prior 100 for README
ANIMATE_DPI_CINEMATIC = 100  # Sol→Centauri path needs sharper labels

STATIC_FIGURE_SIZE_INCHES = (39.0, 39.0)
STATIC_DPI = 150  # was 300

NEIGHBORHOOD_FIGURE_SIZE_INCHES = (12.0, 12.0)
NEIGHBORHOOD_DPI = 150  # was 300


def _parseDimensions(choice: str) -> tuple[Dimension, ...]:
    if choice == 'all':
        return ('2d', '3d')
    if choice in ('2d', '3d'):
        return (choice,)  # type: ignore[return-value]
    raise ValueError(f'Unknown dimension choice: {choice!r}')


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render SOLSYS visualizations (animate=main, static=side).',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    staticParser = subparsers.add_parser('static', help='Side product: static 2D/3D zoom views')
    staticParser.add_argument(
        '--dimension',
        choices=('2d', '3d', 'all'),
        default='all',
        help='Which projection to render (default: all)',
    )
    staticParser.add_argument(
        '--stars',
        default='data/nearby_stars_30.csv',
        help='Path to nearby-stars CSV',
    )

    neighborhoodParser = subparsers.add_parser(
        'neighborhood',
        help='Side product: 3D interstellar neighborhood star map',
    )
    neighborhoodParser.add_argument(
        '--stars',
        default='data/nearby_stars_30.csv',
        help='Path to nearby-stars CSV',
    )
    neighborhoodParser.add_argument(
        '--ly',
        type=float,
        default=10.0,
        help='Maximum distance in light years (default: 10)',
    )
    neighborhoodParser.add_argument(
        '--output',
        default=None,
        help='Output JPG path (default: output/neighborhood/...)',
    )
    neighborhoodParser.add_argument(
        '--show',
        action='store_true',
        help='Also open an interactive matplotlib window',
    )

    animateParser = subparsers.add_parser('animate', help='Main product: 2D/3D GIF animations')
    animateParser.add_argument(
        '--dimension',
        choices=('2d', '3d', 'all'),
        default='all',
        help='Which Solar System animation to render (default: all)',
    )
    animateParser.add_argument(
        '--system',
        choices=(
            'sol',
            'alpha_centauri',
            'sol_centauri',
            'barnards_star',
            'trappist_1',
            'tabbys_star',
            'oumuamua',
            'interstellar',
            'all',
        ),
        default='sol',
        help=(
            'Which scene set to render (default: sol). '
            'sol_centauri = Sol-to-Alpha Centauri cinematic; '
            'interstellar = 1I/2I/3I GIFs; oumuamua = \u02bbOumuamua only (alias).'
        ),
    )
    animateParser.add_argument(
        '--object',
        default='all',
        help=(
            'For --system interstellar: object_id from data/interstellar_objects.csv '
            '(oumuamua, borisov, atlas, or all). Ignored otherwise.'
        ),
    )
    animateParser.add_argument(
        '--stars',
        default='data/nearby_stars_30.csv',
        help='Path to nearby-stars CSV (for multi-star systems)',
    )

    allParser = subparsers.add_parser(
        'all',
        help='Render main animations plus static/neighborhood side products',
    )
    allParser.add_argument(
        '--dimension',
        choices=('2d', '3d', 'all'),
        default='all',
        help='Which projection(s) to render (default: all)',
    )
    allParser.add_argument(
        '--stars',
        default='data/nearby_stars_30.csv',
        help='Path to nearby-stars CSV',
    )
    allParser.add_argument(
        '--ly',
        type=float,
        default=10.0,
        help='Neighborhood map radius in light years (default: 10)',
    )
    allParser.add_argument(
        '--system',
        choices=(
            'sol',
            'alpha_centauri',
            'sol_centauri',
            'barnards_star',
            'trappist_1',
            'tabbys_star',
            'oumuamua',
            'interstellar',
            'all',
        ),
        default='all',
        help='Which scene set(s) to include (default: all)',
    )
    allParser.add_argument(
        '--object',
        default='all',
        help='For interstellar scenes: object_id or all (default: all)',
    )

    blenderParser = subparsers.add_parser(
        'blender',
        help='Export Sol planet catalog state for Blender close-ups (scaffold for flybys)',
    )
    blenderParser.add_argument(
        '--body',
        default='Earth',
        help='PlanetCatalog name to export (default: Earth)',
    )
    blenderParser.add_argument(
        '--frames',
        type=int,
        default=120,
        help='Number of orbit keyframes to sample (default: 120)',
    )
    blenderParser.add_argument(
        '--output-dir',
        default='output/animate/blender',
        help='Directory for body-scene JSON (default: output/animate/blender)',
    )
    blenderParser.add_argument(
        '--load',
        action='store_true',
        help='After export, run Blender to ingest the JSON (requires blender on PATH)',
    )

    return parser


def _runBlenderLoad(scenePath: Path) -> None:
    blenderExecutable = shutil.which('blender')
    if blenderExecutable is None:
        raise SystemExit(
            'blender not found on PATH. Install Blender or omit --load and run:\n'
            f'  blender --background --python {BLENDER_LOAD_SCRIPT} -- {scenePath}'
        )
    completed = subprocess.run(
        [
            blenderExecutable,
            '--background',
            '--python',
            str(BLENDER_LOAD_SCRIPT),
            '--',
            str(scenePath),
        ],
        check=False,
    )
    if completed.returncode != 0:
        raise SystemExit(f'Blender ingest failed with exit code {completed.returncode}')


def _renderAnimations(
    systemChoice: str,
    dimensions: tuple[Dimension, ...],
    starsCsvPath: str,
    objectChoice: str,
) -> None:
    if systemChoice in ('sol', 'all'):
        for dimension in dimensions:
            animateDpi = ANIMATE_DPI_2D if dimension == '2d' else ANIMATE_DPI_3D
            renderAllAnimations(
                dimension,
                figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
                dpi=animateDpi,
            )
    if systemChoice in ('alpha_centauri', 'all'):
        renderAlphaCentauriAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice in ('sol_centauri', 'all'):
        renderSolCentauriCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_CINEMATIC,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice in ('barnards_star', 'all'):
        renderBarnardsStarAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice in ('trappist_1', 'all'):
        renderTrappist1Animations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice in ('tabbys_star', 'all'):
        renderTabbysStarAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice == 'oumuamua':
        renderInterstellarObjectAnimations(
            objectIds=('oumuamua',),
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
        )
    elif systemChoice in ('interstellar', 'all'):
        objectIds = None if objectChoice == 'all' else (objectChoice,)
        renderInterstellarObjectAnimations(
            objectIds=objectIds,
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_2D,
        )


def main(argv: list[str] | None = None) -> None:
    parser = buildParser()
    args = parser.parse_args(argv)

    if args.command == 'neighborhood':
        renderNeighborhood(
            starsCsvPath=args.stars,
            maxDistanceLightYears=args.ly,
            outputPath=args.output,
            showPlot=args.show,
            figureSizeInches=NEIGHBORHOOD_FIGURE_SIZE_INCHES,
            dpi=NEIGHBORHOOD_DPI,
        )
        return

    if args.command == 'blender':
        scenePath = exportPlanetBodyScene(
            args.body,
            frameCount=args.frames,
            outputDirectory=args.output_dir,
        )
        print(f'Exported Blender body scene → {scenePath}')
        if args.load:
            _runBlenderLoad(scenePath)
        else:
            print(
                'Next: dry-run validate with\n'
                f'  {sys.executable} {BLENDER_LOAD_SCRIPT} {scenePath}\n'
                'or ingest with\n'
                f'  blender --background --python {BLENDER_LOAD_SCRIPT} -- {scenePath}\n'
                'Flyby renders: animate.scenes.blender.flyby_scene.renderPlanetFlyby (#12).'
            )
        return

    dimensions = _parseDimensions(args.dimension)
    systemChoice = getattr(args, 'system', 'sol')

    # Main product first when rendering everything.
    if args.command in ('animate', 'all'):
        _renderAnimations(
            systemChoice,
            dimensions,
            starsCsvPath=getattr(args, 'stars', 'data/nearby_stars_30.csv'),
            objectChoice=getattr(args, 'object', 'all'),
        )

    if args.command in ('static', 'all'):
        for dimension in dimensions:
            renderStatic(
                dimension,
                starsCsvPath=args.stars,
                figureSizeInches=STATIC_FIGURE_SIZE_INCHES,
                dpi=STATIC_DPI,
            )

    if args.command == 'all':
        renderNeighborhood(
            starsCsvPath=args.stars,
            maxDistanceLightYears=args.ly,
            figureSizeInches=NEIGHBORHOOD_FIGURE_SIZE_INCHES,
            dpi=NEIGHBORHOOD_DPI,
        )


if __name__ == '__main__':
    main()
