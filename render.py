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
from animate.scenes.asteroseismology_cinematic import (
    renderAsteroseismologyCinematicAnimations,
)
from animate.scenes.barnards_star import renderBarnardsStarAnimations
from animate.scenes.blender.export_body import exportBodyScene
from animate.scenes.blender.flyby_scene import renderPlanetFlyby, renderPlanetSpin
from animate.scenes.cepheid_ladder_cinematic import renderCepheidLadderCinematicAnimations
from animate.scenes.interstellar_objects import renderInterstellarObjectAnimations
from animate.scenes.kpg_cinematic import renderKpgCinematicAnimations
from animate.scenes.pulsar_cinematic import renderPulsarCinematicAnimations
from animate.scenes.rr_lyrae_cinematic import renderRrLyraeCinematicAnimations
from animate.scenes.sol_centauri_cinematic import renderSolCentauriCinematicAnimations
from animate.scenes.sol_trappist_cinematic import renderSolTrappistCinematicAnimations
from animate.scenes.solar_cycle_cinematic import renderSolarCycleCinematicAnimations
from animate.scenes.tabbys_star import renderTabbysStarAnimations
from animate.scenes.tabbys_star_cinematic import renderTabbysStarCinematicAnimations
from animate.scenes.transit_cinematic import renderTransitCinematicAnimations
from animate.scenes.trappist_1 import renderTrappist1Animations
from animate.scenes.type_ia_cinematic import renderTypeIaCandleCinematicAnimations
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
ANIMATE_DPI_TRANSIT = 84  # full-frame photosphere; keeps the GIF near gallery size

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
            'sol_trappist',
            'barnards_star',
            'trappist_1',
            'tabbys_star',
            'tabbys_star_cinematic',
            'transit_cinematic',
            'asteroseismology_cinematic',
            'solar_cycle_cinematic',
            'cepheid_ladder_cinematic',
            'type_ia_cinematic',
            'pulsar_cinematic',
            'rr_lyrae_cinematic',
            'kpg_cinematic',
            'oumuamua',
            'interstellar',
            'all',
        ),
        default='sol',
        help=(
            'Which scene set to render (default: sol). '
            'sol_centauri = Sol-to-Alpha Centauri cinematic; '
            'sol_trappist = Sol-to-TRAPPIST-1 cinematic; '
            'tabbys_star_cinematic = Tabby lightcurve cinema (Kepler + photosphere occultation); '
            'transit_cinematic = TRAPPIST-1 b transit cinema (TESS fold); '
            'asteroseismology_cinematic = red giant weighed by its ringing (Kepler FFT); '
            'solar_cycle_cinematic = the Sun\u2019s spots counted and placed (SILSO + butterfly); '
            'cepheid_ladder_cinematic = Leavitt\u2019s law fitted from OGLE-IV Cepheids '
            '(period\u2013luminosity \u2192 Magellanic Cloud distances); '
            'type_ia_cinematic = Type Ia standard candles (Pantheon+ Hubble diagram); '
            'pulsar_cinematic = pulsar lighthouse (EPN folded profiles + ATNF ages); '
            'rr_lyrae_cinematic = RR Lyrae / horizontal-branch clocks '
            '(OGLE-IV Magellanic Bailey diagram); '
            'kpg_cinematic = K–Pg / Chicxulub camera-move impact, schematic fireball (Earth pack); '
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
    animateParser.add_argument(
        '--blender-bodies',
        action='store_true',
        help=(
            'For --system sol_centauri / sol_trappist: composite Blender spin-loop '
            'frames (lazy-loaded; requires prior: blender --spin)'
        ),
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
            'sol_trappist',
            'barnards_star',
            'trappist_1',
            'tabbys_star',
            'tabbys_star_cinematic',
            'transit_cinematic',
            'asteroseismology_cinematic',
            'solar_cycle_cinematic',
            'cepheid_ladder_cinematic',
            'type_ia_cinematic',
            'pulsar_cinematic',
            'rr_lyrae_cinematic',
            'kpg_cinematic',
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
        help='Export Sol planet/moon catalog state or render Blender close-up flybys',
    )
    blenderParser.add_argument(
        '--body',
        default='Earth',
        help='PlanetCatalog or MoonCatalog name to export / flyby (default: Earth)',
    )
    blenderParser.add_argument(
        '--frames',
        type=int,
        default=120,
        help='Orbit keyframes for export, or flyby frames when --flyby (default: 120 / 72)',
    )
    blenderParser.add_argument(
        '--output-dir',
        default='output/animate/blender',
        help='Directory for body-scene JSON / flyby GIFs (default: output/animate/blender)',
    )
    blenderParser.add_argument(
        '--load',
        action='store_true',
        help='After export, run Blender to ingest the JSON (requires blender on PATH)',
    )
    blenderParser.add_argument(
        '--flyby',
        action='store_true',
        help='Render polished light/dark flyby GIFs via Blender (requires blender on PATH)',
    )
    blenderParser.add_argument(
        '--spin',
        action='store_true',
        help=(
            'Render fixed-camera RGBA day/night spin loops for cinematic reuse '
            '(requires blender on PATH)'
        ),
    )
    blenderParser.add_argument(
        '--theme',
        choices=('light', 'dark', 'all'),
        default='all',
        help='Theme when using --flyby / --spin (default: all)',
    )
    blenderParser.add_argument(
        '--pipeline',
        action='store_true',
        help=(
            'End-to-end: Earth+Moon Blender spin loops, then Sol→Centauri cinematic '
            'with --blender-bodies. Ignores --body.'
        ),
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


def _runBlenderCinematicPipeline(
    *,
    theme: str,
    frames: int,
    outputDirectory: str,
    starsCsvPath: str,
) -> None:
    """Earth+Moon spin loops, then Sol→Centauri cinematic compositing those frames."""
    spinFrames = 48 if frames == 120 else frames
    for bodyName in ('Earth', 'Moon'):
        print(f'[pipeline] Blender spin loop → {bodyName}')
        paths = renderPlanetSpin(
            bodyName,
            theme=theme,
            frameCount=spinFrames,
            outputDirectory=outputDirectory,
        )
        for path in paths:
            print(f'Spin ready → {path}')

    print('[pipeline] Sol→Centauri cinematic with Blender spin billboards')
    renderSolCentauriCinematicAnimations(
        figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
        dpi=ANIMATE_DPI_CINEMATIC,
        starsCsvPath=starsCsvPath,
        useBlenderBodies=True,
    )
    print('[pipeline] done')


def _renderLightcurveCinemas(systemChoice: str, starsCsvPath: str) -> None:
    """Flux-timeline episodes. Most need their Blender packs first, so never `all`."""
    if systemChoice == 'tabbys_star_cinematic':
        # Requires prior: blender --body "Tabby's Star" --spin.
        renderTabbysStarCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_CINEMATIC,
            starsCsvPath=starsCsvPath,
        )
    if systemChoice == 'transit_cinematic':
        # Requires prior: blender --body "TRAPPIST-1" and "TRAPPIST-1 b" --spin.
        renderTransitCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'asteroseismology_cinematic':
        # Requires prior: blender --body "KIC 7944142" and "Sun" --spin.
        renderAsteroseismologyCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'solar_cycle_cinematic':
        # Requires prior: blender --body "Sun" --spin.
        renderSolarCycleCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'cepheid_ladder_cinematic':
        # Catalogue and photometry only: no Blender pack needed.
        renderCepheidLadderCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'type_ia_cinematic':
        # Catalogue and photometry only: no Blender pack needed.
        renderTypeIaCandleCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'pulsar_cinematic':
        # Catalogue and folded profiles only: no Blender pack needed.
        renderPulsarCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )
    if systemChoice == 'rr_lyrae_cinematic':
        # Catalogue and photometry only: no Blender pack needed.
        renderRrLyraeCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_TRANSIT,
        )


def _renderAnimations(
    systemChoice: str,
    dimensions: tuple[Dimension, ...],
    starsCsvPath: str,
    objectChoice: str,
    *,
    useBlenderBodies: bool = False,
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
            useBlenderBodies=useBlenderBodies,
        )
    if systemChoice in ('sol_trappist', 'all'):
        renderSolTrappistCinematicAnimations(
            figureSizeInches=ANIMATE_FIGURE_SIZE_INCHES,
            dpi=ANIMATE_DPI_CINEMATIC,
            starsCsvPath=starsCsvPath,
            useBlenderBodies=useBlenderBodies,
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
    _renderLightcurveCinemas(systemChoice, starsCsvPath)
    if systemChoice == 'kpg_cinematic':
        # Requires the Earth Blender pack (textures already in-repo).
        renderKpgCinematicAnimations()
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


def _handleBlenderCommand(args: argparse.Namespace) -> None:
    if args.pipeline:
        _runBlenderCinematicPipeline(
            theme=args.theme,
            frames=args.frames,
            outputDirectory=args.output_dir,
            starsCsvPath='data/nearby_stars_30.csv',
        )
        return

    if args.spin:
        spinFrames = 48 if args.frames == 120 else args.frames
        paths = renderPlanetSpin(
            args.body,
            theme=args.theme,
            frameCount=spinFrames,
            outputDirectory=args.output_dir,
        )
        for path in paths:
            print(f'Spin ready → {path}')
        return

    if args.flyby:
        flybyFrames = 72 if args.frames == 120 else args.frames
        gifPaths = renderPlanetFlyby(
            args.body,
            theme=args.theme,
            frameCount=flybyFrames,
            outputDirectory=args.output_dir,
        )
        for gifPath in gifPaths:
            print(f'Flyby ready → {gifPath}')
        return

    scenePath = exportBodyScene(
        args.body,
        frameCount=args.frames,
        outputDirectory=args.output_dir,
    )
    print(f'Exported Blender body scene → {scenePath}')
    if args.load:
        _runBlenderLoad(scenePath)
        return
    print(
        'Next: dry-run validate with\n'
        f'  {sys.executable} {BLENDER_LOAD_SCRIPT} {scenePath}\n'
        'or ingest with\n'
        f'  blender --background --python {BLENDER_LOAD_SCRIPT} -- {scenePath}\n'
        'or render flybys with\n'
        f'  {sys.executable} render.py blender --body {args.body} --flyby\n'
        'or render spin loops with\n'
        f'  {sys.executable} render.py blender --body {args.body} --spin\n'
        'or run the full Earth/Moon → cinematic pipeline with\n'
        f'  {sys.executable} render.py blender --pipeline'
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
        _handleBlenderCommand(args)
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
            useBlenderBodies=bool(getattr(args, 'blender_bodies', False)),
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
