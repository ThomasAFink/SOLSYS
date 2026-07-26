"""Single CLI entry point for all SOLSYS visualizations."""

from __future__ import annotations

import argparse
from typing import Literal

from animate import renderAllAnimations
from static import renderAll as renderStatic
from static import renderNeighborhood

Dimension = Literal['2d', '3d']


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
        help="Which projection to render (default: all)",
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
        help="Which animation to render (default: all)",
    )

    allParser = subparsers.add_parser(
        'all',
        help='Render main animations plus static/neighborhood side products',
    )
    allParser.add_argument(
        '--dimension',
        choices=('2d', '3d', 'all'),
        default='all',
        help="Which projection(s) to render (default: all)",
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = buildParser()
    args = parser.parse_args(argv)

    if args.command == 'neighborhood':
        renderNeighborhood(
            starsCsvPath=args.stars,
            maxDistanceLightYears=args.ly,
            outputPath=args.output,
            showPlot=args.show,
        )
        return

    dimensions = _parseDimensions(args.dimension)

    # Main product first when rendering everything.
    if args.command in ('animate', 'all'):
        for dimension in dimensions:
            renderAllAnimations(dimension)

    if args.command in ('static', 'all'):
        for dimension in dimensions:
            renderStatic(dimension, starsCsvPath=args.stars)

    if args.command == 'all':
        renderNeighborhood(starsCsvPath=args.stars, maxDistanceLightYears=args.ly)


if __name__ == '__main__':
    main()
