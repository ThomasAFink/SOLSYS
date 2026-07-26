"""Single CLI entry point for solar-system static views and animations."""

from __future__ import annotations

import argparse
from typing import Literal

from interstellar_scale import renderAll as renderStatic
from solar_system_animation import renderAllAnimations

Dimension = Literal['2d', '3d']


def _parseDimensions(choice: str) -> tuple[Dimension, ...]:
    if choice == 'all':
        return ('2d', '3d')
    if choice in ('2d', '3d'):
        return (choice,)  # type: ignore[return-value]
    raise ValueError(f'Unknown dimension choice: {choice!r}')


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Render solar-system visualizations (static views and animations).',
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    staticParser = subparsers.add_parser('static', help='Render static 2D/3D zoom views')
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

    animateParser = subparsers.add_parser('animate', help='Render 2D/3D GIF animations')
    animateParser.add_argument(
        '--dimension',
        choices=('2d', '3d', 'all'),
        default='all',
        help="Which animation to render (default: all)",
    )

    allParser = subparsers.add_parser('all', help='Render static views and animations')
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

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = buildParser()
    args = parser.parse_args(argv)
    dimensions = _parseDimensions(args.dimension)

    if args.command in ('static', 'all'):
        for dimension in dimensions:
            renderStatic(dimension, starsCsvPath=args.stars)

    if args.command in ('animate', 'all'):
        for dimension in dimensions:
            renderAllAnimations(dimension)


if __name__ == '__main__':
    main()
