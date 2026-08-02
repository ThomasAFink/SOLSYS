"""Barnard's Star top-down exoplanet animations (shared single-host animator)."""

from __future__ import annotations

from .exoplanet_system import (
    DEFAULT_DPI,
    DEFAULT_FIGURE_SIZE_INCHES,
    ExoplanetSystemSceneConfig,
    renderExoplanetSystemAnimations,
)

BARNARDS_STAR_COLOR = '#E06040'

BARNARDS_STAR_PLANETS_CONFIG = ExoplanetSystemSceneConfig(
    title="Barnard's Star planets",
    starLabel="Barnard's Star",
    starColor=BARNARDS_STAR_COLOR,
    minAxisLimitAu=0.05,
    # ~600 frames: d≈4 orbits, b≈3, e≈1.3 — compact sub-Earths stay readable.
    animationSpeed=0.015,
    footerNote='Nearby M dwarf · four sub-Earth planets · ~6 ly · system_id=barnards_star',
)


def renderBarnardsStarAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    renderExoplanetSystemAnimations(
        systemId='barnards_star',
        filenameStem='barnards_star_planets',
        outputDirectory='output/animate/barnards_star',
        config=BARNARDS_STAR_PLANETS_CONFIG,
        figureSizeInches=figureSizeInches,
        dpi=dpi,
        starsCsvPath=starsCsvPath,
    )
