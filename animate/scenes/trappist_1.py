"""TRAPPIST-1 top-down exoplanet animations (shared single-host animator)."""

from __future__ import annotations

from .exoplanet_system import (
    DEFAULT_DPI,
    DEFAULT_FIGURE_SIZE_INCHES,
    ExoplanetSystemSceneConfig,
    renderExoplanetSystemAnimations,
)

TRAPPIST_1_STAR_COLOR = '#FF6B4A'

TRAPPIST_1_PLANETS_CONFIG = ExoplanetSystemSceneConfig(
    title='TRAPPIST-1 planets',
    starLabel='TRAPPIST-1',
    starColor=TRAPPIST_1_STAR_COLOR,
    minAxisLimitAu=0.09,
    # ~600 frames: b≈5 orbits, e≈1.5, h≈0.5 — inner planets stay readable.
    animationSpeed=0.012,
    footerNote='Ultracool dwarf · seven Earth-sized planets · ~40.7 ly · system_id=trappist_1',
)


def renderTrappist1Animations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
) -> None:
    renderExoplanetSystemAnimations(
        systemId='trappist_1',
        filenameStem='trappist_1_planets',
        outputDirectory='output/animate/trappist_1',
        config=TRAPPIST_1_PLANETS_CONFIG,
        figureSizeInches=figureSizeInches,
        dpi=dpi,
        starsCsvPath=starsCsvPath,
    )
