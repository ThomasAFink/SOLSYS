"""Scatter-point counts per belt/group for each zoom level."""

from __future__ import annotations


class PointDensityConfig:
    """Scatter-point counts per belt/group for each zoom level."""

    DENSITIES_BY_VIEW: dict[str, tuple[int, int, int, int, int]] = {
        '0_inner_solar_system': (20000, 4000, 4000, 10000, 50000),
        '1_inner_solar_system_with_jupiter': (10000, 2000, 1000, 10000, 50000),
        '2_solar_system_with_kuiper_belt': (500, 20, 15, 10000, 50000),
        '3_solar_system_with_oort_cloud': (20, 10, 100, 100, 50000),
        '4_solar_system_with_alpha_centauri': (10, 5, 5, 50, 5000),
        '5_solar_system_with_nearest_stars_10': (2, 2, 2, 20, 2000),
        'inner_solar_system': (20000, 4000, 4000, 10000, 50000),
        'inner_solar_system_with_jupiter': (10000, 2000, 2000, 10000, 50000),
        'solar_system_with_kuiper_belt': (200, 100, 50, 10000, 50000),
        'solar_system_with_oort_cloud': (20, 10, 10, 100, 50000),
        'solar_system_with_alpha_centauri': (10, 5, 5, 50, 5000),
        'solar_system_with_nearest_stars_10': (2, 2, 2, 20, 2000),
        'solar_system_with_nearest_stars_25': (1, 1, 1, 10, 1000),
        'solar_system_with_nearest_stars_30': (1, 1, 1, 10, 1000),
        'default': (1, 1, 1, 10, 1000),
    }

    @classmethod
    def forView(cls, viewId: str) -> dict[str, int]:
        densities = cls.DENSITIES_BY_VIEW.get(viewId, cls.DENSITIES_BY_VIEW['default'])
        return {
            'asteroidBelt': densities[0],
            'trojansAndGreeks': densities[1],
            'hildas': densities[2],
            'kuiperBelt': densities[3],
            'oortCloud': densities[4],
        }
