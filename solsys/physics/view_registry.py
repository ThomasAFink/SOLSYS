"""Registry of static 2D/3D zoom views and star-distance limits."""

from __future__ import annotations

from typing import Dict, List, Tuple

from solsys.physics.view_definition import ViewDefinition


class ViewRegistry:
    VIEWS_2D: List[ViewDefinition] = [
        ViewDefinition('0_inner_solar_system', -3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
        ViewDefinition('1_inner_solar_system_with_jupiter', -6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
        ViewDefinition('2_solar_system_with_kuiper_belt', -70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
        ViewDefinition('3_solar_system_with_oort_cloud', -100000, 100000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
        ViewDefinition('4_solar_system_with_alpha_centauri', -280000, 125000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
        ViewDefinition('5_solar_system_with_nearest_stars_10', -632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
        ViewDefinition('6_solar_system_with_nearest_stars_25', -1584189.9811, 1584189.9811, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years'),
        ViewDefinition('7_solar_system_with_nearest_stars_30', -1897232.3126, 1897232.3126, 80, 'solar_system_with_nearest_stars_30', 'Interstellar Neighbors Within 30 Light Years'),
    ]

    VIEWS_3D: List[ViewDefinition] = [
        ViewDefinition('0_inner_solar_system', -3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
        ViewDefinition('1_inner_solar_system_with_jupiter', -6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
        ViewDefinition('2_solar_system_with_kuiper_belt', -70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
        ViewDefinition('3_solar_system_with_oort_cloud', -100000, 100000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
        ViewDefinition('4_solar_system_with_alpha_centauri', -280000, 280000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
        ViewDefinition('5_solar_system_with_nearest_stars_10', -632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
        ViewDefinition('6_solar_system_with_nearest_stars_25', -1584188.9811, 1584188.9811, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years'),
        ViewDefinition('7_solar_system_with_nearest_stars_30', -1897232.3126, 1897232.3126, 80, 'solar_system_with_nearest_stars_30', 'Interstellar Neighbors Within 30 Light Years'),
    ]

    STAR_DISTANCE_LIMITS_LY: Dict[str, float] = {
        '5_solar_system_with_nearest_stars_10': 10,
        '6_solar_system_with_nearest_stars_25': 25.05,
        '7_solar_system_with_nearest_stars_30': 30,
        '4_solar_system_with_alpha_centauri': 5,
        'solar_system_with_nearest_stars_10': 10,
        'solar_system_with_nearest_stars_25': 25.05,
        'solar_system_with_nearest_stars_30': 30,
        'solar_system_with_alpha_centauri': 5,
    }

    @classmethod
    def axisLimitsForView(cls, viewId: str) -> Tuple[float, float]:
        for view in cls.VIEWS_2D:
            if view.viewId == viewId:
                return view.axisMinAu, view.axisMaxAu
        return -3.5, 3.5

    @classmethod
    def titleForView(cls, viewId: str) -> str:
        for view in cls.VIEWS_2D:
            if view.viewId == viewId:
                return view.title
        return 'Solar System'

    @classmethod
    def maxStarDistanceLy(cls, viewId: str) -> float:
        return cls.STAR_DISTANCE_LIMITS_LY.get(viewId, 25.05)
