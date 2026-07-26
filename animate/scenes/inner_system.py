"""Fixed inner-system 2D animation scene."""

from animate.solar_system_animator import SolarSystemAnimator


def create(style: str = 'default') -> SolarSystemAnimator:
    return SolarSystemAnimator(dimension='2d', style=style)
