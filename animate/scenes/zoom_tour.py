"""Staged Oort-to-inner 3D zoom-tour animation scene."""

from animate.solar_system_animator import SolarSystemAnimator


def create(style: str = 'default') -> SolarSystemAnimator:
    return SolarSystemAnimator(dimension='3d', style=style)
