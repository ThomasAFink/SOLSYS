"""Animated population motion (no plotting)."""

from solsys.motion.animated_asteroid_population import (
    AnimatedAsteroidPopulation,
    AsteroidPopulationCounts,
)
from solsys.motion.mean_anomaly import meanAnomalyAtFrame, planetMeanAnomalyRad

__all__ = [
    'AnimatedAsteroidPopulation',
    'AsteroidPopulationCounts',
    'meanAnomalyAtFrame',
    'planetMeanAnomalyRad',
]
