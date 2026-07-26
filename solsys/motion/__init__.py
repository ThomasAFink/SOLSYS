"""Animated population motion (no plotting)."""

from solsys.motion.mean_anomaly import meanAnomalyAtFrame, planetMeanAnomalyRad
from solsys.motion.animated_asteroid_population import AnimatedAsteroidPopulation, AsteroidPopulationCounts

__all__ = [
    'AnimatedAsteroidPopulation',
    'AsteroidPopulationCounts',
    'meanAnomalyAtFrame',
    'planetMeanAnomalyRad',
]
