"""Planet orbital catalog."""

from __future__ import annotations

from dataclasses import dataclass

from solsys.physics.astronomical_constants import AstronomicalConstants


@dataclass(frozen=True)
class PlanetOrbit:
    name: str
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    color: str
    diameterKm: int
    orbitalPeriodDays: float


class PlanetCatalog:
    def __init__(self, constants: AstronomicalConstants):
        self.constants = constants
        self.planets: dict[str, PlanetOrbit] = self._buildCatalog()

    def _buildCatalog(self) -> dict[str, PlanetOrbit]:
        constants = self.constants
        return {
            'Mercury': PlanetOrbit('Mercury', 0.387, 0.205, 7.0, 'gray', 4879, 88),
            'Venus': PlanetOrbit('Venus', 0.723, 0.007, 3.4, 'yellow', 12104, 224.7),
            'Earth': PlanetOrbit('Earth', 1.00, 0.017, 0.0, 'blue', 12742, 365.2),
            'Mars': PlanetOrbit('Mars', 1.52, 0.093, 1.85, 'red', 6779, 687),
            'Jupiter': PlanetOrbit('Jupiter', 5.20, 0.048, 1.3, 'orange', 139822, 4331),
            'Saturn': PlanetOrbit('Saturn', 9.58, 0.056, 2.49, 'gold', 116464, 10747),
            'Uranus': PlanetOrbit('Uranus', 19.22, 0.046, 0.77, 'lightblue', 50724, 30589),
            'Neptune': PlanetOrbit('Neptune', 30.05, 0.010, 1.77, 'blue', 49244, 59800),
            'Pluto': PlanetOrbit(
                'Pluto',
                constants.plutoSemiMajorAxis,
                constants.plutoEccentricity,
                17.16,
                'brown',
                2376,
                90560,
            ),
        }
