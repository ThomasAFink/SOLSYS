"""Nearby-star catalog loaded from CSV with RA/Dec positions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.orbit_calculator import OrbitCalculator


class StarCatalog:
    def __init__(self, csvPath: str, constants: AstronomicalConstants):
        self.constants = constants
        self.starsDataFrame = self._loadStars(csvPath)

    def _loadStars(self, csvPath: str) -> pd.DataFrame:
        starsFrame = pd.read_csv(csvPath)
        starsFrame['distanceAu'] = starsFrame['Distance (ly)'] * self.constants.lightYearToAu
        coordinates = starsFrame.apply(
            lambda row: OrbitCalculator.parseRightAscensionAndDeclination(row['RA'], row['Dec']),
            axis=1,
        )
        starsFrame['rightAscensionDeg'] = coordinates.apply(lambda pair: pair[0])
        starsFrame['declinationDeg'] = coordinates.apply(lambda pair: pair[1])
        cartesianCoords = starsFrame.apply(
            lambda row: OrbitCalculator.equatorialToCartesianAu(
                row['rightAscensionDeg'], row['declinationDeg'], row['distanceAu']
            )
            if pd.notna(row['rightAscensionDeg']) and pd.notna(row['declinationDeg'])
            else (np.nan, np.nan, np.nan),
            axis=1,
        )
        starsFrame['positionX'] = cartesianCoords.apply(lambda coord: coord[0])
        starsFrame['positionY'] = cartesianCoords.apply(lambda coord: coord[1])
        starsFrame['positionZ'] = cartesianCoords.apply(lambda coord: coord[2])
        return starsFrame

    def vegaRow(self) -> pd.Series:
        return self.starsDataFrame[
            self.starsDataFrame['System'].str.startswith('Vega', na=False)
        ].iloc[0]

    def starsWithinLightYears(self, maxDistanceLy: float) -> pd.DataFrame:
        return self.starsDataFrame[
            (self.starsDataFrame['Distance (ly)'] <= maxDistanceLy)
            & self.starsDataFrame['positionX'].notna()
            & self.starsDataFrame['positionY'].notna()
        ]
