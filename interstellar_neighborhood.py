"""3D neighborhood map of stars within a given light-year radius."""

from __future__ import annotations

import os
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from solsys_core import AstronomicalConstants, OrbitCalculator, StarCatalog

FIGURE_SIZE_INCHES = (12, 12)
MAX_DISTANCE_LIGHT_YEARS = 10


class InterstellarNeighborhoodVisualizer:
    def __init__(self, starsCsvPath: str, maxDistanceLightYears: float = MAX_DISTANCE_LIGHT_YEARS):
        self.constants = AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self.maxDistanceLightYears = maxDistanceLightYears
        self.orbitCalculator = OrbitCalculator()

    def starsWithinRadius(self) -> pd.DataFrame:
        starsFrame = self.starCatalog.starsDataFrame
        nearbyStars = starsFrame[
            (starsFrame['Distance (ly)'] <= self.maxDistanceLightYears)
            & starsFrame['rightAscensionDeg'].notna()
            & starsFrame['declinationDeg'].notna()
        ].copy()
        nearbyStars = nearbyStars.iloc[1:].copy()

        cartesianCoords = nearbyStars.apply(
            lambda row: self._equatorialToCartesianLightYears(
                row['rightAscensionDeg'],
                row['declinationDeg'],
                row['Distance (ly)'],
            ),
            axis=1,
        )
        nearbyStars['positionX'] = cartesianCoords.apply(lambda coord: coord[0])
        nearbyStars['positionY'] = cartesianCoords.apply(lambda coord: coord[1])
        nearbyStars['positionZ'] = cartesianCoords.apply(lambda coord: coord[2])
        return nearbyStars.dropna(subset=['positionX', 'positionY', 'positionZ'])

    @staticmethod
    def _equatorialToCartesianLightYears(
        rightAscensionDeg: float, declinationDeg: float, distanceLightYears: float
    ) -> Tuple[float, float, float]:
        rightAscensionRad = np.radians(rightAscensionDeg)
        declinationRad = np.radians(declinationDeg)
        positionX = distanceLightYears * np.cos(declinationRad) * np.cos(rightAscensionRad)
        positionY = distanceLightYears * np.cos(declinationRad) * np.sin(rightAscensionRad)
        positionZ = distanceLightYears * np.sin(declinationRad)
        return positionX, positionY, positionZ

    def render(self, showPlot: bool = True) -> None:
        nearbyStars = self.starsWithinRadius()
        validCoordinates = nearbyStars[['positionX', 'positionY', 'positionZ']].dropna()
        maxRangeLightYears = float(np.max(np.abs(validCoordinates.values)))

        figure = plt.figure(figsize=FIGURE_SIZE_INCHES)
        axes = figure.add_subplot(111, projection='3d')
        axes.set_facecolor('black')

        for index, starRow in validCoordinates.iterrows():
            starName = (
                nearbyStars.loc[index, 'Star or (sub-) brown dwarf']
                if pd.notnull(nearbyStars.loc[index, 'Star or (sub-) brown dwarf'])
                else nearbyStars.loc[index, 'System']
            )
            markerColor = 'yellow' if 'Sun' in str(starName) else 'white'
            axes.scatter(
                starRow['positionX'],
                starRow['positionY'],
                starRow['positionZ'],
                color=markerColor,
                s=100,
                depthshade=True,
                marker='o',
            )
            axes.text(
                starRow['positionX'],
                starRow['positionY'],
                starRow['positionZ'],
                str(starName),
                color='white',
                fontsize=5,
                ha='center',
            )

        axes.scatter(0, 0, 0, color='yellow', s=200, depthshade=True, marker='o')
        axes.text(0, 0, 0, 'Sun', color='white', fontsize=8, ha='center')
        axes.set_xlim([-maxRangeLightYears, maxRangeLightYears])
        axes.set_ylim([-maxRangeLightYears, maxRangeLightYears])
        axes.set_zlim([-maxRangeLightYears, maxRangeLightYears])
        axes.axis('off')
        axes.set_title('3D Interstellar Neighborhood with Spheres', color='white')

        if showPlot:
            plt.show()
        else:
            plt.close(figure)


if __name__ == '__main__':
    visualizer = InterstellarNeighborhoodVisualizer('data/nearby_stars_30.csv')
    visualizer.render()
