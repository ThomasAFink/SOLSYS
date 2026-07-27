"""Side-product: dark 3D interstellar neighborhood star map."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from solsys.physics import AstronomicalConstants, StarCatalog

# Fallbacks when calling without going through render.py
DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 150
NEIGHBORHOOD_MAX_DISTANCE_LY = 10.0


class InterstellarNeighborhoodVisualizer:
    """Dark 3D star map in light-year coordinates (no planets/belts)."""

    def __init__(
        self,
        starsCsvPath: str,
        maxDistanceLightYears: float = NEIGHBORHOOD_MAX_DISTANCE_LY,
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
    ):
        self.constants = AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self.maxDistanceLightYears = maxDistanceLightYears
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi

    def starsWithinRadius(self) -> pd.DataFrame:
        nearbyStars = self.starCatalog.starsWithinLightYears(self.maxDistanceLightYears).copy()
        # Exclude the Sun row — plotted at the origin separately.
        nearbyStars = nearbyStars[
            ~nearbyStars['System'].astype(str).str.contains('Solar System', na=False)
        ].copy()
        lightYearToAu = self.constants.lightYearToAu
        nearbyStars['positionXLy'] = nearbyStars['positionX'] / lightYearToAu
        nearbyStars['positionYLy'] = nearbyStars['positionY'] / lightYearToAu
        nearbyStars['positionZLy'] = nearbyStars['positionZ'] / lightYearToAu
        return nearbyStars.dropna(subset=['positionXLy', 'positionYLy', 'positionZLy'])

    @staticmethod
    def _starLabel(starRow: pd.Series) -> str:
        if 'StarName' in starRow.index and pd.notnull(starRow['StarName']):
            return str(starRow['StarName'])
        return str(starRow['System'])

    def render(
        self,
        outputPath: str,
        showPlot: bool = False,
    ) -> None:
        nearbyStars = self.starsWithinRadius()
        coordinates = nearbyStars[['positionXLy', 'positionYLy', 'positionZLy']]
        maxRangeLightYears = float(np.max(np.abs(coordinates.values)))

        figure = plt.figure(figsize=self.figureSizeInches)
        axes = figure.add_subplot(111, projection='3d')
        axes.set_facecolor('black')
        figure.patch.set_facecolor('black')

        for _index, starRow in nearbyStars.iterrows():
            starName = self._starLabel(starRow)
            axes.scatter(
                starRow['positionXLy'],
                starRow['positionYLy'],
                starRow['positionZLy'],
                color='white',
                s=100,
                depthshade=True,
                marker='o',
            )
            axes.text(
                starRow['positionXLy'],
                starRow['positionYLy'],
                starRow['positionZLy'],
                starName,
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
        axes.set_title(
            f'3D Interstellar Neighborhood ({self.maxDistanceLightYears:g} light years)',
            color='white',
        )

        os.makedirs(os.path.dirname(outputPath) or '.', exist_ok=True)
        figure.savefig(outputPath, dpi=self.dpi, facecolor=figure.get_facecolor())
        print(f'Saved to {outputPath}')
        if showPlot:
            plt.show()
        else:
            plt.close(figure)


def renderNeighborhood(
    starsCsvPath: str = 'data/nearby_stars_30.csv',
    maxDistanceLightYears: float = NEIGHBORHOOD_MAX_DISTANCE_LY,
    outputPath: str | None = None,
    showPlot: bool = False,
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
) -> None:
    if outputPath is None:
        lyLabel = f'{maxDistanceLightYears:g}'.replace('.', 'p')
        outputPath = f'output/neighborhood/interstellar_neighborhood_{lyLabel}ly.jpg'
    visualizer = InterstellarNeighborhoodVisualizer(
        starsCsvPath,
        maxDistanceLightYears,
        figureSizeInches=figureSizeInches,
        dpi=dpi,
    )
    visualizer.render(outputPath=outputPath, showPlot=showPlot)
