"""Confirmed interstellar visitors (1I / 2I / 3I) from CSV orbital elements."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_INTERSTELLAR_OBJECTS_CSV = 'data/interstellar_objects.csv'


@dataclass(frozen=True)
class InterstellarObject:
    objectId: str
    designation: str
    displayName: str
    perihelionAu: float
    eccentricity: float
    inclinationDeg: float
    longitudeAscendingNodeDeg: float
    argumentOfPerihelionDeg: float
    earthClosestApproachAu: float | None
    colorLight: str
    colorDark: str
    maxHeliocentricAu: float
    trueAnomalySpanDeg: float
    highlight: str  # earth_flyby | perihelion
    notes: str = ''

    def colorForStyle(self, isDark: bool) -> str:
        return self.colorDark if isDark else self.colorLight


class InterstellarObjectCatalog:
    """Load hyperbolic interstellar-object elements from CSV."""

    def __init__(self, csvPath: str = DEFAULT_INTERSTELLAR_OBJECTS_CSV):
        path = Path(csvPath)
        if not path.is_file():
            raise FileNotFoundError(f'Interstellar objects CSV not found: {csvPath}')
        self._frame = pd.read_csv(path)
        self._byId: dict[str, InterstellarObject] = {}
        for _, row in self._frame.iterrows():
            objectId = str(row['object_id'])
            earthClosest = row['earth_closest_approach_au']
            earthClosestAu = None if pd.isna(earthClosest) else float(earthClosest)
            self._byId[objectId] = InterstellarObject(
                objectId=objectId,
                designation=str(row['designation']),
                displayName=str(row['display_name']),
                perihelionAu=float(row['perihelion_au']),
                eccentricity=float(row['eccentricity']),
                inclinationDeg=float(row['inclination_deg']),
                longitudeAscendingNodeDeg=float(row['longitude_ascending_node_deg']),
                argumentOfPerihelionDeg=float(row['argument_of_perihelion_deg']),
                earthClosestApproachAu=earthClosestAu,
                colorLight=str(row['color_light']),
                colorDark=str(row['color_dark']),
                maxHeliocentricAu=float(row['max_heliocentric_au']),
                trueAnomalySpanDeg=float(row['true_anomaly_span_deg']),
                highlight=str(row['highlight']),
                notes='' if pd.isna(row.get('notes')) else str(row['notes']),
            )

    def listObjectIds(self) -> list[str]:
        return list(self._byId.keys())

    def load(self, objectId: str) -> InterstellarObject:
        try:
            return self._byId[objectId]
        except KeyError as error:
            known = ', '.join(self.listObjectIds()) or '(none)'
            raise KeyError(
                f'Unknown interstellar object_id: {objectId!r}. Known: {known}'
            ) from error

    def all(self) -> tuple[InterstellarObject, ...]:
        return tuple(self._byId[objectId] for objectId in self.listObjectIds())
