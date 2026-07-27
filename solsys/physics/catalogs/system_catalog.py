"""Star-system catalog: systems, member stars, stellar orbits, planets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from solsys.physics.astronomical_constants import AstronomicalConstants
from solsys.physics.catalogs.star_catalog import StarCatalog

DEFAULT_SYSTEMS_CSV = 'data/systems.csv'
DEFAULT_STELLAR_ORBITS_CSV = 'data/stellar_orbits.csv'
DEFAULT_PLANETS_CSV = 'data/planets.csv'
DEFAULT_STARS_CSV = 'data/nearby_stars_30.csv'


@dataclass(frozen=True)
class StarMember:
    starUuid: str
    systemId: str
    displaySystem: str
    starName: str
    massSolar: float | None
    distanceLy: float | None
    positionX: float | None
    positionY: float | None
    positionZ: float | None


@dataclass(frozen=True)
class StellarOrbit:
    systemId: str
    starUuid: str
    role: str
    periodDays: float
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    longitudeAscendingNodeDeg: float
    argumentPeriapsisDeg: float
    meanAnomalyDegEpoch: float
    notes: str = ''


@dataclass(frozen=True)
class SystemPlanet:
    planetId: str
    systemId: str
    hostStarUuid: str
    name: str
    semiMajorAxisAu: float
    eccentricity: float
    inclinationDeg: float
    longitudeAscendingNodeDeg: float
    argumentPeriapsisDeg: float
    orbitalPeriodDays: float
    color: str
    diameterKm: float
    confidence: str
    notes: str = ''


@dataclass(frozen=True)
class StarSystem:
    systemId: str
    displayName: str
    barycenterPolicy: str
    notes: str
    stars: tuple[StarMember, ...]
    stellarOrbits: tuple[StellarOrbit, ...]
    planets: tuple[SystemPlanet, ...]

    def starByUuid(self, starUuid: str) -> StarMember | None:
        for star in self.stars:
            if star.starUuid == starUuid:
                return star
        return None

    def orbitByUuid(self, starUuid: str) -> StellarOrbit | None:
        for orbit in self.stellarOrbits:
            if orbit.starUuid == starUuid:
                return orbit
        return None

    def planetsForHost(self, hostStarUuid: str) -> tuple[SystemPlanet, ...]:
        return tuple(planet for planet in self.planets if planet.hostStarUuid == hostStarUuid)


class SystemCatalog:
    """Load star systems by system_id (e.g. alpha_centauri = A+B+Proxima)."""

    def __init__(
        self,
        starsCsvPath: str = DEFAULT_STARS_CSV,
        systemsCsvPath: str = DEFAULT_SYSTEMS_CSV,
        stellarOrbitsCsvPath: str = DEFAULT_STELLAR_ORBITS_CSV,
        planetsCsvPath: str = DEFAULT_PLANETS_CSV,
        constants: AstronomicalConstants | None = None,
    ):
        self.constants = constants or AstronomicalConstants()
        self.starCatalog = StarCatalog(starsCsvPath, self.constants)
        self._systemsFrame = pd.read_csv(systemsCsvPath)
        self._orbitsFrame = pd.read_csv(stellarOrbitsCsvPath)
        self._planetsFrame = pd.read_csv(planetsCsvPath)

    def listSystemIds(self) -> list[str]:
        return [str(value) for value in self._systemsFrame['system_id'].tolist()]

    def load(self, systemId: str) -> StarSystem:
        systemRows = self._systemsFrame[self._systemsFrame['system_id'] == systemId]
        if systemRows.empty:
            raise KeyError(f'Unknown system_id: {systemId!r}')
        systemRow = systemRows.iloc[0]

        starsFrame = self.starCatalog.starsDataFrame
        if 'system_id' not in starsFrame.columns:
            raise KeyError('nearby stars CSV is missing system_id column')
        memberRows = starsFrame[starsFrame['system_id'] == systemId]
        stars = tuple(self._starMemberFromRow(row) for _, row in memberRows.iterrows())

        orbitRows = self._orbitsFrame[self._orbitsFrame['system_id'] == systemId]
        orbits = tuple(self._stellarOrbitFromRow(row) for _, row in orbitRows.iterrows())

        planetRows = self._planetsFrame[self._planetsFrame['system_id'] == systemId]
        planets = tuple(self._planetFromRow(row) for _, row in planetRows.iterrows())

        return StarSystem(
            systemId=str(systemRow['system_id']),
            displayName=str(systemRow['display_name']),
            barycenterPolicy=str(systemRow['barycenter_policy']),
            notes=self._optionalStr(systemRow.get('notes')),
            stars=stars,
            stellarOrbits=orbits,
            planets=planets,
        )

    @staticmethod
    def _optionalFloat(value) -> float | None:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _optionalStr(value, default: str = '') -> str:
        if value is None or pd.isna(value):
            return default
        text = str(value).strip()
        return default if text.lower() == 'nan' else text

    def _starMemberFromRow(self, row: pd.Series) -> StarMember:
        return StarMember(
            starUuid=str(row['UUID']),
            systemId=str(row['system_id']),
            displaySystem=self._optionalStr(row.get('System')),
            starName=self._optionalStr(row.get('StarName')),
            massSolar=self._optionalFloat(row.get('Mass')),
            distanceLy=self._optionalFloat(row.get('Distance (ly)')),
            positionX=self._optionalFloat(row.get('positionX')),
            positionY=self._optionalFloat(row.get('positionY')),
            positionZ=self._optionalFloat(row.get('positionZ')),
        )

    def _stellarOrbitFromRow(self, row: pd.Series) -> StellarOrbit:
        return StellarOrbit(
            systemId=str(row['system_id']),
            starUuid=str(row['star_uuid']),
            role=str(row['role']),
            periodDays=float(row['period_days']),
            semiMajorAxisAu=float(row['semi_major_axis_au']),
            eccentricity=float(row['eccentricity']),
            inclinationDeg=float(row['inclination_deg']),
            longitudeAscendingNodeDeg=float(row['longitude_ascending_node_deg']),
            argumentPeriapsisDeg=float(row['argument_periapsis_deg']),
            meanAnomalyDegEpoch=float(row['mean_anomaly_deg_epoch']),
            notes=self._optionalStr(row.get('notes')),
        )

    def _planetFromRow(self, row: pd.Series) -> SystemPlanet:
        return SystemPlanet(
            planetId=str(row['planet_id']),
            systemId=str(row['system_id']),
            hostStarUuid=str(row['host_star_uuid']),
            name=str(row['name']),
            semiMajorAxisAu=float(row['semi_major_axis_au']),
            eccentricity=float(row['eccentricity']),
            inclinationDeg=float(row['inclination_deg']),
            longitudeAscendingNodeDeg=float(row['longitude_ascending_node_deg']),
            argumentPeriapsisDeg=float(row['argument_periapsis_deg']),
            orbitalPeriodDays=float(row['orbital_period_days']),
            color=str(row['color']),
            diameterKm=float(row['diameter_km']),
            confidence=str(row['confidence']),
            notes=self._optionalStr(row.get('notes')),
        )


def defaultDataPaths(repoRoot: str | Path | None = None) -> dict[str, str]:
    root = Path(repoRoot) if repoRoot else Path('.')
    return {
        'starsCsvPath': str(root / DEFAULT_STARS_CSV),
        'systemsCsvPath': str(root / DEFAULT_SYSTEMS_CSV),
        'stellarOrbitsCsvPath': str(root / DEFAULT_STELLAR_ORBITS_CSV),
        'planetsCsvPath': str(root / DEFAULT_PLANETS_CSV),
    }
