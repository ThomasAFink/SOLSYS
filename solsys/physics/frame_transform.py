"""Sol ↔ Alpha Centauri barycenter frame transforms.

Sol scenes use a heliocentric / Sol-barycentric XYZ frame in AU from
``OrbitCalculator.equatorialToCartesianAu`` (RA/Dec → Cartesian).

Alpha Centauri top-down scenes use a separate **AB-barycentric** frame whose
XY plane is the binary orbital plane (face-on schematic). Sky inclination and
Ω are stored on the A/B stellar orbits and are applied here when mapping into
Sol XYZ — they are not applied in the 2D Centauri schematics themselves.

Transform (Centauri → Sol):

1. Rotate orbital-plane coordinates with the same ``(i, Ω)`` convention as
   ``OrbitCalculator.ellipticalPosition`` (argument of periapsis stays in the
   Centauri-plane schematic, matching ``bodyPositionInOrbitalPlane``).
2. Translate by the mass-weighted AB barycenter in Sol XYZ
   (``barycenter_policy=ab_mass_weighted``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solsys.physics.catalogs.system_catalog import StarMember, StarSystem, StellarOrbit

ALPHA_CENTAURI_SYSTEM_ID = 'alpha_centauri'
PRIMARY_ROLE = 'primary'
SECONDARY_ROLE = 'secondary'


def orbitalPlaneToSolRotation(
    inclinationDeg: float,
    longitudeAscendingNodeDeg: float,
) -> np.ndarray:
    """Return 3×3 matrix mapping orbital-plane coords → Sol XYZ.

    Columns are the local +X / +Y / +Z unit vectors expressed in Sol XYZ.
    Matches ``OrbitCalculator.ellipticalPosition`` for in-plane ``(x, y, 0)``.
    """
    inclinationRad = np.radians(inclinationDeg)
    ascendingNodeRad = np.radians(longitudeAscendingNodeDeg)
    cosI = float(np.cos(inclinationRad))
    sinI = float(np.sin(inclinationRad))
    cosO = float(np.cos(ascendingNodeRad))
    sinO = float(np.sin(ascendingNodeRad))

    # Local +X along the line of nodes projection used by ellipticalPosition.
    axisX = np.array([cosO, sinO, 0.0], dtype=float)
    axisY = np.array([-cosI * sinO, cosI * cosO, sinI], dtype=float)
    axisZ = np.array([sinI * sinO, -sinI * cosO, cosI], dtype=float)
    return np.column_stack((axisX, axisY, axisZ))


def abBarycenterSolPositionAu(system: StarSystem) -> np.ndarray:
    """Mass-weighted α Cen A–B barycenter in Sol XYZ (AU)."""
    if system.systemId != ALPHA_CENTAURI_SYSTEM_ID:
        raise ValueError(
            f'Expected system_id={ALPHA_CENTAURI_SYSTEM_ID!r}, got {system.systemId!r}'
        )
    if system.barycenterPolicy != 'ab_mass_weighted':
        raise ValueError(
            f'Unsupported barycenter_policy={system.barycenterPolicy!r}; '
            "expected 'ab_mass_weighted'"
        )

    primaryOrbit = _requireOrbit(system, PRIMARY_ROLE)
    secondaryOrbit = _requireOrbit(system, SECONDARY_ROLE)
    primary = _requireStar(system, primaryOrbit.starUuid, PRIMARY_ROLE)
    secondary = _requireStar(system, secondaryOrbit.starUuid, SECONDARY_ROLE)

    massPrimary = primary.massSolar
    massSecondary = secondary.massSolar
    if massPrimary is None or massSecondary is None:
        raise ValueError('α Cen A and B must have Mass values for ab_mass_weighted barycenter')
    if massPrimary <= 0.0 or massSecondary <= 0.0:
        raise ValueError('α Cen A and B masses must be positive')

    positionPrimary = _requirePosition(primary, PRIMARY_ROLE)
    positionSecondary = _requirePosition(secondary, SECONDARY_ROLE)
    totalMass = massPrimary + massSecondary
    return (massPrimary * positionPrimary + massSecondary * positionSecondary) / totalMass


def _requireOrbit(system: StarSystem, role: str) -> StellarOrbit:
    for orbit in system.stellarOrbits:
        if orbit.role == role:
            return orbit
    raise KeyError(f'No stellar orbit with role={role!r} in {system.systemId!r}')


def _requireStar(system: StarSystem, starUuid: str, role: str) -> StarMember:
    star = system.starByUuid(starUuid)
    if star is None:
        raise KeyError(f'No star uuid={starUuid!r} for role={role!r}')
    return star


def _requirePosition(star: StarMember, role: str) -> np.ndarray:
    if star.positionX is None or star.positionY is None or star.positionZ is None:
        raise ValueError(f'Missing Sol-frame position for role={role!r} ({star.starName!r})')
    return np.array([star.positionX, star.positionY, star.positionZ], dtype=float)


def _asNx3(positionsAu: np.ndarray | list | tuple) -> np.ndarray:
    array = np.asarray(positionsAu, dtype=float)
    if array.ndim == 1:
        if array.shape[0] == 2:
            array = np.append(array, 0.0)
        elif array.shape[0] != 3:
            raise ValueError(f'Expected shape (2,) or (3,), got {array.shape}')
        return array.reshape(1, 3)
    if array.ndim != 2:
        raise ValueError(f'Expected 1D or 2D positions, got ndim={array.ndim}')
    if array.shape[1] == 2:
        zeros = np.zeros((array.shape[0], 1), dtype=float)
        return np.hstack((array, zeros))
    if array.shape[1] == 3:
        return array
    raise ValueError(f'Expected N×2 or N×3 positions, got shape {array.shape}')


@dataclass(frozen=True)
class SolCentauriFrameTransform:
    """Map α Cen AB-barycentric orbital-plane coords ↔ Sol XYZ (AU)."""

    originSolAu: np.ndarray
    rotationCentauriToSol: np.ndarray
    inclinationDeg: float
    longitudeAscendingNodeDeg: float

    @classmethod
    def fromStarSystem(cls, system: StarSystem) -> SolCentauriFrameTransform:
        """Build from a loaded ``alpha_centauri`` ``StarSystem``."""
        primaryOrbit = _requireOrbit(system, PRIMARY_ROLE)
        secondaryOrbit = _requireOrbit(system, SECONDARY_ROLE)
        if (
            primaryOrbit.inclinationDeg != secondaryOrbit.inclinationDeg
            or primaryOrbit.longitudeAscendingNodeDeg != secondaryOrbit.longitudeAscendingNodeDeg
        ):
            raise ValueError('α Cen A and B must share inclination and Ω for the AB frame')

        origin = abBarycenterSolPositionAu(system)
        rotation = orbitalPlaneToSolRotation(
            primaryOrbit.inclinationDeg,
            primaryOrbit.longitudeAscendingNodeDeg,
        )
        return cls(
            originSolAu=origin,
            rotationCentauriToSol=rotation,
            inclinationDeg=float(primaryOrbit.inclinationDeg),
            longitudeAscendingNodeDeg=float(primaryOrbit.longitudeAscendingNodeDeg),
        )

    def toSol(self, centauriPositionsAu: np.ndarray | list | tuple) -> np.ndarray:
        """Map Centauri orbital-plane coords (N×2/N×3) → Sol XYZ (N×3)."""
        local = _asNx3(centauriPositionsAu)
        rotated = local @ self.rotationCentauriToSol.T
        return rotated + self.originSolAu

    def toCentauri(self, solPositionsAu: np.ndarray | list | tuple) -> np.ndarray:
        """Map Sol XYZ (N×3) → Centauri orbital-plane coords (N×3; Z out of plane)."""
        array = np.asarray(solPositionsAu, dtype=float)
        if array.ndim == 1:
            if array.shape[0] != 3:
                raise ValueError(f'Expected shape (3,), got {array.shape}')
            array = array.reshape(1, 3)
        elif array.ndim != 2 or array.shape[1] != 3:
            raise ValueError(f'Expected N×3 Sol positions, got shape {array.shape}')
        relative = array - self.originSolAu
        return relative @ self.rotationCentauriToSol
