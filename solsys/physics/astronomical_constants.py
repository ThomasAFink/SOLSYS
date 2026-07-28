"""Astronomical constants used across SOLSYS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AstronomicalConstants:
    plutoSemiMajorAxis: float = 39.482
    plutoEccentricity: float = 0.2488
    asteroidBeltInnerAu: float = 2.2
    asteroidBeltOuterAu: float = 3.2
    kuiperBeltInnerAu: float = 30.0
    kuiperBeltOuterAu: float = 55.0
    jupiterSemiMajorAxisAu: float = 5.2
    jupiterInclinationDeg: float = 1.3
    jupiterEccentricity: float = 0.0489
    oortCloudInnerAu: float = 2000.0
    oortCloudOuterAu: float = 100000.0
    lightYearToAu: float = 63241.077
    # ʻOumuamua elements also live in data/interstellar_objects.csv (preferred for new scenes).
    oumuamuaEccentricity: float = 1.2011
    oumuamuaPerihelionAu: float = 0.2559
    oumuamuaInclinationDeg: float = 122.74
    oumuamuaLongitudeAscendingNodeDeg: float = 24.60
    oumuamuaArgumentOfPerihelionDeg: float = 241.69
    # Published Earth flyby (2017-10-14); used for callouts / Earth phasing in the flyby scene.
    oumuamuaEarthClosestApproachAu: float = 0.1618
    auToKm: float = 149597870.7

    @property
    def plutoPerihelionAu(self) -> float:
        return self.plutoSemiMajorAxis * (1 - self.plutoEccentricity)

    @property
    def plutoAphelionAu(self) -> float:
        return self.plutoSemiMajorAxis * (1 + self.plutoEccentricity)
