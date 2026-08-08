"""Body-centered camera path for Blender planet flybys (host + tests, no bpy)."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class FlybyCameraSample:
    frame: int
    cameraAu: tuple[float, float, float]
    bodyRotationDeg: float


def flybyCameraLocation(
    frame: int,
    frameCount: int,
    displayRadiusAu: float,
    *,
    distanceScale: float = 4.4,
    elevationDeg: float = 16.0,
    azimuthStartDeg: float = 35.0,
    azimuthSweepDeg: float = 220.0,
) -> tuple[float, float, float]:
    """Camera position on an elevated arc around a body at the origin."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')
    progress = frame / (frameCount - 1)
    azimuthRad = math.radians(azimuthStartDeg + progress * azimuthSweepDeg)
    elevationRad = math.radians(elevationDeg)
    distance = displayRadiusAu * distanceScale
    horizontal = distance * math.cos(elevationRad)
    return (
        horizontal * math.cos(azimuthRad),
        horizontal * math.sin(azimuthRad),
        distance * math.sin(elevationRad),
    )


def buildFlybyCameraPath(
    displayRadiusAu: float,
    frameCount: int = 72,
    *,
    bodySpinDeg: float = 140.0,
) -> tuple[FlybyCameraSample, ...]:
    samples: list[FlybyCameraSample] = []
    for frame in range(frameCount):
        progress = frame / (frameCount - 1)
        samples.append(
            FlybyCameraSample(
                frame=frame,
                cameraAu=flybyCameraLocation(frame, frameCount, displayRadiusAu),
                bodyRotationDeg=progress * bodySpinDeg,
            )
        )
    return tuple(samples)


def buildSpinCameraPath(
    displayRadiusAu: float,
    frameCount: int = 48,
    *,
    elevationDeg: float = 18.0,
    azimuthDeg: float = 35.0,
) -> tuple[FlybyCameraSample, ...]:
    """Fixed camera + full 360° body spin (seamless loop for cinematic reuse)."""
    if frameCount < 2:
        raise ValueError('frameCount must be >= 2')
    cameraAu = flybyCameraLocation(
        0,
        2,
        displayRadiusAu,
        elevationDeg=elevationDeg,
        azimuthStartDeg=azimuthDeg,
        azimuthSweepDeg=0.0,
    )
    samples: list[FlybyCameraSample] = []
    for frame in range(frameCount):
        # Omit 360° on the last frame so index 0 and N line up when looping.
        samples.append(
            FlybyCameraSample(
                frame=frame,
                cameraAu=cameraAu,
                bodyRotationDeg=(frame / frameCount) * 360.0,
            )
        )
    return tuple(samples)
