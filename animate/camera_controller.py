"""Camera zoom and visibility helpers for the main animation product."""

from __future__ import annotations

import numpy as np

from animate.animation_styles import (
    AXIS_LIMIT_2D_AU,
    MAX_CAMERA_DISTANCE_AU,
    MIN_CAMERA_DISTANCE_AU,
    VISIBILITY_FADE_SPAN_AU,
    ZOOM_IN_FRAME_FRACTION,
    ZOOM_STAGES,
)


class CameraController:
    """Staged log-zoom camera for 3D; fixed axis limit for 2D."""

    def __init__(self, is3d: bool, animationFrames: int, baseAnimationSpeed: float):
        self.is3d = is3d
        self.animationFrames = animationFrames
        self.baseAnimationSpeed = baseAnimationSpeed

    def distanceAu(self, frame: int) -> float:
        if not self.is3d:
            return AXIS_LIMIT_2D_AU

        zoomFrameCount = max(int(self.animationFrames * ZOOM_IN_FRAME_FRACTION), 1)
        if frame >= zoomFrameCount:
            return MIN_CAMERA_DISTANCE_AU

        zoomProgress = frame / zoomFrameCount
        for stageIndex in range(len(ZOOM_STAGES) - 1):
            progressStart, distanceStartAu = ZOOM_STAGES[stageIndex]
            progressEnd, distanceEndAu = ZOOM_STAGES[stageIndex + 1]
            if zoomProgress <= progressEnd:
                segmentSpan = progressEnd - progressStart
                segmentProgress = (
                    (zoomProgress - progressStart) / segmentSpan if segmentSpan > 0 else 1.0
                )
                logDistance = np.log(distanceStartAu) + segmentProgress * (
                    np.log(distanceEndAu) - np.log(distanceStartAu)
                )
                return float(np.exp(logDistance))

        return MIN_CAMERA_DISTANCE_AU

    def animationSpeed(self, cameraDistanceAu: float) -> float:
        if not self.is3d:
            return self.baseAnimationSpeed
        zoomFactor = (cameraDistanceAu / MAX_CAMERA_DISTANCE_AU) ** 0.15
        return max(0.75, zoomFactor) * self.baseAnimationSpeed

    @staticmethod
    def visibilityAlpha(positionX, positionY, positionZ, cameraDistanceAu: float) -> np.ndarray:
        distances = np.sqrt(positionX**2 + positionY**2 + positionZ**2)
        return np.clip(0.8 * (1 - distances / (cameraDistanceAu * 1.5)), 0.05, 0.8)

    @staticmethod
    def groupFadeAlpha(
        cameraDistanceAu: float,
        showBelowAu: float | None,
        showAboveAu: float | None,
        baseAlpha: float,
    ) -> float:
        if showBelowAu is not None and cameraDistanceAu > showBelowAu + VISIBILITY_FADE_SPAN_AU:
            return 0.0
        if showAboveAu is not None and cameraDistanceAu < showAboveAu - VISIBILITY_FADE_SPAN_AU:
            return 0.0

        fadeAlpha = baseAlpha
        if showBelowAu is not None and cameraDistanceAu > showBelowAu:
            fadeProgress = (cameraDistanceAu - showBelowAu) / VISIBILITY_FADE_SPAN_AU
            fadeAlpha *= max(0.0, 1.0 - fadeProgress)
        if showAboveAu is not None and cameraDistanceAu < showAboveAu:
            fadeProgress = (showAboveAu - cameraDistanceAu) / VISIBILITY_FADE_SPAN_AU
            fadeAlpha *= max(0.0, 1.0 - fadeProgress)
        return fadeAlpha
