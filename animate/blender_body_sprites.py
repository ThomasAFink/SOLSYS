"""Load Blender spin-loop RGBA frames for the Sol→Centauri cinematic.

Assets come from ``render.py blender --spin`` (fixed camera, full rotation,
transparent PNGs). The cinematic indexes into that loop each ``update()`` —
rendered together, not GIF-concatenated afterward.

Earth: free spin by animation frame.
Moon: tidally locked — spin index follows orbital mean anomaly (one turn / orbit).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from animate.scenes.blender.flyby_scene import spinFramesDirectory

DEFAULT_DISPLAY_RESOLUTION = 384


def loadSpinLoopFrames(
    bodyName: str,
    theme: str,
    *,
    outputDirectory: Path | str = 'output/animate/blender',
) -> list[np.ndarray] | None:
    """Load persistent ``*_spin_<theme>/frame_*.png`` as float RGBA arrays."""
    directory = spinFramesDirectory(bodyName, theme, outputDirectory=outputDirectory)
    paths = sorted(directory.glob('frame_*.png'))
    if not paths:
        return None

    frames: list[np.ndarray] = []
    for path in paths:
        with Image.open(path) as image:
            frames.append(np.asarray(image.convert('RGBA'), dtype=np.float32) / 255.0)
    return frames


def _resizeRgba(rgba: np.ndarray, size: int) -> np.ndarray:
    if rgba.shape[0] == size and rgba.shape[1] == size:
        return rgba
    image = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


def tidalLockFrameIndex(orbitalPhaseRad: float, frameCount: int) -> int:
    """Map orbital mean anomaly → spin-loop index (synchronous rotation)."""
    return int(np.floor(tidalLockFramePosition(orbitalPhaseRad, frameCount))) % frameCount


def tidalLockFramePosition(orbitalPhaseRad: float, frameCount: int) -> float:
    """Fractional spin-loop position for smooth tidally locked blending."""
    if frameCount <= 0:
        raise ValueError('frameCount must be positive')
    turns = float(orbitalPhaseRad) / (2.0 * np.pi)
    return turns * frameCount


def _blendSpinFrames(frames: list[np.ndarray], position: float) -> np.ndarray:
    """Linear blend between neighboring loop frames (seamless at wrap)."""
    count = len(frames)
    wrapped = position % count
    index0 = int(np.floor(wrapped)) % count
    index1 = (index0 + 1) % count
    # Fractional part of floor is always in [0, 1) — no near-1.0 branch needed.
    blend = float(wrapped - np.floor(wrapped))
    if blend <= 1e-6:
        return frames[index0]
    return frames[index0] * (1.0 - blend) + frames[index1] * blend


class BlenderBodySpriteAtlas:
    """Theme-scoped Earth/Moon Blender spin loops for in-animation billboards."""

    def __init__(self, theme: str, *, outputDirectory: Path | str = 'output/animate/blender'):
        self.theme = theme
        self.earth = loadSpinLoopFrames('Earth', theme, outputDirectory=outputDirectory)
        self.moon = loadSpinLoopFrames('Moon', theme, outputDirectory=outputDirectory)

    @property
    def hasEarth(self) -> bool:
        return bool(self.earth)

    @property
    def hasMoon(self) -> bool:
        return bool(self.moon)

    def earthFrame(
        self, frame: int, *, resolution: int = DEFAULT_DISPLAY_RESOLUTION
    ) -> np.ndarray | None:
        if not self.earth:
            return None
        # Fractional position so Earth also eases between PNG samples.
        position = float(frame) % len(self.earth)
        rgba = _blendSpinFrames(self.earth, position)
        return _resizeRgba(rgba, resolution)

    def moonFrame(
        self,
        frame: int,
        *,
        orbitalPhaseRad: float | None = None,
        resolution: int = DEFAULT_DISPLAY_RESOLUTION,
    ) -> np.ndarray | None:
        """Moon disk. Prefer ``orbitalPhaseRad`` for tidal lock; ``frame`` is fallback."""
        if not self.moon:
            return None
        if orbitalPhaseRad is None:
            position = float(frame) % len(self.moon)
        else:
            position = tidalLockFramePosition(orbitalPhaseRad, len(self.moon))
        rgba = _blendSpinFrames(self.moon, position)
        return _resizeRgba(rgba, resolution)
