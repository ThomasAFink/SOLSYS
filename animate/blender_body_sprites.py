"""Load Blender spin-loop RGBA frames for the Sol→Centauri cinematic.

Assets come from ``render.py blender --spin`` (fixed camera, full rotation,
transparent PNGs). The cinematic indexes into that loop each ``update()`` —
rendered together, not GIF-concatenated afterward.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from animate.scenes.blender.flyby_scene import spinFramesDirectory

DEFAULT_DISPLAY_RESOLUTION = 384
# Advance Luna slower than Earth so the opening reads as orbit + slow spin.
MOON_FRAME_STRIDE = 3


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
        rgba = self.earth[int(frame) % len(self.earth)]
        return _resizeRgba(rgba, resolution)

    def moonFrame(
        self, frame: int, *, resolution: int = DEFAULT_DISPLAY_RESOLUTION
    ) -> np.ndarray | None:
        if not self.moon:
            return None
        rgba = self.moon[(int(frame) // MOON_FRAME_STRIDE) % len(self.moon)]
        return _resizeRgba(rgba, resolution)
