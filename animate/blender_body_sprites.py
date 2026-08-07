"""Load Blender flyby frames as circular sprites for matplotlib cinematic bodies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from animate.scenes.blender.export_body import bodyOutputDirectory, bodyStem

DEFAULT_SPRITE_FRAMES = 24
DEFAULT_SPRITE_RESOLUTION = 96


def flybyGifPath(kind: str, bodyName: str, theme: str) -> Path:
    stem = bodyStem(bodyName)
    return bodyOutputDirectory(kind, bodyName) / f'{stem}_flyby_{theme}.gif'


def _circularAlpha(height: int, width: int, *, fill: float = 0.42) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width]
    radius = min(height, width) * fill
    distance = np.sqrt((xx - (width - 1) / 2.0) ** 2 + (yy - (height - 1) / 2.0) ** 2)
    edge = max(radius * 0.08, 1.0)
    return np.clip((radius - distance) / edge, 0.0, 1.0).astype(np.float32)


def frameToCircularRgba(image: Image.Image, *, size: int = DEFAULT_SPRITE_RESOLUTION) -> np.ndarray:
    """RGB frame → small RGBA sprite with a soft circular alpha (planet-centered)."""
    rgb = image.convert('RGB').resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    alpha = _circularAlpha(size, size)
    return np.dstack([array, alpha]).astype(np.float32)


def loadBodySpriteFrames(
    kind: str,
    bodyName: str,
    theme: str,
    *,
    maxFrames: int = DEFAULT_SPRITE_FRAMES,
    resolution: int = DEFAULT_SPRITE_RESOLUTION,
) -> list[np.ndarray] | None:
    """Return RGBA float sprites from a flyby GIF, or None if the asset is missing."""
    path = flybyGifPath(kind, bodyName, theme)
    if not path.is_file():
        return None

    frames: list[np.ndarray] = []
    with Image.open(path) as gif:
        total = int(getattr(gif, 'n_frames', 1))
        if total <= 0:
            return None
        if total <= maxFrames:
            indices = list(range(total))
        else:
            indices = [
                int(round(index * (total - 1) / (maxFrames - 1))) for index in range(maxFrames)
            ]
        for index in indices:
            gif.seek(index)
            frames.append(frameToCircularRgba(gif, size=resolution))
    return frames


class BlenderBodySpriteAtlas:
    """Theme-scoped Earth/Moon sprite sequences for in-animation billboards."""

    def __init__(self, theme: str, *, maxFrames: int = DEFAULT_SPRITE_FRAMES):
        self.theme = theme
        self.earth = loadBodySpriteFrames('planet', 'Earth', theme, maxFrames=maxFrames)
        self.moon = loadBodySpriteFrames('moon', 'Moon', theme, maxFrames=maxFrames)

    @property
    def hasEarth(self) -> bool:
        return bool(self.earth)

    @property
    def hasMoon(self) -> bool:
        return bool(self.moon)

    def earthFrame(self, frame: int) -> np.ndarray | None:
        if not self.earth:
            return None
        return self.earth[int(frame) % len(self.earth)]

    def moonFrame(self, frame: int) -> np.ndarray | None:
        if not self.moon:
            return None
        return self.moon[int(frame) % len(self.moon)]
