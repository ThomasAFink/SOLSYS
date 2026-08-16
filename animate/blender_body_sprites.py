"""Load Blender spin-loop RGBA frames for the Sol→Centauri cinematic.

Assets come from ``render.py blender --spin`` (fixed camera, full rotation,
transparent PNGs). The cinematic indexes into that loop each ``update()`` —
rendered together, not GIF-concatenated afterward.

Earth / planets / asteroids: free spin by animation frame.
Moons: tidally locked — spin index follows orbital mean anomaly (one turn / orbit).

Spin loops are lazy-loaded by catalog name so the cinematic only pays for bodies
that enter a zoom stage with a visible billboard.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from animate.scenes.blender.flyby_scene import spinFramesDirectory

DEFAULT_DISPLAY_RESOLUTION = 384

# Moons that should use orbital phase for tidal lock (vs free spin).
TIDALLY_LOCKED_BODIES = frozenset(
    {
        'Moon',
        'Phobos',
        'Deimos',
        'Io',
        'Europa',
        'Ganymede',
        'Callisto',
        'Titan',
        'Enceladus',
        'Rhea',
        'Titania',
        'Oberon',
        'Triton',
        'Charon',
    }
)


def diskRadiusFraction(rgba: np.ndarray) -> float:
    """Opaque disk radius in a sprite, as a fraction of its half-width.

    Sprites are rendered with transparent margin around the body, so scenes that
    place something on the disk have to measure it rather than assume it fills
    the frame.
    """
    alpha = np.asarray(rgba)[..., 3]
    height, width = alpha.shape
    solid = alpha > 0.5
    if not solid.any():
        return 1.0
    yy, xx = np.mgrid[0:height, 0:width]
    radius = np.sqrt((xx - (width - 1) * 0.5) ** 2 + (yy - (height - 1) * 0.5) ** 2)
    return float(radius[solid].max() / (min(width, height) * 0.5))


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


def spinLoopAvailable(
    bodyName: str,
    theme: str,
    *,
    outputDirectory: Path | str = 'output/animate/blender',
) -> bool:
    """True when at least one spin-loop PNG exists (does not load pixels)."""
    directory = spinFramesDirectory(bodyName, theme, outputDirectory=outputDirectory)
    return any(directory.glob('frame_*.png'))


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
    """Theme-scoped Blender spin loops for in-animation billboards (lazy-loaded)."""

    def __init__(self, theme: str, *, outputDirectory: Path | str = 'output/animate/blender'):
        self.theme = theme
        self.outputDirectory = Path(outputDirectory)
        # None = missing on disk; list = loaded frames. Absent key = not probed yet.
        self._frames: dict[str, list[np.ndarray] | None] = {}
        self._available: dict[str, bool] = {}

    def hasBody(self, catalogName: str) -> bool:
        """True when a spin-loop directory with PNGs exists for ``catalogName``."""
        if catalogName in self._available:
            return self._available[catalogName]
        present = spinLoopAvailable(catalogName, self.theme, outputDirectory=self.outputDirectory)
        self._available[catalogName] = present
        return present

    @property
    def hasEarth(self) -> bool:
        return self.hasBody('Earth')

    @property
    def hasMoon(self) -> bool:
        return self.hasBody('Moon')

    @property
    def earth(self) -> list[np.ndarray] | None:
        """Eager Earth frames (tests / legacy). Prefer ``bodyFrame``."""
        return self._ensureLoaded('Earth')

    @property
    def moon(self) -> list[np.ndarray] | None:
        """Eager Moon frames (tests / legacy). Prefer ``bodyFrame``."""
        return self._ensureLoaded('Moon')

    def loadedBodyNames(self) -> tuple[str, ...]:
        return tuple(name for name, frames in self._frames.items() if frames)

    def _ensureLoaded(self, catalogName: str) -> list[np.ndarray] | None:
        if catalogName in self._frames:
            return self._frames[catalogName]
        if not self.hasBody(catalogName):
            self._frames[catalogName] = None
            return None
        frames = loadSpinLoopFrames(catalogName, self.theme, outputDirectory=self.outputDirectory)
        self._frames[catalogName] = frames
        if frames:
            print(f'Blender spin loop loaded: {catalogName} ({len(frames)} frames)')
        return frames

    def bodyFrame(
        self,
        catalogName: str,
        frame: int,
        *,
        orbitalPhaseRad: float | None = None,
        resolution: int = DEFAULT_DISPLAY_RESOLUTION,
    ) -> np.ndarray | None:
        """RGBA disk for ``catalogName``, lazy-loading the spin loop on first use."""
        frames = self._ensureLoaded(catalogName)
        if not frames:
            return None
        useTidal = catalogName in TIDALLY_LOCKED_BODIES and orbitalPhaseRad is not None
        if useTidal:
            position = tidalLockFramePosition(orbitalPhaseRad, len(frames))
        else:
            position = float(frame) % len(frames)
        rgba = _blendSpinFrames(frames, position)
        return _resizeRgba(rgba, resolution)

    def earthFrame(
        self, frame: int, *, resolution: int = DEFAULT_DISPLAY_RESOLUTION
    ) -> np.ndarray | None:
        return self.bodyFrame('Earth', frame, resolution=resolution)

    def moonFrame(
        self,
        frame: int,
        *,
        orbitalPhaseRad: float | None = None,
        resolution: int = DEFAULT_DISPLAY_RESOLUTION,
    ) -> np.ndarray | None:
        """Moon disk. Prefer ``orbitalPhaseRad`` for tidal lock; ``frame`` is fallback."""
        return self.bodyFrame(
            'Moon',
            frame,
            orbitalPhaseRad=orbitalPhaseRad,
            resolution=resolution,
        )
