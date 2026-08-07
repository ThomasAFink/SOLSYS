"""Render Earth/Moon globes from Blender texture packs for the cinematic.

Each animation frame samples the equirectangular color maps (same packs Blender
uses) into a lit circular disk. No flyby-GIF splicing — the disk is generated
and drawn while ``FuncAnimation`` paints the frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from animate.scenes.blender.body_appearance import appearanceForCatalogName

DEFAULT_GLOBE_RESOLUTION = 384
# ~2.4°/frame ≈ one Earth day every ~7.5 s at 20 fps — readable day/night crawl.
EARTH_SPIN_DEG_PER_FRAME = 2.4
MOON_SPIN_DEG_PER_FRAME = 0.15


def _loadRgbMap(path) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0


def _sampleEquirectangular(texture: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Bilinear sample equirectangular RGB. lon/lat in radians."""
    height, width = texture.shape[:2]
    u = (lon / (2.0 * np.pi) + 0.5) % 1.0
    v = np.clip(0.5 - lat / np.pi, 0.0, 1.0)
    x = u * (width - 1)
    y = v * (height - 1)
    x0 = np.floor(x).astype(np.int32)
    y0 = np.floor(y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, width - 1)
    y1 = np.clip(y0 + 1, 0, height - 1)
    fx = (x - x0)[..., None]
    fy = (y - y0)[..., None]
    c00 = texture[y0, x0]
    c10 = texture[y0, x1]
    c01 = texture[y1, x0]
    c11 = texture[y1, x1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def renderGlobeDisk(
    colorMap: np.ndarray,
    *,
    spinDeg: float,
    sunDirection: np.ndarray,
    cloudMap: np.ndarray | None = None,
    resolution: int = DEFAULT_GLOBE_RESOLUTION,
    ambient: float = 0.18,
) -> np.ndarray:
    """Orthographic dayside disk (RGBA float) from an equirectangular color map."""
    # Supersample then Lanczos-downscale for clean limb anti-aliasing in GIFs.
    outSize = int(resolution)
    size = outSize * 2
    axis = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    radius2 = xx * xx + yy * yy
    mask = radius2 <= 1.0
    zz = np.zeros_like(xx)
    zz[mask] = np.sqrt(np.clip(1.0 - radius2[mask], 0.0, 1.0))

    # Camera looks down -Z; +Y is up. Spin about body north.
    spin = np.deg2rad(spinDeg)
    cosS = float(np.cos(spin))
    sinS = float(np.sin(spin))
    bx = xx * cosS + zz * sinS
    by = yy
    bz = -xx * sinS + zz * cosS

    lon = np.arctan2(bx, bz)
    lat = np.arcsin(np.clip(by, -1.0, 1.0))
    rgb = np.zeros((size, size, 3), dtype=np.float32)
    if np.any(mask):
        sampled = _sampleEquirectangular(colorMap, lon[mask], lat[mask])
        if cloudMap is not None:
            clouds = _sampleEquirectangular(cloudMap, lon[mask], lat[mask])
            cloudWeight = np.clip(clouds.mean(axis=-1, keepdims=True), 0.0, 1.0) * 0.85
            sampled = sampled * (1.0 - cloudWeight) + cloudWeight
        rgb[mask] = sampled

    sun = np.asarray(sunDirection, dtype=np.float32)
    sunNorm = float(np.linalg.norm(sun))
    if sunNorm < 1e-9:
        sun = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    else:
        sun = sun / sunNorm
    # Shade in view space so the terminator stays camera-stable while the
    # texture spins underneath (avoids the jumpy flyby-camera look).
    normal = np.stack([xx, yy, zz], axis=-1)
    lambert = np.clip(normal @ sun, 0.0, 1.0)
    light = ambient + (1.0 - ambient) * lambert
    rgb *= light[..., None]

    alpha = np.zeros((size, size), dtype=np.float32)
    edge = 2.5 / size
    alpha[mask] = np.clip((1.0 - np.sqrt(radius2[mask])) / edge, 0.0, 1.0)
    hiRes = np.dstack([np.clip(rgb, 0.0, 1.0), alpha]).astype(np.float32)
    image = Image.fromarray((hiRes * 255.0).astype(np.uint8), mode='RGBA')
    image = image.resize((outSize, outSize), Image.Resampling.LANCZOS)
    return np.asarray(image, dtype=np.float32) / 255.0


@dataclass
class BodyGlobePack:
    catalogName: str
    colorMap: np.ndarray
    cloudMap: np.ndarray | None
    spinDegPerFrame: float

    def disk(
        self,
        frame: int,
        sunDirection: np.ndarray,
        *,
        resolution: int = DEFAULT_GLOBE_RESOLUTION,
    ) -> np.ndarray:
        return renderGlobeDisk(
            self.colorMap,
            spinDeg=float(frame) * self.spinDegPerFrame,
            sunDirection=sunDirection,
            cloudMap=self.cloudMap,
            resolution=resolution,
        )


def loadBodyGlobePack(catalogName: str) -> BodyGlobePack | None:
    appearance = appearanceForCatalogName(catalogName)
    if appearance is None:
        return None
    color = _loadRgbMap(appearance.textures.color)
    if color is None:
        return None
    clouds = _loadRgbMap(appearance.textures.clouds)
    spin = EARTH_SPIN_DEG_PER_FRAME if catalogName == 'Earth' else MOON_SPIN_DEG_PER_FRAME
    return BodyGlobePack(
        catalogName=catalogName,
        colorMap=color,
        cloudMap=clouds,
        spinDegPerFrame=spin,
    )


class BlenderBodySpriteAtlas:
    """Earth/Moon texture-pack globes for in-animation drawing."""

    def __init__(self, theme: str):
        self.theme = theme
        self.earth = loadBodyGlobePack('Earth')
        self.moon = loadBodyGlobePack('Moon')
        # Daylight comes from +X in the opening Sol view; fine for billboards.
        self.sunDirection = np.array([0.75, 0.15, 0.64], dtype=np.float32)

    @property
    def hasEarth(self) -> bool:
        return self.earth is not None

    @property
    def hasMoon(self) -> bool:
        return self.moon is not None

    def earthFrame(
        self, frame: int, *, resolution: int = DEFAULT_GLOBE_RESOLUTION
    ) -> np.ndarray | None:
        if self.earth is None:
            return None
        return self.earth.disk(frame, self.sunDirection, resolution=resolution)

    def moonFrame(
        self, frame: int, *, resolution: int = DEFAULT_GLOBE_RESOLUTION
    ) -> np.ndarray | None:
        if self.moon is None:
            return None
        return self.moon.disk(frame, self.sunDirection, resolution=resolution)
