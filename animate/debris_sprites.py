"""Soft debris / dust-clump RGBA sprites for stellar occultation.

These are **not** named asteroids (Ceres, Vesta, …). They are translucent,
irregular aggregates meant to cross in front of a photosphere billboard
(e.g. Tabby's Star lightcurve cinema, issue #73 / packs issue #78).

Layout::

    data/textures/debris/
        clump_a.png   # RGBA, soft alpha
        clump_b.png
        clump_c.png

Generate with::

    .venv/bin/python -m animate.debris_sprites
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEBRIS_ROOT = REPO_ROOT / 'data' / 'textures' / 'debris'
DEFAULT_SPRITE_SIZE = 768


@dataclass(frozen=True)
class DebrisClumpSpec:
    """One occulting debris sprite."""

    clumpId: str
    catalogName: str
    seed: int
    # Warm dust vs cooler gray rock-dust.
    rgb: tuple[float, float, float]
    stretch: tuple[float, float]
    density: float


DEBRIS_CLUMPS: tuple[DebrisClumpSpec, ...] = (
    DebrisClumpSpec(
        'clump_a',
        'Debris Clump A',
        seed=101,
        # Near-silhouette browns — must read against a bright photosphere.
        rgb=(0.14, 0.10, 0.07),
        stretch=(1.35, 0.78),
        density=0.72,
    ),
    DebrisClumpSpec(
        'clump_b',
        'Debris Clump B',
        seed=202,
        rgb=(0.10, 0.09, 0.08),
        stretch=(0.85, 1.25),
        density=0.78,
    ),
    DebrisClumpSpec(
        'clump_c',
        'Debris Clump C',
        seed=303,
        rgb=(0.12, 0.08, 0.05),
        stretch=(1.15, 1.05),
        density=0.68,
    ),
)


def debrisRoot(root: Path | str | None = None) -> Path:
    return Path(root) if root is not None else DEFAULT_DEBRIS_ROOT


def debrisSpritePath(clumpId: str, *, root: Path | str | None = None) -> Path:
    return debrisRoot(root) / f'{clumpId}.png'


def debrisClumpIds() -> tuple[str, ...]:
    return tuple(spec.clumpId for spec in DEBRIS_CLUMPS)


def debrisCatalogNames() -> tuple[str, ...]:
    return tuple(spec.catalogName for spec in DEBRIS_CLUMPS)


def specForCatalogName(catalogName: str) -> DebrisClumpSpec | None:
    for spec in DEBRIS_CLUMPS:
        if spec.catalogName == catalogName or spec.clumpId == catalogName:
            return spec
    return None


def debrisSpriteAvailable(catalogName: str, *, root: Path | str | None = None) -> bool:
    spec = specForCatalogName(catalogName)
    if spec is None:
        return False
    return debrisSpritePath(spec.clumpId, root=root).is_file()


def loadDebrisSprite(
    catalogName: str,
    *,
    root: Path | str | None = None,
    size: int | None = None,
) -> np.ndarray | None:
    """Load float RGBA in ``[0, 1]``; optional square resize."""
    spec = specForCatalogName(catalogName)
    if spec is None:
        return None
    path = debrisSpritePath(spec.clumpId, root=root)
    if not path.is_file():
        return None
    with Image.open(path) as image:
        rgba = np.asarray(image.convert('RGBA'), dtype=np.float32) / 255.0
    if size is not None and (rgba.shape[0] != size or rgba.shape[1] != size):
        pil = Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA')
        pil = pil.resize((size, size), Image.Resampling.LANCZOS)
        rgba = np.asarray(pil, dtype=np.float32) / 255.0
    return rgba


def _hash2(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    x = ix.astype(np.float64) + seed * 17.13
    y = iy.astype(np.float64) + seed * 31.77
    n = np.sin(x * 127.1 + y * 311.7) * 43758.5453
    return n - np.floor(n)


def _valueNoise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx = x - x0
    fy = y - y0
    ux = fx * fx * (3.0 - 2.0 * fx)
    uy = fy * fy * (3.0 - 2.0 * fy)
    a = _hash2(x0, y0, seed)
    b = _hash2(x0 + 1, y0, seed)
    c = _hash2(x0, y0 + 1, seed)
    d = _hash2(x0 + 1, y0 + 1, seed)
    return a + (b - a) * ux + (c - a) * uy + (a - b - c + d) * ux * uy


def _fbm(
    x: np.ndarray,
    y: np.ndarray,
    *,
    octaves: int,
    seed: int,
    lac: float = 2.05,
    gain: float = 0.5,
) -> np.ndarray:
    amp = 1.0
    freq = 1.0
    total = np.zeros_like(x, dtype=np.float64)
    norm = 0.0
    for octave in range(octaves):
        total += amp * _valueNoise(x * freq, y * freq, seed + octave * 19)
        norm += amp
        amp *= gain
        freq *= lac
    return total / norm


def renderDebrisClumpRgba(spec: DebrisClumpSpec, *, size: int = DEFAULT_SPRITE_SIZE) -> np.ndarray:
    """Procedural soft dust aggregate (RGBA float)."""
    axis = (np.arange(size) + 0.5) / size
    uu, vv = np.meshgrid(axis, axis)
    # Centered coords; stretch makes clumps oblong, not round blobs.
    x = (uu - 0.5) * 2.0 * spec.stretch[0]
    y = (vv - 0.5) * 2.0 * spec.stretch[1]
    radius = np.sqrt(x * x + y * y)

    base = _fbm(x * 3.2 + 4.0, y * 3.2 - 2.0, octaves=6, seed=spec.seed)
    mid = _fbm(x * 7.0 - 1.0, y * 7.0 + 3.0, octaves=5, seed=spec.seed + 11)
    fine = _fbm(x * 18.0, y * 18.0, octaves=4, seed=spec.seed + 23)
    warp = _fbm(x * 2.0, y * 2.0, octaves=3, seed=spec.seed + 41)
    xw = x + (warp - 0.5) * 0.55
    yw = y + (mid - 0.5) * 0.45
    warped = _fbm(xw * 4.0, yw * 4.0, octaves=5, seed=spec.seed + 7)

    # Soft irregular mask: envelope + noise threshold (not a hard ellipse).
    envelope = np.clip(1.0 - radius * (0.68 + 0.32 * (1.0 - spec.density)), 0.0, 1.0) ** 1.2
    field = 0.45 * warped + 0.30 * base + 0.25 * mid
    density = np.clip((field - (0.34 - 0.14 * spec.density)) * (3.4 + 1.8 * spec.density), 0.0, 1.0)
    alpha = envelope * density
    # Soft outer falloff + speckled rock grains (keep peak opacity high for star contrast).
    alpha *= 0.72 + 0.28 * fine
    alpha = np.clip(alpha * (0.92 + 0.35 * spec.density), 0.0, 1.0)
    # Mild lift in midtones so thin wisps still dim the photosphere.
    alpha = np.clip(alpha**0.72, 0.0, 1.0)

    rgb = np.empty((size, size, 3), dtype=np.float64)
    grain = (fine - 0.5) * 0.06
    rgb[..., 0] = np.clip(spec.rgb[0] + 0.05 * base + grain, 0.0, 1.0)
    rgb[..., 1] = np.clip(spec.rgb[1] + 0.04 * mid + grain * 0.8, 0.0, 1.0)
    rgb[..., 2] = np.clip(spec.rgb[2] + 0.03 * warped + grain * 0.5, 0.0, 1.0)
    # Dark core so the clump silhouettes against a bright photosphere.
    core = np.clip(1.0 - radius * 0.85, 0.22, 0.78)
    rgb *= core[..., None]

    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    return rgba.astype(np.float32)


def writeDebrisClumpTextures(
    *,
    root: Path | str | None = None,
    size: int = DEFAULT_SPRITE_SIZE,
) -> list[Path]:
    """Write ``clump_*.png`` under the debris texture root."""
    outRoot = debrisRoot(root)
    outRoot.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for spec in DEBRIS_CLUMPS:
        rgba = renderDebrisClumpRgba(spec, size=size)
        path = debrisSpritePath(spec.clumpId, root=outRoot)
        Image.fromarray((np.clip(rgba, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA').save(
            path, optimize=True
        )
        written.append(path)
        print(f'wrote {path} ({path.stat().st_size // 1024} KiB)')
    return written


def writeOccultationQa(
    *,
    debrisRootPath: Path | str | None = None,
    sunFramePath: Path | str | None = None,
    outputPath: Path | str | None = None,
    theme: str = 'dark',
) -> Path:
    """Composite Debris Clump A over a Sol spin frame (visual acceptance for #78)."""
    sunPath = (
        Path(sunFramePath)
        if sunFramePath is not None
        else REPO_ROOT
        / 'output'
        / 'animate'
        / 'blender'
        / 'stars'
        / 'sun'
        / f'sun_spin_{theme}'
        / 'frame_0000.png'
    )
    out = (
        Path(outputPath)
        if outputPath is not None
        else REPO_ROOT / 'output' / 'animate' / 'debris' / f'qa_occultation_{theme}.png'
    )
    if not sunPath.is_file():
        raise FileNotFoundError(f'Sol spin frame missing for QA: {sunPath}')

    clump = loadDebrisSprite('Debris Clump A', root=debrisRootPath, size=420)
    if clump is None:
        raise FileNotFoundError('Debris Clump A sprite missing — run generate first')

    with Image.open(sunPath) as image:
        star = np.asarray(image.convert('RGBA'), dtype=np.float32) / 255.0

    canvas = star.copy()
    # Center the clump slightly offset so cover is obvious but not total eclipse.
    height, width = canvas.shape[:2]
    ch, cw = clump.shape[:2]
    y0 = (height - ch) // 2 - height // 18
    x0 = (width - cw) // 2 + width // 22
    y1, x1 = y0 + ch, x0 + cw
    ys0, xs0 = max(0, y0), max(0, x0)
    ys1, xs1 = min(height, y1), min(width, x1)
    cy0, cx0 = ys0 - y0, xs0 - x0
    cy1, cx1 = cy0 + (ys1 - ys0), cx0 + (xs1 - xs0)
    region = canvas[ys0:ys1, xs0:xs1]
    sprite = clump[cy0:cy1, cx0:cx1]
    alpha = sprite[..., 3:4]
    region[..., :3] = sprite[..., :3] * alpha + region[..., :3] * (1.0 - alpha)
    region[..., 3:4] = np.clip(region[..., 3:4] + alpha * (1.0 - region[..., 3:4]), 0.0, 1.0)

    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8), mode='RGBA').save(out)
    print(f'wrote QA occultation {out}')
    return out


def main() -> None:
    writeDebrisClumpTextures()
    for theme in ('dark', 'light'):
        try:
            writeOccultationQa(theme=theme)
        except FileNotFoundError as error:
            print(f'skip QA ({theme}): {error}')


if __name__ == '__main__':
    main()
