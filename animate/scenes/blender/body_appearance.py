"""Shared body appearance / texture packs for Blender close-ups.

Designed so planets, moons, and asteroids share one layout under
``data/textures/bodies/<bodyId>/`` and one resolution API from catalog names.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TEXTURE_BODIES_ROOT = REPO_ROOT / 'data' / 'textures' / 'bodies'


@dataclass(frozen=True)
class BodyTextureMaps:
    """Optional equirectangular maps. Missing paths fall back to catalog color."""

    color: Path | None = None
    specular: Path | None = None
    clouds: Path | None = None
    normal: Path | None = None

    def existingMaps(self) -> dict[str, Path]:
        maps: dict[str, Path] = {}
        for key, path in (
            ('color', self.color),
            ('specular', self.specular),
            ('clouds', self.clouds),
            ('normal', self.normal),
        ):
            if path is not None and path.is_file():
                maps[key] = path.resolve()
        return maps


@dataclass(frozen=True)
class BodyAtmosphere:
    """Optional limb-haze shell (planets/moons with air; off for asteroids)."""

    enabled: bool = False
    scale: float = 1.04
    colorRgba: tuple[float, float, float, float] = (0.45, 0.72, 1.0, 1.0)
    strength: float = 1.1
    fresnelBlend: float = 0.18

    def toJobDict(self) -> dict:
        return {
            'enabled': self.enabled,
            'scale': self.scale,
            'colorRgba': list(self.colorRgba),
            'strength': self.strength,
            'fresnelBlend': self.fresnelBlend,
        }


@dataclass(frozen=True)
class BodyAppearance:
    """Visual pack for one body id, linked to one or more catalog names."""

    bodyId: str
    kind: str  # planet | moon | asteroid | dwarf_planet | …
    catalogNames: tuple[str, ...]
    textures: BodyTextureMaps
    roughness: float = 0.55
    specular: float = 0.25
    atmosphere: BodyAtmosphere = BodyAtmosphere()

    def toJobDict(self) -> dict:
        maps = {key: str(path) for key, path in self.textures.existingMaps().items()}
        payload = {
            'bodyId': self.bodyId,
            'kind': self.kind,
            'textures': maps,
            'roughness': self.roughness,
            'specular': self.specular,
        }
        if self.atmosphere.enabled:
            payload['atmosphere'] = self.atmosphere.toJobDict()
        return payload


def _packDir(bodyId: str) -> Path:
    return TEXTURE_BODIES_ROOT / bodyId


def _optionalMap(bodyId: str, stem: str) -> Path | None:
    directory = _packDir(bodyId)
    for suffix in ('.png', '.jpg', '.jpeg', '.tif', '.tiff'):
        candidate = directory / f'{stem}{suffix}'
        if candidate.is_file():
            return candidate
    return None


def _mapsForBodyId(bodyId: str) -> BodyTextureMaps:
    return BodyTextureMaps(
        color=_optionalMap(bodyId, 'color'),
        specular=_optionalMap(bodyId, 'specular'),
        clouds=_optionalMap(bodyId, 'clouds'),
        normal=_optionalMap(bodyId, 'normal'),
    )


# Registry: add Jupiter / Ceres / other moons here as packs land under data/textures/bodies/.
_BODY_APPEARANCES: tuple[BodyAppearance, ...] = (
    BodyAppearance(
        bodyId='earth',
        kind='planet',
        catalogNames=('Earth',),
        textures=_mapsForBodyId('earth'),
        # Matte-leaning: oceans stay readable without mirror glare in EEVEE.
        roughness=0.58,
        specular=0.18,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.038,
            colorRgba=(0.40, 0.68, 1.0, 1.0),
            strength=0.85,
            fresnelBlend=0.14,
        ),
    ),
    BodyAppearance(
        bodyId='moon',
        kind='moon',
        catalogNames=('Moon',),
        textures=_mapsForBodyId('moon'),
        # Airless regolith: matte, no atmosphere / clouds.
        roughness=0.82,
        specular=0.04,
        atmosphere=BodyAtmosphere(enabled=False),
    ),
)


def _catalogIndex() -> dict[str, BodyAppearance]:
    index: dict[str, BodyAppearance] = {}
    for appearance in _BODY_APPEARANCES:
        # Re-resolve maps at lookup time so tests can drop files into place.
        resolved = BodyAppearance(
            bodyId=appearance.bodyId,
            kind=appearance.kind,
            catalogNames=appearance.catalogNames,
            textures=_mapsForBodyId(appearance.bodyId),
            roughness=appearance.roughness,
            specular=appearance.specular,
            atmosphere=appearance.atmosphere,
        )
        for catalogName in appearance.catalogNames:
            index[catalogName] = resolved
    return index


def appearanceForCatalogName(catalogName: str) -> BodyAppearance | None:
    """Return a texture pack for a PlanetCatalog / MoonCatalog / asteroid name."""
    return _catalogIndex().get(catalogName)


def registeredCatalogNames() -> tuple[str, ...]:
    return tuple(sorted(_catalogIndex()))
