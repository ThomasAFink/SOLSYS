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
class BodyAppearance:
    """Visual pack for one body id, linked to one or more catalog names."""

    bodyId: str
    kind: str  # planet | moon | asteroid | dwarf_planet | …
    catalogNames: tuple[str, ...]
    textures: BodyTextureMaps
    roughness: float = 0.55
    specular: float = 0.25

    def toJobDict(self) -> dict:
        maps = {key: str(path) for key, path in self.textures.existingMaps().items()}
        return {
            'bodyId': self.bodyId,
            'kind': self.kind,
            'textures': maps,
            'roughness': self.roughness,
            'specular': self.specular,
        }


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


# Registry: add Moon / Jupiter / Ceres here as packs land under data/textures/bodies/.
_BODY_APPEARANCES: tuple[BodyAppearance, ...] = (
    BodyAppearance(
        bodyId='earth',
        kind='planet',
        catalogNames=('Earth',),
        textures=_mapsForBodyId('earth'),
        # Oceans read a bit glossier once a color map is present.
        roughness=0.48,
        specular=0.35,
    ),
    # Future examples (no files yet → colorRgba fallback until packs exist):
    # BodyAppearance('moon', 'moon', ('Moon',), _mapsForBodyId('moon'), roughness=0.85, specular=0.05),
    # BodyAppearance('jupiter', 'planet', ('Jupiter',), _mapsForBodyId('jupiter')),
    # BodyAppearance('ceres', 'asteroid', ('Ceres',), _mapsForBodyId('ceres'), roughness=0.9, specular=0.05),
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
        )
        for catalogName in appearance.catalogNames:
            index[catalogName] = resolved
    return index


def appearanceForCatalogName(catalogName: str) -> BodyAppearance | None:
    """Return a texture pack for a PlanetCatalog / MoonCatalog / asteroid name."""
    return _catalogIndex().get(catalogName)


def registeredCatalogNames() -> tuple[str, ...]:
    return tuple(sorted(_catalogIndex()))
