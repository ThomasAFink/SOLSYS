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
    rings: Path | None = None

    def existingMaps(self) -> dict[str, Path]:
        maps: dict[str, Path] = {}
        for key, path in (
            ('color', self.color),
            ('specular', self.specular),
            ('clouds', self.clouds),
            ('normal', self.normal),
            ('rings', self.rings),
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
class BodyRings:
    """Equatorial ring annulus (Saturn / ice giants). Scales are × body radius."""

    enabled: bool = False
    innerScale: float = 1.2
    outerScale: float = 2.3
    tiltDeg: float = 0.0
    opacity: float = 1.0

    def toJobDict(self) -> dict:
        return {
            'enabled': self.enabled,
            'innerScale': self.innerScale,
            'outerScale': self.outerScale,
            'tiltDeg': self.tiltDeg,
            'opacity': self.opacity,
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
    rings: BodyRings = BodyRings()

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
        if self.rings.enabled:
            payload['rings'] = self.rings.toJobDict()
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
        rings=_optionalMap(bodyId, 'rings'),
    )


def _planet(
    bodyId: str,
    catalogName: str,
    *,
    roughness: float,
    specular: float,
    atmosphere: BodyAtmosphere | None = None,
    rings: BodyRings | None = None,
) -> BodyAppearance:
    return BodyAppearance(
        bodyId=bodyId,
        kind='planet',
        catalogNames=(catalogName,),
        textures=_mapsForBodyId(bodyId),
        roughness=roughness,
        specular=specular,
        atmosphere=atmosphere or BodyAtmosphere(enabled=False),
        rings=rings or BodyRings(enabled=False),
    )


def _moon(
    bodyId: str,
    catalogName: str,
    *,
    roughness: float = 0.82,
    specular: float = 0.04,
    atmosphere: BodyAtmosphere | None = None,
) -> BodyAppearance:
    """Airless by default; Titan may pass a thin haze shell."""
    return BodyAppearance(
        bodyId=bodyId,
        kind='moon',
        catalogNames=(catalogName,),
        textures=_mapsForBodyId(bodyId),
        roughness=roughness,
        specular=specular,
        atmosphere=atmosphere or BodyAtmosphere(enabled=False),
    )


def _asteroid(
    bodyId: str,
    catalogName: str,
    *,
    kind: str = 'asteroid',
    roughness: float = 0.88,
    specular: float = 0.04,
) -> BodyAppearance:
    """Airless small body / dwarf planet (no atmosphere or clouds)."""
    return BodyAppearance(
        bodyId=bodyId,
        kind=kind,
        catalogNames=(catalogName,),
        textures=_mapsForBodyId(bodyId),
        roughness=roughness,
        specular=specular,
        atmosphere=BodyAtmosphere(enabled=False),
    )


def _star(bodyId: str, *catalogNames: str) -> BodyAppearance:
    """Emissive photosphere (no key lamp, no fresnel atmosphere shell)."""
    return BodyAppearance(
        bodyId=bodyId,
        kind='star',
        catalogNames=catalogNames,
        textures=_mapsForBodyId(bodyId),
        # Emission-driven in render_flyby; no fresnel shell (reads as a hard pixelated ring).
        roughness=0.95,
        specular=0.0,
        atmosphere=BodyAtmosphere(enabled=False),
    )


# Registry: stars, planets, major moons, named asteroids / dwarfs.
_BODY_APPEARANCES: tuple[BodyAppearance, ...] = (
    _star('sun', 'Sun'),
    _star('alpha_centauri_a', 'Alpha Centauri A'),
    _star('alpha_centauri_b', 'Alpha Centauri B'),
    _star('proxima_centauri', 'Proxima Centauri'),
    _planet(
        'mercury',
        'Mercury',
        roughness=0.78,
        specular=0.06,
    ),
    _planet(
        'venus',
        'Venus',
        roughness=0.62,
        specular=0.12,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.055,
            colorRgba=(0.95, 0.85, 0.45, 1.0),
            strength=1.35,
            fresnelBlend=0.22,
        ),
    ),
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
    _planet(
        'mars',
        'Mars',
        roughness=0.72,
        specular=0.08,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.028,
            colorRgba=(0.85, 0.55, 0.35, 1.0),
            strength=0.45,
            fresnelBlend=0.12,
        ),
    ),
    # Proxima planets — gallery-grade packs for the cinematic finale (#65).
    _planet(
        'proxima_b',
        'Proxima b',
        roughness=0.62,
        specular=0.14,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.03,
            colorRgba=(0.55, 0.35, 0.45, 1.0),
            strength=0.55,
            fresnelBlend=0.12,
        ),
    ),
    _planet(
        'proxima_d',
        'Proxima d',
        roughness=0.78,
        specular=0.06,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.02,
            colorRgba=(0.45, 0.55, 0.65, 1.0),
            strength=0.30,
            fresnelBlend=0.10,
        ),
    ),
    _planet(
        'jupiter',
        'Jupiter',
        roughness=0.48,
        specular=0.22,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.02,
            colorRgba=(0.75, 0.70, 0.55, 1.0),
            strength=0.55,
            fresnelBlend=0.16,
        ),
    ),
    _planet(
        'saturn',
        'Saturn',
        roughness=0.50,
        specular=0.20,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.018,
            colorRgba=(0.85, 0.78, 0.55, 1.0),
            strength=0.50,
            fresnelBlend=0.15,
        ),
        rings=BodyRings(
            enabled=True,
            innerScale=1.15,
            outerScale=2.25,
            tiltDeg=26.7,
            opacity=1.0,
        ),
    ),
    _planet(
        'uranus',
        'Uranus',
        roughness=0.45,
        specular=0.18,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.02,
            colorRgba=(0.55, 0.85, 0.90, 1.0),
            strength=0.60,
            fresnelBlend=0.16,
        ),
        rings=BodyRings(
            enabled=True,
            innerScale=1.55,
            outerScale=2.05,
            tiltDeg=97.8,
            opacity=0.55,
        ),
    ),
    _planet(
        'neptune',
        'Neptune',
        roughness=0.45,
        specular=0.20,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.022,
            colorRgba=(0.35, 0.50, 0.95, 1.0),
            strength=0.70,
            fresnelBlend=0.17,
        ),
        rings=BodyRings(
            enabled=True,
            innerScale=1.45,
            outerScale=2.35,
            tiltDeg=28.3,
            opacity=0.65,
        ),
    ),
    _planet(
        'pluto',
        'Pluto',
        roughness=0.80,
        specular=0.05,
    ),
    # --- Moons (MoonCatalog) ---
    _moon('moon', 'Moon'),
    _moon('phobos', 'Phobos', roughness=0.88, specular=0.03),
    _moon('deimos', 'Deimos', roughness=0.88, specular=0.03),
    _moon('io', 'Io', roughness=0.70, specular=0.08),
    _moon('europa', 'Europa', roughness=0.55, specular=0.18),
    _moon('ganymede', 'Ganymede', roughness=0.72, specular=0.08),
    _moon('callisto', 'Callisto', roughness=0.80, specular=0.05),
    _moon(
        'titan',
        'Titan',
        roughness=0.65,
        specular=0.10,
        atmosphere=BodyAtmosphere(
            enabled=True,
            scale=1.045,
            colorRgba=(0.95, 0.70, 0.35, 1.0),
            strength=1.10,
            fresnelBlend=0.20,
        ),
    ),
    _moon('enceladus', 'Enceladus', roughness=0.45, specular=0.22),
    _moon('rhea', 'Rhea', roughness=0.78, specular=0.06),
    _moon('titania', 'Titania', roughness=0.80, specular=0.05),
    _moon('oberon', 'Oberon', roughness=0.80, specular=0.05),
    _moon('triton', 'Triton', roughness=0.70, specular=0.10),
    _moon('charon', 'Charon', roughness=0.80, specular=0.05),
    # --- Asteroids / dwarf planets (FamousAsteroidCatalog) ---
    _asteroid('ceres', 'Ceres', kind='dwarf_planet', roughness=0.78, specular=0.06),
    _asteroid('vesta', 'Vesta', roughness=0.82, specular=0.05),
    _asteroid('pallas', 'Pallas', roughness=0.85, specular=0.04),
    _asteroid('psyche', 'Psyche', roughness=0.55, specular=0.28),
    _asteroid('bennu', 'Bennu', roughness=0.90, specular=0.03),
    _asteroid('eros', 'Eros', roughness=0.86, specular=0.04),
    _asteroid('haumea', 'Haumea', kind='dwarf_planet', roughness=0.55, specular=0.18),
    _asteroid('makemake', 'Makemake', kind='dwarf_planet', roughness=0.70, specular=0.10),
    _asteroid('eris', 'Eris', kind='dwarf_planet', roughness=0.60, specular=0.16),
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
            rings=appearance.rings,
        )
        for catalogName in appearance.catalogNames:
            index[catalogName] = resolved
    return index


def appearanceForCatalogName(catalogName: str) -> BodyAppearance | None:
    """Return a texture pack for a catalog / CLI body name (planets, moons, Sun, …)."""
    return _catalogIndex().get(catalogName)


def registeredStarCatalogNames() -> tuple[str, ...]:
    """Star packs (Sol, α Cen A/B, Proxima)."""
    return tuple(
        sorted(
            {
                name
                for appearance in _catalogIndex().values()
                for name in appearance.catalogNames
                if appearance.kind == 'star'
            }
        )
    )


def registeredCatalogNames() -> tuple[str, ...]:
    return tuple(sorted(_catalogIndex()))


def registeredPlanetCatalogNames() -> tuple[str, ...]:
    """PlanetCatalog-facing packs (excludes moons)."""
    return tuple(
        sorted(
            name
            for appearance in _catalogIndex().values()
            for name in appearance.catalogNames
            if appearance.kind == 'planet'
        )
    )


def registeredMoonCatalogNames() -> tuple[str, ...]:
    """MoonCatalog-facing packs."""
    return tuple(
        sorted(
            name
            for appearance in _catalogIndex().values()
            for name in appearance.catalogNames
            if appearance.kind == 'moon'
        )
    )


def registeredAsteroidCatalogNames() -> tuple[str, ...]:
    """FamousAsteroidCatalog-facing packs (asteroids + dwarf planets)."""
    return tuple(
        sorted(
            name
            for appearance in _catalogIndex().values()
            for name in appearance.catalogNames
            if appearance.kind in {'asteroid', 'dwarf_planet'}
        )
    )
