# Body surface textures (Blender close-ups)

Shared texture packs for Blender planet / moon / asteroid close-ups.

## Layout

```text
data/textures/
└── bodies/
    ├── earth/
    │   ├── color.png          # equirectangular color map
    │   └── clouds.png         # cloud layer (RGBA; alpha = coverage)
    ├── moon/
    │   └── color.png          # equirectangular color map (airless)
    ├── sun/                   # Sol photosphere (emissive star close-up)
    ├── mercury/ … pluto/      # Sol planet packs (color; rings where applicable)
    ├── io/ … charon/          # major moon packs (airless; Titan haze in code)
    └── ceres/ … eris/         # asteroids / dwarf planets (airless)
```

Optional maps per body (same filename convention when added):

| File | Role |
|------|------|
| `color.png` / `color.jpg` | Base color (equirectangular) |
| `specular.png` | Ocean / metalness-style mask (optional) |
| `clouds.png` | Cloud layer (optional; alpha or luminance = coverage) |
| `normal.png` | Normal map (optional) |
| `rings.png` | Radial ring strip with alpha (long axis = inner→outer; Saturn / ice giants) |

Host code resolves packs via `animate.scenes.blender.body_appearance` (catalog name → `bodyId` → paths). Bodies without a pack keep the flat catalog `colorRgba` fallback.

Atmosphere is not a texture file: enable a fresnel limb-haze shell per body in `BodyAtmosphere` (Earth/Venus/Mars/gas giants/Titan on; Mercury/Moon/Pluto and other moons off). Clouds are a texture map on the same pack and mix over the surface color when present. Rings use `BodyRings` + optional `rings.png`.

## Earth

- **Color:** `bodies/earth/color.png` (2048×1024)
  - **Source:** NASA Scientific Visualization Studio — [Blue Marble seamless mosaic (SVS 2915)](https://svs.gsfc.nasa.gov/2915) (`bluemarble-2048.png`)
  - **Credit:** NASA Goddard Space Flight Center / Earth Observatory (Reto Stöckli et al.); MODIS / Terra
- **Clouds:** `bodies/earth/clouds.png` (2048×1024 RGBA; thinned coverage mask)
  - **Source:** [Live Cloud Maps](https://clouds.matteason.co.uk/) 2048×1024 `clouds-alpha.png` (NASA/NOAA satellite cloud data; static pack for reproducible renders)
  - **Credit:** Matt Eason / Live Cloud Maps; underlying imagery from NASA & NOAA sources
- **License:** NASA media generally in the U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/))

## Moon

- **Color:** `bodies/moon/color.png` (2048×1024)
  - **Source:** NASA SVS — [CGI Moon Kit / LRO LROC WAC color (SVS 4720)](https://svs.gsfc.nasa.gov/4720) (`lroc_color_2k.jpg`)
  - **Credit:** NASA/GSFC Scientific Visualization Studio; LRO / LROC
- **Atmosphere / clouds:** none (airless pack in `body_appearance.py`)
- **License:** NASA media generally in the U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/))

## Sun (Sol)

Equirectangular photosphere map for the Blender star close-up (`kind=star`). Rendered emissive (not a matte ball under a key lamp); no fresnel atmosphere shell (that read as a hard ring).

| Body | Pack notes | Primary source |
|------|------------|----------------|
| Sun | Photosphere granulation (emissive; no atmosphere shell) | [Solar System Scope](https://www.solarsystemscope.com/textures/) [`8k_sun.jpg`](https://www.solarsystemscope.com/textures/download/8k_sun.jpg) (4096×2048) |

- **SSS textures:** free with attribution to [Solar System Scope](https://www.solarsystemscope.com/textures/).
- Sol only for now; α Cen A/B / Proxima packs are a follow-up.

## Sol planets (Mercury–Neptune + Pluto)

2k equirectangular color maps under `bodies/<bodyId>/color.png`, registered in `body_appearance.py`.

| Body | Pack notes | Primary source |
|------|------------|----------------|
| Mercury | Airless, matte | [Solar System Scope](https://www.solarsystemscope.com/textures/) `2k_mercury.jpg` (based on NASA / USGS maps) |
| Venus | Cloud-deck color + thick yellowish limb haze | SSS `2k_venus_atmosphere.jpg` |
| Mars | Thin dusty atmosphere shell | SSS `2k_mars.jpg` |
| Jupiter | Subtle cream limb haze | SSS `2k_jupiter.jpg` |
| Saturn | Rings required (`rings.png` + `BodyRings` tilt 26.7°) | SSS `2k_saturn.jpg` + `2k_saturn_ring_alpha.png` |
| Uranus | Subtler rings (synthetic strip) + cyan haze | SSS `2k_uranus.jpg`; rings generated for gallery readability |
| Neptune | Subtler rings (synthetic strip) + blue haze | SSS `2k_neptune.jpg`; rings generated for gallery readability |
| Pluto | Airless New Horizons color | [NASA 3D Resources](https://github.com/nasa/NASA-3D-Resources) `Images and Textures/Pluto/Pluto.jpg` |

- **Solar System Scope textures:** free for personal and commercial use with attribution to [Solar System Scope](https://www.solarsystemscope.com/textures/) (underlying spacecraft data from NASA / USGS / ESA missions).
- **NASA Pluto map:** U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/)).
- Ice-giant `rings.png` files are lightweight synthetic radial alpha strips (not spacecraft mosaics) so Uranus/Neptune show readable ring systems in EEVEE flybys.

## Major moons

Equirectangular color maps under `bodies/<bodyId>/color.png` for every `MoonCatalog` entry (except Luna, documented above). Registered as `kind=moon` in `body_appearance.py` — airless (no clouds) unless noted.

| Body | Pack notes | Primary source |
|------|------------|----------------|
| Io | Volcanic color | [NASA 3D Resources](https://github.com/nasa/NASA-3D-Resources) `Jupiter - Io (A)` |
| Europa | Icy, higher specular | NASA 3D Resources `Jupiter - Europa` |
| Ganymede | Cratered ice/rock | NASA 3D Resources `Jupiter - Ganymede` |
| Callisto | Dark cratered | NASA 3D Resources `Jupiter - Callisto` |
| Titan | Low-detail surface + orange haze shell | NASA 3D Resources `Saturn - Titan` |
| Enceladus | Bright ice | NASA 3D Resources `Saturn - Enceladus` |
| Rhea | Icy regolith | NASA 3D Resources `Saturn - Rhea` |
| Phobos / Deimos | Irregular Mars moons | NASA 3D Resources `Mars - Phobos` / `Mars - Deimos` |
| Titania / Oberon | Uranian ice moons | NASA 3D Resources `Uranus - Titania` / `Uranus - Oberon` |
| Triton | Neptune moon | NASA 3D Resources `Neptune - Triton` |
| Charon | Pluto companion | NASA 3D Resources `Pluto - Charon` |

- **License:** NASA media generally in the U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/)).
- Titan’s spacecraft map is low-contrast; the EEVEE orange limb haze carries most of the recognizable look.

## Asteroids & dwarf planets

Equirectangular color maps for priority `FamousAsteroidCatalog` bodies. Registered as `kind=asteroid` or `kind=dwarf_planet` (atmosphere/clouds always off). Sphere + texture is enough for v1 (no irregular mesh).

| Body | Pack notes | Primary source |
|------|------------|----------------|
| Ceres | Dwarf planet | [Solar System Scope](https://www.solarsystemscope.com/textures/) `2k_ceres_fictional.jpg` (Dawn-based) |
| Vesta | Dawn HAMO clear mosaic, downsampled 2k | USGS Planetary Maps [`Vesta_Dawn_FC_HAMO_Mosaic_Global_74ppd.tif`](https://planetarymaps.usgs.gov/mosaic/Vesta_Dawn_FC_HAMO_Mosaic_Global_74ppd.tif) (NASA/JPL-Caltech/UCLA/MPS/DLR/IDA) |
| Pallas / Psyche | Gallery-grade procedural cratered maps (no public 2k pack yet) | Generated in-repo |
| Bennu / Eros | Gallery-grade procedural (stretch NEOs) | Generated in-repo |
| Haumea / Makemake / Eris | Dwarf planets | SSS `2k_*_fictional.jpg` |

- **SSS textures:** free with attribution to [Solar System Scope](https://www.solarsystemscope.com/textures/).
- **Vesta Dawn mosaic:** NASA / USGS public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/)).
- Procedural packs are placeholders for belt linger readability until spacecraft mosaics are packaged.

Do not replace vendor textures without updating this README attribution.
