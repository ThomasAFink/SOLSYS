# Body surface textures (Blender close-ups)

Shared texture packs for Blender planet / moon / asteroid close-ups.

## Layout

```text
data/textures/
└── bodies/
    ├── earth/
    │   ├── color.png          # equirectangular color map
    │   └── clouds.png         # cloud layer (RGBA; alpha = coverage)
    ├── moon/                  # future
    ├── jupiter/               # future
    └── ceres/                 # future (asteroids use the same layout)
```

Optional maps per body (same filename convention when added):

| File | Role |
|------|------|
| `color.png` / `color.jpg` | Base color (equirectangular) |
| `specular.png` | Ocean / metalness-style mask (optional) |
| `clouds.png` | Cloud layer (optional; alpha or luminance = coverage) |
| `normal.png` | Normal map (optional) |

Host code resolves packs via `animate.scenes.blender.body_appearance` (catalog name → `bodyId` → paths). Bodies without a pack keep the flat catalog `colorRgba` fallback.

Atmosphere is not a texture file: enable a fresnel limb-haze shell per body in `BodyAtmosphere` (Earth on; asteroids stay off). Clouds are a texture map on the same pack and mix over the surface color when present.

## Earth

- **Color:** `bodies/earth/color.png` (2048×1024)
  - **Source:** NASA Scientific Visualization Studio — [Blue Marble seamless mosaic (SVS 2915)](https://svs.gsfc.nasa.gov/2915) (`bluemarble-2048.png`)
  - **Credit:** NASA Goddard Space Flight Center / Earth Observatory (Reto Stöckli et al.); MODIS / Terra
- **Clouds:** `bodies/earth/clouds.png` (2048×1024 RGBA; thinned coverage mask)
  - **Source:** [Live Cloud Maps](https://clouds.matteason.co.uk/) 2048×1024 `clouds-alpha.png` (NASA/NOAA satellite cloud data; static pack for reproducible renders)
  - **Credit:** Matt Eason / Live Cloud Maps; underlying imagery from NASA & NOAA sources
- **License:** NASA media generally in the U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/))

Do not replace vendor textures without updating this README attribution.