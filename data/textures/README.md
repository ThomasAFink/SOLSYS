# Body surface textures (Blender close-ups)

Shared texture packs for Blender planet / moon / asteroid close-ups.

## Layout

```text
data/textures/
└── bodies/
    ├── earth/
    │   └── color.png          # equirectangular color map
    ├── moon/                  # future
    ├── jupiter/               # future
    └── ceres/                 # future (asteroids use the same layout)
```

Optional maps per body (same filename convention when added):

| File | Role |
|------|------|
| `color.png` / `color.jpg` | Base color (equirectangular) |
| `specular.png` | Ocean / metalness-style mask (optional) |
| `clouds.png` | Cloud layer (optional; planets with atmosphere) |
| `normal.png` | Normal map (optional) |

Host code resolves packs via `animate.scenes.blender.body_appearance` (catalog name → `bodyId` → paths). Bodies without a pack keep the flat catalog `colorRgba` fallback.

Atmosphere is not a texture file: enable a fresnel limb-haze shell per body in `BodyAtmosphere` (Earth on; asteroids stay off). Optional `clouds.png` can layer later on the same pack.

## Earth

- **File:** `bodies/earth/color.png` (2048×1024)
- **Source:** NASA Scientific Visualization Studio — [Blue Marble seamless mosaic (SVS 2915)](https://svs.gsfc.nasa.gov/2915) (`bluemarble-2048.png`)
- **Credit:** NASA Goddard Space Flight Center / Earth Observatory (Reto Stöckli et al.); MODIS / Terra
- **License:** NASA media generally in the U.S. public domain ([NASA image use policy](https://www.nasa.gov/nasa-brand-center/images-and-media/))

Do not replace vendor textures without updating this README attribution.