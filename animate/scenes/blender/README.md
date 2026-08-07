# Blender planet close-ups

Pipeline for Blender-based planet flybys / zoom-ins that plug into the SOLSYS animation product.

## Pipeline

**Catalog → body JSON → flyby job JSON → Blender PNG frames → GIF**

| Stage | Module | Runs in | Role |
|-------|--------|---------|------|
| 1. Build | `body_scene.py` | SOLSYS venv | Load one PlanetCatalog / MoonCatalog body into `solsys.blender_body_scene/v1` |
| 2. Export | `export_body.py` | SOLSYS venv | Write `output/animate/blender/{planets,moons}/<body>/*_body_scene.json` |
| 3. Ingest | `load_body.py` | Blender (`bpy`) or host dry-run | Debug UV sphere + orbit keyframes (**stdlib-only**) |
| 4. Appearance | `body_appearance.py` | SOLSYS venv | Shared texture packs under `data/textures/bodies/<id>/` (planets/moons/asteroids) |
| 5. Camera | `flyby_camera.py` | SOLSYS venv | Body-centered elevated turntable orbit + spin samples |
| 6. Close-up job | `flyby_scene.py` | SOLSYS venv | Theme job JSON (`solsys.blender_flyby_job/v1`) + GIF assembly |
| 7. Render | `render_flyby.py` | Blender (`bpy`) | Close-up shading / textures, lights, EEVEE PNG sequence |

Why JSON jobs (not a live `bpy` import of `solsys`)?

- Blender ships its own Python; it should not need the SOLSYS venv.
- CI can test export / job validation / GIF assembly without Blender.

## CLI

Export body-scene JSON (scaffold):

```bash
.venv/bin/python render.py blender --body Earth
```

Render the first polished Earth flyby (light + dark GIFs; needs `blender` on `PATH`):

```bash
.venv/bin/python render.py blender --body Earth --flyby
.venv/bin/python render.py blender --body Earth --flyby --theme dark
```

Outputs (organized by kind):

```text
output/animate/blender/
├── planets/earth/   # *_body_scene.json, *_flyby_{light,dark}.gif (+ jobs)
└── moons/moon/
```

Earth uses NASA Blue Marble color + cloud maps under `data/textures/bodies/earth/` (see `data/textures/README.md`) plus a shared fresnel **atmosphere** shell from `BodyAtmosphere`. Moon uses an LRO LROC color pack under `data/textures/bodies/moon/` (airless). Future Jupiter / asteroid packs drop into the same folder layout and register in `body_appearance.py` (clouds/atmosphere optional per body).

```bash
.venv/bin/python render.py blender --body Moon --flyby
```

### Debug ingest (orbit view, not the flyby)

```bash
blender --python animate/scenes/blender/load_body.py -- \
  output/animate/blender/planets/earth/earth_body_scene.json
```

That also writes `earth_body_scene.blend` (Blender may keep `*.blend1` as a save backup).
