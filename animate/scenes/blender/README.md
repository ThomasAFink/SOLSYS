# Blender planet close-ups

Pipeline for Blender-based planet flybys / zoom-ins that plug into the SOLSYS animation product.

## Chosen export path

**Catalog orbit state → JSON keyframes → Blender ingest**

| Stage | Module | Runs in | Role |
|-------|--------|---------|------|
| 1. Build | `body_scene.py` | SOLSYS venv | Load one `PlanetCatalog` body; sample Keplerian positions (`OrbitCalculator.ellipticalPosition`) into a versioned JSON scene |
| 2. Export | `export_body.py` | SOLSYS venv | Write `output/animate/blender/<body>_body_scene.json` |
| 3. Ingest | `load_body.py` | Blender (`bpy`) or host dry-run | Create UV sphere + material + location keyframes from JSON (**stdlib-only** so Blender’s Python can run it) |
| 4. Flyby | `flyby_scene.py` | SOLSYS → Blender | Extension point for issue #12 (camera path, light/dark renders) |

Why JSON keyframes (not a live `bpy` import of `solsys`)?

- Blender ships its own Python; it should not need the SOLSYS venv.
- CI can test export + JSON validation without Blender.
- Flyby scenes (#12) can re-export and re-ingest the same schema.

Schema id: `solsys.blender_body_scene/v1`

## CLI

Export Earth (default) for Blender:

```bash
.venv/bin/python render.py blender --body Earth
```

Validate JSON without Blender (dry-run):

```bash
.venv/bin/python animate/scenes/blender/load_body.py output/animate/blender/earth_body_scene.json
```

Ingest inside Blender (headless):

```bash
blender --background --python animate/scenes/blender/load_body.py -- \
  output/animate/blender/earth_body_scene.json
```

Open the GUI to look at it (omit `--background`), then press **Space** / use the timeline to scrub:

```bash
blender --python animate/scenes/blender/load_body.py -- \
  output/animate/blender/earth_body_scene.json
```

Tip: press **Numpad 0** for camera view, or select **Earth** in the outliner and **View → Frame Selected**. The body orbits near ~1 AU.

Or from the CLI (requires `blender` on `PATH`):

```bash
.venv/bin/python render.py blender --body Earth --load
```

## Extension point (#12)

`flyby_scene.renderPlanetFlyby` is the stable call site for the first polished flyby:

- prepare export (already works via `preparePlanetFlybyExport`)
- add camera path / shading / light+dark output under `output/animate/blender/`
- link gallery entries in the root README
