# SOLSYS

<img width="500" align="right" alt="Inner Solar System Animation" src="https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/animate/2d/inner_solar_system_light.gif" />

**SOLSYS** visualizes the solar system with corrected Keplerian physics — from the inner planets out past the Kuiper Belt and Oort Cloud to nearby stars.

The **main product** is animation: light/dark GIFs of the inner system and a staged 3D zoom from the Oort cloud inward. A **side product** renders static multi-zoom JPGs and a light-year neighborhood star map. Shared libraries hold orbit math, catalogs, and asteroid-population motion so both products stay in sync.

Built with `numpy`, `pandas`, and `matplotlib`. Orbital elements include elliptical planet orbits, spherical asteroid/Kuiper/Oort shells, Jupiter Trojans/Greeks, Hilda clusters, moons, named asteroids, and `'Oumuamua`'s hyperbolic trajectory from JPL elements.

## What's included

| Layer | Path | Role |
|-------|------|------|
| CLI | `render.py` | Single entry point: `animate` / `static` / `neighborhood` / `all` |
| Main product | `animate/` | GIF animations (2D inner system + 3D zoom tour); Blender close-ups planned under `animate/scenes/blender/` |
| Side product | `static/` | Multi-zoom JPGs (2D top-down / 3D) and neighborhood star map |
| Shared physics | `solsys/physics/` | Constants, orbits, catalogs, belt generators, view registry |
| Shared motion | `solsys/motion/` | Animated asteroid / Hilda / Trojan / Kuiper / Oort populations |

**Outputs**

- Main: `output/animate/2d/`, `output/animate/3d/`
- Side: `output/2d/`, `output/3d/`, `output/neighborhood/`

## Getting started

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Render everything (~several minutes for animations):

```bash
export MPLBACKEND=Agg
.venv/bin/python render.py all
```

Or render by product:

```bash
.venv/bin/python render.py animate --dimension all
.venv/bin/python render.py animate --dimension 3d
.venv/bin/python render.py static --dimension 2d
.venv/bin/python render.py neighborhood --ly 10
```

## File structure

```text
SOLSYS/
├── render.py                                      # CLI entry point
├── requirements.txt
├── README.md
├── data/
│   └── nearby_stars_30.csv                        # Nearby-star catalog (RA/Dec, distances)
│
├── animate/                                       # MAIN PRODUCT — GIF animations
│   ├── __init__.py
│   ├── solar_system_animator.py                   # SolarSystemAnimator, renderAllAnimations
│   ├── camera_controller.py                       # CameraController
│   ├── animation_styles.py                        # Light/dark styles and timing constants
│   └── scenes/
│       ├── inner_system.py                        # Fixed 2D inner-system scene
│       ├── zoom_tour.py                          # Staged 3D Oort → inner zoom
│       └── blender/                               # Future planet close-ups
│           └── README.md
│
├── static/                                        # SIDE PRODUCT — still images
│   ├── __init__.py
│   ├── solar_system_visualizer.py                 # SolarSystemVisualizer
│   ├── interstellar_neighborhood_visualizer.py    # InterstellarNeighborhoodVisualizer
│   ├── dimension_plotter.py                       # DimensionPlotter
│   └── hilda_point_generator.py                   # HildaPointGenerator
│
├── solsys/                                        # Shared libraries (no plotting)
│   ├── physics/
│   │   ├── astronomical_constants.py              # AstronomicalConstants
│   │   ├── orbit_calculator.py                    # OrbitCalculator
│   │   ├── belt_point_generator.py                # BeltPointGenerator
│   │   ├── view_definition.py                     # ViewDefinition
│   │   ├── view_registry.py                       # ViewRegistry
│   │   ├── point_density_config.py                # PointDensityConfig
│   │   └── catalogs/
│   │       ├── planet_catalog.py                  # PlanetCatalog, PlanetOrbit
│   │       ├── moon_catalog.py                    # MoonCatalog, MoonOrbit
│   │       ├── famous_asteroid_catalog.py         # FamousAsteroidCatalog, FamousAsteroidOrbit
│   │       └── star_catalog.py                    # StarCatalog
│   └── motion/
│       ├── mean_anomaly.py                        # meanAnomalyAtFrame, planetMeanAnomalyRad
│       └── animated_asteroid_population.py        # AnimatedAsteroidPopulation, AsteroidPopulationCounts
│
└── output/
    ├── animate/
    │   ├── 2d/                                    # inner_solar_system_{light,dark}.gif
    │   └── 3d/                                    # solar_system_{light,dark}.gif
    ├── 2d/                                        # Static top-down zoom JPGs
    ├── 3d/                                        # Static perspective zoom JPGs
    └── neighborhood/                              # interstellar_neighborhood_*ly.jpg
```

## Architecture

```text
render.py
   ├── animate/          ← main product (matplotlib GIFs)
   │      └── uses solsys.motion + solsys.physics
   └── static/           ← side product (matplotlib JPGs)
          └── uses solsys.physics

solsys.physics   → constants, Kepler/hyperbola math, catalogs
solsys.motion    → moving asteroid fields for animation frames
```

2D views are a top-down projection of the same XYZ geometry used in 3D.

## Gallery

### Animations (main)

**Inner Solar System**

| Light | Dark |
|-------|------|
| ![Inner 2D light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/animate/2d/inner_solar_system_light.gif?raw=true) | ![Inner 2D dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/animate/2d/inner_solar_system_dark.gif?raw=true) |

**Solar System Zoom**

| Light | Dark |
|-------|------|
| ![Zoom 3D light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/animate/3d/solar_system_light.gif?raw=true) | ![Zoom 3D dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/animate/3d/solar_system_dark.gif?raw=true) |

### Static zoom views (side)

| Inner Solar System With Jupiter | Solar System With Kuiper Belt |
|---------------------------------|-------------------------------|
| ![Inner](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/1_inner_solar_system_with_jupiter.jpg?raw=true) | ![Kuiper](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/2_solar_system_with_kuiper_belt.jpg?raw=true) |

| Solar System With Oort Cloud | Solar System with Alpha Centauri |
|------------------------------|----------------------------------|
| ![Oort](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/3_solar_system_with_oort_cloud.jpg?raw=true) | ![Alpha Cen](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/4_solar_system_with_alpha_centauri.jpg?raw=true) |

| Neighbors Within 10 Light Years | Neighbors Within 25 Light Years |
|---------------------------------|---------------------------------|
| ![10 ly](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/5_solar_system_with_nearest_stars_10.jpg?raw=true) | ![25 ly](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/6_solar_system_with_nearest_stars_25.jpg?raw=true) |

### Matching 3D static views

| Inner Solar System With Jupiter | Solar System With Kuiper Belt |
|---------------------------------|-------------------------------|
| ![3D Inner](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/1_inner_solar_system_with_jupiter.jpg?raw=true) | ![3D Kuiper](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/2_solar_system_with_kuiper_belt.jpg?raw=true) |

| Solar System With Oort Cloud | Solar System with Alpha Centauri |
|------------------------------|----------------------------------|
| ![3D Oort](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/3_solar_system_with_oort_cloud.jpg?raw=true) | ![3D Alpha](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/4_solar_system_with_alpha_centauri.jpg?raw=true) |

## Physics notes

- Planet and asteroid positions use Keplerian ellipses (and a hyperbola for `'Oumuamua`).
- Asteroid / Kuiper / Oort fields use spherical shells in 3D; 2D is the XY projection.
- Hildas follow a 3:2 resonance angular rate relative to Jupiter.
- Jupiter Trojans and Greeks sit near the L4 / L5 Lagrange longitudes.
- Moon orbits are exaggerated for visibility at solar-system zoom levels.

## Roadmap

- Additional animation scenes under `animate/scenes/`
- Blender-based planet close-ups / flybys in `animate/scenes/blender/`
