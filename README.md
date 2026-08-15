# SOLSYS

![Kepler's three laws](docs/keplers_three_laws.png)

**SOLSYS** visualizes the solar system with corrected Keplerian physics — from the inner planets out past the Kuiper Belt and Oort Cloud to nearby stars.

The **main product** is animation: light/dark GIFs of the inner system and a staged 3D zoom from the Oort cloud inward. A **side product** renders static multi-zoom JPGs (light/dark) and a light-year neighborhood star map. Shared libraries hold orbit math, catalogs, and asteroid-population motion so both products stay in sync.

Built with `numpy`, `pandas`, and `matplotlib`. Orbital elements include elliptical planet orbits, spherical asteroid/Kuiper/Oort shells, Jupiter Trojans/Greeks, Hilda clusters, moons, named asteroids, and hyperbolic **interstellar visitors** (1I/ʻOumuamua, 2I/Borisov, 3I/ATLAS) from `data/interstellar_objects.csv`.

## What's included

| Layer | Path | Role |
|-------|------|------|
| CLI | `render.py` | Single entry point: `animate` / `static` / `neighborhood` / `blender` / `all` |
| Main product | `animate/` | GIF animations (2D inner system + 3D zoom tour); Blender close-ups scaffold under `animate/scenes/blender/` |
| Side product | `static/` | Multi-zoom JPGs light/dark (2D top-down / 3D) and neighborhood star map |
| Shared physics | `solsys/physics/` | Constants, orbits, catalogs, belt generators, view registry |
| Shared motion | `solsys/motion/` | Animated asteroid / Hilda / Trojan / Kuiper / Oort populations |

**Outputs**

- Main: `output/animate/2d/`, `output/animate/3d/`
- Side: `output/2d/`, `output/3d/`, `output/neighborhood/`

## File structure

```text
SOLSYS/
├── render.py                                      # CLI entry point
├── requirements.txt
├── requirements-dev.txt                           # pre-commit, ruff, radon, xenon
├── pyproject.toml                                 # Ruff config
├── .pre-commit-config.yaml                        # Git hook definitions
├── .githooks/
│   └── commit-msg                                 # type(SOLSYS[-N]): subject
├── README.md
├── docs/
│   ├── keplers_three_laws.png                     # Kepler laws diagram
│   └── generate_keplers_laws_figure.py            # Regenerates the diagram
├── data/
│   ├── nearby_stars_30.csv                        # Star catalog (RA/Dec, distances, system_id; includes hosts beyond 30 ly)
│   ├── systems.csv                                # Star systems (sol, alpha_centauri, barnards_star, trappist_1, tabbys_star, …)
│   ├── stellar_orbits.csv                         # Multi-star orbits vs system barycenter
│   ├── planets.csv                                # Exoplanets linked by host_star_uuid
│   ├── interstellar_objects.csv                   # 1I/2I/3I hyperbolic visitors
│   ├── tabbys_star_lightcurve.csv                 # Downsampled Kepler LC for Tabby's Star inset
│   └── textures/                                  # Body maps + debris occultation sprites
│       ├── README.md                              # Attribution + pack layout
│       ├── bodies/{earth,moon,…}/…                # NASA / SSS / procedural equirect packs
│       └── debris/clump_{a,b,c}.png               # Soft dust clumps for stellar occultation (#78)
│
├── animate/                                       # MAIN PRODUCT — GIF animations
│   ├── __init__.py
│   ├── solar_system_animator.py                   # SolarSystemAnimator, renderAllAnimations
│   ├── camera_controller.py                       # CameraController
│   ├── animation_styles.py                        # Light/dark styles and timing constants
│   ├── blender_body_sprites.py                    # Lazy Blender spin loops for cinematic Sol billboards
│   └── scenes/
│       ├── inner_system.py                        # Fixed 2D inner-system scene
│       ├── zoom_tour.py                          # Staged 3D Oort → inner zoom
│       ├── exoplanet_system.py                    # Shared single-host exoplanet top-down animator
│       ├── alpha_centauri.py                      # A–B close-up, wide triple; Proxima via exoplanet_system
│       ├── sol_centauri_cinematic.py              # Sol → α Cen AB cinematic (uses frame transform)
│       ├── sol_trappist_cinematic.py              # Sol → TRAPPIST-1 cinematic (Sol XYZ host)
│       ├── barnards_star.py                       # Barnard's Star planets via exoplanet_system
│       ├── trappist_1.py                          # TRAPPIST-1 planets via exoplanet_system
│       ├── tabbys_star.py                         # Tabby's Star dust-cloud dimming schematic
│       ├── tabbys_star_cinematic.py               # Tabby's lightcurve cinema (#73; LC + occultation)
│       ├── interstellar_objects.py                # 1I/2I/3I hyperbolic passages (side + oblique)
│       └── blender/                               # Blender close-up pipeline (catalog → JSON → bpy)
│           ├── README.md                          # Pipeline docs + CLI examples
│           ├── body_scene.py                      # Versioned body-scene schema + PlanetCatalog build
│           ├── export_body.py                     # Write blender/{planets,moons}/<body>/*.json
│           ├── load_body.py                       # Blender ingest (stdlib + optional bpy)
│           ├── body_appearance.py                 # Shared texture packs (planets/moons/asteroids)
│           ├── flyby_camera.py                    # Body-centered close-up camera path
│           ├── flyby_scene.py                     # Host close-up orchestration + GIF assembly
│           └── render_flyby.py                    # Blender EEVEE PNG close-up renderer
│
├── static/                                        # SIDE PRODUCT — still images
│   ├── __init__.py
│   ├── solar_system_visualizer.py                 # SolarSystemVisualizer
│   ├── interstellar_neighborhood_visualizer.py    # InterstellarNeighborhoodVisualizer
│   ├── dimension_plotter.py                       # DimensionPlotter
│   └── hilda_point_generator.py                   # HildaPointGenerator
│
├── tests/                                         # Unit tests (stdlib unittest)
│   ├── test_frame_transform.py                    # Sol ↔ α Cen frame transform
│   ├── test_sol_centauri_cinematic.py             # Sol → α Cen cinematic helpers
│   ├── test_sol_trappist_cinematic.py             # Sol → TRAPPIST-1 cinematic helpers
│   ├── test_tabbys_cinematic.py                   # Tabby's lightcurve cinema (#73)
│   ├── test_blender_body_sprites.py               # Blender sprites + cinematic billboards
│   └── test_blender_pipeline.py                   # Blender export / ingest scaffold
│
├── solsys/                                        # Shared libraries (no plotting)
│   ├── physics/
│   │   ├── astronomical_constants.py              # AstronomicalConstants
│   │   ├── orbit_calculator.py                    # OrbitCalculator
│   │   ├── frame_transform.py                     # Sol ↔ α Cen AB barycenter frame transform
│   │   ├── belt_point_generator.py                # BeltPointGenerator
│   │   ├── view_definition.py                     # ViewDefinition
│   │   ├── view_registry.py                       # ViewRegistry
│   │   ├── point_density_config.py                # PointDensityConfig
│   │   └── catalogs/
│   │       ├── planet_catalog.py                  # PlanetCatalog, PlanetOrbit (Sol, hard-coded)
│   │       ├── moon_catalog.py                    # MoonCatalog, MoonOrbit
│   │       ├── famous_asteroid_catalog.py         # FamousAsteroidCatalog, FamousAsteroidOrbit
│   │       ├── star_catalog.py                    # StarCatalog
│   │       ├── system_catalog.py                  # SystemCatalog (multi-star systems)
│   │       └── interstellar_object_catalog.py     # InterstellarObjectCatalog (1I/2I/3I)
│   └── motion/
│       ├── mean_anomaly.py                        # meanAnomalyAtFrame, planetMeanAnomalyRad
│       └── animated_asteroid_population.py        # AnimatedAsteroidPopulation, AsteroidPopulationCounts
│
└── output/
    ├── animate/
    │   ├── 2d/                                    # inner_solar_system_{light,dark}.gif
    │   ├── 3d/                                    # solar_system_{light,dark}.gif
│   ├── alpha_centauri/                        # ab / system_wide / proxima_planets GIFs
│   ├── sol_centauri/                          # classic cinematic GIFs
│   │   └── blender/                           # cinematic with textured Earth/Moon
│   ├── sol_trappist/                          # Sol → TRAPPIST-1 cinematic GIFs
│   │   └── blender/                           # cinematic with TRAPPIST b–h billboards
│   ├── barnards_star/                         # barnards_star_planets_{light,dark}.gif
│   ├── trappist_1/                            # trappist_1_planets_{light,dark}.gif
│   ├── tabbys_star/                           # dust schematic + cinematic/
│   │   └── cinematic/                         # tabbys_star_cinematic_{light,dark}.gif
│   ├── interstellar_objects/                  # {oumuamua,borisov,atlas}_{side,oblique}_{light,dark}.gif
│   └── blender/{planets,moons}/<body>/        # body JSON + *_flyby_{light,dark}.gif
    ├── 2d/                                        # Static top-down zoom JPGs (*_{light,dark}.jpg)
    ├── 3d/                                        # Static perspective zoom JPGs (*_{light,dark}.jpg)
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
| ![Inner 2D light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/2d/inner_solar_system_light.gif?raw=true) | ![Inner 2D dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/2d/inner_solar_system_dark.gif?raw=true) |

**Sol → Alpha Centauri cinematic (Blender bodies)**

| Light | Dark |
|-------|------|
| ![Sol Centauri blender light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol_centauri/blender/sol_centauri_cinematic_blender_light.gif?raw=true) | ![Sol Centauri blender dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol_centauri/blender/sol_centauri_cinematic_blender_dark.gif?raw=true) |

**Sol → TRAPPIST-1 cinematic (Blender bodies)**

| Light | Dark |
|-------|------|
| ![Sol TRAPPIST blender light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol_trappist/blender/sol_trappist_cinematic_blender_light.gif?raw=true) | ![Sol TRAPPIST blender dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol_trappist/blender/sol_trappist_cinematic_blender_dark.gif?raw=true) |

**Tabby's Star lightcurve cinema (Kepler + photosphere occultation)**

| Light | Dark |
|-------|------|
| ![Tabby cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/tabbys_star/cinematic/tabbys_star_cinematic_light.gif?raw=true) | ![Tabby cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/tabbys_star/cinematic/tabbys_star_cinematic_dark.gif?raw=true) |

**Earth Blender close-up**

| Light | Dark |
|-------|------|
| ![Earth close-up light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/earth/earth_flyby_light.gif?raw=true) | ![Earth close-up dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/earth/earth_flyby_dark.gif?raw=true) |

**Moon Blender close-up**

| Light | Dark |
|-------|------|
| ![Moon close-up light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/moon/moon_flyby_light.gif?raw=true) | ![Moon close-up dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/moon/moon_flyby_dark.gif?raw=true) |

**Jupiter Blender close-up**

| Light | Dark |
|-------|------|
| ![Jupiter close-up light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/jupiter/jupiter_flyby_light.gif?raw=true) | ![Jupiter close-up dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/jupiter/jupiter_flyby_dark.gif?raw=true) |

**Saturn Blender close-up (rings)**

| Light | Dark |
|-------|------|
| ![Saturn close-up light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/saturn/saturn_flyby_light.gif?raw=true) | ![Saturn close-up dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/saturn/saturn_flyby_dark.gif?raw=true) |

**Galilean moons + Titan**

| Body | Light | Dark |
|------|-------|------|
| Io | ![Io light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/io/io_flyby_light.gif?raw=true) | ![Io dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/io/io_flyby_dark.gif?raw=true) |
| Europa | ![Europa light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/europa/europa_flyby_light.gif?raw=true) | ![Europa dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/europa/europa_flyby_dark.gif?raw=true) |
| Ganymede | ![Ganymede light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/ganymede/ganymede_flyby_light.gif?raw=true) | ![Ganymede dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/ganymede/ganymede_flyby_dark.gif?raw=true) |
| Callisto | ![Callisto light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/callisto/callisto_flyby_light.gif?raw=true) | ![Callisto dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/callisto/callisto_flyby_dark.gif?raw=true) |
| Titan | ![Titan light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/titan/titan_flyby_light.gif?raw=true) | ![Titan dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/moons/titan/titan_flyby_dark.gif?raw=true) |

**Ceres + Vesta**

| Body | Light | Dark |
|------|-------|------|
| Ceres | ![Ceres light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/dwarf_planets/ceres/ceres_flyby_light.gif?raw=true) | ![Ceres dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/dwarf_planets/ceres/ceres_flyby_dark.gif?raw=true) |
| Vesta | ![Vesta light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/asteroids/vesta/vesta_flyby_light.gif?raw=true) | ![Vesta dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/asteroids/vesta/vesta_flyby_dark.gif?raw=true) |

**Stars**

| Body | Light | Dark |
|------|-------|------|
| Sun | ![Sun light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/sun/sun_flyby_light.gif?raw=true) | ![Sun dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/sun/sun_flyby_dark.gif?raw=true) |
| Alpha Centauri A | ![α Cen A light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/alpha_centauri_a/alpha_centauri_a_flyby_light.gif?raw=true) | ![α Cen A dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/alpha_centauri_a/alpha_centauri_a_flyby_dark.gif?raw=true) |
| Alpha Centauri B | ![α Cen B light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/alpha_centauri_b/alpha_centauri_b_flyby_light.gif?raw=true) | ![α Cen B dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/alpha_centauri_b/alpha_centauri_b_flyby_dark.gif?raw=true) |
| Proxima Centauri | ![Proxima light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/proxima_centauri/proxima_centauri_flyby_light.gif?raw=true) | ![Proxima dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/proxima_centauri/proxima_centauri_flyby_dark.gif?raw=true) |
| Tabby's Star | ![Tabby's light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/tabbys_star/tabbys_star_spin_light.gif?raw=true) | ![Tabby's dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/tabbys_star/tabbys_star_spin_dark.gif?raw=true) |
| Proxima b | ![Proxima b light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/proxima_b/proxima_b_flyby_light.gif?raw=true) | ![Proxima b dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/proxima_b/proxima_b_flyby_dark.gif?raw=true) |
| Proxima d | ![Proxima d light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/proxima_d/proxima_d_flyby_light.gif?raw=true) | ![Proxima d dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/proxima_d/proxima_d_flyby_dark.gif?raw=true) |
| TRAPPIST-1 b | ![TRAPPIST-1 b light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_b/trappist-1_b_flyby_light.gif?raw=true) | ![TRAPPIST-1 b dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_b/trappist-1_b_flyby_dark.gif?raw=true) |
| TRAPPIST-1 c | ![TRAPPIST-1 c light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_c/trappist-1_c_flyby_light.gif?raw=true) | ![TRAPPIST-1 c dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_c/trappist-1_c_flyby_dark.gif?raw=true) |
| TRAPPIST-1 d | ![TRAPPIST-1 d light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_d/trappist-1_d_flyby_light.gif?raw=true) | ![TRAPPIST-1 d dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_d/trappist-1_d_flyby_dark.gif?raw=true) |
| TRAPPIST-1 e | ![TRAPPIST-1 e light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_e/trappist-1_e_flyby_light.gif?raw=true) | ![TRAPPIST-1 e dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_e/trappist-1_e_flyby_dark.gif?raw=true) |
| TRAPPIST-1 f | ![TRAPPIST-1 f light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_f/trappist-1_f_flyby_light.gif?raw=true) | ![TRAPPIST-1 f dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_f/trappist-1_f_flyby_dark.gif?raw=true) |
| TRAPPIST-1 g | ![TRAPPIST-1 g light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_g/trappist-1_g_flyby_light.gif?raw=true) | ![TRAPPIST-1 g dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_g/trappist-1_g_flyby_dark.gif?raw=true) |
| TRAPPIST-1 h | ![TRAPPIST-1 h light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_h/trappist-1_h_flyby_light.gif?raw=true) | ![TRAPPIST-1 h dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/planets/trappist-1_h/trappist-1_h_flyby_dark.gif?raw=true) |

### Static zoom views (side)

**Inner Solar System With Jupiter**

| Light | Dark |
|-------|------|
| ![Inner light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/1_inner_solar_system_with_jupiter_light.jpg?raw=true) | ![Inner dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/1_inner_solar_system_with_jupiter_dark.jpg?raw=true) |

**Solar System With Kuiper Belt**

| Light | Dark |
|-------|------|
| ![Kuiper light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/2_solar_system_with_kuiper_belt_light.jpg?raw=true) | ![Kuiper dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/2_solar_system_with_kuiper_belt_dark.jpg?raw=true) |

**Solar System With Oort Cloud**

| Light | Dark |
|-------|------|
| ![Oort light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/3_solar_system_with_oort_cloud_light.jpg?raw=true) | ![Oort dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/3_solar_system_with_oort_cloud_dark.jpg?raw=true) |

**Solar System with Alpha Centauri**

| Light | Dark |
|-------|------|
| ![Alpha Cen light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/4_solar_system_with_alpha_centauri_light.jpg?raw=true) | ![Alpha Cen dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/4_solar_system_with_alpha_centauri_dark.jpg?raw=true) |

**Neighbors Within 10 Light Years**

| Light | Dark |
|-------|------|
| ![10 ly light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/5_solar_system_with_nearest_stars_10_light.jpg?raw=true) | ![10 ly dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/5_solar_system_with_nearest_stars_10_dark.jpg?raw=true) |

**Neighbors Within 25 Light Years**

| Light | Dark |
|-------|------|
| ![25 ly light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/6_solar_system_with_nearest_stars_25_light.jpg?raw=true) | ![25 ly dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/2d/6_solar_system_with_nearest_stars_25_dark.jpg?raw=true) |

### Matching 3D static views

**Inner Solar System With Jupiter**

| Light | Dark |
|-------|------|
| ![3D Inner light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/1_inner_solar_system_with_jupiter_light.jpg?raw=true) | ![3D Inner dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/1_inner_solar_system_with_jupiter_dark.jpg?raw=true) |

**Solar System With Kuiper Belt**

| Light | Dark |
|-------|------|
| ![3D Kuiper light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/2_solar_system_with_kuiper_belt_light.jpg?raw=true) | ![3D Kuiper dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/2_solar_system_with_kuiper_belt_dark.jpg?raw=true) |

**Solar System With Oort Cloud**

| Light | Dark |
|-------|------|
| ![3D Oort light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/3_solar_system_with_oort_cloud_light.jpg?raw=true) | ![3D Oort dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/3_solar_system_with_oort_cloud_dark.jpg?raw=true) |

**Solar System with Alpha Centauri**

| Light | Dark |
|-------|------|
| ![3D Alpha light](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/4_solar_system_with_alpha_centauri_light.jpg?raw=true) | ![3D Alpha dark](https://github.com/ThomasAFink/visualization_of_the_solar_system_on_an_interstellar_scale/blob/main/output/3d/4_solar_system_with_alpha_centauri_dark.jpg?raw=true) |

## Getting started

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/pre-commit install
.venv/bin/pre-commit install --hook-type commit-msg
```

### Git hooks (pre-commit)

After install, commits are gated by:

- **pre-commit** — Ruff format + lint on `animate/`, `static/`, `solsys/`, `render.py`
- **commit-msg** — enforces the commit message rules below

CI mirrors the Ruff gate on push/PR to `main` via the `✨ Ruff` workflow (`.github/workflows/ruff.yml`), alongside `🪐 Tests` and `🛡️ CodeQL`.

### Commit message rules

First line must match:

```text
type(SOLSYS): subject
type(SOLSYS-123): subject
```

| Part | Rule |
|------|------|
| `type` | One of `feat`, `fix`, `chore`, `docs`, `clean` |
| Scope | `SOLSYS` or `SOLSYS-<ticket>` (digits only) |
| Subject | Required non-empty text after `: ` |

Examples:

```text
feat(SOLSYS): add light/dark static themes
fix(SOLSYS-42): correct Hilda resonance rate
docs(SOLSYS): document commit message rules
chore(SOLSYS): bump ruff
clean(SOLSYS): remove unused Pluto study script
```

Merge / revert / fixup / squash commits are allowed through unchanged.

Manual checks:

```bash
.venv/bin/pre-commit run --all-files
.venv/bin/ruff format animate static solsys render.py
.venv/bin/ruff check animate static solsys render.py
.venv/bin/radon cc animate static solsys render.py -s -a
.venv/bin/python -m unittest discover -s tests -v
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
.venv/bin/python render.py animate --system alpha_centauri
.venv/bin/python render.py animate --system sol_centauri
.venv/bin/python render.py animate --system sol_centauri --blender-bodies
.venv/bin/python render.py animate --system barnards_star
.venv/bin/python render.py animate --system trappist_1
.venv/bin/python render.py animate --system tabbys_star
.venv/bin/python render.py blender --body "Tabby's Star" --spin --theme all
.venv/bin/python render.py animate --system tabbys_star_cinematic
.venv/bin/python render.py animate --system interstellar
.venv/bin/python render.py animate --system interstellar --object borisov
.venv/bin/python render.py animate --system oumuamua
.venv/bin/python render.py static --dimension 2d
.venv/bin/python render.py neighborhood --ly 10
.venv/bin/python render.py blender --body Earth
.venv/bin/python render.py blender --body Earth --load    # optional debug ingest
.venv/bin/python render.py blender --body Earth --flyby   # light/dark close-up GIFs
.venv/bin/python render.py blender --body Moon --flyby    # Moon close-up GIFs
.venv/bin/python render.py blender --body Jupiter --flyby # textured gas giant
.venv/bin/python render.py blender --body Saturn --flyby  # rings + textured globe
.venv/bin/python render.py blender --body Europa --flyby  # Galilean moon close-up
.venv/bin/python render.py blender --body Titan --flyby   # Titan + haze shell
.venv/bin/python render.py blender --body Ceres --flyby   # dwarf planet
.venv/bin/python render.py blender --body Vesta --flyby   # main-belt asteroid
.venv/bin/python render.py blender --body Sun --flyby     # emissive photosphere
.venv/bin/python render.py blender --body "Alpha Centauri A" --flyby
.venv/bin/python render.py blender --body "Proxima Centauri" --spin
.venv/bin/python render.py blender --body Saturn --spin   # RGBA spin (rings) for cinematic
.venv/bin/python render.py blender --body Earth --spin    # RGBA day/night loop for cinematic
.venv/bin/python render.py blender --body Moon --spin     # Moon spin loop for cinematic
.venv/bin/python render.py blender --pipeline             # Earth+Moon spins + blender cinematic
```

Blender close-ups: `render.py blender` exports Keplerian body-scene JSON (`#11`). `render.py blender --flyby` renders a body-centered close-up via EEVEE (`#12`/`#36`/`#41`/`#46`/`#47`/`#48`/`#50`) to `output/animate/blender/{planets,moons,asteroids,dwarf_planets,stars}/<body>/` (`*_flyby_{light,dark}.gif`). `render.py blender --spin` writes a fixed-camera transparent PNG day/night loop (`*_spin_<theme>/frame_*.png`) for cinematic reuse. Texture packs for Sol, planets, major moons, and named asteroids/dwarfs live under `data/textures/bodies/`.

Blender → cinematic workflow:

```bash
.venv/bin/python render.py blender --body Earth --spin
.venv/bin/python render.py blender --body Moon --spin
.venv/bin/python render.py animate --system sol_centauri --blender-bodies
# or one-shot:
.venv/bin/python render.py blender --pipeline
```

Sol → Centauri cinematic with `--blender-bodies` (issues #42 / #49 / #65 / #67) keeps the same camera tour but composites Blender spin frames **each `FuncAnimation` frame** (not GIF concat) for every Sol body that has a spin pack — planets (incl. ringed giants), major moons, and named asteroids/dwarfs. Loops are lazy-loaded by zoom stage; missing packs fall back to catalog dots, while floored disks stay textured through the Kuiper linger (Earth/Moon keep the no-blue/gray-dot rule). Gallery preview uses the Blender outputs under `output/animate/sol_centauri/blender/`.

Alpha Centauri (issue #1) is one `system_id` covering A, B, and Proxima. Animations render to `output/animate/alpha_centauri/`:

- `alpha_centauri_ab_{light,dark}.gif` — A–B binary close-up (±28 AU)
- `alpha_centauri_system_{light,dark}.gif` — wide triple with Proxima (~8.7 kau)
- `proxima_planets_{light,dark}.gif` — confirmed Proxima planets (via shared `exoplanet_system` animator)

Sol → Alpha Centauri cinematic (issue #10) flies from our solar system to the A–B close-up, zooms back out, then finishes on Proxima and its planets — all in Sol XYZ via `SolCentauriFrameTransform`:

- `output/animate/sol_centauri/sol_centauri_cinematic_{light,dark}.gif` — classic scatter-dot bodies
- `output/animate/sol_centauri/blender/sol_centauri_cinematic_blender_{light,dark}.gif` — Blender spin billboards for Sol planets/moons/rings/asteroids (`--blender-bodies` or `blender --pipeline`)

Earth Blender close-up (issues #12 / #36) is a body-centered EEVEE view of Earth with NASA Blue Marble texture (light/dark). Render with `render.py blender --body Earth --flyby` / `--spin` to `output/animate/blender/planets/earth/`:

- `earth_flyby_{light,dark}.gif` — textured sphere, clouds, thin atmosphere limb, elevated turntable camera + spin
- `earth_spin_{light,dark}/frame_*.png` — fixed-camera RGBA day/night loop for the cinematic (`--spin`)

Moon Blender close-up (issue #41) uses the shared pack path with an LRO LROC color map (airless — no atmosphere/clouds) under `output/animate/blender/moons/moon/`:

- `moon_flyby_{light,dark}.gif` — textured lunar sphere, elevated turntable camera + spin
- `moon_spin_{light,dark}/frame_*.png` — fixed-camera RGBA spin loop for the cinematic (`--spin`)

Sol planet packs (issue #46) extend the same path to Mercury–Neptune + Pluto (color maps; atmosphere where it helps). Saturn/Uranus/Neptune add a shared ring annulus:

- `jupiter_flyby_{light,dark}.gif` — textured Jupiter + subtle limb haze
- `saturn_flyby_{light,dark}.gif` / `saturn_spin_{light,dark}/` — rings + globe
- Other bodies: `render.py blender --body <Name> --flyby`

Major moon packs (issue #47) cover every `MoonCatalog` body (Galileans, Titan, Enceladus, Rhea, Phobos/Deimos, Titania/Oberon, Triton, Charon):

- `io|europa|ganymede|callisto|titan_flyby_{light,dark}.gif` — priority gallery moons
- `--spin` writes transparent RGBA loops under `output/animate/blender/moons/<body>/`
- Airless by default; Titan uses a thin orange haze shell

Asteroid / dwarf packs (issue #48) wire `FamousAsteroidCatalog` into the Blender CLI (`--body Ceres`, `--body Vesta`, …):

- `dwarf_planets/ceres/` and `asteroids/vesta/` flyby + spin outputs
- Also registered: Pallas, Psyche, Bennu, Eros, Haumea, Makemake, Eris
- Airless (no atmosphere/clouds); sphere + texture for v1

Cinematic billboards (issue #49) extend `--blender-bodies` beyond Earth/Moon:

- Planets + Saturn/Uranus/Neptune ring spins during their Sol zoom stages
- Major moons when the parent system is framed tightly enough
- Named asteroids (Ceres/Vesta/…) during belt / Kuiper visibility windows
- Requires `render.py blender --body <Name> --spin` assets under `output/animate/blender/`

Star close-ups (`kind=star`, issues #50 / #59):

- `render.py blender --body Sun --flyby` / `--spin` → `output/animate/blender/stars/sun/`
- `render.py blender --body "Alpha Centauri A"` / `"Alpha Centauri B"` / `"Proxima Centauri"` / `"Tabby's Star"` → `stars/<id>/`
- Emissive photosphere (not a yellow diffuse planet under a key lamp; no atmosphere shell)
- With `--blender-bodies`, the Sol→α Cen cinematic composites A/B (AB hold) and Proxima (dive) spin billboards when on-screen

Staged Sol zoom-out beats (issue #51) — blender mode only (`--blender-bodies`); classic dotted GIFs unchanged:

| Beat | Approx. scale | Notes |
|------|---------------|--------|
| Earth → Moon | 0.04 → 0.16 AU | Day/night hold, reveal Luna, then ~1 full lunar orbit |
| Near-Sun | ~2.4 AU | Large Sol photosphere billboard + look-at Sol (textured spin stays through outer Sol) |
| Inner planets | ~6.5 AU | Mercury / Venus / Mars pass |
| Belt + Jupiter | ~7.8 AU | Ceres hero + Jupiter system |
| Saturn | ~18 AU | Rings linger floor |
| Outer / Kuiper | ~42 AU | Ice giants + Pluto, then Oort pullback |

Staged α Cen arrival beats (issue #63) — blender mode only; classic arrival timing unchanged:

| Beat | Approx. scale | Notes |
|------|---------------|--------|
| AB approach | cruise → ~32 AU | Fly to α Cen; A/B resolve |
| AB hold | ~32 AU | Textured α Cen A + B |
| Triple wide | ~12 000 AU | A/B + Proxima framed |
| Proxima dive | → ~2 AU | Zoom onto Proxima |
| Proxima wide hold | ~2 AU | System overview before the inner close-up |
| Inner planets | ~0.055 AU | b/d finale; Proxima photosphere stays textured |

Beats ease in/out via half-width plateaus (no GIF splicing). HUD titles switch per beat.

Barnard's Star (issue #15) is a nearby M dwarf (~6 ly) with four confirmed sub-Earth planets. Animations render via `exoplanet_system` to `output/animate/barnards_star/`:

- `barnards_star_planets_{light,dark}.gif` — compact planets d, b, c, e

TRAPPIST-1 (issue #7) is a single-host ultracool dwarf (~40.7 ly) with seven confirmed planets. Animations render via `exoplanet_system` to `output/animate/trappist_1/`:

- `trappist_1_planets_{light,dark}.gif` — compact resonant chain (b–h)

Blender close-ups (issue #69): `render.py blender --body "TRAPPIST-1 b"` … `"TRAPPIST-1 h" --flyby` / `--spin` write under `output/animate/blender/planets/trappist-1_<letter>/`.

Sol → TRAPPIST-1 cinematic (issue #71): `--system sol_trappist` reuses the Sol opening / pullback from the α Cen tour, then cruises to TRAPPIST-1 at true Sol XYZ (~40.7 ly; no AB-style frame transform), holds the seven-planet chain with a schematic habitable-zone band, then sequential single-planet portraits of temperate candidates **e** then **f**, and returns to the full b–h finale. With `--blender-bodies`, billboards use the TRAPPIST spin packs (host stays a scatter marker until a star pack exists):

- `output/animate/sol_trappist/sol_trappist_cinematic_{light,dark}.gif` — classic scatter-dot bodies
- `output/animate/sol_trappist/blender/sol_trappist_cinematic_blender_{light,dark}.gif` — Blender Sol + TRAPPIST planet billboards

Tabby's Star (issue #17 / Boyajian's Star / KIC 8462852) has no confirmed planets. The schematic visualizes the leading **uneven circumstellar dust / debris** explanation for its irregular Kepler dips (not a megastructure). Animations render to `output/animate/tabbys_star/`:

- `tabbys_star_dust_{light,dark}.gif` — top-down schematic; real Kepler LC (`data/tabbys_star_lightcurve.csv`); orbiting dust clumps cross the LOS at observed dip times

**Lightcurve cinema** (issue #73) is a different grammar from Sol→destination odysseys: the Kepler flux **is** the timeline. A dedicated F3V Blender photosphere (`render.py blender --body "Tabby's Star" --spin`) sits behind soft debris sprites (#78) that occult the disk on real dips:

- `cinematic/tabbys_star_cinematic_{light,dark}.gif` — measurement-led edit (no Sol open / cruise)

CLI: `render.py animate --system tabbys_star_cinematic` (requires the Tabby's spin pack first).

ʻOumuamua–Earth flyby (issue #2) and interstellar visitors (issue #5) render from
`data/interstellar_objects.csv` via `InterstellarObjectCatalog` to
`output/animate/interstellar_objects/`:

- `{oumuamua,borisov,atlas}_{side,oblique}_{light,dark}.gif`

CLI: `render.py animate --system interstellar` (all) or `--object oumuamua|borisov|atlas`.
`--system oumuamua` remains as an alias for ʻOumuamua only.

## Physics notes

- Planet and asteroid positions use Keplerian ellipses (and a hyperbola for `'Oumuamua`).
- Asteroid / Kuiper / Oort fields use spherical shells in 3D; 2D is the XY projection.
- Hildas follow a 3:2 resonance angular rate relative to Jupiter.
- Jupiter Trojans and Greeks sit near the L4 / L5 Lagrange longitudes.
- Moon orbits are exaggerated for visibility at solar-system zoom levels.
- **Coordinate frames (multi-star):** Sol scenes use a heliocentric / Sol-barycentric frame in AU, with +X/+Y/+Z from the equatorial Cartesian mapping in `OrbitCalculator.equatorialToCartesianAu` (RA/Dec → XYZ). Alpha Centauri scenes use a separate **α Cen AB-barycentric** frame in AU, plotted face-on in the binary orbital plane (sky inclination stored in CSV but not applied to the 2D schematic). `SolCentauriFrameTransform` in `solsys/physics/frame_transform.py` maps between those frames: mass-weighted AB barycenter origin from `nearby_stars_30.csv`, and rotation from the Centauri orbital plane into Sol XYZ using the A/B `(i, Ω)` convention shared with `OrbitCalculator.ellipticalPosition` (`toSol` / `toCentauri`). The Sol → Centauri cinematic (`animate/scenes/sol_centauri_cinematic.py`, `--system sol_centauri`) consumes this API to place A/B in Sol XYZ while the camera travels from the neighborhood into the binary.

## Roadmap

- Additional star systems via `SystemCatalog` / `data/systems.csv` (`exoplanet_system.py` for planet disks; dedicated scenes for dust / other phenomena)
- More Blender close-ups (additional moons, asteroids, planets) via shared `data/textures/bodies/` packs
- Optional TRAPPIST-1 host-star Blender photosphere pack for arrival billboards
- Additional destination cinematics (Barnard's Star) on the Sol→TRAPPIST pattern when a new odyssey chapter is wanted
