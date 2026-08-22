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
│   ├── trappist_1_tess_lightcurve.csv             # TESS S70 2-min PDCSAP for the transit fold (#95)
│   ├── kic_7944142_kepler_lightcurve.csv          # Kepler Q0–Q17 long cadence for the mode spectrum (#169)
│   ├── silso_sunspot_number_monthly.csv           # SILSO v2.0 monthly sunspot number, 1749– (#102)
│   ├── sunspot_groups_carrington.csv              # Group latitudes/areas per rotation, 1874–2019 (#102)
│   ├── ogle_magellanic_cepheids.csv               # OGLE-IV fundamental-mode Cepheids, LMC + SMC (#159)
│   ├── gaia_cepheid_lightcurves.csv               # Gaia DR3 epoch photometry for three LMC Cepheids (#159)
│   ├── ogle_magellanic_rrlyrae.csv                # OGLE-IV Magellanic RRab + RRc (#160)
│   ├── ogle_rrlyrae_lightcurves.csv               # OGLE-IV I-band photometry for three LMC RR Lyrae (#160)
│   ├── pantheonplus_type_ia.csv                   # Pantheon+ Type Ia supernovae, one row per SN (#126)
│   ├── type_ia_lightcurves.csv                    # B-band LCs for SN 2011fe, 2000cn, 2005eq (#126)
│   ├── epn_pulsar_profiles.csv                    # Folded 1.4 GHz Stokes I for Crab, Vela, B0329+54 (#103)
│   ├── atnf_pulsars.csv                           # ATNF periods and characteristic ages (#103)
│   ├── chicxulub_kpg.csv                          # Chicxulub location, size, speed, age (#86)
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
│       ├── transit_cinematic.py                   # TRAPPIST-1 b transit cinema (#95; real TESS fold)
│       ├── asteroseismology_cinematic.py          # red giant weighed by its ringing (#169; Kepler FFT)
│       ├── solar_cycle_cinematic.py               # the Sun's spots counted and placed (#102; SILSO + butterfly)
│       ├── cepheid_ladder_cinematic.py            # Leavitt's law fitted from OGLE-IV Cepheids (#159)
│       ├── rr_lyrae_cinematic.py                  # RR Lyrae / horizontal-branch clocks (#160)
│       ├── type_ia_cinematic.py                   # Type Ia standard candles, Pantheon+ Hubble diagram (#126)
│       ├── pulsar_cinematic.py                    # pulsar lighthouse, EPN profiles + ATNF ages (#103)
│       ├── kpg_cinematic.py                       # K–Pg / Chicxulub camera-move impact (#86 / #210)
│       ├── interstellar_objects.py                # 1I/2I/3I hyperbolic passages (side + oblique)
│       └── blender/                               # Blender close-up pipeline (catalog → JSON → bpy)
│           ├── README.md                          # Pipeline docs + CLI examples
│           ├── body_scene.py                      # Versioned body-scene schema + PlanetCatalog build
│           ├── export_body.py                     # Write blender/{planets,moons}/<body>/*.json
│           ├── load_body.py                       # Blender ingest (stdlib + optional bpy)
│           ├── body_appearance.py                 # Shared texture packs (planets/moons/asteroids)
│           ├── flyby_camera.py                    # Body-centered close-up camera path
│           ├── flyby_scene.py                     # Host close-up orchestration + GIF assembly
│           ├── render_flyby.py                    # Blender EEVEE PNG close-up renderer
│           └── render_kpg.py                      # Blender Earth + fireball / ejecta (#86 / #210)
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
│   ├── test_transit_cinematic.py                  # TRAPPIST-1 b transit fold vs TESS data (#95)
│   ├── test_asteroseismology_cinematic.py         # numax / Dnu / R / M re-derived from Kepler (#169)
│   ├── test_solar_cycle_cinematic.py              # cycle timing, disk projection, Spörer drift (#102)
│   ├── test_cepheid_ladder_cinematic.py           # Leavitt slope, Wesenheit tightening, SMC distance (#159)
│   ├── test_rr_lyrae_cinematic.py                 # Bailey split, Wesenheit, SMC offset vs EBs (#160)
│   ├── test_type_ia_cinematic.py                  # Δm15, Phillips, Hubble-diagram slope (#126)
│   ├── test_pulsar_cinematic.py                   # W50, playhead clocks, Crab age vs SN 1054 (#103)
│   ├── test_kpg_cinematic.py                      # Chicxulub vector, dive, schematic contact (#86 / #210)
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
│   │   └── cinematic/                         # trappist_1_transit_cinematic_{light,dark}.gif
│   ├── kic_7944142/
│   │   └── cinematic/                         # kic_7944142_asteroseismology_{light,dark}.gif
│   ├── sol/
│   │   └── cinematic/                         # sol_solar_cycle_{light,dark}.gif
│   ├── magellanic/
│   │   └── cinematic/                         # magellanic_cepheid_ladder_{light,dark}.gif
│   │                                          # magellanic_rr_lyrae_{light,dark}.gif
│   ├── type_ia/
│   │   └── cinematic/                         # type_ia_standard_candle_{light,dark}.gif
│   ├── pulsar/
│   │   └── cinematic/                         # pulsar_lighthouse_{light,dark}.gif
│   ├── earth/
│   │   └── cinematic/                         # earth_kpg_{light,dark}.gif
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

**TRAPPIST-1 b transit cinema (real TESS photometry, revealed by folding)**

| Light | Dark |
|-------|------|
| ![TRAPPIST transit cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/trappist_1/cinematic/trappist_1_transit_cinematic_light.gif?raw=true) | ![TRAPPIST transit cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/trappist_1/cinematic/trappist_1_transit_cinematic_dark.gif?raw=true) |

**Asteroseismology cinema (Kepler red giant, weighed by its own ringing)**

| Light | Dark |
|-------|------|
| ![Asteroseismology cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/kic_7944142/cinematic/kic_7944142_asteroseismology_light.gif?raw=true) | ![Asteroseismology cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/kic_7944142/cinematic/kic_7944142_asteroseismology_dark.gif?raw=true) |

**Solar cycle cinema (real sunspot groups on the disk, opening into the butterfly)**

| Light | Dark |
|-------|------|
| ![Solar cycle cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol/cinematic/sol_solar_cycle_light.gif?raw=true) | ![Solar cycle cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/sol/cinematic/sol_solar_cycle_dark.gif?raw=true) |

**Cepheid ladder cinema (three real folded light curves, then Leavitt's law fitted across two galaxies)**

| Light | Dark |
|-------|------|
| ![Cepheid ladder cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/magellanic/cinematic/magellanic_cepheid_ladder_light.gif?raw=true) | ![Cepheid ladder cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/magellanic/cinematic/magellanic_cepheid_ladder_dark.gif?raw=true) |

**RR Lyrae cinema (three real folded light curves, then Bailey's diagram across two galaxies)**

| Light | Dark |
|-------|------|
| ![RR Lyrae cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/magellanic/cinematic/magellanic_rr_lyrae_light.gif?raw=true) | ![RR Lyrae cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/magellanic/cinematic/magellanic_rr_lyrae_dark.gif?raw=true) |

**Type Ia standard-candle cinema (three real light curves, then a Hubble diagram of 1,543 supernovae)**

| Light | Dark |
|-------|------|
| ![Type Ia cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/type_ia/cinematic/type_ia_standard_candle_light.gif?raw=true) | ![Type Ia cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/type_ia/cinematic/type_ia_standard_candle_dark.gif?raw=true) |

**Pulsar lighthouse cinema (three real folded profiles, then a catalogue of clocks)**

| Light | Dark |
|-------|------|
| ![Pulsar cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/pulsar/cinematic/pulsar_lighthouse_light.gif?raw=true) | ![Pulsar cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/pulsar/cinematic/pulsar_lighthouse_dark.gif?raw=true) |

**K–Pg cinema (camera move into Chicxulub, schematic fireball and ejecta)**

| Light | Dark |
|-------|------|
| ![K-Pg cinema light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/earth/cinematic/earth_kpg_light.gif?raw=true) | ![K-Pg cinema dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/earth/cinematic/earth_kpg_dark.gif?raw=true) |

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
| TRAPPIST-1 | ![TRAPPIST-1 light](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/trappist-1/trappist-1_spin_light.gif?raw=true) | ![TRAPPIST-1 dark](https://github.com/ThomasAFink/SOLSYS/blob/main/output/animate/blender/stars/trappist-1/trappist-1_spin_dark.gif?raw=true) |
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
.venv/bin/python render.py blender --body "TRAPPIST-1" --spin --theme all
.venv/bin/python render.py blender --body "TRAPPIST-1 b" --spin --theme all
.venv/bin/python render.py animate --system transit_cinematic

# Asteroseismology cinema (needs the KIC 7944142 + Sun spin packs first)
.venv/bin/python render.py blender --body "KIC 7944142" --spin --theme all
.venv/bin/python render.py blender --body Sun --spin --theme all
.venv/bin/python render.py animate --system asteroseismology_cinematic

# Solar cycle cinema (needs the Sun spin pack first)
.venv/bin/python render.py blender --body Sun --spin --theme all
.venv/bin/python render.py animate --system solar_cycle_cinematic

# Cepheid ladder cinema (catalogue + photometry only, no Blender pack needed)
.venv/bin/python render.py animate --system cepheid_ladder_cinematic

# RR Lyrae cinema (catalogue + photometry only, no Blender pack needed)
.venv/bin/python render.py animate --system rr_lyrae_cinematic

# Type Ia standard-candle cinema (catalogue + photometry only, no Blender pack needed)
.venv/bin/python render.py animate --system type_ia_cinematic

# Pulsar lighthouse cinema (catalogue + folded profiles only, no Blender pack needed)
.venv/bin/python render.py animate --system pulsar_cinematic

# K–Pg cinema (needs blender on PATH; uses the in-repo Earth texture pack)
.venv/bin/python render.py animate --system kpg_cinematic
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
- `render.py blender --body "Alpha Centauri A"` / `"Alpha Centauri B"` / `"Proxima Centauri"` / `"Tabby's Star"` / `"TRAPPIST-1"` → `stars/<id>/`
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

Sol → TRAPPIST-1 cinematic (issue #71): `--system sol_trappist` reuses the Sol opening / pullback from the α Cen tour, then cruises to TRAPPIST-1 at true Sol XYZ (~40.7 ly; no AB-style frame transform), holds the seven-planet chain with a schematic habitable-zone band, then sequential single-planet portraits of temperate candidates **e** then **f**, and returns to the full b–h finale. With `--blender-bodies`, billboards use the TRAPPIST spin packs, including the M8V host photosphere (issue #88) once the camera is tight enough for the chain (~0.2 AU); the cruise keeps a scatter marker, where a world-fixed disk would be sub-pixel:

- `output/animate/sol_trappist/sol_trappist_cinematic_{light,dark}.gif` — classic scatter-dot bodies
- `output/animate/sol_trappist/blender/sol_trappist_cinematic_blender_{light,dark}.gif` — Blender Sol + TRAPPIST planet billboards

Tabby's Star (issue #17 / Boyajian's Star / KIC 8462852) has no confirmed planets. The schematic visualizes the leading **uneven circumstellar dust / debris** explanation for its irregular Kepler dips (not a megastructure). Animations render to `output/animate/tabbys_star/`:

- `tabbys_star_dust_{light,dark}.gif` — top-down schematic; real Kepler LC (`data/tabbys_star_lightcurve.csv`); orbiting dust clumps cross the LOS at observed dip times

**Lightcurve cinema** (issue #73) is a different grammar from Sol→destination odysseys: the Kepler flux **is** the timeline. A dedicated F3V Blender photosphere (`render.py blender --body "Tabby's Star" --spin`) sits behind soft debris sprites (#78) that occult the disk on real dips:

- `cinematic/tabbys_star_cinematic_{light,dark}.gif` — measurement-led edit (no Sol open / cruise)

CLI: `render.py animate --system tabbys_star_cinematic` (requires the Tabby's spin pack first).

**Transit cinema** (issue #95) keeps that measurement grammar but changes the explanation: the dip is a **planet**, not dust. It also refuses the textbook cheat of drawing a transit you could never actually see. Every flux point is observed — TESS Sector 70, 2-minute PDCSAP from MAST (`data/trappist_1_tess_lightcurve.csv`) — and at TESS's 1.37% point-to-point scatter, TRAPPIST-1 b's 0.74% transit is **invisible**. The film is built around the step that fixes that:

1. **Stream** — the real light curve scrolls past while b crosses the M8V photosphere (#88) as a silhouette (#69) at the published radius ratio. Nothing shows in the data.
2. **Fold** — the 13 observed transit windows slide together onto one phase axis; the quiet baseline fades away.
3. **Reveal** — stacked in 10-minute phase bins, the dip appears at 10σ, deeper than the geometric (Rp/R★)² because the disk is limb darkened.

- `output/animate/trappist_1/cinematic/trappist_1_transit_cinematic_{light,dark}.gif`

The b ephemeris is measured from that light curve by box-least-squares with no catalog input (period 1.510919 d vs 1.510826 d published), and the tests re-derive the fold from the committed CSV: a dip at phase 0 above 5σ, no dip half a period away, and no single transit clearing a detection on its own. Regenerate the CSV with lightkurve (a dev-only tool, not a runtime dependency) if it ever needs refreshing.

CLI: `render.py animate --system transit_cinematic` (requires the TRAPPIST-1 and TRAPPIST-1 b spin packs).

**Asteroseismology cinema** (issue #169) closes the measurement trilogy with a signal that is plainly visible and still means nothing until you change domain. KIC 7944142 (HD 176694, Kp 7.8) is a red giant whose surface heaves by ~670 ppm; four years of Kepler long-cadence PDCSAP (`data/kic_7944142_kepler_lightcurve.csv`, 18 quarters from MAST) show that wobble directly. What the wobble *is* only appears after a Fourier transform:

1. **Wobble** — real photometry scrolls past while the photosphere brightens and dims with it (exaggerated ×150, and labelled as such, since a few hundred ppm is invisible on screen).
2. **Transform** — the strip becomes a power spectrum; granulation and shot noise are divided out by a continuum anchored on medians either side of the modes.
3. **Envelope** — the wobble was never noise. It is a hump of pure tones centred on νmax.
4. **Fold** — the same trick that found the planet in #95, now applied in frequency: wrap the spectrum every Δν and the overtones stack into ridges.
5. **Payoff** — those two frequencies plus a temperature give the star through the scaling relations, with the Sun drawn beside it at the measured scale.

- `output/animate/kic_7944142/cinematic/kic_7944142_asteroseismology_{light,dark}.gif`

Nothing is precomputed: the spectrum, νmax, Δν, radius and mass are all derived at runtime from the committed CSV with numpy's FFT alone (quarter gaps zero-filled on a regular grid, 91% duty cycle). The film measures νmax = 76.4 µHz and Δν = 6.92 µHz against published 74.75 and 6.993 (Yu et al. 2018), giving R = 8.8 R☉ and M = 1.8 M☉ against 8.38 and 1.59 — the captions show both, since a boxcar envelope peak is coarser than a fitted one and mass goes as νmax³/Δν⁴, which amplifies that gap. Tests re-derive all of it from the CSV, including a check that folding on the measured Δν concentrates power while wrong spacings do not.

CLI: `render.py animate --system asteroseismology_cinematic` (requires the `KIC 7944142` and `Sun` spin packs).

**Solar cycle cinema** (issue #102) turns the same grammar on the one star that is not a point of light. Three films measured stars we can only count photons from; here the measurement is older and much more literal — someone looked at the disk and wrote down what was on it. Two observed series drive it: SILSO's monthly sunspot number since 1749 (`data/silso_sunspot_number_monthly.csv`) and the Mandal et al. 2020 cross-calibrated composite of group positions and areas since 1874 (`data/sunspot_groups_carrington.csv`, sampled one observed day per Carrington rotation):

1. **Minimum** — cycle 23 opens on an almost blank Sun. Each disk carries the groups actually recorded that day, at their measured heliographic latitude and central meridian distance, sized by their measured area.
2. **Maximum** — four years later the same face is crowded, and the count peaks at 244 in July 2000.
3. **Century** — the strip pulls back to all 275 years. The mean cycle runs 11.0 years, but individual cycles run 9.1 to 13.7: the clock keeps bad time.
4. **Butterfly** — the count flattens onto the equator line it was counting up from, and the wings open in its place. A number says how many spots there were, never where.
5. **Payoff** — averaged over every cycle since 1874, spots open at ±22° and close at ±10°. Spörer's law, measured rather than asserted.

- `output/animate/sol/cinematic/sol_solar_cycle_{light,dark}.gif`

The disks are not decorated. Latitude and central meridian distance are projected with B0, the tilt of the solar axis toward Earth on that date (computed from the date, Meeus ch. 29), and the tests check that projection against the catalogue's own record of how far each group sat from disk centre: the median disagreement is 0.0017 solar radii over 8,408 groups, and dropping B0 makes it eleven times worse. Each group is drawn as the circle its measured area implies and then widened ×2.5 together, labelled on screen, because a 300 msh group really is about a fiftieth of the disk across. Cycle minima, mean length, the featured cycle's peak and the latitude drift are all measured at runtime from the two CSVs. The flare/CME beat the issue sketched is deliberately absent: there is no observed series behind it here, and a drawn coronal mass ejection would be the one invented thing in a film whose whole claim is that nothing is.

CLI: `render.py animate --system solar_cycle_cinematic` (requires the `Sun` spin pack).

**Cepheid ladder cinema** (issue #159) is the first rung of the distance ladder, and the first film here whose subject is another galaxy. Two committed datasets drive it: 4,952 fundamental-mode classical Cepheids in the Magellanic Clouds from the OGLE-IV Collection of Variable Stars (Soszyński et al. 2015, `data/ogle_magellanic_cepheids.csv`) and Gaia DR3 epoch photometry for three of them (`data/gaia_cepheid_lightcurves.csv`):

1. **Pulse** — one Cepheid, OGLE-LMC-CEP-3592, folded on its catalogued 3.00-day period. The points are individual Gaia transits; the line is a three-harmonic fit through them.
2. **Trio** — two more arrive, at 9.99 and 34.45 days. Every playhead runs on its own clock, so in the same 36 days of star time the fast one completes twelve cycles and the slow one barely manages one — and the slow one is 3.2 magnitudes brighter. All three sit in the same galaxy, so that cannot be a distance effect.
3. **Leavitt** — the period–luminosity plane fills with 2,315 LMC Cepheids and the ridge is fitted on screen.
4. **Wesenheit** — mean I slides into the reddening-free combination W = I − 1.55 (V − I) a fraction at a time, and the ridge tightens from 0.146 to 0.079 mag as the dust is divided out.
5. **Clouds** — the SMC's 2,637 Cepheids arrive as a second, parallel ridge, offset 0.461 mag. That offset is a distance ratio of 1.24.

- `output/animate/magellanic/cinematic/magellanic_cepheid_ladder_{light,dark}.gif`

The fit is live: every frame refits whatever is currently on screen, so the slope and scatter printed during the reddening morph are the real values for a partly corrected magnitude, not an interpolation between two remembered numbers. The LMC Wesenheit slope comes out at −3.313 mag/dex, which is the value OGLE reports for this sample, with 0.079 mag scatter over 2,231 stars after a 3σ clip that rejects under 4% of them. Feeding the 0.461 mag offset the LMC's geometric distance from detached eclipsing binaries (18.477 ± 0.026, Pietrzyński et al. 2019 — 49.6 kpc) puts the SMC at 61.3 kpc, against the 62.4 kpc measured the same geometric way (Graczyk et al. 2020): a 0.04 mag disagreement, which is roughly what the metallicity difference between the clouds and the SMC's line-of-sight depth should cost. The two clouds' slopes differ by 0.14 mag/dex for the same reasons, and the film says so rather than fitting a common slope to hide it. Nothing is precomputed — the folds, the Fourier fits, both ridges, the offset and the distance are all derived at runtime from the two CSVs.

CLI: `render.py animate --system cepheid_ladder_cinematic` (no Blender pack needed).

**RR Lyrae cinema** (issue #160) is the other clock on the first rung: metal-poor horizontal-branch stars, two pulsation modes, nearly a standard candle. Two committed datasets drive it: 42,364 single-mode RR Lyrae in the Magellanic Clouds from the OGLE-IV Collection of Variable Stars (Soszyński et al. 2016, `data/ogle_magellanic_rrlyrae.csv`) and I-band photometry for three LMC stars (`data/ogle_rrlyrae_lightcurves.csv`):

1. **Pulse** — OGLE-LMC-RRLYR-03686, an RRc, folded at 0.284 days. A sine, not a sawtooth.
2. **Trio** — two RRab arrive, at 0.489 and 0.650 days. Every playhead runs on its own clock, and Bailey's 1902 result is already visible: the longer fundamental has the smaller bump (0.70 vs 0.40 mag).
3. **Bailey** — period against I-band amplitude fills with 27,199 LMC RRab and 9,390 RRc. Median amplitude 0.54 vs 0.27 mag. Period is the mode.
4. **Candle** — mean I slides into W = I − 1.55 (V − I). The RRab ridge tightens from 0.162 to 0.133 mag.
5. **Clouds** — the SMC is 0.368 mag farther on this clock: 58.7 kpc against 62.4 kpc from eclipsing binaries.

- `output/animate/magellanic/cinematic/magellanic_rr_lyrae_{light,dark}.gif`

Nothing is precomputed. The folds, the Bailey split, both ridges and the offset are derived at runtime. The film does not hide the shortfall: RR Lyrae still need a metallicity term the Cepheid film already absorbed, so this clock puts the SMC too close. Tests re-derive the mode split, the Wesenheit tightening, the playhead ratios and the caption numbers from the two CSVs.

CLI: `render.py animate --system rr_lyrae_cinematic` (no Blender pack needed).

**Type Ia standard-candle cinema** (issue #126) is the second rung of the distance ladder, standing on the Cepheid film. Two committed datasets drive it: 1,543 unique Type Ia supernovae from Pantheon+ / SH0ES (Scolnic et al. 2022, `data/pantheonplus_type_ia.csv`) and B-band light curves for three of them from the Open Supernova Catalog (`data/type_ia_lightcurves.csv`):

1. **Pulse** — SN 2011fe in M101, walking a real B-band light curve. It fades 0.98 mag in 15 days.
2. **Trio** — SN 2000cn (fast, Δm15 = 1.54) and SN 2005eq (slow, Δm15 = 0.89) arrive. Placed at cz/70 km/s/Mpc so brightness is not a distance effect, the slow one is 0.67 mag brighter. Phillips 1993.
3. **Stretch** — the time axis stretches until every decline matches 2011fe. That is the SALT2 x1 correction: in the Hubble-flow sample, stretch vs brightness slopes −0.19 mag per unit x1.
4. **Hubble** — the diagram fills with 1,543 standardized supernovae. The Hubble-flow slope is 5.24 mag per dex of redshift against 5 from the inverse-square law, with 0.14 mag scatter over 476 SN.
5. **Ruler** — 43 of them exploded in Cepheid hosts. The light curve is the ruler; the first rung is what sets its length.

- `output/animate/type_ia/cinematic/type_ia_standard_candle_{light,dark}.gif`

Nothing is precomputed. Δm15 is interpolated from the committed photometry, the Phillips slope and the Hubble-diagram slope are fitted at render time from the Pantheon+ table, and the captions quote those fits. The film does not quote a Hubble constant: a q0-less cz/H0 is the wrong estimator, and the zero point is the Cepheid rung the previous film already measured. Tests re-derive the inverse-square slope, the Phillips sign, the stretch collapse and the caption numbers from the two CSVs.

CLI: `render.py animate --system type_ia_cinematic` (no Blender pack needed).

**Pulsar lighthouse cinema** (issue #103) is a neutron star treated as a clock. Two committed datasets drive it: folded 1.4 GHz Stokes I profiles for the Crab, Vela and B0329+54 from the European Pulsar Network (`data/epn_pulsar_profiles.csv`) and 2,052 pulsars with a measured period and characteristic age from the ATNF catalogue via VizieR B/psr (`data/atnf_pulsars.csv`):

1. **Pulse** — the Crab, folded at 1.4 GHz. A 33.4 ms period, a 1.5% duty cycle, and a second spike (the interpulse) half a turn later.
2. **Trio** — Vela (89.3 ms) and B0329+54 (0.715 s) arrive. Every playhead runs on its own clock, slowed ×7 so the motion is readable: in the same seconds of star time the Crab races and B0329+54 crawls.
3. **Beam** — a schematic lighthouse whose wedge is the measured W50. Rotation is locked to the Crab playhead; when the wedge sweeps Earth, the pulse is at its peak.
4. **Ages** — the period–age plane fills with 2,052 pulsars. P-dot is recovered as P/2τ from the catalogue's own characteristic age. The Crab sits at 4.20×10⁻¹³ s/s.
5. **Remnant** — that clock reads 1,260 years. SN 1054 was 972 years ago. Characteristic age assumes the star was born spinning infinitely fast, so it overshoots the historical year; four catalogue objects are younger, none of them a 33 ms radio pulsar.

- `output/animate/pulsar/cinematic/pulsar_lighthouse_{light,dark}.gif`

Nothing is precomputed. Duty cycle is the fraction of the folded profile above half maximum after a baseline subtraction, P-dot is P/2τ, and the captions quote those measurements. The lighthouse is labelled schematic: the wedge width is the measured W50, the tilt is not a fitted magnetic obliquity. Tests re-derive the periods, the duty cycles, the interpulse, the playhead ratios, the Crab P-dot and the SN 1054 comparison from the two CSVs.

CLI: `render.py animate --system pulsar_cinematic` (no Blender pack needed).

**K–Pg cinema** (issues #86 / #210) is a camera move, not a chart. Stay on Earth. A committed one-row table (`data/chicxulub_kpg.csv`) plus the catalogue Earth diameter drive it: Chicxulub at 21.4°N, 89.5°W, 66.0 Ma, a 10 km body at 20 km/s, a 180 km crater.

1. **Quiet** — Late Cretaceous Earth, modern Blue Marble as a labelled stand-in. The Yucatán faces the camera.
2. **Approach** — the camera dives. The rock is true scale (10 / 12,742 of Earth's diameter), so it is a speck until the last radii. The inbound clock is distance / 20 km/s.
3. **Strike** — contact. A fireball and a 45° ejecta curtain, labelled Hollywood-adjacent schematic, not a hydro simulation.
4. **Veil** — a dust plume, then the disk darkens and the light returns. That veil is not a climate model.

- `output/animate/earth/cinematic/earth_kpg_{light,dark}.gif`

CLI: `render.py animate --system kpg_cinematic` (requires `blender` on PATH).

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

Season backlog (opened after 0.3.18):

- **Measurement / stay-with-object:** Betelgeuse (#84/#85); Tabby (#73); white-dwarf pollution (#104); Venus transit/eclipse (#105); technosignature LC (#112); Listen waterfall (#113); Dyson limits (#114); EHT shadow (#123); kilonova/GW (#124); FRB (#125); solar neutrinos (#127); Mars dust year (#131); biosignature false positives (#136); magnetar flare (#145); blazar/AGN (#146); SEPs (#148); carbon-star wind (#161); transit retrieval cartoon (#168)
- **Deep time:** Moon-forming (#106); Snowball/O₂ (#107); Apophis (#108); Cambrian bumper (#162); Chicxulub core zoom (#163)
- **Encounters / formation:** Solar flybys (#96); formation (#97); Oort rain (#120); interstellar vs Oort taxonomy (#135)
- **Cosmic timeline:** Local Group (#87); Hubble Deep Field (#141)
- **Mission timeline:** Cassini (#93); Voyager (#94); New Horizons (#98); Parker (#99); Galileo/Juno (#100); Apollo (#101)
- **Stay-with-body / Sol system craft:** Io (#128); Enceladus plumes (#129); Venus hell-twin (#130); Saturn hexagon (#132); ring shepherds (#133); Jupiter Trojans (#134); ocean worlds (#137); Proxima terminator (#117); TRAPPIST resonance (#118); Mercury spin/hollows (#149); Uranus seasons (#150); Neptune GDS (#151); Titan methane (#152); Pluto haze/heart (#153); binary asteroids (#154); quasi-satellites (#155)
- **Stellar populations:** open cluster dissolve (#156); globular core (#157); blue stragglers (#158)
- **Geometry:** Lagrange (#116); lensing (#119); Arecibo (#115); parallax (#142); aberration (#143); horseshoe/quasi-sat (#155)
- **History of science:** Kepler/Mars (#139); Le Verrier/Neptune (#140)
- **Earth / life analogy:** night-lights (#138); extremophiles↔ocean worlds (#164); ozone hole (#165); cosmic-ray shower (#147)
- **Observing craft:** adaptive optics (#166); speckle/masking (#167)
- **Scale / catalog:** Gaia (#109); HZ ladder (#110); rogue planets (#111); units (#122); scale continuum (#170)
- **Craft / perception / literacy:** true vs false-color Sol (#121); sky-map projections (#144); error bars (#171); data provenance bumper (#172); GIF compression honesty (#194); sim vs data manifesto (#195)
- **Cosmology / dark sector teaching:** CMB peaks (#173); BAO (#174); weak lensing mass map (#175); neutrino hierarchy cartoon (#176); proton-decay non-detection (#177)
- **More Sol-system craft shorts:** Ceres Occator (#178); Vesta Rheasilvia (#179); Hyperion tumble (#180); Iapetus two-tone (#181); Miranda coronae (#182); centaur rings (#183); lunar gardening (#193)
- **More transients / stars:** interstellar meteor careful (#184); brown dwarf weather (#185); YSO flicker/jet (#186); CV/nova (#187); microlensing planet anomaly (#188); ZTF/LSST alert stream (#189)
- **Earth space environment:** Van Allen (#190); Schumann bumper (#191); constellation streaks vs astronomy (#192)
- **ISM:** Nebulas (#81)
- **Odyssey (optional):** Sol → Barnard's (#89)
- **Craft / inventory:** more Blender packs (#90); more star portraits (#82); more SystemCatalog systems (#91)
- **Wild-card:** something entirely different (#83) — brief before code

Older craft notes:

- Additional destination cinematics reuse the Sol→TRAPPIST pattern when a new odyssey chapter is wanted (#89)
