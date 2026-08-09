"""Sol → Alpha Centauri cinematic using SolCentauriFrameTransform."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import proj3d
from solsys.motion import AnimatedAsteroidPopulation, AsteroidPopulationCounts
from solsys.motion.mean_anomaly import planetMeanAnomalyRad
from solsys.physics import (
    AstronomicalConstants,
    FamousAsteroidCatalog,
    OrbitCalculator,
    PlanetCatalog,
    SolCentauriFrameTransform,
)
from solsys.physics.catalogs.moon_catalog import MoonCatalog
from solsys.physics.catalogs.planet_catalog import PlanetOrbit
from solsys.physics.catalogs.system_catalog import (
    StarSystem,
    StellarOrbit,
    SystemCatalog,
    SystemPlanet,
)

from animate.animation_styles import ASTEROID_RENDER_STYLES
from animate.blender_body_sprites import BlenderBodySpriteAtlas
from animate.scenes.blender.body_appearance import appearanceForCatalogName
from animate.scenes.exoplanet_system import bodyPositionInOrbitalPlane, orbitPathInOrbitalPlane

DEFAULT_FIGURE_SIZE_INCHES = (12.0, 12.0)
DEFAULT_DPI = 100
ANIMATION_FPS = 20
ANIMATION_FRAMES = 1800
ANIMATION_SPEED_AB = 90.0
ANIMATION_SPEED_PROXIMA_STAR = 520000.0
ANIMATION_SPEED_PROXIMA_PLANETS = 0.28
ANIMATION_SPEED_SOL = 3.5
# Keep Sol orbits calm until the outer system, then ramp hard for giant / Pluto motion.
ANIMATION_SPEED_SOL_NEAR = 3.5
ANIMATION_SPEED_SOL_FAR = 520.0
ANIMATION_SPEED_SOL_RAMP_START_AU = 14.0
ANIMATION_SPEED_SOL_FAR_HALF_AU = 42.0
NEIGHBORHOOD_MAX_LY = 8.0
# Nearby catalog stars at true Sol XYZ (CSV has no RGB — tint from Stellarclass).
FIELD_STARS_MAX_LY = 30.0
# Only draw once the camera is wide enough for stellar distances to matter.
FIELD_STARS_VISIBLE_ABOVE_AU = 2000.0
# Approximate blackbody tints from the Morgan–Keenan letter (muted for background use).
SPECTRAL_FIELD_COLORS = {
    'O': '#9BB0FF',
    'B': '#AABFFF',
    'A': '#CAD7FF',
    'F': '#F8F7FF',
    'G': '#FFE7A0',
    'K': '#FFC98A',
    'M': '#FFB06B',
    'L': '#C08060',
    'T': '#9080A0',
    'Y': '#706878',
    'D': '#D8E6FF',  # white dwarfs (class strings often start with D)
}
# Sol opening: Earth+Moon → near-Sun → inner → tighter outer linger, then interstellar pullback.
# Moon display radius ≈ 0.128 AU (50× exaggerated); leave a little margin for the label.
SOL_EARTH_HALF_AU = 0.16
# Blender open: Earth-only frame (Moon orbit ≈ 0.128 AU stays off-screen).
SOL_EARTH_CLOSE_HALF_AU = 0.042
# Fixed world radii for textured globes (must NOT scale up with camera half-width).
EARTH_GLOBE_RADIUS_AU = SOL_EARTH_HALF_AU * 0.18
# Luna uses bodyScale 0.35 against this base in the billboard path.
# Visual (not physical) billboard scales vs Earth — keep giants/asteroids modest.
BLENDER_PLANET_BODY_SCALE = {
    'Mercury': 0.34,
    'Venus': 0.78,
    'Earth': 1.0,
    'Mars': 0.45,
    'Jupiter': 1.65,
    'Saturn': 1.35,
    'Uranus': 1.05,
    'Neptune': 1.0,
    'Pluto': 0.32,
    # Proxima finale — smaller than the Proxima photosphere at every finale zoom.
    'Proxima b': 0.35,
    'Proxima d': 0.22,
}
# Cap on-screen size so Proxima planets stay under the star disk.
BLENDER_PLANET_MAX_FRAC = {
    'Proxima b': 0.028,
    'Proxima d': 0.020,
}
BLENDER_MOON_BODY_SCALE = {
    'Moon': 0.35,
    'Phobos': 0.10,
    'Deimos': 0.08,
    'Io': 0.20,
    'Europa': 0.18,
    'Ganymede': 0.22,
    'Callisto': 0.20,
    'Titan': 0.22,
    'Enceladus': 0.10,
    'Rhea': 0.12,
    'Titania': 0.12,
    'Oberon': 0.12,
    'Triton': 0.14,
    'Charon': 0.14,
}
# Named asteroids / dwarfs — small belt markers (not Earth-class disks).
BLENDER_ASTEROID_BODY_SCALE = {
    'Ceres': 0.12,
    'Vesta': 0.09,
    'Pallas': 0.08,
    'Psyche': 0.08,
    'Bennu': 0.05,
    'Eros': 0.06,
    'Haumea': 0.10,
    'Makemake': 0.10,
    'Eris': 0.11,
}
# Star photosphere billboards (world scale vs Earth globe).
# Sol: Near-Sun hero disk, but radius must stay inside Mercury's orbit (~0.39 AU).
# α Cen A/B: readable at AB hold (~32 AU).
# Proxima: MUST stay smaller than Proxima d/b orbits (~0.029 / 0.049 AU) or the
# planets crawl across the photosphere instead of orbiting around it.
BLENDER_STAR_BODY_SCALE = {
    'Sun': 5.5,
    'Alpha Centauri A': 48.0,
    'Alpha Centauri B': 40.0,
    # Under Proxima d's orbit (~0.029 AU); larger than b/d planet billboards.
    'Proxima Centauri': 0.75,
}
# Cap on-screen size for non-Sol stars so the Proxima dive does not fill the frame.
BLENDER_STAR_MAX_FRAC = {
    'Alpha Centauri A': 0.055,
    'Alpha Centauri B': 0.048,
    # Keep under planet orbit fracs through the wide→inner tighten.
    'Proxima Centauri': 0.055,
}
# Sol outer-system readability floor (figure fraction). Raw world scale alone
# shrinks below floored planet disks at Saturn/Kuiper and the Sun vanishes.
BLENDER_SUN_OUTER_MIN_FRAC = 0.008
BLENDER_SUN_OUTER_FLOOR_HALF_AU = 20.0
# Below this on-screen (floored) fraction, non-Earth/Moon packs fall back to catalog dots.
# Matches the default paint floor so world-fixed disks stay textured through Sol zoom-out.
BLENDER_MIN_BILLBOARD_FRAC = 0.0035
# Soft floors so rings/asteroids stay barely readable without dominating the frame.
BLENDER_RING_LINGER_MIN_FRAC = 0.008
BLENDER_ASTEROID_BELT_MIN_FRAC = 0.004
# Hide Luna until the camera is wide enough that its exaggerated orbit fits.
SOL_MOON_REVEAL_HALF_AU = 0.11
SOL_NEAR_SUN_HALF_AU = 2.4
SOL_INNER_HALF_AU = 6.5
# Linger on the main belt / Jupiter's orbit (same idea as the Kuiper hold).
SOL_BELT_LINGER_HALF_AU = 7.8
# Saturn rings beat — wide enough for the annulus without jumping to Kuiper.
SOL_SATURN_LINGER_HALF_AU = 18.0
# Outer linger frames Neptune/Pluto with Kuiper just coming into view.
SOL_OUTER_LINGER_HALF_AU = 42.0
SOL_HALF_WIDTH_AU = SOL_OUTER_LINGER_HALF_AU
# Leave Earth look-at and ease toward Sol before the Near-Sun plateau.
SOL_LEAVE_EARTH_FOCUS_HALF_AU = 1.15
# Textured star billboards by body (half-width AU window). Outside → scatter marker.
# Sol min is 0: as soon as Sol is on-screen in blender mode we use the photosphere spin.
BLENDER_STAR_BILLBOARD_HALF_AU_BY_BODY = {
    'Sun': (0.0, min(95.0, SOL_OUTER_LINGER_HALF_AU * 2.2)),
    'Alpha Centauri A': (8.0, 140.0),
    'Alpha Centauri B': (8.0, 140.0),
    # Min 0: keep the photosphere through the Proxima-planet finale (0.055 AU).
    # The old 0.08 floor dropped texture and snapped to a tiny scatter marker.
    'Proxima Centauri': (0.0, 80.0),
}
# Back-compat alias used by Sol-draw path + tests (Sol window).
BLENDER_STAR_BILLBOARD_HALF_AU = BLENDER_STAR_BILLBOARD_HALF_AU_BY_BODY['Sun']
# Wide enough to bring most of the 30 ly catalog into the Sol neighborhood frame.
START_HALF_WIDTH_LY = 25.0
AB_HALF_WIDTH_AU = 32.0
WIDE_HALF_WIDTH_AU = 12000.0
# Wide frame shows Proxima c; then ease into the inner planets for the finale.
PROXIMA_WIDE_HALF_AU = 2.0
# Close-up on b (a≈0.049 AU) and d (a≈0.029 AU) — orbit of b fills most of the frame.
PROXIMA_INNER_HALF_AU = 0.055
PROXIMA_HALF_WIDTH_AU = PROXIMA_INNER_HALF_AU
# Keep Sol and α Cen framed during the pan; dive only after focus has settled.
AB_FOCUS_ARRIVE = 0.68
AB_CRUISE_END = 0.58
AB_FRAME_PADDING = 1.25
# Intermediate zoom stops so the final dive is not one huge log jump.
AB_DIVE_WAYPOINTS_AU = (2500.0, 180.0)
# Dive straight from the triple-system wide frame into Proxima (no Sol re-frame zoom-out).
PROXIMA_DIVE_WAYPOINTS_AU = (3000.0, 400.0, 40.0, 8.0)
# Earth → belt/Jupiter linger.
SOL_TO_BELT_WAYPOINTS_AU = (
    0.40,
    0.75,
    1.2,
    1.8,
    SOL_NEAR_SUN_HALF_AU,
    4.0,
    SOL_INNER_HALF_AU,
)
SOL_TO_BELT_WEIGHTS = (1.0, 1.2, 1.4, 1.5, 1.5, 1.7, 1.9, 2.2)
# Belt linger → outer/Kuiper linger.
SOL_BELT_TO_OUTER_WAYPOINTS_AU = (12.0, 22.0, 32.0)
SOL_BELT_TO_OUTER_WEIGHTS = (2.0, 2.4, 2.6, 2.2)
# Pull back: leave Kuiper → linger through Oort scales → neighborhood.
PULLBACK_WAYPOINTS_AU = (
    70.0,
    120.0,
    250.0,
    500.0,
    1000.0,
    2000.0,
    4500.0,
    12000.0,
    40000.0,
    100000.0,
)
PULLBACK_WEIGHTS = (1.4, 2.0, 2.4, 2.8, 3.0, 3.0, 2.6, 2.0, 1.6, 1.3, 1.0)
# Timeline: Earth → belt linger → outer linger → Oort pullback (classic dotted mode).
SOL_EARTH_DWELL_END = 0.03
# Blender open: hold Earth-close for ~2 day/night spins (48 PNG samples/turn),
# ease out to Earth+Moon, then hold for ~1 full lunar orbit before Sol beats (#51).
SOL_EARTH_SPIN_HOLD_END = 0.055
SOL_EARTH_MOON_REVEAL_END = 0.095
SOL_EARTH_BLENDER_DWELL_END = 0.16
# Lunar clock scale at Earth+Moon open (× Sol motion days). Tuned so the
# Earth+Moon plateau covers ~one sidereal month before the Sol pullback.
LUNAR_OPEN_MOTION_SCALE = 0.067
# Blender-only Sol beats (classic mode keeps SOL_BELT_* / SOL_OUTER_* below).
# Shifted later to make room for the longer Earth+Moon hold; belt hold shortened.
SOL_BEAT_NEAR_SUN_ARRIVE = 0.185
SOL_BEAT_NEAR_SUN_HOLD_END = 0.210
SOL_BEAT_INNER_ARRIVE = 0.230
SOL_BEAT_INNER_HOLD_END = 0.250
SOL_BEAT_BELT_ARRIVE = 0.275
SOL_BEAT_BELT_HOLD_END = 0.355
SOL_BEAT_SATURN_ARRIVE = 0.390
SOL_BEAT_SATURN_HOLD_END = 0.425
SOL_BEAT_OUTER_ARRIVE = 0.48
SOL_BELT_ARRIVE = 0.16
SOL_BELT_HOLD_END = 0.28
SOL_OUTER_ARRIVE = 0.40
SOL_HOLD_END = 0.54
PULLBACK_END = 0.70
AB_TRAVEL_END = 0.76
# Linger on the resolved A–B binary before pulling out to the triple view.
AB_HOLD_END = 0.85
# Pull out to the A–B + Proxima triple view, then linger before diving to Proxima.
WIDE_OUT_ARRIVE = 0.89
WIDE_OUT_END = 0.935
PROXIMA_TRAVEL_END = 0.945
# Longer, slower tighten onto b/d so the final zoom does not rush.
PROXIMA_INNER_ARRIVE = 0.97
# Blender-only α Cen arrival beats (#63). Classic dotted mode keeps the constants above.
# AB approach → AB hold → triple wide hold → Proxima dive → Proxima-wide hold → inner.
# Inner b/d hold is intentionally long (~5% of the GIF) so the finale can breathe.
ARRIVAL_AB_TRAVEL_END = 0.76
ARRIVAL_AB_HOLD_END = 0.815
ARRIVAL_WIDE_OUT_ARRIVE = 0.845
ARRIVAL_WIDE_HOLD_END = 0.875
ARRIVAL_PROXIMA_DIVE_END = 0.905
ARRIVAL_PROXIMA_WIDE_HOLD_END = 0.925
ARRIVAL_PROXIMA_INNER_ARRIVE = 0.945
CAMERA_ELEVATION_DEG = 22.0
SOL_ELEVATION_DEG = 28.0
# Top-down open so the Moon orbit fills the viewport (3D foreshortening shrinks it).
EARTH_OPEN_ELEVATION_DEG = 62.0
# Blender Earth-close: slightly flatter so the globe fills the frame cleanly.
EARTH_CLOSE_ELEVATION_DEG = 48.0
PROXIMA_ELEVATION_DEG = 58.0
OUTPUT_DIRECTORY = 'output/animate/sol_centauri'
BLENDER_OUTPUT_DIRECTORY = 'output/animate/sol_centauri/blender'

SOL_PLANET_NAMES = (
    'Mercury',
    'Venus',
    'Earth',
    'Mars',
    'Jupiter',
    'Saturn',
    'Uranus',
    'Neptune',
    'Pluto',
)
LABELED_SOL_PLANETS = ('Earth', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Pluto')
INNER_BELT_VISIBLE_BELOW_AU = 12.0
KUIPER_VISIBLE_BELOW_AU = 90.0
# Fixed world-space Oort (like Kuiper): pan out through it — not locked to the camera frame.
OORT_VISIBLE_ABOVE_AU = 55.0
OORT_VISIBLE_BELOW_AU = 200000.0
OORT_WORLD_INNER_AU = 70.0
OORT_WORLD_OUTER_AU = 100000.0
# Keep Sol readable: only draw populations in a moving annulus (not a filled ball).
POPULATION_ANNULUS_INNER_FRAC = 0.24
POPULATION_ANNULUS_OUTER_FRAC = 0.98
POPULATION_CORE_CLEAR_FRAC = 0.12

STAR_COLORS = {
    'primary': '#F6D56A',
    'secondary': '#E8A05A',
    'proxima': '#E07060',
    'sun': '#F6D56A',
    'traveler': '#7EC8FF',
}

LABELED_STAR_SUBSTRINGS = (
    'Barnard',
    'Sirius',
    'Lalande',
    'Luyten',
    'Ross 154',
    'Epsilon Eridani',
    'Procyon',
    '61 Cygni',
)


def smoothstep(progress: float) -> float:
    clamped = float(np.clip(progress, 0.0, 1.0))
    return clamped * clamped * (3.0 - 2.0 * clamped)


def smootherstep(progress: float) -> float:
    """Ken Perlin's smootherstep — gentler acceleration than smoothstep."""
    clamped = float(np.clip(progress, 0.0, 1.0))
    return clamped**3 * (clamped * (clamped * 6.0 - 15.0) + 10.0)


def timelineProgress(frame: int, animationFrames: int) -> float:
    if animationFrames <= 1:
        return 1.0
    return float(np.clip(frame / (animationFrames - 1), 0.0, 1.0))


def segmentProgress(linear: float, start: float, end: float) -> float:
    if linear <= start:
        return 0.0
    if linear >= end:
        return 1.0
    return smootherstep((linear - start) / (end - start))


def logLerp(startAu: float, endAu: float, progress: float) -> float:
    clamped = float(np.clip(progress, 0.0, 1.0))
    return float(np.exp(np.log(startAu) + clamped * (np.log(endAu) - np.log(startAu))))


def lerpAngleDeg(startDeg: float, endDeg: float, progress: float) -> float:
    """Interpolate degrees along the shortest arc (avoids 180°+ view flips)."""
    delta = ((endDeg - startDeg + 180.0) % 360.0) - 180.0
    return float(startDeg + float(np.clip(progress, 0.0, 1.0)) * delta)


def solAnimationSpeed(halfWidthAu: float) -> float:
    """Keep Sol orbits calm until the outer system, then ramp hard for giant / Pluto motion."""
    if halfWidthAu <= ANIMATION_SPEED_SOL_RAMP_START_AU:
        return ANIMATION_SPEED_SOL_NEAR
    lo = ANIMATION_SPEED_SOL_RAMP_START_AU
    hi = ANIMATION_SPEED_SOL_FAR_HALF_AU
    logT = (np.log(halfWidthAu) - np.log(lo)) / (np.log(hi) - np.log(lo))
    # Power > 1 keeps the ramp late, then climbs quickly toward the outer linger.
    eased = smootherstep(float(np.clip(logT, 0.0, 1.0))) ** 1.65
    return float(
        np.exp(
            np.log(ANIMATION_SPEED_SOL_NEAR)
            + eased * (np.log(ANIMATION_SPEED_SOL_FAR) - np.log(ANIMATION_SPEED_SOL_NEAR))
        )
    )


def innerBeltRenderParams(
    halfWidthAu: float, style: dict
) -> tuple[float, float, float, float] | None:
    """Continuous main-belt / Trojan emphasis from near-Sun through Kuiper fade-out.

    Peaks at the belt linger scale and falls smoothly — no discrete mode jumps.
    Returns (beltSize, beltAlpha, clusterSize, clusterAlpha) or None when hidden.
    """
    innerCutoff = INNER_BELT_VISIBLE_BELOW_AU * 8.0
    # Show as soon as the main-belt annulus (~2.2 AU) enters the camera frame.
    beltEnterAu = 1.9
    if halfWidthAu < beltEnterAu or halfWidthAu > innerCutoff:
        return None

    peakAu = SOL_BELT_LINGER_HALF_AU
    if halfWidthAu <= peakAu:
        rise = smootherstep((halfWidthAu - beltEnterAu) / max(peakAu - beltEnterAu, 1e-6))
        # Keep the linger readable without a solid dust cloud.
        beltBoost = 1.15 + 2.35 * rise
        beltAlphaScale = 0.55 + 0.45 * rise
        clusterBoost = 0.80 + 1.70 * rise
        clusterAlphaScale = 0.45 + 0.40 * rise
    else:
        # Ease down through the outer system — still a faint ring at the Kuiper linger.
        fall = (1.0 - float(np.clip((halfWidthAu - peakAu) / 52.0, 0.0, 1.0))) ** 1.45
        # Matched to the peak values above so the join at peakAu is C0-continuous.
        beltBoost = 0.35 + 3.15 * fall
        beltAlphaScale = 0.18 + 0.82 * fall
        clusterBoost = 0.20 + 2.30 * fall
        clusterAlphaScale = 0.12 + 0.73 * fall

    beltAlpha = min(0.72, style['beltAlpha'] * beltAlphaScale)
    clusterAlpha = min(0.65, style['clusterAlpha'] * clusterAlphaScale)
    if beltAlpha < 0.04 and clusterAlpha < 0.03:
        return None

    floorBlend = float(np.clip((14.0 - halfWidthAu) / 12.0, 0.0, 1.0))
    # Keep a readable residual size at Kuiper scale (tiny floors vanish into the Sun).
    kuiperResidual = float(
        np.clip(1.0 - abs(halfWidthAu - SOL_OUTER_LINGER_HALF_AU) / 22.0, 0.0, 1.0)
    )
    sizeFloor = 0.22 + 0.55 * floorBlend + 0.45 * kuiperResidual
    clusterFloor = 0.18 + 0.45 * floorBlend + 0.30 * kuiperResidual
    return (
        max(style['beltSize'] * beltBoost, sizeFloor),
        beltAlpha,
        max(style['clusterSize'] * clusterBoost, clusterFloor),
        clusterAlpha,
    )


def kuiperRenderParams(halfWidthAu: float, style: dict) -> tuple[float, float] | None:
    """Continuous Kuiper size/alpha — rise into the linger, then ease out toward Oort."""
    if halfWidthAu < 18.0 or halfWidthAu > KUIPER_VISIBLE_BELOW_AU * 1.8:
        return None

    if halfWidthAu < 30.0:
        rise = smootherstep((halfWidthAu - 18.0) / 12.0)
        boost = 3.5 + 3.5 * rise
        alphaScale = 0.85 + 0.50 * rise
    elif halfWidthAu <= 55.0:
        boost = 7.0
        alphaScale = 1.35
    else:
        fall = float(np.clip(1.0 - (halfWidthAu - 55.0) / 50.0, 0.0, 1.0))
        boost = 4.0 + 3.0 * fall
        alphaScale = 0.70 + 0.65 * fall

    alpha = min(0.95, style['kuiperAlpha'] * alphaScale)
    if halfWidthAu > OORT_WORLD_INNER_AU:
        alpha *= max(0.0, 1.0 - (halfWidthAu - OORT_WORLD_INNER_AU) / 80.0)
    if alpha <= 0.03:
        return None
    sizeFloor = 1.2 + 1.3 * float(np.clip((40.0 - abs(halfWidthAu - 42.0)) / 22.0, 0.0, 1.0))
    return max(style['kuiperSize'] * boost, sizeFloor), alpha


def stagedLogDive(
    startAu: float,
    endAu: float,
    waypointsAu: tuple[float, ...],
    progress: float,
    weights: tuple[float, ...] | None = None,
) -> float:
    """Log-zoom through intermediate scales so large dives don't feel like a snap.

    Optional per-segment weights stretch time on chosen scales (e.g. inner system).
    """
    scales = (startAu, *waypointsAu, endAu)
    if progress <= 0.0:
        return float(scales[0])
    if progress >= 1.0:
        return float(scales[-1])
    segmentCount = len(scales) - 1
    if weights is None:
        segmentWeights = np.ones(segmentCount, dtype=float)
    else:
        if len(weights) != segmentCount:
            raise ValueError(f'weights length {len(weights)} != segments {segmentCount}')
        segmentWeights = np.asarray(weights, dtype=float)
    cumulative = np.cumsum(segmentWeights)
    total = float(cumulative[-1])
    target = float(np.clip(progress, 0.0, 1.0)) * total
    segmentIndex = int(np.searchsorted(cumulative, target, side='left'))
    segmentIndex = min(max(segmentIndex, 0), segmentCount - 1)
    startWeight = 0.0 if segmentIndex == 0 else float(cumulative[segmentIndex - 1])
    local = (target - startWeight) / max(float(segmentWeights[segmentIndex]), 1e-12)
    return logLerp(scales[segmentIndex], scales[segmentIndex + 1], local)


def travelProgress(
    frame: int,
    animationFrames: int,
    *,
    abTravelEnd: float = AB_TRAVEL_END,
) -> float:
    """0 while still at Sol → 1 once the camera has arrived at α Cen AB."""
    linear = timelineProgress(frame, animationFrames)
    return segmentProgress(linear, PULLBACK_END, abTravelEnd)


def proximaTravelProgress(
    frame: int,
    animationFrames: int,
    *,
    wideHoldEnd: float = WIDE_OUT_END,
    proximaDiveEnd: float = PROXIMA_TRAVEL_END,
) -> float:
    """0 at triple-wide hold end → 1 once the dive reaches Proxima-wide scale."""
    linear = timelineProgress(frame, animationFrames)
    return segmentProgress(linear, wideHoldEnd, proximaDiveEnd)


def spectralClassColor(stellarClass: object, *, fallback: str) -> str:
    """Map a catalog Stellarclass string to a faint spectral tint (no RGB in the CSV)."""
    text = str(stellarClass or '').strip().upper().replace('−', '-')
    if not text:
        return fallback
    # Skip leading digits / junk; take the first MK letter.
    for character in text:
        if character.isalpha():
            return SPECTRAL_FIELD_COLORS.get(character, fallback)
    return fallback


def parseApparentMagnitude(rawMagnitude: object) -> float | None:
    """Parse catalog V (or J-tagged) magnitude strings like '−1.46' or '10.7 J'."""
    text = str(rawMagnitude or '').replace('−', '-').replace('\xa0', ' ').strip()
    if not text or text.lower() in {'nan', 'none'}:
        return None
    token = text.split()[0]
    try:
        return float(token)
    except ValueError:
        return None


class SolCentauriCinematicAnimator:
    """Flight from our solar system through α Cen AB to Proxima planets."""

    def __init__(
        self,
        system: StarSystem,
        style: str = 'default',
        figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
        dpi: int = DEFAULT_DPI,
        starsCsvPath: str = 'data/nearby_stars_30.csv',
        *,
        useBlenderBodies: bool = False,
    ):
        if system.systemId != 'alpha_centauri':
            raise ValueError(f'Expected alpha_centauri, got {system.systemId!r}')

        self.system = system
        self.figureSizeInches = figureSizeInches
        self.dpi = dpi
        self.animationFrames = ANIMATION_FRAMES
        self.useBlenderBodies = useBlenderBodies
        self.constants = AstronomicalConstants()
        self.orbitCalculator = OrbitCalculator()
        self.planetCatalog = PlanetCatalog(self.constants)
        self.moonCatalog = MoonCatalog()
        self.transform = SolCentauriFrameTransform.fromStarSystem(system)
        self.barycenterSolAu = np.asarray(self.transform.originSolAu, dtype=float)
        self.distanceLy = float(np.linalg.norm(self.barycenterSolAu) / self.constants.lightYearToAu)
        self.primaryOrbit = self._requireOrbit('primary')
        self.secondaryOrbit = self._requireOrbit('secondary')
        self.proximaOrbit = self._requireOrbit('wide_companion')
        self.proximaPlanets = system.planetsForHost(self.proximaOrbit.starUuid)
        self.asteroidPopulation = AnimatedAsteroidPopulation(
            self.constants,
            AsteroidPopulationCounts(
                asteroidBelt=480,
                hildas=140,
                trojansAndGreeks=90,
                kuiperBelt=900,
                oortCloud=4500,
            ),
            includeKuiperAndOort=True,
            useSphericalShell3d=True,
        )
        self.famousAsteroidCatalog = FamousAsteroidCatalog()
        self._fixedOortX: np.ndarray | None = None
        self._fixedOortY: np.ndarray | None = None
        self._fixedOortZ: np.ndarray | None = None

        plt.style.use(style)
        self.isDark = style == 'dark_background'
        self.labelColor = '#F0F0F0' if self.isDark else '#202020'
        self.fieldStarColor = '#E8E8E8' if self.isDark else '#505050'
        self.pathColor = '#9EC9FF' if self.isDark else '#2F6FBF'
        self.hudColor = '#D8EEFF' if self.isDark else '#103050'
        self.orbitColor = '#B0B0B0' if self.isDark else '#606060'
        self.renderStyle = ASTEROID_RENDER_STYLES['dark' if self.isDark else 'light']

        self.solEarthHalfWidthAu = SOL_EARTH_HALF_AU
        self.solHalfWidthAu = SOL_HALF_WIDTH_AU
        self.startHalfWidthAu = START_HALF_WIDTH_LY * self.constants.lightYearToAu
        self.abHalfWidthAu = AB_HALF_WIDTH_AU
        self.wideHalfWidthAu = WIDE_HALF_WIDTH_AU
        self.proximaWideHalfWidthAu = PROXIMA_WIDE_HALF_AU
        self.proximaInnerHalfWidthAu = PROXIMA_INNER_HALF_AU
        self.proximaHalfWidthAu = PROXIMA_INNER_HALF_AU
        pathAzimuth = float(
            np.degrees(np.arctan2(self.barycenterSolAu[1], self.barycenterSolAu[0]))
        )
        self.travelAzimuthDeg = pathAzimuth + 55.0
        self.solAzimuthDeg = pathAzimuth + 35.0
        # Keep AB/Proxima on the travel camera — orbital-plane "arrival" spins caused flippy orbits.

        self.fieldStars = self._loadFieldStars(starsCsvPath)
        self.primaryOrbitPathSol = self._orbitPathSol(self.primaryOrbit)
        self.secondaryOrbitPathSol = self._orbitPathSol(self.secondaryOrbit)
        self.proximaOrbitPathSol = self._orbitPathSol(self.proximaOrbit)
        self.solPlanetPaths = {
            name: self._planetOrbitPath(self.planetCatalog.planets[name])
            for name in SOL_PLANET_NAMES
        }
        self.proximaPlanetPathsLocal = {
            planet.planetId: orbitPathInOrbitalPlane(
                self.orbitCalculator,
                planet.semiMajorAxisAu,
                planet.eccentricity,
                planet.argumentPeriapsisDeg,
            )
            for planet in self.proximaPlanets
        }

        self._viewFocus = np.zeros(3)
        self._viewHalfWidthAu = self.solEarthHalfWidthAu
        self.figure = plt.figure(figsize=figureSizeInches, dpi=dpi, layout='none')
        self.axes = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), projection='3d')
        # 2D overlay for Blender spin-loop frames (drawn after the 3D camera is set).
        self.bodyOverlay = self.figure.add_axes((0.0, 0.0, 1.0, 1.0), facecolor='none', zorder=20)
        self.bodyOverlay.set_axis_off()
        self.bodyOverlay.patch.set_alpha(0.0)
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)
        # (name, center, frame, halfWidth, openCloseup, bodyScale, orbitalPhaseRad|None)
        self._pendingBlenderBodies: list[
            tuple[str, np.ndarray, int, float, bool, float, float | None]
        ] = []
        # (name, center, fontsize, bodyScale)
        self._pendingBlenderLabels: list[tuple[str, np.ndarray, float, float]] = []
        # Billboard paint zorder by (name, x, y, z) after depth sort in overlay flush.
        self._blenderBodyPaintZorder: dict[tuple[str, float, float, float], int] = {}
        self.blenderSprites: BlenderBodySpriteAtlas | None = None
        if self.useBlenderBodies:
            themeName = 'dark' if self.isDark else 'light'
            self.blenderSprites = BlenderBodySpriteAtlas(themeName)
            print(
                'Blender spin loops: lazy-load by zoom stage '
                f'(theme={themeName}; Earth/Moon probed: '
                f'Earth={"on" if self.blenderSprites.hasEarth else "missing"}, '
                f'Moon={"on" if self.blenderSprites.hasMoon else "missing"})'
            )

    def _requireOrbit(self, role: str) -> StellarOrbit:
        for orbit in self.system.stellarOrbits:
            if orbit.role == role:
                return orbit
        raise KeyError(f'No stellar orbit with role={role!r}')

    def _loadFieldStars(self, starsCsvPath: str) -> pd.DataFrame:
        catalog = SystemCatalog(starsCsvPath=starsCsvPath).starCatalog
        stars = catalog.starsWithinLightYears(FIELD_STARS_MAX_LY).copy()
        stars = stars[~stars['System'].astype(str).str.contains('Solar System', na=False)]
        if 'system_id' in stars.columns:
            stars = stars[stars['system_id'] != 'alpha_centauri']
        stars = stars.dropna(subset=['positionX', 'positionY', 'positionZ']).copy()
        if stars.empty:
            return stars

        fallback = self.fieldStarColor
        colors = [
            spectralClassColor(value, fallback=fallback)
            for value in stars.get('Stellarclass', pd.Series(dtype=str))
        ]
        magnitudes = [
            parseApparentMagnitude(value)
            for value in stars.get('Apparent magnitude (v)', pd.Series(dtype=str))
        ]
        # Brighter stars larger/more opaque; faint catalog objects stay smaller.
        sizes: list[float] = []
        alphas: list[float] = []
        for magnitude in magnitudes:
            if magnitude is None:
                sizes.append(14.0)
                alphas.append(0.55)
                continue
            sizes.append(float(np.clip(34.0 - 1.15 * magnitude, 5.0, 48.0)))
            alphas.append(float(np.clip(0.85 - 0.022 * magnitude, 0.22, 0.95)))
        stars['fieldColor'] = colors
        stars['fieldSize'] = sizes
        stars['fieldAlpha'] = alphas
        return stars

    def _planetOrbitPath(self, planet: PlanetOrbit) -> np.ndarray:
        trueAnomaly = np.linspace(0.0, 2.0 * np.pi, 180)
        positionX, positionY, positionZ = self.orbitCalculator.ellipticalPosition(
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.inclinationDeg,
            trueAnomaly,
        )
        return np.column_stack(
            (np.asarray(positionX), np.asarray(positionY), np.asarray(positionZ))
        )

    def _orbitPathSol(self, orbit: StellarOrbit) -> np.ndarray:
        pathX, pathY = orbitPathInOrbitalPlane(
            self.orbitCalculator,
            orbit.semiMajorAxisAu,
            orbit.eccentricity,
            orbit.argumentPeriapsisDeg,
        )
        return self.transform.toSol(np.column_stack((pathX, pathY)))

    def _starPositionSol(self, orbit: StellarOrbit, frame: int, speed: float) -> np.ndarray:
        positionX, positionY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            orbit.semiMajorAxisAu,
            orbit.eccentricity,
            orbit.periodDays,
            orbit.argumentPeriapsisDeg,
            orbit.meanAnomalyDegEpoch,
            frame,
            speed,
        )
        return self.transform.toSol([positionX, positionY])[0]

    def _proximaPositionSol(self, frame: int) -> np.ndarray:
        return self._starPositionSol(self.proximaOrbit, frame, ANIMATION_SPEED_PROXIMA_STAR)

    def _proximaPlanetPositionSol(self, planet: SystemPlanet, frame: int) -> np.ndarray:
        proximaLocalX, proximaLocalY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            self.proximaOrbit.semiMajorAxisAu,
            self.proximaOrbit.eccentricity,
            self.proximaOrbit.periodDays,
            self.proximaOrbit.argumentPeriapsisDeg,
            self.proximaOrbit.meanAnomalyDegEpoch,
            frame,
            ANIMATION_SPEED_PROXIMA_STAR,
        )
        offsetX, offsetY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.orbitalPeriodDays,
            planet.argumentPeriapsisDeg,
            0.0,
            frame,
            ANIMATION_SPEED_PROXIMA_PLANETS,
        )
        return self.transform.toSol([proximaLocalX + offsetX, proximaLocalY + offsetY])[0]

    def _framedHalfWidthAu(self, focus: np.ndarray, anchor: np.ndarray) -> float:
        """Half-width that keeps ``focus``'s origin landmark and ``anchor`` on screen."""
        distOrigin = float(np.linalg.norm(focus))
        distAnchor = float(np.linalg.norm(focus - anchor))
        floor = 0.35 * float(np.linalg.norm(anchor))
        return AB_FRAME_PADDING * max(distOrigin, distAnchor, floor)

    def _abFocusProgress(self, abProgress: float) -> float:
        """Settle the look-at point on AB before the final zoom dive finishes."""
        return smootherstep(min(1.0, abProgress / AB_FOCUS_ARRIVE))

    def _abLegHalfWidthAu(self, focus: np.ndarray, abProgress: float) -> float:
        """Cruise with Sol+α Cen framed; staged dive into the binary afterward."""
        cruiseHalf = self._framedHalfWidthAu(focus, self.barycenterSolAu)
        if abProgress <= AB_CRUISE_END:
            blend = abProgress / AB_CRUISE_END if AB_CRUISE_END > 0 else 1.0
            return logLerp(self.startHalfWidthAu, cruiseHalf, smootherstep(blend))

        focusAtCruiseEnd = self._abFocusProgress(AB_CRUISE_END) * self.barycenterSolAu
        cruiseEndHalf = self._framedHalfWidthAu(focusAtCruiseEnd, self.barycenterSolAu)
        dive = segmentProgress(abProgress, AB_CRUISE_END, 1.0)
        return stagedLogDive(cruiseEndHalf, self.abHalfWidthAu, AB_DIVE_WAYPOINTS_AU, dive)

    def _earthDwellEnd(self) -> float:
        """Timeline fraction when the Earth(-Moon) open ends and the Sol dive begins."""
        return SOL_EARTH_BLENDER_DWELL_END if self.useBlenderBodies else SOL_EARTH_DWELL_END

    def _abTravelEnd(self) -> float:
        return ARRIVAL_AB_TRAVEL_END if self.useBlenderBodies else AB_TRAVEL_END

    def _abHoldEnd(self) -> float:
        return ARRIVAL_AB_HOLD_END if self.useBlenderBodies else AB_HOLD_END

    def _wideOutArrive(self) -> float:
        return ARRIVAL_WIDE_OUT_ARRIVE if self.useBlenderBodies else WIDE_OUT_ARRIVE

    def _wideHoldEnd(self) -> float:
        return ARRIVAL_WIDE_HOLD_END if self.useBlenderBodies else WIDE_OUT_END

    def _proximaDiveEnd(self) -> float:
        return ARRIVAL_PROXIMA_DIVE_END if self.useBlenderBodies else PROXIMA_TRAVEL_END

    def _proximaWideHoldEnd(self) -> float:
        """Blender: linger at Proxima-wide before inner tighten. Classic: no extra hold."""
        return ARRIVAL_PROXIMA_WIDE_HOLD_END if self.useBlenderBodies else PROXIMA_TRAVEL_END

    def _proximaInnerArrive(self) -> float:
        return ARRIVAL_PROXIMA_INNER_ARRIVE if self.useBlenderBodies else PROXIMA_INNER_ARRIVE

    def _travelProgress(self, frame: int) -> float:
        return travelProgress(frame, self.animationFrames, abTravelEnd=self._abTravelEnd())

    def _proximaTravelProgress(self, frame: int) -> float:
        return proximaTravelProgress(
            frame,
            self.animationFrames,
            wideHoldEnd=self._wideHoldEnd(),
            proximaDiveEnd=self._proximaDiveEnd(),
        )

    def _solOpeningHalfWidthAu(self, linear: float) -> float:
        """Earth → belt/Jupiter linger → outer/Kuiper linger."""
        dwellEnd = self._earthDwellEnd()
        if self.useBlenderBodies and linear < dwellEnd:
            # Earth-only day/night hold → reveal Luna → short Earth+Moon beat.
            if linear < SOL_EARTH_SPIN_HOLD_END:
                return SOL_EARTH_CLOSE_HALF_AU
            if linear < SOL_EARTH_MOON_REVEAL_END:
                reveal = segmentProgress(linear, SOL_EARTH_SPIN_HOLD_END, SOL_EARTH_MOON_REVEAL_END)
                return logLerp(
                    SOL_EARTH_CLOSE_HALF_AU,
                    self.solEarthHalfWidthAu,
                    smootherstep(reveal),
                )
            return self.solEarthHalfWidthAu
        if self.useBlenderBodies:
            return self._blenderSolOpeningHalfWidthAu(linear, dwellEnd)
        if linear < SOL_BELT_ARRIVE:
            dive = segmentProgress(linear, dwellEnd, SOL_BELT_ARRIVE)
            if dive >= 1.0 - 1e-9:
                return SOL_BELT_LINGER_HALF_AU
            return stagedLogDive(
                self.solEarthHalfWidthAu,
                SOL_BELT_LINGER_HALF_AU,
                SOL_TO_BELT_WAYPOINTS_AU,
                dive,
                weights=SOL_TO_BELT_WEIGHTS,
            )
        if linear < SOL_BELT_HOLD_END:
            return SOL_BELT_LINGER_HALF_AU
        if linear < SOL_OUTER_ARRIVE:
            dive = segmentProgress(linear, SOL_BELT_HOLD_END, SOL_OUTER_ARRIVE)
            if dive >= 1.0 - 1e-9:
                return self.solHalfWidthAu
            return stagedLogDive(
                SOL_BELT_LINGER_HALF_AU,
                self.solHalfWidthAu,
                SOL_BELT_TO_OUTER_WAYPOINTS_AU,
                dive,
                weights=SOL_BELT_TO_OUTER_WEIGHTS,
            )
        return self.solHalfWidthAu

    def _blenderSolOpeningHalfWidthAu(self, linear: float, dwellEnd: float) -> float:
        """Staged Sol zoom-out beats (#51): Near-Sun → inner → belt → Saturn → Kuiper."""
        if linear < SOL_BEAT_NEAR_SUN_ARRIVE:
            dive = segmentProgress(linear, dwellEnd, SOL_BEAT_NEAR_SUN_ARRIVE)
            return stagedLogDive(
                self.solEarthHalfWidthAu,
                SOL_NEAR_SUN_HALF_AU,
                (0.40, 0.75, 1.2, 1.8),
                dive,
                weights=(1.0, 1.2, 1.4, 1.6, 1.8),
            )
        if linear < SOL_BEAT_NEAR_SUN_HOLD_END:
            return SOL_NEAR_SUN_HALF_AU
        if linear < SOL_BEAT_INNER_ARRIVE:
            dive = segmentProgress(linear, SOL_BEAT_NEAR_SUN_HOLD_END, SOL_BEAT_INNER_ARRIVE)
            return logLerp(SOL_NEAR_SUN_HALF_AU, SOL_INNER_HALF_AU, smootherstep(dive))
        if linear < SOL_BEAT_INNER_HOLD_END:
            return SOL_INNER_HALF_AU
        if linear < SOL_BEAT_BELT_ARRIVE:
            dive = segmentProgress(linear, SOL_BEAT_INNER_HOLD_END, SOL_BEAT_BELT_ARRIVE)
            return logLerp(SOL_INNER_HALF_AU, SOL_BELT_LINGER_HALF_AU, smootherstep(dive))
        if linear < SOL_BEAT_BELT_HOLD_END:
            return SOL_BELT_LINGER_HALF_AU
        if linear < SOL_BEAT_SATURN_ARRIVE:
            dive = segmentProgress(linear, SOL_BEAT_BELT_HOLD_END, SOL_BEAT_SATURN_ARRIVE)
            return logLerp(SOL_BELT_LINGER_HALF_AU, SOL_SATURN_LINGER_HALF_AU, smootherstep(dive))
        if linear < SOL_BEAT_SATURN_HOLD_END:
            return SOL_SATURN_LINGER_HALF_AU
        if linear < SOL_BEAT_OUTER_ARRIVE:
            dive = segmentProgress(linear, SOL_BEAT_SATURN_HOLD_END, SOL_BEAT_OUTER_ARRIVE)
            return logLerp(SOL_SATURN_LINGER_HALF_AU, self.solHalfWidthAu, smootherstep(dive))
        return self.solHalfWidthAu

    def _pullbackHalfWidthAu(self, linear: float) -> float:
        pull = segmentProgress(linear, SOL_HOLD_END, PULLBACK_END)
        # Linger through early Oort scales before racing to the neighborhood.
        pull = pull**1.55
        return stagedLogDive(
            self.solHalfWidthAu,
            self.startHalfWidthAu,
            PULLBACK_WAYPOINTS_AU,
            pull,
            weights=PULLBACK_WEIGHTS,
        )

    def _scaleHalfWidthAu(self, frame: int) -> float:
        """Camera scale for Sol motion timing — no body positions (avoids feedback loops)."""
        linear = timelineProgress(frame, self.animationFrames)
        if linear <= SOL_HOLD_END:
            return self._solOpeningHalfWidthAu(linear)
        if linear <= PULLBACK_END:
            return self._pullbackHalfWidthAu(linear)
        return self.startHalfWidthAu

    def _ensureSolMotionClock(self) -> None:
        """Accumulate zoom-dependent Sol animation days so speed changes don't jump orbits."""
        if getattr(self, '_solMotionDaysByFrame', None) is not None:
            return
        days = np.zeros(self.animationFrames, dtype=float)
        accumulated = 0.0
        for frame in range(self.animationFrames):
            accumulated += solAnimationSpeed(self._scaleHalfWidthAu(frame))
            days[frame] = accumulated
        self._solMotionDaysByFrame = days

    def _solMotionDays(self, frame: int) -> float:
        self._ensureSolMotionClock()
        index = int(np.clip(frame, 0, self.animationFrames - 1))
        return float(self._solMotionDaysByFrame[index])

    def _solSpeedEquivalent(self, frame: int) -> float:
        """Equivalent constant speed for APIs that multiply by frame."""
        if frame <= 0:
            return solAnimationSpeed(self._scaleHalfWidthAu(0))
        return self._solMotionDays(frame) / float(frame)

    def _planetPositionAu(self, planetName: str, frame: int) -> np.ndarray:
        planet = self.planetCatalog.planets[planetName]
        meanAnomaly = planetMeanAnomalyRad(planet.orbitalPeriodDays, 1, self._solMotionDays(frame))
        positionX, positionY, positionZ = self.orbitCalculator.ellipticalPosition(
            planet.semiMajorAxisAu,
            planet.eccentricity,
            planet.inclinationDeg,
            meanAnomaly,
        )
        return np.array([float(positionX), float(positionY), float(positionZ)], dtype=float)

    def _solOpeningCameraState(self, frame: int, linear: float) -> tuple[np.ndarray, float]:
        """Earth+Moon dwell → near-Sun → inner system → full Sol belts."""
        halfWidthAu = self._solOpeningHalfWidthAu(linear)
        earth = self._planetPositionAu('Earth', frame)
        if self.useBlenderBodies:
            # Earth/Moon open stays on Earth; then ease to Sol for the Near-Sun beat+.
            if halfWidthAu <= SOL_LEAVE_EARTH_FOCUS_HALF_AU:
                focus = earth
            elif halfWidthAu >= SOL_NEAR_SUN_HALF_AU:
                focus = np.zeros(3)
            else:
                blend = (halfWidthAu - SOL_LEAVE_EARTH_FOCUS_HALF_AU) / (
                    SOL_NEAR_SUN_HALF_AU - SOL_LEAVE_EARTH_FOCUS_HALF_AU
                )
                focus = (1.0 - smootherstep(blend)) * earth
            return focus, halfWidthAu
        # Classic: keep Earth framed until the Sun fits, then ease look-at back to Sol.
        if halfWidthAu <= SOL_NEAR_SUN_HALF_AU:
            focus = earth
        elif halfWidthAu >= SOL_INNER_HALF_AU:
            focus = np.zeros(3)
        else:
            blend = (halfWidthAu - SOL_NEAR_SUN_HALF_AU) / (
                SOL_INNER_HALF_AU - SOL_NEAR_SUN_HALF_AU
            )
            focus = (1.0 - smootherstep(blend)) * earth
        return focus, halfWidthAu

    def _cameraState(self, frame: int) -> tuple[np.ndarray, float]:
        linear = timelineProgress(frame, self.animationFrames)
        abProgress = self._travelProgress(frame)
        proximaProgress = self._proximaTravelProgress(frame)
        proximaSol = self._proximaPositionSol(frame)
        abHoldEnd = self._abHoldEnd()
        wideOutArrive = self._wideOutArrive()
        wideHoldEnd = self._wideHoldEnd()
        proximaDiveEnd = self._proximaDiveEnd()
        proximaWideHoldEnd = self._proximaWideHoldEnd()
        proximaInnerArrive = self._proximaInnerArrive()

        if linear <= SOL_HOLD_END:
            return self._solOpeningCameraState(frame, linear)

        if linear <= PULLBACK_END:
            return np.zeros(3), self._pullbackHalfWidthAu(linear)

        if linear <= abHoldEnd:
            focus = self._abFocusProgress(abProgress) * self.barycenterSolAu
            return focus, self._abLegHalfWidthAu(focus, abProgress)

        if linear <= wideOutArrive:
            pull = segmentProgress(linear, abHoldEnd, wideOutArrive)
            return self.barycenterSolAu.copy(), stagedLogDive(
                self.abHalfWidthAu, self.wideHalfWidthAu, (400.0, 3000.0), pull
            )

        if linear <= wideHoldEnd:
            # Hold on the three-star view: A–B with Proxima on its wide orbit.
            return self.barycenterSolAu.copy(), self.wideHalfWidthAu

        # After the triple pause, dive straight onto Proxima (not the AB–Proxima midpoint).
        if linear <= proximaDiveEnd:
            return proximaSol.copy(), stagedLogDive(
                self.wideHalfWidthAu,
                self.proximaWideHalfWidthAu,
                PROXIMA_DIVE_WAYPOINTS_AU,
                proximaProgress,
            )

        if linear <= proximaWideHoldEnd:
            # Blender arrival beat: linger at Proxima-wide before the b/d close-up.
            return proximaSol.copy(), self.proximaWideHalfWidthAu

        if linear <= proximaInnerArrive:
            tighten = segmentProgress(linear, proximaWideHoldEnd, proximaInnerArrive)
            # Bias time toward the start of the dive so the last stretch eases in.
            tighten = tighten**1.35
            return proximaSol.copy(), logLerp(
                self.proximaWideHalfWidthAu, self.proximaInnerHalfWidthAu, tighten
            )

        return proximaSol.copy(), self.proximaInnerHalfWidthAu

    def update(self, frame: int):
        self.axes.clear()
        self.bodyOverlay.clear()
        self.bodyOverlay.set_axis_off()
        self.bodyOverlay.patch.set_alpha(0.0)
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)
        self._pendingBlenderBodies = []
        self._pendingBlenderLabels = []
        self._blenderBodyPaintZorder = {}
        for textArtist in list(self.figure.texts):
            textArtist.remove()
        focus, halfWidthAu = self._cameraState(frame)
        self._viewFocus = focus
        self._viewHalfWidthAu = halfWidthAu
        abProgress = self._travelProgress(frame)
        proximaProgress = self._proximaTravelProgress(frame)
        linear = timelineProgress(frame, self.animationFrames)

        self._drawPath(frame, focus, abProgress, proximaProgress, halfWidthAu, linear)
        self._drawSolarSystem(frame, halfWidthAu, abProgress)
        self._drawFieldStars(halfWidthAu)
        self._drawAlphaCentauri(frame, halfWidthAu, abProgress, linear)
        self._drawProxima(frame, halfWidthAu, linear)
        self._applyAxes(focus, halfWidthAu, abProgress, proximaProgress, linear)
        # Globes need a finished 3D projection — paint them into this frame now.
        self._flushBlenderBodyOverlays(halfWidthAu)
        self._flushBlenderBodyLabels(halfWidthAu)
        return []

    def _inView(self, position: np.ndarray, margin: float = 0.95) -> bool:
        focus = getattr(self, '_viewFocus', np.zeros(3))
        halfWidthAu = getattr(self, '_viewHalfWidthAu', np.inf)
        return bool(
            np.all(np.abs(np.asarray(position, dtype=float) - focus) <= halfWidthAu * margin)
        )

    def _label3d(
        self,
        position: np.ndarray,
        text: str,
        *,
        color: str,
        fontsize: float = 8,
        alpha: float = 1.0,
        clipOn: bool = True,
    ) -> None:
        if not self._inView(position):
            return
        self.axes.text(
            float(position[0]),
            float(position[1]),
            float(position[2]),
            text,
            color=color,
            fontsize=fontsize,
            alpha=alpha,
            clip_on=clipOn,
        )

    def _drawPath(
        self,
        frame: int,
        focus: np.ndarray,
        abProgress: float,
        proximaProgress: float,
        halfWidthAu: float,
        linear: float,
    ) -> None:
        if halfWidthAu < 200.0 and abProgress < 0.05 and linear < self._abHoldEnd():
            return

        if linear < self._wideHoldEnd():
            path = np.vstack((np.zeros(3), self.barycenterSolAu))
            self.axes.plot(
                path[:, 0], path[:, 1], path[:, 2], color=self.pathColor, linewidth=1.2, alpha=0.4
            )
            # Only show the traveler while cruising — not during the tight AB dive.
            if 0.0 < abProgress < AB_CRUISE_END:
                self._drawTraveler(focus)
            return

        # Hide the AB→Proxima path once we're in the Proxima system itself.
        if halfWidthAu < 40.0:
            return
        proximaSol = self._proximaPositionSol(frame)
        path = np.vstack((self.barycenterSolAu, proximaSol))
        self.axes.plot(
            path[:, 0], path[:, 1], path[:, 2], color=self.pathColor, linewidth=1.0, alpha=0.45
        )
        if 0.0 < proximaProgress < AB_CRUISE_END:
            self._drawTraveler(focus)

    def _drawTraveler(self, traveler: np.ndarray) -> None:
        if not self._inView(traveler):
            return
        self.axes.scatter(
            [traveler[0]],
            [traveler[1]],
            [traveler[2]],
            color=STAR_COLORS['traveler'],
            s=55,
            depthshade=False,
            zorder=6,
            marker='D',
        )
        self._label3d(traveler, '  camera', color=STAR_COLORS['traveler'], fontsize=7, alpha=0.9)

    def _drawStarMarker(
        self,
        position: np.ndarray,
        color: str,
        size: float,
        *,
        glow: bool = True,
        zorder: int = 5,
    ) -> None:
        """Core star marker with a soft multi-layer halo."""
        if glow and size >= 20.0:
            for scale, alpha in ((3.6, 0.10), (2.2, 0.20), (1.45, 0.35)):
                self.axes.scatter(
                    [position[0]],
                    [position[1]],
                    [position[2]],
                    color=color,
                    s=size * scale,
                    alpha=alpha,
                    depthshade=False,
                    zorder=max(1, zorder - 1),
                )
        self.axes.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=color,
            s=size,
            depthshade=True,
            zorder=zorder,
        )

    def _drawSolarSystem(self, frame: int, halfWidthAu: float, abProgress: float) -> None:
        if abProgress > 0.85 and halfWidthAu < 80.0:
            return

        sunPosition = np.zeros(3)
        sunSize = 340.0 if halfWidthAu < 3.0 else (270.0 if halfWidthAu < 80.0 else 150.0)
        sunInView = self._inView(sunPosition, margin=1.05)
        wantSun = sunInView or (SOL_NEAR_SUN_HALF_AU <= halfWidthAu < 5000.0)
        starBillboardMin, starBillboardMax = BLENDER_STAR_BILLBOARD_HALF_AU
        queuedSun = False
        # Blender mode: textured photosphere for every on-screen Sol during the Sol tour.
        # Do not draw a scatter marker first — that reads as a non→textured pop.
        if (
            wantSun
            and self.useBlenderBodies
            and starBillboardMin <= halfWidthAu <= starBillboardMax
        ):
            queuedSun = self._queueBlenderBody(
                'Sun',
                sunPosition,
                frame,
                halfWidthAu,
                openCloseup=True,
                bodyScale=BLENDER_STAR_BODY_SCALE['Sun'],
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
        if wantSun:
            if not queuedSun:
                self._drawStarMarker(
                    sunPosition,
                    STAR_COLORS['sun'],
                    sunSize,
                    zorder=self._scatterDepthZorder(sunPosition, base=5),
                )
            if sunInView and halfWidthAu < 120.0:
                if queuedSun:
                    self._pendingBlenderLabels.append(
                        ('Sun', sunPosition.copy(), 10.0, BLENDER_STAR_BODY_SCALE['Sun'])
                    )
                else:
                    self._label3d(sunPosition, '  Sun', color=self.labelColor, fontsize=10)
            elif sunInView and halfWidthAu > 150.0:
                self._label3d(sunPosition, '  Sol (Sun)', color=self.labelColor, fontsize=9)

        if halfWidthAu <= 140.0:
            self._drawSolPlanets(frame, halfWidthAu)
            self._drawSolMoons(frame, halfWidthAu)
            # Main belt / Hildas / Trojans as soon as their orbits fit in frame (~2 AU).
            if halfWidthAu >= 1.9:
                self._drawSolAsteroidPopulations(frame, halfWidthAu)
        elif halfWidthAu <= OORT_VISIBLE_BELOW_AU:
            self._drawSolAsteroidPopulations(frame, halfWidthAu)

    def _drawSolPlanets(self, frame: int, halfWidthAu: float) -> None:
        for name in SOL_PLANET_NAMES:
            self._drawOneSolPlanet(name, frame, halfWidthAu)

    def _solPlanetMarkerSize(self, name: str, halfWidthAu: float) -> float:
        if name == 'Earth' and halfWidthAu < SOL_NEAR_SUN_HALF_AU:
            return 110.0 if halfWidthAu <= SOL_EARTH_HALF_AU * 1.5 else 70.0
        if name == 'Pluto' and halfWidthAu >= 20.0:
            return 52.0
        if name == 'Jupiter':
            return 55.0 if 4.5 <= halfWidthAu <= 12.0 else 36.0
        if name in LABELED_SOL_PLANETS:
            return 22.0
        return 12.0

    def _drawOneSolPlanet(self, name: str, frame: int, halfWidthAu: float) -> None:
        planet = self.planetCatalog.planets[name]
        position = self._planetPositionAu(name, frame)
        earthOnly = halfWidthAu <= SOL_EARTH_HALF_AU * 2.2
        if earthOnly and name != 'Earth':
            return
        if not earthOnly:
            path = self.solPlanetPaths[name]
            plutoOuter = name == 'Pluto' and halfWidthAu >= 20.0
            self.axes.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=planet.color if plutoOuter else self.orbitColor,
                linewidth=1.6 if plutoOuter else 0.7,
                alpha=0.85 if plutoOuter else 0.45,
            )
        if name != 'Earth' and not self._inView(position, margin=1.2):
            return
        earthOpen = halfWidthAu <= SOL_EARTH_HALF_AU * 1.5
        plutoOuter = name == 'Pluto' and halfWidthAu >= 20.0
        bodyScale = self._blenderPlanetBodyScale(name)
        saturnHero = (
            self.useBlenderBodies
            and name == 'Saturn'
            and SOL_SATURN_LINGER_HALF_AU * 0.9 <= halfWidthAu <= SOL_SATURN_LINGER_HALF_AU * 1.15
        )
        jupiterHero = (
            self.useBlenderBodies
            and name == 'Jupiter'
            and SOL_BELT_LINGER_HALF_AU * 0.85 <= halfWidthAu <= SOL_BELT_LINGER_HALF_AU * 1.25
        )
        queuedBlender = self._queueBlenderBody(
            name,
            position,
            frame,
            halfWidthAu,
            openCloseup=(earthOpen and name == 'Earth') or saturnHero or jupiterHero,
            bodyScale=bodyScale,
            orbitalPhaseRad=None,
            suppressDotFallback=name == 'Earth' or saturnHero,
        )
        # Earth: never fall back to the catalog-blue scatter when a spin pack exists.
        # Other planets: dots when the pack is missing or the disk is too tiny.
        if not queuedBlender and not (name == 'Earth' and self._blenderBodyAvailable(name)):
            self.axes.scatter(
                [position[0]],
                [position[1]],
                [position[2]],
                color=planet.color,
                s=self._solPlanetMarkerSize(name, halfWidthAu),
                depthshade=True,
                zorder=self._scatterDepthZorder(position, base=6 if plutoOuter else 4),
            )
        if name in LABELED_SOL_PLANETS or (
            halfWidthAu < SOL_INNER_HALF_AU and name in ('Mercury', 'Venus', 'Mars')
        ):
            fontsize = 12 if earthOpen and name == 'Earth' else (10 if plutoOuter else 8)
            if queuedBlender:
                # Overlay text above the spin billboard — axes labels sit under the disk.
                self._pendingBlenderLabels.append(
                    (name, position.copy(), float(fontsize), bodyScale)
                )
            else:
                self._label3d(
                    position,
                    f'  {name}',
                    color=self.labelColor,
                    fontsize=fontsize,
                )

    def _drawSolMoons(self, frame: int, halfWidthAu: float) -> None:
        moonScale = self.moonCatalog.displayScaleForCameraAu(halfWidthAu)
        if moonScale <= 0.0:
            return
        parentNames = ('Earth',) if halfWidthAu < 2.0 else ('Earth', 'Mars', 'Jupiter', 'Saturn')
        for parentName in parentNames:
            planetPosition = self._planetPositionAu(parentName, frame)
            for moon in self.moonCatalog.forPlanet(parentName):
                self._drawOneSolMoon(moon, planetPosition, frame, moonScale, halfWidthAu)

    def _drawOneSolMoon(
        self,
        moon,
        planetPosition: np.ndarray,
        frame: int,
        moonScale: float,
        halfWidthAu: float,
    ) -> None:
        # Earth-only open: keep Luna off-screen until the reveal zoom.
        if moon.name == 'Moon' and self.useBlenderBodies and halfWidthAu < SOL_MOON_REVEAL_HALF_AU:
            return
        orbitRadiusAu = self.moonCatalog.displayOrbitRadiusAu(moon, moonScale)
        ring = np.linspace(0.0, 2.0 * np.pi, 64)
        moonOpen = moon.name == 'Moon' and halfWidthAu <= SOL_EARTH_HALF_AU * 2.2
        self.axes.plot(
            planetPosition[0] + orbitRadiusAu * np.cos(ring),
            planetPosition[1] + orbitRadiusAu * np.sin(ring),
            np.full_like(ring, planetPosition[2]),
            color=moon.color,
            linewidth=1.6 if moonOpen else (1.0 if moon.name == 'Moon' else 0.6),
            alpha=0.75 if moonOpen else (0.55 if moon.name == 'Moon' else 0.3),
        )
        moonMeanAnomaly = planetMeanAnomalyRad(
            moon.orbitalPeriodDays, 1, self._lunarMotionDays(moon, frame, halfWidthAu)
        )
        moonX, moonY, moonZ = self.moonCatalog.heliocentricPosition(
            planetPosition[0],
            planetPosition[1],
            planetPosition[2],
            moon,
            moonMeanAnomaly,
            moonScale,
        )
        moonPosition = np.array([float(moonX), float(moonY), float(moonZ)], dtype=float)
        bodyScale = BLENDER_MOON_BODY_SCALE.get(moon.name, 0.20)
        # Major moons (not Luna): only billboard when the parent system is framed tightly.
        moonTightEnough = moon.name == 'Moon' or halfWidthAu <= (
            5.5 if moon.parentPlanet in ('Earth', 'Mars') else 11.0
        )
        queuedBlenderMoon = False
        if moonTightEnough:
            queuedBlenderMoon = self._queueBlenderBody(
                moon.name,
                moonPosition,
                frame,
                halfWidthAu,
                openCloseup=moonOpen,
                bodyScale=bodyScale,
                orbitalPhaseRad=float(moonMeanAnomaly),
                suppressDotFallback=moon.name == 'Moon',
            )
        # Luna: never fall back to the catalog-gray scatter when a spin pack exists.
        if not queuedBlenderMoon and not (
            moon.name == 'Moon' and self._blenderBodyAvailable(moon.name)
        ):
            self.axes.scatter(
                [moonPosition[0]],
                [moonPosition[1]],
                [moonPosition[2]],
                color=moon.color,
                s=self.moonCatalog.markerSize3d(
                    moon, 600 if moonOpen else (900 if moon.name == 'Moon' else 1400)
                ),
                depthshade=True,
                zorder=6,
            )
        # Luna: only label in the Earth–Moon open — drop it once the camera
        # reaches the inner system (Earth keeps its planet label as usual).
        if moon.name == 'Moon':
            if moonOpen:
                if queuedBlenderMoon:
                    # Overlay text above the spin billboard so the leading "M" isn't covered.
                    self._pendingBlenderLabels.append(
                        (moon.name, moonPosition.copy(), 11.0, bodyScale)
                    )
                else:
                    self._label3d(
                        moonPosition,
                        f'  {moon.name}',
                        color=self.labelColor,
                        fontsize=11,
                    )
        elif halfWidthAu < 4.0:
            if queuedBlenderMoon:
                self._pendingBlenderLabels.append((moon.name, moonPosition.copy(), 7.0, bodyScale))
            else:
                self._label3d(
                    moonPosition,
                    f'  {moon.name}',
                    color=self.labelColor,
                    fontsize=7,
                )

    def _lunarMotionDays(self, moon, frame: int, halfWidthAu: float) -> float:
        """Sol motion clock is too fast for a readable Earth–Moon opening orbit."""
        days = self._solMotionDays(frame)
        if moon.name != 'Moon' or not self.useBlenderBodies:
            return days
        # At open Luna would crawl on the Sol clock; use a slower open scale, then
        # blend back to full Sol motion by ~2 AU after leaving the Earth+Moon frame.
        openHalf = SOL_EARTH_HALF_AU
        leaveHalf = 2.0
        if halfWidthAu >= leaveHalf:
            return days
        blend = smootherstep((halfWidthAu - openHalf) / max(leaveHalf - openHalf, 1e-6))
        openScale = LUNAR_OPEN_MOTION_SCALE
        return days * (openScale + (1.0 - openScale) * blend)

    def _blenderBodyAvailable(self, catalogName: str) -> bool:
        return (
            self.useBlenderBodies
            and self.blenderSprites is not None
            and self.blenderSprites.hasBody(catalogName)
        )

    def _blenderPlanetBodyScale(self, planetName: str) -> float:
        """World billboard scale vs Earth, lightly padded when the spin includes rings."""
        scale = BLENDER_PLANET_BODY_SCALE.get(planetName, 1.0)
        appearance = appearanceForCatalogName(planetName)
        if appearance is not None and appearance.rings.enabled:
            # Spin PNGs already composite rings; modest pad so the annulus fits
            # without turning Saturn into an Earth-dwarfing disk.
            outer = max(1.0, min(appearance.rings.outerScale, 2.4))
            scale *= 1.0 + 0.35 * (outer - 1.0)
        return scale

    def _blenderRawFracRadius(
        self,
        halfWidthAu: float,
        bodyScale: float,
        *,
        catalogName: str | None = None,
    ) -> float | None:
        """Unfloored on-screen fraction (used for LOD / queue decisions)."""
        radiusAu = self._blenderBillboardRadiusAu(
            halfWidthAu,
            openCloseup=False,
            bodyScale=bodyScale,
            catalogName=catalogName,
        )
        if radiusAu is None:
            return None
        return float(radiusAu / max(2.0 * halfWidthAu, 1e-9))

    def _queueBlenderBody(
        self,
        catalogName: str,
        position: np.ndarray,
        frame: int,
        halfWidthAu: float,
        *,
        openCloseup: bool,
        bodyScale: float,
        orbitalPhaseRad: float | None,
        suppressDotFallback: bool,
    ) -> bool:
        """Queue a spin billboard when the pack exists and the disk is large enough."""
        if not self._blenderBodyAvailable(catalogName):
            return False
        if (
            self._blenderBillboardRadiusAu(
                halfWidthAu,
                openCloseup=openCloseup,
                bodyScale=bodyScale,
                catalogName=catalogName,
            )
            is None
        ):
            return False
        fracRadius = self._blenderBillboardFracRadius(
            halfWidthAu, bodyScale, catalogName=catalogName
        )
        if fracRadius is None:
            return False
        # Earth/Moon keep a painted disk through Sol zoom-out (floor handles tininess).
        # Other packs fall back to catalog dots when they would be sub-pixel.
        if not suppressDotFallback and fracRadius < BLENDER_MIN_BILLBOARD_FRAC:
            return False
        self._pendingBlenderBodies.append(
            (
                catalogName,
                position.copy(),
                frame,
                halfWidthAu,
                openCloseup,
                bodyScale,
                orbitalPhaseRad,
            )
        )
        return True

    def _blenderBillboardRadiusAu(
        self,
        halfWidthAu: float,
        *,
        openCloseup: bool,
        bodyScale: float,
        catalogName: str | None = None,
    ) -> float | None:
        """Fixed world-space globe radius; None → omit the body (no catalog-color dot).

        Radius is anchored to the Earth-open framing so zoom-out shrinks the
        body on screen instead of growing it inside the Moon's orbit.
        """
        del openCloseup  # size is world-fixed; closeup only affects texture resolution
        radiusAu = EARTH_GLOBE_RADIUS_AU * bodyScale
        # Keep textured disks through the Sol zoom-out; drop only once the
        # camera is well past the outer system (never swap to catalog blue/gray).
        # Sol's photosphere billboard may run a little past that generic planet cutoff.
        dropAbove = 100.0
        dropBelow = 0.0
        if catalogName in BLENDER_STAR_BODY_SCALE:
            dropBelow, dropAbove = BLENDER_STAR_BILLBOARD_HALF_AU_BY_BODY.get(
                catalogName, BLENDER_STAR_BILLBOARD_HALF_AU
            )
        if halfWidthAu > dropAbove or halfWidthAu < dropBelow:
            return None
        return radiusAu

    def _cameraDepthKey(self, position: np.ndarray) -> float:
        """Scalar depth along the view axis; larger means closer to the camera.

        Used to painter-sort flat billboard overlays so bodies behind a star
        (Mercury at near-Sun, Proxima b/d at the finale) do not paint on top of it.
        """
        elev = np.deg2rad(self.axes.elev)
        azim = np.deg2rad(self.axes.azim)
        # Unit vector from scene focus toward the camera eye (matplotlib convention).
        eyeDir = np.array(
            [
                np.cos(elev) * np.cos(azim),
                np.cos(elev) * np.sin(azim),
                np.sin(elev),
            ],
            dtype=float,
        )
        focus = getattr(self, '_viewFocus', np.zeros(3))
        return float(np.dot(np.asarray(position, dtype=float) - focus, eyeDir))

    def _scatterDepthZorder(self, position: np.ndarray, *, base: int = 4) -> int:
        """Map camera depth to a discrete axes zorder (farther → lower)."""
        half = max(float(getattr(self, '_viewHalfWidthAu', 1.0)), 1e-6)
        # 0 = far side of the frame, 1 = near side.
        nearness = float(np.clip(0.5 + 0.5 * (self._cameraDepthKey(position) / half), 0.0, 1.0))
        return base + int(round(nearness * 5.0))

    def _projectBlenderOverlay(self, center: np.ndarray) -> tuple[np.ndarray, float] | None:
        """Project a body center to figure-fraction coords + camera depth."""
        x2, y2, _ = proj3d.proj_transform(
            float(center[0]),
            float(center[1]),
            float(center[2]),
            self.axes.get_proj(),
        )
        display = self.axes.transData.transform((x2, y2))
        frac = self.figure.transFigure.inverted().transform(display)
        return frac, self._cameraDepthKey(center)

    def _flushBlenderBodyOverlays(self, halfWidthAu: float) -> None:
        """Project queued bodies and paint Blender spin-loop frames into this frame."""
        if not self._pendingBlenderBodies or self.blenderSprites is None:
            return
        # Far → near so closer disks (and their labels) cover bodies behind stars.
        # Partial occultations keep the free limb: the star PNG's opaque disk covers
        # overlap; do not hard-cull whole planets when only partly behind the star.
        pending = sorted(
            self._pendingBlenderBodies,
            key=lambda item: self._cameraDepthKey(item[1]),
        )
        self._blenderBodyPaintZorder = {}
        for rank, (
            catalogName,
            center,
            frame,
            bodyHalfWidth,
            openCloseup,
            bodyScale,
            orbitalPhaseRad,
        ) in enumerate(pending):
            if (
                self._blenderBillboardRadiusAu(
                    bodyHalfWidth,
                    openCloseup=openCloseup,
                    bodyScale=bodyScale,
                    catalogName=catalogName,
                )
                is None
            ):
                continue
            fracRadius = self._blenderBillboardFracRadius(
                halfWidthAu, bodyScale, catalogName=catalogName
            )
            if fracRadius is None:
                continue
            projected = self._projectBlenderOverlay(center)
            if projected is None:
                continue
            frac, _depth = projected
            if catalogName in BLENDER_STAR_BODY_SCALE:
                resolution = 768
            elif openCloseup:
                resolution = 384
            else:
                resolution = 64 if fracRadius <= 0.01 else 128
            disk = self.blenderSprites.bodyFrame(
                catalogName,
                frame,
                orbitalPhaseRad=orbitalPhaseRad,
                resolution=resolution,
            )
            if disk is None:
                continue
            paintZ = 5 + rank
            self._blenderBodyPaintZorder[
                (catalogName, float(center[0]), float(center[1]), float(center[2]))
            ] = paintZ
            diskU8 = (np.clip(disk, 0.0, 1.0) * 255.0).astype(np.uint8)
            self.bodyOverlay.imshow(
                diskU8,
                extent=(
                    frac[0] - fracRadius,
                    frac[0] + fracRadius,
                    frac[1] - fracRadius,
                    frac[1] + fracRadius,
                ),
                origin='upper',
                interpolation='bilinear',
                zorder=paintZ,
                clip_on=False,
            )
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)

    def _blenderBillboardFracRadius(
        self,
        halfWidthAu: float,
        bodyScale: float,
        *,
        catalogName: str | None = None,
    ) -> float | None:
        """On-screen disk radius in figure fraction (matches overlay paint)."""
        rawFrac = self._blenderRawFracRadius(halfWidthAu, bodyScale, catalogName=catalogName)
        if rawFrac is None:
            return None
        # Stars use pure world scale (no stepped floors) so zoom is monotonic.
        # Non-Sol stars may cap max frac so a dive does not swallow the frame.
        if catalogName in BLENDER_STAR_BODY_SCALE:
            maxFrac = BLENDER_STAR_MAX_FRAC.get(catalogName)
            frac = min(rawFrac, maxFrac) if maxFrac is not None else rawFrac
            # Outer Sol: keep a visible photosphere above shrinking planet floors.
            if (
                catalogName == 'Sun'
                and halfWidthAu >= BLENDER_SUN_OUTER_FLOOR_HALF_AU
                and frac is not None
            ):
                frac = max(frac, BLENDER_SUN_OUTER_MIN_FRAC)
            return frac
        if catalogName in BLENDER_PLANET_MAX_FRAC:
            return min(rawFrac, BLENDER_PLANET_MAX_FRAC[catalogName])
        floor = 0.0035
        if catalogName is not None:
            appearance = appearanceForCatalogName(catalogName)
            if appearance is not None and appearance.rings.enabled and 18.0 <= halfWidthAu <= 70.0:
                floor = BLENDER_RING_LINGER_MIN_FRAC
            elif catalogName in BLENDER_ASTEROID_BODY_SCALE and 3.5 <= halfWidthAu <= 14.0:
                floor = BLENDER_ASTEROID_BELT_MIN_FRAC
        # Ease the readability floor out with zoom so floored planets do not
        # outgrow / bury the Sun at Saturn–Kuiper scales.
        floor *= float(np.clip(12.0 / max(halfWidthAu, 1e-6), 0.2, 1.0))
        # Same floor as overlay paint so labels track the painted disk.
        return max(rawFrac, floor)

    def _blenderBodyLabelPad(self, fracRadius: float) -> float:
        """Gap from disk edge to label; shrinks with Earth as the camera pulls back."""
        # Close-up (~0.09): ~0.03. Inner-Sol floor (0.0035): ~0.0045 — not a fixed
        # 3% figure gap that leaves Earth floating alone on zoom-out.
        return max(0.004, min(0.03, fracRadius * 0.45 + 0.003))

    def _flushBlenderBodyLabels(self, halfWidthAu: float) -> None:
        """Draw deferred body labels above spin billboards (figure fraction)."""
        if not self._pendingBlenderLabels:
            return
        self.bodyOverlay.set_xlim(0.0, 1.0)
        self.bodyOverlay.set_ylim(0.0, 1.0)
        viewHalf = float(halfWidthAu) if halfWidthAu > 0.0 else self._viewHalfWidthAu
        for name, center, fontsize, bodyScale in self._pendingBlenderLabels:
            x2, y2, _ = proj3d.proj_transform(
                float(center[0]),
                float(center[1]),
                float(center[2]),
                self.axes.get_proj(),
            )
            display = self.axes.transData.transform((x2, y2))
            frac = self.figure.transFigure.inverted().transform(display)
            fracRadius = (
                self._blenderBillboardFracRadius(viewHalf, bodyScale, catalogName=name) or 0.0035
            )
            pad = self._blenderBodyLabelPad(fracRadius)
            # Overlay text above the imshow disk. Figure-level text paints under this
            # axes (zorder 20) and gets covered by the globe.
            if name == 'Earth':
                textX = min(frac[0] + fracRadius + pad, 0.94)
                textY = frac[1]
                va = 'center'
            elif name == 'Moon':
                # Moon sits near the right edge; lower-right keeps the leading "M".
                moonPad = max(0.004, min(0.016, pad * 0.55))
                textX = min(frac[0] + fracRadius + moonPad, 0.90)
                textY = max(frac[1] - fracRadius - moonPad, 0.08)
                va = 'top'
            else:
                textX = min(frac[0] + fracRadius + pad, 0.94)
                textY = frac[1]
                va = 'center'
            labelText = {
                'Alpha Centauri A': 'α Cen A',
                'Alpha Centauri B': 'α Cen B',
                'Proxima Centauri': 'Proxima Centauri',
                'Proxima b': 'b',
                'Proxima d': 'd',
            }.get(name, name)
            # Keep labels with their disk in the depth stack (not always on top of stars).
            # Skip labels for disks occulted behind a star (not painted this frame).
            paintKey = (name, float(center[0]), float(center[1]), float(center[2]))
            paintZ = self._blenderBodyPaintZorder.get(paintKey)
            if paintZ is None:
                continue
            self.bodyOverlay.text(
                textX,
                textY,
                labelText,
                color=self.labelColor,
                fontsize=fontsize,
                ha='left',
                va=va,
                clip_on=False,
                zorder=paintZ + 0.5,
            )

    def _drawSolAsteroidPopulations(self, frame: int, halfWidthAu: float) -> None:
        """Asteroid belt, Hildas, Trojans/Greeks, Kuiper, and Oort — same families as sol 3D."""
        style = self.renderStyle
        speed = self._solSpeedEquivalent(frame)
        jupiterMeanAnomaly = planetMeanAnomalyRad(
            self.planetCatalog.planets['Jupiter'].orbitalPeriodDays, 1, self._solMotionDays(frame)
        )
        beltX, beltY, beltZ = self.asteroidPopulation.asteroidBeltPositions(frame, speed)
        hildaX, hildaY, hildaZ = self.asteroidPopulation.hildaPositions(
            frame, jupiterMeanAnomaly, speed
        )
        trojanX, trojanY, trojanZ = self.asteroidPopulation.trojanPositions(
            frame, jupiterMeanAnomaly
        )
        greekX, greekY, greekZ = self.asteroidPopulation.greekPositions(frame, jupiterMeanAnomaly)

        # Inner belts: continuous emphasis curve (peak at belt linger → thin through Kuiper).
        # Never annular-cull — that hid the ~2–3 AU ring mid-zoom.
        beltParams = innerBeltRenderParams(halfWidthAu, style)
        if beltParams is not None:
            beltSize, beltAlpha, clusterSize, clusterAlpha = beltParams
            beltColor = style['beltColor'] if not self.isDark else '#D0D8E4'
            self._scatterPopulation(
                beltX,
                beltY,
                beltZ,
                beltColor,
                beltSize,
                beltAlpha,
            )
            self._scatterPopulation(
                hildaX,
                hildaY,
                hildaZ,
                style['clusterColor'],
                clusterSize,
                clusterAlpha,
            )
            self._scatterPopulation(
                trojanX,
                trojanY,
                trojanZ,
                style['clusterColor'],
                clusterSize,
                clusterAlpha,
            )
            self._scatterPopulation(
                greekX,
                greekY,
                greekZ,
                style['clusterColor'],
                clusterSize,
                clusterAlpha,
            )
            labelAlpha = float(
                np.clip(
                    1.0 - abs(halfWidthAu - SOL_BELT_LINGER_HALF_AU) / 4.5,
                    0.0,
                    1.0,
                )
            )
            if labelAlpha > 0.25:
                self._label3d(
                    np.array([2.8, 1.2, 0.0]),
                    '  Asteroid Belt',
                    color=self.hudColor,
                    fontsize=10,
                    alpha=0.35 + 0.60 * labelAlpha,
                )

        kuiperParams = kuiperRenderParams(halfWidthAu, style)
        if kuiperParams is not None:
            kuiperSize, kuiperAlpha = kuiperParams
            kuiperX, kuiperY, kuiperZ = self.asteroidPopulation.kuiperBeltPositions(frame, speed)
            kuiperColor = '#C8D2E0' if self.isDark else style['kuiperColor']
            self._scatterPopulation(
                kuiperX,
                kuiperY,
                kuiperZ,
                kuiperColor,
                kuiperSize,
                kuiperAlpha,
                halfWidthAu=halfWidthAu,
                annular=True,
            )
            kuiperLabel = float(
                np.clip(1.0 - abs(halfWidthAu - SOL_OUTER_LINGER_HALF_AU) / 18.0, 0.0, 1.0)
            )
            if kuiperLabel > 0.25:
                self._label3d(
                    np.array([42.0, 8.0, 0.0]),
                    '  Kuiper Belt',
                    color=self.hudColor,
                    fontsize=10,
                    alpha=0.35 + 0.60 * kuiperLabel,
                )

        self._drawFamousAsteroids(frame, halfWidthAu)

        if OORT_VISIBLE_ABOVE_AU <= halfWidthAu <= OORT_VISIBLE_BELOW_AU:
            oortX, oortY, oortZ = self._oortWorldPositions()
            oortAlpha = min(0.9, style['oortAlpha'] * 2.4)
            # Fade once we've panned well beyond the cloud (same idea as leaving Kuiper).
            if halfWidthAu > OORT_WORLD_OUTER_AU * 0.7:
                oortAlpha *= max(
                    0.0,
                    1.0 - (halfWidthAu - OORT_WORLD_OUTER_AU * 0.7) / (OORT_WORLD_OUTER_AU * 0.8),
                )
            if oortAlpha > 0.02:
                oortBoost = 8.0 if halfWidthAu < 5000.0 else (5.0 if halfWidthAu < 30000.0 else 3.0)
                oortColor = '#A8B4C4' if self.isDark else style['oortColor']
                # Camera-annulus cull empties the fixed Oort shell once half-width ≫ 10⁵ AU
                # (inner annulus edge overruns the cloud). Drop it on wide neighborhood shots.
                useAnnulus = halfWidthAu < OORT_WORLD_OUTER_AU / POPULATION_ANNULUS_INNER_FRAC
                self._scatterPopulation(
                    oortX,
                    oortY,
                    oortZ,
                    oortColor,
                    max(style['oortSize'] * oortBoost, 2.0),
                    oortAlpha,
                    halfWidthAu=halfWidthAu,
                    annular=useAnnulus,
                )
                # Label near the inner edge while that scale is readable.
                if OORT_WORLD_INNER_AU <= halfWidthAu <= OORT_WORLD_INNER_AU * 12.0:
                    self._label3d(
                        np.array([OORT_WORLD_INNER_AU * 1.6, 0.0, 0.0]),
                        '  Oort Cloud',
                        color=self.hudColor,
                        fontsize=10,
                        alpha=0.95,
                    )

    def _famousAsteroidAlpha(self, category: str, halfWidthAu: float) -> float:
        """Soft visibility so named bodies fade with the population belts (no hard pops)."""
        if category in {'main_belt', 'near_earth', 'trojan'}:
            if halfWidthAu < 1.8:
                return 0.0
            if halfWidthAu < 3.0:
                return smootherstep((halfWidthAu - 1.8) / 1.2)
            if halfWidthAu <= SOL_BELT_LINGER_HALF_AU * 1.35:
                return 1.0
            return (
                max(
                    0.0,
                    1.0 - (halfWidthAu - SOL_BELT_LINGER_HALF_AU * 1.35) / 14.0,
                )
                ** 1.2
            )
        if category == 'kuiper':
            if halfWidthAu < 22.0:
                return 0.0
            if halfWidthAu < 32.0:
                return smootherstep((halfWidthAu - 22.0) / 10.0)
            if halfWidthAu <= 55.0:
                return 1.0
            return max(0.0, 1.0 - (halfWidthAu - 55.0) / 30.0)
        return 0.0

    def _drawFamousAsteroids(self, frame: int, halfWidthAu: float) -> None:
        """Named asteroids / dwarf planets — Ceres, Vesta, Eris, etc."""
        for asteroid in self.famousAsteroidCatalog.asteroids.values():
            alpha = self._famousAsteroidAlpha(asteroid.category, halfWidthAu)
            if alpha < 0.05:
                continue

            orbitX, orbitY, orbitZ = self.orbitCalculator.ellipticalOrbit3d(
                asteroid.semiMajorAxisAu,
                asteroid.eccentricity,
                asteroid.inclinationDeg,
                numPoints=100,
            )
            self.axes.plot(
                orbitX,
                orbitY,
                orbitZ,
                color=asteroid.color,
                linewidth=0.7 if asteroid.diameterKm >= 200 else 0.45,
                alpha=0.22 * alpha,
            )

            meanAnomalyRad = planetMeanAnomalyRad(
                asteroid.orbitalPeriodDays, 1, self._solMotionDays(frame)
            )
            # Keep a stable per-asteroid phase offset (same idea as the static visualizer).
            meanAnomalyRad = (
                meanAnomalyRad + self.famousAsteroidCatalog.initialPhaseRad(asteroid.name)
            ) % (2.0 * np.pi)
            positionX, positionY, positionZ = self.famousAsteroidCatalog.positionAtMeanAnomaly(
                asteroid, meanAnomalyRad, self.orbitCalculator
            )
            position = np.array([float(positionX), float(positionY), float(positionZ)], dtype=float)
            if not self._inView(position, margin=1.1):
                continue

            bodyScale = BLENDER_ASTEROID_BODY_SCALE.get(asteroid.name, 0.22)
            # Ceres is the belt-linger hero during the staged blender zoom-out.
            ceresHero = (
                self.useBlenderBodies
                and asteroid.name == 'Ceres'
                and SOL_BELT_LINGER_HALF_AU * 0.85 <= halfWidthAu <= SOL_BELT_LINGER_HALF_AU * 1.25
            )
            queuedBlender = self._queueBlenderBody(
                asteroid.name,
                position,
                frame,
                halfWidthAu,
                openCloseup=ceresHero,
                bodyScale=bodyScale * (1.35 if ceresHero else 1.0),
                orbitalPhaseRad=None,
                suppressDotFallback=ceresHero,
            )
            if not queuedBlender:
                markerScale = 420.0 if halfWidthAu <= 14.0 else 700.0
                markerSize = max(
                    10.0,
                    float(self.famousAsteroidCatalog.markerSize3d(asteroid, markerScale))
                    * (1.15 if halfWidthAu <= 14.0 else 1.0),
                )
                self.axes.scatter(
                    [position[0]],
                    [position[1]],
                    [position[2]],
                    color=asteroid.color,
                    s=markerSize,
                    alpha=min(0.95, 0.55 + 0.45 * alpha),
                    depthshade=True,
                    zorder=6,
                )
            # Label the larger / well-known bodies while their belt is the focus.
            if alpha >= 0.4 and asteroid.diameterKm >= 100.0:
                fontsize = 8 if asteroid.diameterKm >= 400 else 7
                if queuedBlender:
                    self._pendingBlenderLabels.append(
                        (asteroid.name, position.copy(), float(fontsize), bodyScale)
                    )
                else:
                    self._label3d(
                        position,
                        f'  {asteroid.name}',
                        color=asteroid.color,
                        fontsize=fontsize,
                        alpha=min(0.95, 0.45 + 0.55 * alpha),
                    )

    def _oortWorldPositions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fixed world-space Oort shell — pan out through it like the Kuiper Belt."""
        if self._fixedOortX is None:
            sampleX, sampleY, sampleZ = self.asteroidPopulation.oortCloudPositions(
                0, ANIMATION_SPEED_SOL_NEAR
            )
            radius = np.sqrt(sampleX * sampleX + sampleY * sampleY + sampleZ * sampleZ)
            unitX = sampleX / np.maximum(radius, 1e-12)
            unitY = sampleY / np.maximum(radius, 1e-12)
            unitZ = sampleZ / np.maximum(radius, 1e-12)
            radiusMin = float(np.min(radius))
            radiusMax = float(np.max(radius))
            span = max(radiusMax - radiusMin, 1e-9)
            t = (radius - radiusMin) / span
            # Bias samples outward so the inner decades are not over-weighted.
            t = np.power(np.clip(t, 0.0, 1.0), 0.55)
            # Log-fill from just past Kuiper out to the outer Oort — fixed AU, not camera-relative.
            worldRadius = OORT_WORLD_INNER_AU * (OORT_WORLD_OUTER_AU / OORT_WORLD_INNER_AU) ** t
            self._fixedOortX = unitX * worldRadius
            self._fixedOortY = unitY * worldRadius
            self._fixedOortZ = unitZ * worldRadius
        return self._fixedOortX, self._fixedOortY, self._fixedOortZ

    def _populationAnnulusMask(
        self,
        positionX: np.ndarray,
        positionY: np.ndarray,
        positionZ: np.ndarray,
        halfWidthAu: float,
    ) -> np.ndarray:
        """Keep points in a camera-scale annulus so they don't clot over Sol."""
        radius = np.sqrt(positionX * positionX + positionY * positionY + positionZ * positionZ)
        coreClear = halfWidthAu * POPULATION_CORE_CLEAR_FRAC
        annulusInner = halfWidthAu * POPULATION_ANNULUS_INNER_FRAC
        annulusOuter = halfWidthAu * POPULATION_ANNULUS_OUTER_FRAC
        keepInner = max(coreClear, annulusInner)
        inBox = (
            (np.abs(positionX) <= halfWidthAu * 1.05)
            & (np.abs(positionY) <= halfWidthAu * 1.05)
            & (np.abs(positionZ) <= halfWidthAu * 1.05)
        )
        return inBox & (radius >= keepInner) & (radius <= annulusOuter)

    def _scatterPopulation(
        self,
        positionX: np.ndarray,
        positionY: np.ndarray,
        positionZ: np.ndarray,
        color: str,
        size: float,
        alpha: float,
        *,
        halfWidthAu: float | None = None,
        annular: bool = False,
    ) -> None:
        if annular and halfWidthAu is not None:
            mask = self._populationAnnulusMask(positionX, positionY, positionZ, halfWidthAu)
            if not np.any(mask):
                return
            positionX, positionY, positionZ = positionX[mask], positionY[mask], positionZ[mask]
        self.axes.scatter(
            positionX,
            positionY,
            positionZ,
            color=color,
            s=size,
            alpha=alpha,
            depthshade=True,
            linewidths=0,
        )

    def _fieldStarDrawMask(self, halfWidthAu: float) -> np.ndarray | None:
        """Keep catalog stars whose true Sol XYZ falls inside the current camera box."""
        if self.fieldStars.empty or halfWidthAu < FIELD_STARS_VISIBLE_ABOVE_AU:
            return None

        focus = getattr(self, '_viewFocus', np.zeros(3))
        positionX = self.fieldStars['positionX'].to_numpy(dtype=float)
        positionY = self.fieldStars['positionY'].to_numpy(dtype=float)
        positionZ = self.fieldStars['positionZ'].to_numpy(dtype=float)
        # Match the early 3D neighborhood views: real positions, clipped to the axes.
        margin = halfWidthAu * 1.05
        keep = (
            (np.abs(positionX - focus[0]) <= margin)
            & (np.abs(positionY - focus[1]) <= margin)
            & (np.abs(positionZ - focus[2]) <= margin)
        )
        if not np.any(keep):
            return None
        return keep

    def _drawFieldStars(self, halfWidthAu: float) -> None:
        keep = self._fieldStarDrawMask(halfWidthAu)
        if keep is None:
            return

        visibility = smootherstep(
            float(
                np.clip(
                    (halfWidthAu - FIELD_STARS_VISIBLE_ABOVE_AU) / 40000.0,
                    0.0,
                    1.0,
                )
            )
        )
        themeScale = 0.90 if not self.isDark else 1.0
        visible = self.fieldStars.iloc[np.flatnonzero(keep)]
        colors = visible['fieldColor'].tolist()
        sizes = visible['fieldSize'].to_numpy(dtype=float)
        alphas = visible['fieldAlpha'].to_numpy(dtype=float)
        meanAlpha = float(np.clip(np.mean(alphas) * visibility * themeScale, 0.28, 0.85))
        sizeScale = 1.15 + 0.85 * visibility

        self.axes.scatter(
            visible['positionX'].to_numpy(dtype=float),
            visible['positionY'].to_numpy(dtype=float),
            visible['positionZ'].to_numpy(dtype=float),
            c=colors,
            s=np.maximum(sizes * sizeScale, 6.0),
            alpha=meanAlpha,
            depthshade=False,
            linewidths=0,
            zorder=2,
        )
        if halfWidthAu < 40000.0:
            return
        for _, row in visible.iterrows():
            label = self._fieldStarLabel(row)
            if label is None:
                continue
            self._label3d(
                np.array([row['positionX'], row['positionY'], row['positionZ']], dtype=float),
                f'  {label}',
                color=self.labelColor,
                fontsize=6,
                alpha=0.7 * visibility,
            )

    @staticmethod
    def _fieldStarLabel(row: pd.Series) -> str | None:
        rawName = str(row.get('StarName') or row.get('System') or '')
        name = rawName.replace('\xa0', ' ')
        for fragment in LABELED_STAR_SUBSTRINGS:
            if fragment.lower() in name.lower():
                return fragment
        return None

    def _drawAlphaCentauri(
        self, frame: int, halfWidthAu: float, abProgress: float, linear: float
    ) -> None:
        if halfWidthAu < 100.0 and abProgress < 0.05:
            return
        if linear >= self._proximaDiveEnd():
            return

        primary = self._starPositionSol(self.primaryOrbit, frame, ANIMATION_SPEED_AB)
        secondary = self._starPositionSol(self.secondaryOrbit, frame, ANIMATION_SPEED_AB)
        showBinaryDetail = halfWidthAu < 140.0 and abProgress > 0.2 and linear < self._wideHoldEnd()

        if showBinaryDetail:
            for path, color in (
                (self.primaryOrbitPathSol, STAR_COLORS['primary']),
                (self.secondaryOrbitPathSol, STAR_COLORS['secondary']),
            ):
                self.axes.plot(
                    path[:, 0], path[:, 1], path[:, 2], color=color, linewidth=1.0, alpha=0.65
                )

        if halfWidthAu > 200.0:
            self._drawAbUnresolved(halfWidthAu, linear)
            if linear >= self._abHoldEnd():
                self._drawProximaWideMarker(frame)
            return

        if abProgress < 0.15:
            return

        self._drawAbResolvedPair(primary, secondary, frame, halfWidthAu)
        if abProgress > 0.5:
            self.axes.scatter(
                [self.barycenterSolAu[0]],
                [self.barycenterSolAu[1]],
                [self.barycenterSolAu[2]],
                color=self.pathColor,
                s=18,
                alpha=0.8,
                depthshade=False,
                zorder=4,
            )
            self._label3d(
                self.barycenterSolAu,
                '  AB barycenter',
                color=self.pathColor,
                fontsize=7,
                alpha=0.85,
            )

    def _drawAbResolvedPair(
        self,
        primary: np.ndarray,
        secondary: np.ndarray,
        frame: int,
        halfWidthAu: float,
    ) -> None:
        """A/B photosphere billboards (or scatter markers) once the binary is resolved."""
        markerScale = np.clip(110.0 * (32.0 / max(halfWidthAu, 1.0)), 50.0, 300.0)
        self._drawNamedStar(
            'Alpha Centauri A',
            primary,
            frame,
            halfWidthAu,
            markerColor=STAR_COLORS['primary'],
            markerSize=markerScale,
            classicLabel='  α Cen A',
            labelSize=9.0,
        )
        self._drawNamedStar(
            'Alpha Centauri B',
            secondary,
            frame,
            halfWidthAu,
            markerColor=STAR_COLORS['secondary'],
            markerSize=markerScale * 0.8,
            classicLabel='  α Cen B',
            labelSize=9.0,
        )

    def _drawNamedStar(
        self,
        catalogName: str,
        position: np.ndarray,
        frame: int,
        halfWidthAu: float,
        *,
        markerColor: str,
        markerSize: float,
        classicLabel: str,
        labelSize: float,
    ) -> None:
        """Queue a star spin billboard when available; otherwise scatter + 3D label."""
        queued = False
        if self.useBlenderBodies and catalogName in BLENDER_STAR_BODY_SCALE:
            queued = self._queueBlenderBody(
                catalogName,
                position,
                frame,
                halfWidthAu,
                openCloseup=True,
                bodyScale=BLENDER_STAR_BODY_SCALE[catalogName],
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
        if not queued:
            self._drawStarMarker(
                position,
                markerColor,
                markerSize,
                zorder=self._scatterDepthZorder(position, base=5),
            )
            self._label3d(position, classicLabel, color=self.labelColor, fontsize=int(labelSize))
            return
        self._pendingBlenderLabels.append(
            (catalogName, position.copy(), labelSize, BLENDER_STAR_BODY_SCALE[catalogName])
        )

    def _drawAbUnresolved(self, halfWidthAu: float, linear: float) -> None:
        marker = self.barycenterSolAu
        size = np.clip(52.0 * (self.startHalfWidthAu / halfWidthAu) ** 0.35, 36.0, 190.0)
        label = 'α Cen A/B' if linear >= self._abHoldEnd() else 'α Centauri (next system)'
        self._drawStarMarker(
            marker,
            STAR_COLORS['primary'],
            size,
            zorder=self._scatterDepthZorder(marker, base=5),
        )
        self._label3d(marker, f'  {label}', color=self.labelColor, fontsize=9)

    def _drawProximaWideMarker(self, frame: int) -> None:
        proximaSol = self._proximaPositionSol(frame)
        self.axes.plot(
            self.proximaOrbitPathSol[:, 0],
            self.proximaOrbitPathSol[:, 1],
            self.proximaOrbitPathSol[:, 2],
            color=STAR_COLORS['proxima'],
            linewidth=1.4,
            alpha=0.55,
        )
        self._drawStarMarker(
            proximaSol,
            STAR_COLORS['proxima'],
            140.0,
            zorder=self._scatterDepthZorder(proximaSol, base=5),
        )
        self._label3d(proximaSol, '  Proxima', color=self.labelColor, fontsize=10)

    def _drawProxima(self, frame: int, halfWidthAu: float, linear: float) -> None:
        if linear < self._wideHoldEnd() or halfWidthAu > 80.0:
            return

        proximaSol = self._proximaPositionSol(frame)
        innerCloseup = halfWidthAu <= PROXIMA_WIDE_HALF_AU * 1.05
        starSize = 480.0 if halfWidthAu <= PROXIMA_INNER_HALF_AU * 1.6 else 300.0
        self._drawNamedStar(
            'Proxima Centauri',
            proximaSol,
            frame,
            halfWidthAu,
            markerColor=STAR_COLORS['proxima'],
            markerSize=starSize,
            classicLabel='  Proxima Centauri',
            labelSize=11.0 if innerCloseup else 10.0,
        )

        proximaLocalX, proximaLocalY = bodyPositionInOrbitalPlane(
            self.orbitCalculator,
            self.proximaOrbit.semiMajorAxisAu,
            self.proximaOrbit.eccentricity,
            self.proximaOrbit.periodDays,
            self.proximaOrbit.argumentPeriapsisDeg,
            self.proximaOrbit.meanAnomalyDegEpoch,
            frame,
            ANIMATION_SPEED_PROXIMA_STAR,
        )
        for planet in self.proximaPlanets:
            self._drawOneProximaPlanet(planet, frame, halfWidthAu, proximaLocalX, proximaLocalY)

    def _drawOneProximaPlanet(
        self,
        planet: SystemPlanet,
        frame: int,
        halfWidthAu: float,
        proximaLocalX: float,
        proximaLocalY: float,
    ) -> None:
        confirmed = planet.confidence == 'confirmed'
        # Outer disputed c: keep on the wide frame; hide when zoomed onto b/d.
        if planet.semiMajorAxisAu > halfWidthAu * 1.15:
            return
        alpha = 0.55 if not confirmed else 0.85
        pathX, pathY = self.proximaPlanetPathsLocal[planet.planetId]
        pathLocal = np.column_stack((pathX, pathY))
        pathSol = self.transform.toSol(pathLocal + np.array([proximaLocalX, proximaLocalY]))
        self.axes.plot(
            pathSol[:, 0],
            pathSol[:, 1],
            pathSol[:, 2],
            color=planet.color,
            linewidth=1.8 if confirmed else 1.1,
            alpha=alpha,
        )
        position = self._proximaPlanetPositionSol(planet, frame)
        if not self._inView(position, margin=1.15):
            return
        shortName = planet.name.replace('Proxima ', '')
        suffix = '' if confirmed else ' ?'
        labelSize = 10.0 if confirmed else 8.0
        bodyScale = BLENDER_PLANET_BODY_SCALE.get(planet.name)
        queued = False
        if (
            confirmed
            and bodyScale is not None
            and self.useBlenderBodies
            and halfWidthAu <= PROXIMA_WIDE_HALF_AU * 1.15
        ):
            queued = self._queueBlenderBody(
                planet.name,
                position,
                frame,
                halfWidthAu,
                openCloseup=halfWidthAu <= PROXIMA_INNER_HALF_AU * 2.5,
                bodyScale=bodyScale,
                orbitalPhaseRad=None,
                suppressDotFallback=True,
            )
        if queued:
            self._pendingBlenderLabels.append((planet.name, position.copy(), labelSize, bodyScale))
            return
        # Grow with zoom-in so b/d stay readable as the orbit fills the frame
        # (matplotlib scatter is screen-fixed; without this they look like they shrink).
        baseSize = 56.0 if confirmed else 32.0
        zoomBoost = np.clip(PROXIMA_WIDE_HALF_AU / max(halfWidthAu, 1e-6), 1.0, 12.0)
        markerSize = baseSize * (zoomBoost**0.65)
        self.axes.scatter(
            [position[0]],
            [position[1]],
            [position[2]],
            color=planet.color,
            s=markerSize,
            alpha=1.0 if confirmed else 0.65,
            depthshade=False,
            zorder=self._scatterDepthZorder(position, base=4),
        )
        self._label3d(
            position,
            f'  {shortName}{suffix}',
            color=self.labelColor,
            fontsize=int(labelSize),
            alpha=0.95 if confirmed else 0.7,
        )

    def _solCaption(self, halfWidthAu: float, linear: float) -> tuple[str, str] | None:
        # Sol titles are timeline-gated so Proxima's small AU scale cannot reuse them.
        if linear >= PULLBACK_END:
            return None
        if self.useBlenderBodies:
            staged = self._blenderSolCaption(halfWidthAu, linear)
            if staged is not None:
                return staged
        if halfWidthAu <= SOL_EARTH_HALF_AU * 2.2:
            if self.useBlenderBodies and halfWidthAu < SOL_MOON_REVEAL_HALF_AU:
                return ('Earth', 'A couple of day–night cycles before we find the Moon')
            return ('Earth and the Moon', 'One full lunar orbit before we leave home')
        if halfWidthAu <= SOL_NEAR_SUN_HALF_AU:
            return ('Inner solar system', 'Pulling back past Venus and Mars')
        beltFramed = (
            linear < SOL_BELT_HOLD_END
            or SOL_BELT_LINGER_HALF_AU * 0.85 <= halfWidthAu <= SOL_BELT_LINGER_HALF_AU * 1.25
        )
        if beltFramed and linear < SOL_OUTER_ARRIVE:
            return (
                'Asteroid belt and Jupiter',
                "Main belt inside Jupiter's orbit · Trojans at the Lagrange points",
            )
        if linear < SOL_HOLD_END and halfWidthAu < 28.0:
            return (
                'Outer solar system',
                'Gas giants, ice giants, and the path out to Pluto',
            )
        if linear < SOL_HOLD_END:
            return (
                'Kuiper Belt and Pluto',
                'Icy bodies beyond Neptune · Pluto on its inclined orbit',
            )
        if halfWidthAu < OORT_VISIBLE_ABOVE_AU:
            return ('Leaving the solar system', 'Past Pluto and the Kuiper Belt')
        return ('Through the Oort Cloud', 'A vast shell of icy bodies around Sol')

    def _blenderSolCaption(self, halfWidthAu: float, linear: float) -> tuple[str, str] | None:
        """Beat-specific HUD copy for the staged blender Sol zoom-out (#51)."""
        if linear < SOL_EARTH_BLENDER_DWELL_END:
            if halfWidthAu < SOL_MOON_REVEAL_HALF_AU:
                return ('Earth', 'A couple of day–night cycles before we find the Moon')
            return ('Earth and the Moon', 'One full lunar orbit before we leave home')
        if linear < SOL_BEAT_NEAR_SUN_HOLD_END:
            return ('The Sun', 'Our star up close — then we leave the inner system')
        if linear < SOL_BEAT_INNER_HOLD_END:
            return ('Inner planets', 'Mercury, Venus, and Mars on the road to the belt')
        if linear < SOL_BEAT_BELT_HOLD_END:
            return (
                'Asteroid belt and Jupiter',
                'Ceres in the main belt · Jupiter and its moons beyond',
            )
        if linear < SOL_BEAT_SATURN_HOLD_END:
            return ('Saturn', "Ringed giant — the outer system's signature silhouette")
        if linear < SOL_HOLD_END and halfWidthAu < 28.0:
            return (
                'Outer solar system',
                'Ice giants and the path out toward Pluto',
            )
        if linear < SOL_HOLD_END:
            return (
                'Kuiper Belt and Pluto',
                'Icy bodies beyond Neptune · Pluto on its inclined orbit',
            )
        return None

    def _blenderArrivalCaption(
        self, abProgress: float, halfWidthAu: float, linear: float
    ) -> tuple[str, str] | None:
        """Beat-specific HUD copy for staged blender α Cen arrival (#63)."""
        if linear < PULLBACK_END:
            return None
        remainingLy = (1.0 - abProgress) * self.distanceLy
        if linear < self._abTravelEnd() and abProgress < AB_CRUISE_END:
            return ('Flying toward Alpha Centauri', f'{remainingLy:.2f} light-years remaining')
        if linear < self._abTravelEnd():
            return ('Arriving at Alpha Centauri', f'Scale ~{halfWidthAu:.0f} AU across')
        if linear < self._abHoldEnd():
            return (
                'Alpha Centauri A–B',
                'Textured Rigil Kentaurus and Toliman at their barycenter',
            )
        if linear < self._wideOutArrive():
            return ('Zooming out from A–B', 'Looking for Proxima on its wide orbit')
        if linear < self._wideHoldEnd():
            return (
                'Alpha Centauri triple system',
                'Proxima on its wide ~8.7 kau path around A–B',
            )
        if linear < self._proximaDiveEnd():
            return (
                'Diving in to Proxima Centauri',
                'From the triple system down to Proxima’s planets',
            )
        if linear < self._proximaWideHoldEnd():
            return (
                'Proxima Centauri system',
                'Confirmed planets d & b · disputed c on a wider orbit',
            )
        if halfWidthAu > PROXIMA_INNER_HALF_AU * 1.3:
            return (
                'Proxima Centauri system',
                'Closing in on the inner planets',
            )
        return (
            'Proxima b and d up close',
            'Habitable-zone world b · inner planet d · c lies farther out',
        )

    def _caption(
        self, abProgress: float, proximaProgress: float, halfWidthAu: float, linear: float
    ) -> tuple[str, str]:
        del proximaProgress  # arrival captions are timeline / half-width gated
        solCaption = self._solCaption(halfWidthAu, linear)
        if solCaption is not None:
            return solCaption
        if self.useBlenderBodies:
            arrival = self._blenderArrivalCaption(abProgress, halfWidthAu, linear)
            if arrival is not None:
                return arrival
        remainingLy = (1.0 - abProgress) * self.distanceLy
        if linear < AB_TRAVEL_END and abProgress < AB_CRUISE_END:
            return ('Flying toward Alpha Centauri', f'{remainingLy:.2f} light-years remaining')
        if linear < AB_HOLD_END:
            if halfWidthAu > 80.0:
                return ('Arriving at Alpha Centauri', f'Scale ~{halfWidthAu:.0f} AU across')
            return (
                'Alpha Centauri A–B up close',
                'A and B orbit their shared barycenter',
            )
        if linear < WIDE_OUT_ARRIVE:
            return ('Zooming out from A–B', 'Looking for Proxima on its wide orbit')
        if linear < WIDE_OUT_END:
            return (
                'Alpha Centauri triple system',
                'Proxima orbits the A–B pair on a wide ~8.7 kau path',
            )
        if linear < PROXIMA_TRAVEL_END:
            return (
                'Diving in to Proxima Centauri',
                'From the triple system down to Proxima’s planets',
            )
        if halfWidthAu > PROXIMA_INNER_HALF_AU * 1.3:
            return (
                'Proxima Centauri system',
                'Confirmed planets d & b · disputed c on a wider orbit',
            )
        return (
            'Proxima b and d up close',
            'Habitable-zone world b · inner planet d · c lies farther out',
        )

    def _applyAxes(
        self,
        focus: np.ndarray,
        halfWidthAu: float,
        abProgress: float,
        proximaProgress: float,
        linear: float,
    ) -> None:
        self.axes.set_xlim(focus[0] - halfWidthAu, focus[0] + halfWidthAu)
        self.axes.set_ylim(focus[1] - halfWidthAu, focus[1] + halfWidthAu)
        self.axes.set_zlim(focus[2] - halfWidthAu, focus[2] + halfWidthAu)

        if linear < PULLBACK_END:
            if halfWidthAu <= SOL_NEAR_SUN_HALF_AU:
                if self.useBlenderBodies and halfWidthAu < self.solEarthHalfWidthAu - 1e-9:
                    # Earth-close → Earth+Moon open angle as Luna comes into frame.
                    closeTilt = smootherstep(
                        (halfWidthAu - SOL_EARTH_CLOSE_HALF_AU)
                        / max(self.solEarthHalfWidthAu - SOL_EARTH_CLOSE_HALF_AU, 1e-6)
                    )
                    elev = EARTH_CLOSE_ELEVATION_DEG + closeTilt * (
                        EARTH_OPEN_ELEVATION_DEG - EARTH_CLOSE_ELEVATION_DEG
                    )
                else:
                    # Ease from top-down Earth open down to the usual Sol angle.
                    tilt = smootherstep(
                        (halfWidthAu - SOL_EARTH_HALF_AU)
                        / max(SOL_NEAR_SUN_HALF_AU - SOL_EARTH_HALF_AU, 1e-6)
                    )
                    elev = EARTH_OPEN_ELEVATION_DEG + tilt * (
                        SOL_ELEVATION_DEG - EARTH_OPEN_ELEVATION_DEG
                    )
            else:
                elev = SOL_ELEVATION_DEG
            azim = self.solAzimuthDeg
        elif linear < self._abTravelEnd():
            # Ease Sol → travel once, then hold steady through pan + AB dive (no spin).
            blend = segmentProgress(linear, PULLBACK_END, PULLBACK_END + 0.02)
            elev = SOL_ELEVATION_DEG + blend * (CAMERA_ELEVATION_DEG - SOL_ELEVATION_DEG)
            azim = lerpAngleDeg(self.solAzimuthDeg, self.travelAzimuthDeg, blend)
        elif linear < self._proximaWideHoldEnd():
            # Steady travel camera through AB hold, wide-out, dive, and Proxima-wide hold.
            elev, azim = CAMERA_ELEVATION_DEG, self.travelAzimuthDeg
        else:
            # Gentle tilt-up for the Proxima close-up — azimuth stays put (no orbit flip).
            blend = segmentProgress(linear, self._proximaWideHoldEnd(), self._proximaInnerArrive())
            elev = CAMERA_ELEVATION_DEG + blend * (PROXIMA_ELEVATION_DEG - CAMERA_ELEVATION_DEG)
            azim = self.travelAzimuthDeg

        self.axes.view_init(elev=elev, azim=azim)
        self.axes.set_axis_off()
        self.axes.set_box_aspect((1, 1, 1), zoom=1.0)
        self.axes.set_position((0.0, 0.0, 1.0, 1.0))

        title, subtitle = self._caption(abProgress, proximaProgress, halfWidthAu, linear)
        self.axes.set_title(title, color=self.labelColor, pad=10, y=0.98, fontsize=13)
        self.figure.text(
            0.5,
            0.035,
            subtitle,
            ha='center',
            color=self.hudColor,
            fontsize=10,
            alpha=0.95,
        )

    def saveGif(self, outputPath: str) -> None:
        os.makedirs(os.path.dirname(outputPath) or '.', exist_ok=True)
        animation = FuncAnimation(
            self.figure,
            self.update,
            frames=self.animationFrames,
            interval=1000 // ANIMATION_FPS,
            blit=False,
        )
        self.figure.set_size_inches(*self.figureSizeInches)
        self.figure.set_dpi(self.dpi)
        animation.save(
            outputPath,
            writer=PillowWriter(fps=ANIMATION_FPS),
            savefig_kwargs={'pad_inches': 0, 'facecolor': self.figure.get_facecolor()},
        )
        plt.close(self.figure)
        print(f'Saved {outputPath}')


def renderSolCentauriCinematicAnimations(
    figureSizeInches: tuple[float, float] = DEFAULT_FIGURE_SIZE_INCHES,
    dpi: int = DEFAULT_DPI,
    starsCsvPath: str = 'data/nearby_stars_30.csv',
    *,
    useBlenderBodies: bool = False,
) -> None:
    catalog = SystemCatalog(starsCsvPath=starsCsvPath)
    system = catalog.load('alpha_centauri')
    if useBlenderBodies:
        outputDirectory = BLENDER_OUTPUT_DIRECTORY
        stem = 'sol_centauri_cinematic_blender'
    else:
        outputDirectory = OUTPUT_DIRECTORY
        stem = 'sol_centauri_cinematic'
    for themeName, styleName in (('light', 'default'), ('dark', 'dark_background')):
        outputPath = f'{outputDirectory}/{stem}_{themeName}.gif'
        print(f'Rendering {outputPath}...')
        animator = SolCentauriCinematicAnimator(
            system,
            style=styleName,
            figureSizeInches=figureSizeInches,
            dpi=dpi,
            starsCsvPath=starsCsvPath,
            useBlenderBodies=useBlenderBodies,
        )
        animator.saveGif(outputPath)
    print('Sol ↔ Centauri cinematic animations completed!')
