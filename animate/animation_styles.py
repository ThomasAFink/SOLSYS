"""Render styles and animation timing constants."""

from __future__ import annotations


FIGURE_SIZE_INCHES = (12, 12)
ANIMATION_FPS = 20
CAMERA_ELEVATION_DEG = 25
CAMERA_AZIMUTH_DEG = 120

# 2D: fixed inner-system top-down
AXIS_LIMIT_2D_AU = 6.5
ANIMATION_FRAMES_2D = 1600
ANIMATION_SPEED_2D = 4.0
INNER_PLANET_NAMES = ('Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter')

# 3D: staged zoom from Oort cloud to inner system
ANIMATION_FRAMES_3D = 500
ANIMATION_SPEED_3D = 3.0
MIN_CAMERA_DISTANCE_AU = 3.2
MAX_CAMERA_DISTANCE_AU = 100000.0
ZOOM_IN_FRAME_FRACTION = 0.80
ZOOM_STAGES = (
    (0.00, 100000.0),
    (0.14, 600.0),
    (0.32, 58.0),
    (0.44, 42.0),
    (0.54, 33.0),
    (0.62, 22.0),
    (0.70, 12.0),
    (0.78, 7.5),
    (0.88, 3.2),
    (1.00, 3.2),
)
KUIPER_VISIBLE_BELOW_AU = 48.0
INNER_BELT_VISIBLE_BELOW_AU = 8.5
OORT_VISIBLE_ABOVE_AU = 1500.0
VISIBILITY_FADE_SPAN_AU = 4.0

ASTEROID_RENDER_STYLES = {
    'light': {
        'beltColor': '#707070',
        'clusterColor': '#808080',
        'kuiperColor': '#757575',
        'oortColor': '#A0A0A0',
        'beltSize': 0.5,
        'clusterSize': 0.5,
        'kuiperSize': 0.375,
        'oortSize': 0.25,
        'beltAlpha': 0.55,
        'clusterAlpha': 0.5,
        'kuiperAlpha': 0.45,
        'oortAlpha': 0.25,
    },
    'dark': {
        'beltColor': '#D8D8D8',
        'clusterColor': '#CFCFCF',
        'kuiperColor': '#BDBDBD',
        'oortColor': '#9A9A9A',
        'beltSize': 0.875,
        'clusterSize': 0.75,
        'kuiperSize': 0.625,
        'oortSize': 0.375,
        'beltAlpha': 0.82,
        'clusterAlpha': 0.78,
        'kuiperAlpha': 0.72,
        'oortAlpha': 0.42,
    },
}

