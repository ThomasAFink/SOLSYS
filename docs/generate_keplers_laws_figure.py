"""Generate an original Kepler's-laws diagram for SOLSYS docs.

Each panel uses a different orbital orientation / viewing angle so the
figure does not match classic horizontal textbook stock art.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, PathPatch
from matplotlib.path import Path

OUTPUT_PATH = 'docs/keplers_three_laws.png'
DPI = 220

SEMI_MAJOR_A = 1.0
ECCENTRICITY = 0.52
FOCUS_C = SEMI_MAJOR_A * ECCENTRICITY
SEMI_MINOR_B = SEMI_MAJOR_A * np.sqrt(1.0 - ECCENTRICITY**2)

BG = '#0F1419'
PANEL = '#171D25'
INK = '#E8EEF4'
MUTED = '#9AA7B5'
ACCENT = '#5B9FD4'
STAR = '#FFE08A'
BODY = '#7ED0C8'
SWEEP_NEAR = '#3D6B8C'
SWEEP_FAR = '#2F9B8A'
EMPHASIS = '#F0A56E'


@dataclass(frozen=True)
class ViewAngle:
    """Sky-plane rotation + foreshortening (inclined orbit viewed from above)."""

    rotationDeg: float
    inclineDeg: float = 0.0
    offsetX: float = 0.0
    offsetY: float = 0.0

    def project(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        angle = np.radians(self.rotationDeg)
        cosA, sinA = np.cos(angle), np.sin(angle)
        xr = x * cosA - y * sinA
        yr = x * sinA + y * cosA
        yr = yr * np.cos(np.radians(self.inclineDeg))
        return xr + self.offsetX, yr + self.offsetY

    def point(self, x: float, y: float) -> tuple[float, float]:
        px, py = self.project(np.array([x]), np.array([y]))
        return float(px[0]), float(py[0])


VIEW_LAW1 = ViewAngle(rotationDeg=58.0, inclineDeg=28.0)
VIEW_LAW2 = ViewAngle(rotationDeg=-72.0, inclineDeg=18.0, offsetY=-0.55)
# Mild tilt so a, b, θ stay readable (unlike Laws I–II’s strong angles).
VIEW_LAW3 = ViewAngle(rotationDeg=22.0, inclineDeg=12.0, offsetY=-0.12)


def heliocentric_xy(trueAnomalyRad: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = (SEMI_MAJOR_A * (1.0 - ECCENTRICITY**2)) / (1.0 + ECCENTRICITY * np.cos(trueAnomalyRad))
    return radius * np.cos(trueAnomalyRad), radius * np.sin(trueAnomalyRad)


def draw_orbit(axes, view: ViewAngle, color: str = ACCENT, lw: float = 1.6, scale: float = 1.0) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 500)
    x, y = heliocentric_xy(theta)
    px, py = view.project(x * scale, y * scale)
    axes.plot(px, py, color=color, linewidth=lw, zorder=2)


def draw_sun(axes, size: float = 0.11, center: tuple[float, float] = (0.0, 0.0)) -> None:
    cx, cy = center
    axes.add_patch(Circle((cx, cy), size * 1.7, facecolor=STAR, alpha=0.18, edgecolor='none', zorder=3))
    axes.add_patch(Circle((cx, cy), size, facecolor=STAR, edgecolor='#E6C35A', linewidth=0.8, zorder=5))


def draw_body(
    axes,
    view: ViewAngle,
    trueAnomalyRad: float,
    radius: float = 0.05,
    label: str | None = None,
    labelOffset=(0.10, 0.08),
) -> tuple[float, float]:
    x, y = heliocentric_xy(np.array([trueAnomalyRad]))
    px, py = view.point(float(x[0]), float(y[0]))
    axes.add_patch(Circle((px, py), radius, facecolor=BODY, edgecolor='#4EA89F', linewidth=0.7, zorder=6))
    if label:
        axes.text(px + labelOffset[0], py + labelOffset[1], label, color=INK, fontsize=10, ha='left', va='bottom')
    return px, py


def sector_patch(
    view: ViewAngle,
    thetaStart: float,
    thetaEnd: float,
    facecolor: str,
    scale: float = 1.0,
) -> PathPatch:
    n = 72
    anomalies = np.linspace(thetaStart, thetaEnd, n)
    x, y = heliocentric_xy(anomalies)
    px, py = view.project(x * scale, y * scale)
    sun = view.point(0.0, 0.0)
    vertices = [sun, *zip(px, py), sun]
    codes = [Path.MOVETO, *([Path.LINETO] * n), Path.CLOSEPOLY]
    return PathPatch(
        Path(vertices, codes),
        facecolor=facecolor,
        edgecolor=ACCENT,
        linewidth=0.8,
        alpha=0.55,
        zorder=1,
    )


def areal_integral(thetaStart: float, thetaEnd: float, samples: int = 400) -> float:
    anomalies = np.linspace(thetaStart, thetaEnd, samples)
    radius = (SEMI_MAJOR_A * (1.0 - ECCENTRICITY**2)) / (1.0 + ECCENTRICITY * np.cos(anomalies))
    return 0.5 * float(np.trapezoid(radius**2, anomalies))


def equal_area_far_width(nearStart: float, nearEnd: float) -> float:
    targetArea = areal_integral(nearStart, nearEnd)
    low, high = 0.05, 1.2
    for _ in range(40):
        mid = 0.5 * (low + high)
        area = areal_integral(np.pi - mid, np.pi + mid)
        if area < targetArea:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def style_panel(axes) -> None:
    axes.set_facecolor(PANEL)
    # Identical square data windows; panel boxes are already equal squares.
    axes.set_xlim(-1.70, 1.70)
    axes.set_ylim(-1.70, 1.70)
    axes.set_xticks([])
    axes.set_yticks([])
    for spine in axes.spines.values():
        spine.set_visible(False)


def law1(axes) -> None:
    view = VIEW_LAW1
    style_panel(axes)
    draw_orbit(axes, view)
    draw_sun(axes, center=view.point(0.0, 0.0))
    draw_body(axes, view, 2.35, label='planet', labelOffset=(0.12, -0.02))

    emptyFocus = view.point(-2.0 * FOCUS_C, 0.0)
    sun = view.point(0.0, 0.0)
    axes.plot(
        [emptyFocus[0]],
        [emptyFocus[1]],
        'o',
        markersize=5,
        markerfacecolor='none',
        markeredgecolor=MUTED,
        zorder=5,
    )
    axes.text(emptyFocus[0], emptyFocus[1] - 0.20, 'empty focus', color=MUTED, fontsize=8, ha='center')
    axes.text(sun[0], sun[1] - 0.28, 'Sun (focus)', color=MUTED, fontsize=8, ha='center')

    axes.text(
        -1.55,
        1.45,
        'Law I — Sun at one focus',
        color=INK,
        fontsize=10,
        fontweight='bold',
        ha='left',
    )
    axes.text(
        -1.55,
        1.22,
        f'rotated {view.rotationDeg:.0f}° · inclined {view.inclineDeg:.0f}°',
        color=MUTED,
        fontsize=8,
        ha='left',
    )


def law2(axes) -> None:
    view = VIEW_LAW2
    style_panel(axes)

    nearStart, nearEnd = -0.70, 0.70
    farHalf = equal_area_far_width(nearStart, nearEnd)
    farStart, farEnd = np.pi - farHalf, np.pi + farHalf

    axes.add_patch(sector_patch(view, nearStart, nearEnd, SWEEP_NEAR, scale=0.88))
    axes.add_patch(sector_patch(view, farStart, farEnd, SWEEP_FAR, scale=0.88))
    draw_orbit(axes, view, scale=0.88)
    draw_sun(axes, center=view.point(0.0, 0.0))
    for anomaly in (nearStart, nearEnd, farStart, farEnd):
        x, y = heliocentric_xy(np.array([anomaly]))
        px, py = view.project(x * 0.88, y * 0.88)
        axes.add_patch(
            Circle(
                (float(px[0]), float(py[0])),
                0.035,
                facecolor=BODY,
                edgecolor='#4EA89F',
                linewidth=0.7,
                zorder=6,
            )
        )

    nearLabelX, nearLabelY = view.project(np.array([0.48]), np.array([0.0]))
    farLabelX, farLabelY = view.project(np.array([-0.92]), np.array([0.0]))
    axes.text(
        float(nearLabelX[0]),
        float(nearLabelY[0]),
        r'$\Delta t$',
        color=INK,
        fontsize=12,
        ha='center',
        fontweight='bold',
    )
    axes.text(
        float(farLabelX[0]),
        float(farLabelY[0]),
        r'$\Delta t$',
        color=INK,
        fontsize=12,
        ha='center',
        fontweight='bold',
    )

    axes.text(
        -1.55,
        1.52,
        'Law II — equal areas in equal times',
        color=INK,
        fontsize=10,
        fontweight='bold',
        ha='left',
    )
    axes.text(
        -1.55,
        1.32,
        f'rotated {view.rotationDeg:.0f}° · inclined {view.inclineDeg:.0f}°',
        color=MUTED,
        fontsize=8,
        ha='left',
    )
    axes.text(
        0.0,
        -1.50,
        'same time interval → same swept area',
        color=MUTED,
        fontsize=9,
        ha='center',
    )


def law3(axes) -> None:
    """Law III with clear a / b / θ / T — mild view so geometry stays readable."""
    view = VIEW_LAW3
    style_panel(axes)
    draw_orbit(axes, view)
    sun = view.point(0.0, 0.0)
    draw_sun(axes, center=sun)

    center = view.point(-FOCUS_C, 0.0)
    peri = view.point(SEMI_MAJOR_A * (1.0 - ECCENTRICITY), 0.0)
    apo = view.point(-SEMI_MAJOR_A * (1.0 + ECCENTRICITY), 0.0)
    minorTop = view.point(-FOCUS_C, SEMI_MINOR_B)
    minorBot = view.point(-FOCUS_C, -SEMI_MINOR_B)

    axes.plot([apo[0], peri[0]], [apo[1], peri[1]], color=MUTED, lw=0.9, ls='--', zorder=1)
    axes.plot([minorBot[0], minorTop[0]], [minorBot[1], minorTop[1]], color=MUTED, lw=0.9, ls='--', zorder=1)

    # Semi-major a: center → aphelion (away from the Sun), like the classic diagram.
    axes.annotate(
        '',
        xy=apo,
        xytext=center,
        arrowprops=dict(arrowstyle='<->', color=EMPHASIS, lw=1.8),
        zorder=3,
    )
    aMid = (0.45 * center[0] + 0.55 * apo[0], 0.45 * center[1] + 0.55 * apo[1])
    alongX, alongY = apo[0] - center[0], apo[1] - center[1]
    alongLen = max(np.hypot(alongX, alongY), 1e-6)
    perpX, perpY = -alongY / alongLen, alongX / alongLen
    axes.text(
        aMid[0] + perpX * 0.20,
        aMid[1] + perpY * 0.20,
        r'$a$',
        color=EMPHASIS,
        fontsize=14,
        fontstyle='italic',
        ha='center',
        va='center',
    )

    axes.annotate(
        '',
        xy=minorTop,
        xytext=center,
        arrowprops=dict(arrowstyle='<->', color=ACCENT, lw=1.4),
        zorder=3,
    )
    bMid = (0.5 * (center[0] + minorTop[0]), 0.5 * (center[1] + minorTop[1]))
    axes.text(
        bMid[0] - 0.14,
        bMid[1],
        r'$b$',
        color=ACCENT,
        fontsize=13,
        fontstyle='italic',
        ha='right',
        va='center',
    )

    axes.text(0.85, 1.00, r'$b^{2} = a^{2}(1 - e^{2})$', color=MUTED, fontsize=10, ha='center')

    planetAnomaly = 1.05
    px, py = draw_body(axes, view, planetAnomaly, radius=0.05)
    axes.plot([sun[0], px], [sun[1], py], color=INK, lw=1.0, zorder=3)
    axes.text(px + 0.14, py + 0.08, r'$T$', color=EMPHASIS, fontsize=14, fontstyle='italic', ha='left', va='bottom')

    periDir = np.array([peri[0] - sun[0], peri[1] - sun[1]], dtype=float)
    planetDir = np.array([px - sun[0], py - sun[1]], dtype=float)
    periAngle = np.degrees(np.arctan2(periDir[1], periDir[0]))
    planetAngle = np.degrees(np.arctan2(planetDir[1], planetDir[0]))
    theta1, theta2 = sorted((periAngle, planetAngle))
    if theta2 - theta1 > 180:
        theta1, theta2 = theta2, theta1 + 360
    arcRadius = 0.32
    axes.add_patch(
        Arc(
            sun,
            arcRadius * 2,
            arcRadius * 2,
            angle=0,
            theta1=theta1,
            theta2=theta2,
            color=INK,
            lw=1.1,
            zorder=4,
        )
    )
    midAngle = np.radians(0.5 * (theta1 + theta2))
    axes.text(
        sun[0] + arcRadius * 1.45 * np.cos(midAngle),
        sun[1] + arcRadius * 1.45 * np.sin(midAngle),
        r'$\theta$',
        color=INK,
        fontsize=12,
        ha='center',
        va='center',
    )

    axes.text(
        -1.55,
        1.45,
        r'Law III — $T^{2} \propto a^{3}$',
        color=INK,
        fontsize=10,
        fontweight='bold',
        ha='left',
    )
    axes.text(
        -1.55,
        1.22,
        f'rotated {view.rotationDeg:.0f}° · inclined {view.inclineDeg:.0f}°',
        color=MUTED,
        fontsize=8,
        ha='left',
    )
    axes.text(
        0.0,
        -1.42,
        r'$T$ = orbital period',
        color=MUTED,
        fontsize=8,
        ha='center',
    )
    axes.text(
        0.0,
        -1.58,
        r'$a$ = semi-major   ·   $b$ = semi-minor',
        color=MUTED,
        fontsize=8,
        ha='center',
    )


def main() -> None:
    os.makedirs(os.path.dirname(OUTPUT_PATH) or '.', exist_ok=True)

    figureWidth = 16.0
    figureHeight = 7.2
    figure = plt.figure(figsize=(figureWidth, figureHeight), facecolor=BG)

    figure.text(
        0.5,
        0.955,
        "Kepler's laws — the physics behind SOLSYS orbits",
        ha='center',
        va='top',
        color=INK,
        fontsize=16,
        fontweight='bold',
    )
    figure.text(
        0.5,
        0.905,
        'Same Keplerian model, shown at three different sky orientations (not the textbook head-on view)',
        ha='center',
        va='top',
        color=MUTED,
        fontsize=10,
    )

    # Three equal square panels in a row with equal horizontal gaps.
    panelCount = 3
    gap = 0.025
    rowLeft = 0.04
    rowRight = 0.04
    usableWidth = 1.0 - rowLeft - rowRight - gap * (panelCount - 1)
    panelWidth = usableWidth / panelCount
    panelHeight = panelWidth * figureWidth / figureHeight
    # Sit the row under the title, vertically centered in the remaining space.
    contentTop = 0.86
    contentBottom = 0.04
    panelBottom = contentBottom + max(0.0, (contentTop - contentBottom - panelHeight) / 2.0)

    axesList = []
    for index in range(panelCount):
        left = rowLeft + index * (panelWidth + gap)
        axesList.append(figure.add_axes([left, panelBottom, panelWidth, panelHeight]))

    law1(axesList[0])
    law2(axesList[1])
    law3(axesList[2])

    figure.savefig(OUTPUT_PATH, dpi=DPI, facecolor=BG)
    plt.close(figure)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
