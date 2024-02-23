import numpy as np
import matplotlib.pyplot as plt

# Constants for the orbits and labels
ORBIT_POINTS = 1000  # Number of points to plot for each orbit
PLANET_ORBITS = [0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30]  # Semi-major axes of the planets in AU
PLUTO_PERIHELION = 29.7  # Pluto's closest point to the Sun in AU
PLUTO_APHELION = 49.5  # Pluto's farthest point from the Sun in AU
PLUTO_ECCENTRICITY = 0.25  # Pluto's orbital eccentricity
PLUTO_SEMI_MAJOR_AXIS = (PLUTO_PERIHELION + PLUTO_APHELION) / 2  # Semi-major axis of Pluto's orbit
KUIPER_BELT_INNER = 30  # Inner edge of the Kuiper Belt in AU
KUIPER_BELT_OUTER = 50  # Outer edge of the Kuiper Belt in AU
KUIPER_BELT_POINTS = 20000  # Number of points to represent the Kuiper Belt

def calculate_ellipse(eccentricity, semi_major_axis, theta):
    """
    Calculate the x, y coordinates of an ellipse based on eccentricity and semi-major axis.
    """
    b = semi_major_axis * np.sqrt(1 - eccentricity**2)  # Semi-minor axis
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

theta = np.linspace(0, 2 * np.pi, ORBIT_POINTS)
x, y = calculate_ellipse(PLUTO_ECCENTRICITY, PLUTO_SEMI_MAJOR_AXIS, theta)

kuiper_belt_r = np.random.uniform(KUIPER_BELT_INNER, KUIPER_BELT_OUTER, KUIPER_BELT_POINTS)
kuiper_belt_theta = np.random.uniform(0, 2 * np.pi, KUIPER_BELT_POINTS)
kuiper_belt_x = kuiper_belt_r * np.cos(kuiper_belt_theta)
kuiper_belt_y = kuiper_belt_r * np.sin(kuiper_belt_theta)

fig, ax = plt.subplots(figsize=(39, 39))

# Plot the Sun at the center
sun = plt.Circle((0, 0), 0.05, color='yellow', fill=True)
ax.add_artist(sun)

# Plot the orbits of the major planets
for orbit in PLANET_ORBITS:
    circle = plt.Circle((0, 0), orbit, color='black', fill=False)
    ax.add_artist(circle)

# Plot the elliptical orbit of Pluto
ax.plot(x, y, color='blue')

# Scatter the adjusted points for the Kuiper Belt
ax.scatter(kuiper_belt_x, kuiper_belt_y, color='gray', s=5)

# Mark the perihelion and aphelion of Pluto's orbit
ax.plot(PLUTO_PERIHELION, 0, 'bo')  # Perihelion
ax.plot(-PLUTO_APHELION, 0, 'bo')  # Aphelion

# Annotations and labels with increased font size
font_size = 48
ax.annotate('Kuiper Belt', xy=(KUIPER_BELT_OUTER, 0), xytext=(KUIPER_BELT_OUTER+5, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
ax.annotate("Pluto's aphelion (49.5 AU)", xy=(-PLUTO_APHELION, 0), xytext=(-PLUTO_APHELION-25, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
ax.annotate("Pluto's perihelion (29.7 AU)", xy=(PLUTO_PERIHELION, 0), xytext=(PLUTO_PERIHELION+10, -10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)

ax.set_xlim([-70, 70])
ax.set_ylim([-70, 70])
ax.set_aspect('equal', 'box')
ax.axis('off')
plt.title('Relationship of Pluto’s orbit to the Kuiper Belt', fontsize=62)
plt.savefig("pluto_orbit.jpg", dpi=300)
#plt.show()
