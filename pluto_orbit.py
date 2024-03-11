import numpy as np
import matplotlib.pyplot as plt

# Constants for the orbits, labels, and new belts
ORBIT_POINTS = 1000
PLANET_ORBITS = [0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30]
PLUTO_PERIHELION = 29.7
PLUTO_APHELION = 49.5
PLUTO_ECCENTRICITY = 0.25
PLUTO_SEMI_MAJOR_AXIS = (PLUTO_PERIHELION + PLUTO_APHELION) / 2
KUIPER_BELT_INNER = 30
KUIPER_BELT_OUTER = 50
KUIPER_BELT_POINTS = 20000
ASTEROID_BELT_INNER = 2.2
ASTEROID_BELT_OUTER = 3.2
ASTEROID_BELT_POINTS = 250
TROJAN_GREEK_POINTS = 50  # Making Hilda's points match Trojan and Greek for consistency
JUPITER_SEMI_MAJOR_AXIS = 5.2
TROJAN_GREEK_ANGLE = np.deg2rad(60)  # 60 degrees in radians
TROJAN_GREEK_SPREAD = np.pi / 3  # 1/6 of the orbit in radians
TROJAN_GREEK_WIDTH = 0.5  # Approximate width of the belts in AU
HILDA_POINTS = 50  # Adjusted to match the Trojan and Greek points
HILDA_SPREAD = np.pi / 3.5  # 1/7 of the orbit in radians for length

def calculate_ellipse(eccentricity, semi_major_axis, theta):
    b = semi_major_axis * np.sqrt(1 - eccentricity**2)
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

# Generate asteroid belt points
asteroid_belt_r = np.random.uniform(ASTEROID_BELT_INNER, ASTEROID_BELT_OUTER, ASTEROID_BELT_POINTS)
asteroid_belt_theta = np.random.uniform(0, 2 * np.pi, ASTEROID_BELT_POINTS)
asteroid_belt_x = asteroid_belt_r * np.cos(asteroid_belt_theta)
asteroid_belt_y = asteroid_belt_r * np.sin(asteroid_belt_theta)

# Trojan and Greek points
trojan_r = np.random.uniform(JUPITER_SEMI_MAJOR_AXIS - TROJAN_GREEK_WIDTH, JUPITER_SEMI_MAJOR_AXIS + TROJAN_GREEK_WIDTH, TROJAN_GREEK_POINTS)
greek_r = np.random.uniform(JUPITER_SEMI_MAJOR_AXIS - TROJAN_GREEK_WIDTH, JUPITER_SEMI_MAJOR_AXIS + TROJAN_GREEK_WIDTH, TROJAN_GREEK_POINTS)
trojan_theta = np.linspace(TROJAN_GREEK_ANGLE - TROJAN_GREEK_SPREAD / 2, TROJAN_GREEK_ANGLE + TROJAN_GREEK_SPREAD / 2, TROJAN_GREEK_POINTS)
greek_theta = np.linspace(TROJAN_GREEK_ANGLE + np.pi - TROJAN_GREEK_SPREAD / 2, TROJAN_GREEK_ANGLE + np.pi + TROJAN_GREEK_SPREAD / 2, TROJAN_GREEK_POINTS) + 20.15

trojan_x = trojan_r * np.cos(trojan_theta)
trojan_y = trojan_r * np.sin(trojan_theta)
greek_x = greek_r * np.cos(greek_theta)
greek_y = greek_r * np.sin(greek_theta)

# Adjusted Hilda asteroids to match the width and number of Trojan and Greek points, placed between the asteroid belt and Jupiter's orbit
hilda_theta = np.linspace(-HILDA_SPREAD / 2, HILDA_SPREAD / 2, HILDA_POINTS) + np.pi
hilda_r = np.random.uniform(ASTEROID_BELT_OUTER, JUPITER_SEMI_MAJOR_AXIS, HILDA_POINTS)
hilda_x = hilda_r * np.cos(hilda_theta)
hilda_y = hilda_r * np.sin(hilda_theta)

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

# Scatter the points for the Kuiper Belt, asteroid belt, Trojans, Greeks, and Hilda asteroids
ax.scatter(kuiper_belt_x, kuiper_belt_y, color='gray', s=5)  # Kuiper Belt
ax.scatter(asteroid_belt_x, asteroid_belt_y, color='gray', s=5)  # Asteroid Belt
ax.scatter(trojan_x, trojan_y, color='gray', s=5)  # Trojan Asteroids
ax.scatter(greek_x, greek_y, color='gray', s=5)  # Greek Asteroids
ax.scatter(hilda_x, hilda_y, color='gray', s=5)  # Hilda Asteroids, color matched to Trojan/Greek

# Add Jupiter as a red dot
# Plotting each planet with the specified colors
ax.plot(PLANET_ORBITS[0] * np.cos(5), PLANET_ORBITS[0] * np.sin(5), 'o', markersize=1, color='darkgrey')  # Mercury
ax.plot(PLANET_ORBITS[1] * np.cos(160), PLANET_ORBITS[1] * np.sin(160), 'o', markersize=3, color='orange')    # Venus
ax.plot(PLANET_ORBITS[2] * np.cos(100), PLANET_ORBITS[2] * np.sin(100), 'o', markersize=3, color='blue')      # Earth
ax.plot(PLANET_ORBITS[3] * np.cos(10), PLANET_ORBITS[3] * np.sin(10), 'o', markersize=2, color='red')       # Mars
# Jupiter is already added as a red dot, we can change its color if needed
ax.plot(PLANET_ORBITS[4] * np.cos(0), PLANET_ORBITS[4] * np.sin(0), 'o', markersize=10, color='orange')   # Jupiter
ax.plot(PLANET_ORBITS[5] * np.cos(40), PLANET_ORBITS[5] * np.sin(40), 'o', markersize=8, color='beige')     # Saturn
ax.plot(PLANET_ORBITS[6] * np.cos(200), PLANET_ORBITS[6] * np.sin(200), 'o', markersize=7, color='blue')      # Uranus
ax.plot(PLANET_ORBITS[7] * np.cos(60), PLANET_ORBITS[7] * np.sin(60), 'o', markersize=6, color='blue')      # Neptune


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
