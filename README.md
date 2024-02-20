# Visualization of Pluto's Orbit and the Kuiper Belt

This Python script provides a detailed visualization of the orbital paths of the major planets in our solar system, with a special focus on Pluto and its relationship with the Kuiper Belt. Utilizing `numpy` for mathematical calculations and `matplotlib` for plotting, the script generates a comprehensive diagram showing the orbits of the planets, Pluto's elliptical orbit, and the scattered distribution of the Kuiper Belt.

## Getting Started

To run this script, you need Python installed on your system along with the `numpy` and `matplotlib` libraries. These dependencies can be installed using pip:

```bash
pip install numpy numpy
pip install numpy matplotlib
```


## Code Explanation
**Importing Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
```

**Constants Definition**

These constants define various parameters for the orbits of the major planets, Pluto, and the Kuiper Belt. The semi-major axes of the planet orbits and specific parameters related to Pluto's orbit and the Kuiper Belt are set here.

```python
ORBIT_POINTS = 1000  # Number of points to plot for each orbit
PLANET_ORBITS = [0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30]  # Semi-major axes of the planets in AU
PLUTO_PERIHELION = 29.7  # Pluto's closest point to the Sun in AU
PLUTO_APHELION = 49.5  # Pluto's farthest point from the Sun in AU
PLUTO_ECCENTRICITY = 0.25  # Pluto's orbital eccentricity
PLUTO_SEMI_MAJOR_AXIS = (PLUTO_PERIHELION + PLUTO_APHELION) / 2  # Semi-major axis of Pluto's orbit
KUIPER_BELT_INNER = 30  # Inner edge of the Kuiper Belt in AU
KUIPER_BELT_OUTER = 50  # Outer edge of the Kuiper Belt in AU
KUIPER_BELT_POINTS = 20000  # Number of points to represent the Kuiper Belt
```

**Function to Calculate Ellipse**

This function calculates the x and y coordinates for points on an ellipse, used to plot Pluto's orbit. It takes the eccentricity and semi-major axis of the ellipse, along with an array of angle values (theta), to compute the coordinates.

```python
def calculate_ellipse(eccentricity, semi_major_axis, theta):
    """
    Calculate the x, y coordinates of an ellipse based on eccentricity and semi-major axis.
    """
    b = semi_major_axis * np.sqrt(1 - eccentricity**2)  # Semi-minor axis
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y
```

**Generating Orbits and Kuiper Belt**

Generates the coordinates for Pluto's orbit and randomly distributed points within the Kuiper Belt. For Pluto, it calculates an elliptical orbit. For the Kuiper Belt, it generates points within specified radial distances to simulate its scattered disk.

```python
theta = np.linspace(0, 2 * np.pi, ORBIT_POINTS)
x, y = calculate_ellipse(PLUTO_ECCENTRICITY, PLUTO_SEMI_MAJOR_AXIS, theta)

kuiper_belt_r = np.random.uniform(KUIPER_BELT_INNER, KUIPER_BELT_OUTER, KUIPER_BELT_POINTS)
kuiper_belt_theta = np.random.uniform(0, 2 * np.pi, KUIPER_BELT_POINTS)
kuiper_belt_x = kuiper_belt_r * np.cos(kuiper_belt_theta)
kuiper_belt_y = kuiper_belt_r * np.sin(kuiper_belt_theta)
```


**Plotting**

```python
fig, ax = plt.subplots(figsize=(39, 39))
```

Plots the Sun at the center of the solar system.
```python
# Plot the Sun at the center
sun = plt.Circle((0, 0), 0.05, color='yellow', fill=True)
ax.add_artist(sun)
```

Loops through the PLANET_ORBITS array to plot circular orbits for each of the major planets.
```python
# Plot the orbits of the major planets
for orbit in PLANET_ORBITS:
    circle = plt.Circle((0, 0), orbit, color='black', fill=False)
    ax.add_artist(circle)
```

Plots Pluto's elliptical orbit and scatters points for the Kuiper Belt.
```python
# Plot the elliptical orbit of Pluto
ax.plot(x, y, color='blue')

# Mark the perihelion and aphelion of Pluto's orbit
ax.plot(PLUTO_PERIHELION, 0, 'bo')  # Perihelion
ax.plot(-PLUTO_APHELION, 0, 'bo')  # Aphelion

# Scatter the adjusted points for the Kuiper Belt
ax.scatter(kuiper_belt_x, kuiper_belt_y, color='gray', s=5)
```

Adds annotations for key features such as Pluto's perihelion and aphelion, and labels the Kuiper Belt. It also adjusts the plot's appearance for better visualization.
```python
# Annotations and labels with increased font size
font_size = 48
ax.annotate('Kuiper Belt', xy=(KUIPER_BELT_OUTER, 0), xytext=(KUIPER_BELT_OUTER+5, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
ax.annotate("Pluto's aphelion (49.5 AU)", xy=(-PLUTO_APHELION, 0), xytext=(-PLUTO_APHELION-25, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
ax.annotate("Pluto's perihelion (29.7 AU)", xy=(PLUTO_PERIHELION, 0), xytext=(PLUTO_PERIHELION+10, -10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
```

Saves the generated plot as a high-resolution image for use in presentations, educational materials, or personal study.
```python
ax.set_xlim([-70, 70])
ax.set_ylim([-70, 70])
ax.set_aspect('equal', 'box')
ax.axis('off')
plt.title('Relationship of Pluto’s orbit to the Kuiper Belt', fontsize=62)
plt.savefig("orbit.jpg", dpi=300)
plt.show()
```
