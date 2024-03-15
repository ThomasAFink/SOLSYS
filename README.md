# Visualization of The Solar System on an Interstellar Scale

This Python script provides a detailed visualization of the orbital paths of the major planets in our solar system, with a special focus on Pluto and its relationship with the Kuiper Belt. Utilizing `numpy` for mathematical calculations and `matplotlib` for plotting, the script generates a comprehensive diagram showing the orbits of the planets, Pluto's elliptical orbit, and the scattered distribution of the Kuiper Belt.

| Inner Solar System With Jupiter | Solar System With Kuiper Belt |
|---------------|-----------------------------|
| ![Inner Solar System With Jupiter](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/1_inner_solar_system_with_jupiter.jpg?raw=true) | ![Solar System With Kuiper Belt](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/2_solar_system_with_kuiper_belt.jpg?raw=true) |

| Solar System With Oort Cloud | Solar System with Alpha Centauri |
|-----------------------------|-----------------------------|
| ![Solar System With Oort Cloud](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/3_solar_system_with_oort_cloud.jpg?raw=true) | ![Solar System with Alpha Centauri](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/4_solar_system_with_alpha_centauri.jpg?raw=true) |

| Interstellar Neighbors Within 10 Light Years | Interstellar Neighbors Within 25 Light Years |
|-----------------------------|-----------------------------|
| ![Interstellar Neighbors Within 10 Light Years](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/5_solar_system_with_nearest_stars_10.jpg?raw=true) | ![Interstellar Neighbors Within 25 Light Years](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/2d/6_solar_system_with_nearest_stars_25.jpg?raw=true) |


## Getting Started

To run this script, you need Python installed on your system along with the `numpy` and `matplotlib` libraries. These dependencies can be installed using pip:

```bash
pip3 install numpy
pip3 install pandas
pip3 install matplotlib
pip3 install mpl_toolkits
```


## Code Explanation
**Importing Libraries**

```python
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

**Constants Definition**

These constants define various parameters for the orbits of the major planets, Pluto, and the various asteroid belts. The semi-major axes of the planet orbits and specific parameters related to Pluto's orbit and the Kuiper Belt are set here.

```python
ORBIT_POINTS = 1000
PLANET_ORBITS = [0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30]
PLUTO_PERIHELION = 29.7
PLUTO_APHELION = 49.5
PLUTO_ECCENTRICITY = 0.25
PLUTO_SEMI_MAJOR_AXIS = (PLUTO_PERIHELION + PLUTO_APHELION) / 2
KUIPER_BELT_INNER = 30
KUIPER_BELT_OUTER = 50
ASTEROID_BELT_INNER = 2.2
ASTEROID_BELT_OUTER = 3.2
JUPITER_SEMI_MAJOR_AXIS = 5.2
JUPITER_INCLINATION = 1.3
JUPITER_ECCENTRICITY = 0.0489
TROJANS_GREEKS_ANGLE = np.deg2rad(60)
TROJANS_GREEKS_SPREAD = np.pi / 3
TROJANS_GREEKS_WIDTH = 0.5
HILDAS_INNER = ASTEROID_BELT_OUTER+0.25
HILDAS_OUTER = JUPITER_SEMI_MAJOR_AXIS-0.25
OORT_CLOUD_INNER = 2000
OORT_CLOUD_OUTER = 100000

if limit[3] == 'inner_solar_system':
    ASTEROID_BELT_POINTS = 20000
    TROJANS_GREEKS_POINTS = 4000
    HILDAS_POINTS = 4000
    KUIPER_BELT_POINTS = 10000
    OORT_CLOUD_POINTS = 50000

elif limit[3] == 'inner_solar_system_with_jupiter':
    ASTEROID_BELT_POINTS = 10000
    TROJANS_GREEKS_POINTS = 2000
    HILDAS_POINTS = 2000
    KUIPER_BELT_POINTS = 10000
    OORT_CLOUD_POINTS = 50000

elif limit[3] == 'solar_system_with_kuiper_belt':
    ASTEROID_BELT_POINTS = 100
    TROJANS_GREEKS_POINTS = 10
    HILDAS_POINTS = 100
    KUIPER_BELT_POINTS = 10000
    OORT_CLOUD_POINTS = 50000

elif limit[3] == 'solar_system_with_oort_cloud':
    ASTEROID_BELT_POINTS = 20
    TROJANS_GREEKS_POINTS = 10
    HILDAS_POINTS = 10
    KUIPER_BELT_POINTS = 100
    OORT_CLOUD_POINTS = 50000

elif limit[3] == 'solar_system_with_alpha_centauri':
    ASTEROID_BELT_POINTS = 10
    TROJANS_GREEKS_POINTS = 5
    HILDAS_POINTS = 5
    KUIPER_BELT_POINTS = 50
    OORT_CLOUD_POINTS = 5000  

elif  limit[3] == 'solar_system_with_nearest_stars_10':
    ASTEROID_BELT_POINTS = 2
    TROJANS_GREEKS_POINTS = 2
    HILDAS_POINTS = 2
    KUIPER_BELT_POINTS = 20
    OORT_CLOUD_POINTS = 2000

elif  limit[3] == 'solar_system_with_nearest_stars_25':
    ASTEROID_BELT_POINTS = 1
    TROJANS_GREEKS_POINTS = 1
    HILDAS_POINTS = 1
    KUIPER_BELT_POINTS = 10
    OORT_CLOUD_POINTS = 1000
```

**Function to Calculate Ellipse**

This function calculates the x and y coordinates for points on an ellipse, used to plot Pluto's orbit. It takes the eccentricity and semi-major axis of the ellipse, along with an array of angle values (theta), to compute the coordinates.

Equation 1: Calculates the semi-minor axis (b) of the ellipse based on its semi-major axis (a) and eccentricity (ε). The eccentricity measures the deviation of the ellipse from a perfect circle, with 0 representing a circle. The semi-minor axis is derived using the relationship between the eccentricity and the semi-major axis.
- ![Equation 1](https://latex.codecogs.com/png.latex?b%20%3D%20a%20%5Csqrt%7B1%20-%20%5Cepsilon%5E2%7D)

Equation 2: Computes the radial distance (r) from the center of the ellipse to a point on its edge, given an angle (theta) from the major axis. This formula adjusts for the ellipse's eccentricity, which influences the radial distance across the ellipse.
- ![Equation 2](https://latex.codecogs.com/png.latex?r%20%3D%20%5Cfrac%7Ba%281%20-%20%5Cepsilon%5E2%29%7D%7B1%20%2B%20%5Cepsilon%20%5Ccos%28%5Ctheta%29%7D)

Equation 3: Determines the x-coordinate of a point on the ellipse, based on the radial distance (r) and the angle (theta), showing how this distance is projected onto the x-axis.
- ![Equation 3](https://latex.codecogs.com/png.latex?x%20%3D%20r%20%5Ccos%28%5Ctheta%29)

Equation 4: Calculates the y-coordinate of a point on the ellipse, using the radial distance (r) and the angle (theta), which illustrates the projection of this distance onto the y-axis.
- ![Equation 4](https://latex.codecogs.com/png.latex?y%20%3D%20r%20%5Csin%28%5Ctheta%29)


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

## 3D Visualization of Pluto's Orbit and the Kuiper Belt

The following images provide different perspectives on Pluto's orbit and the Kuiper Belt, showcasing the 3D modeling capabilities of `matplotlib`. These images can be generated from various viewing angles to illustrate the complex spatial relationships within this region of our solar system.

| Inner Solar System With Jupiter | Solar System With Kuiper Belt |
|---------------|-----------------------------|
| ![Inner Solar System With Jupiter](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/1_inner_solar_system_with_jupiter.jpg?raw=true) | ![Solar System With Kuiper Belt](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/2_solar_system_with_kuiper_belt.jpg?raw=true) |

| Solar System With Oort Cloud | Solar System with Alpha Centauri |
|-----------------------------|-----------------------------|
| ![Solar System With Oort Cloud](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/3_solar_system_with_oort_cloud.jpg?raw=true) | ![Solar System with Alpha Centauri](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/4_solar_system_with_alpha_centauri.jpg?raw=true) |

| Interstellar Neighbors Within 10 Light Years | Interstellar Neighbors Within 25 Light Years |
|-----------------------------|-----------------------------|
| ![Interstellar Neighbors Within 10 Light Years](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/5_solar_system_with_nearest_stars_10.jpg?raw=true) | ![Interstellar Neighbors Within 25 Light Years](https://github.com/ThomasAFink/visualization_of_plutos_orbit_and_the_kuiper_belt/blob/main/output/3d/6_solar_system_with_nearest_stars_25.jpg?raw=true) |

## Conclusion
This script is an educational tool that visualizes the orbits within our solar system, with a focus on Pluto and the Kuiper Belt. It demonstrates the power of numpy and matplotlib in creating complex scientific visualizations. Feel free to modify the constants and functions to explore other celestial mechanics or solar system objects.

