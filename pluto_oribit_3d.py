import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_ellipse(eccentricity, semi_major_axis, theta):
    """
    Calculate the x, y coordinates of an ellipse based on eccentricity and semi-major axis.
    """
    b = semi_major_axis * np.sqrt(1 - eccentricity**2)  # Semi-minor axis
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def calculate_3d_ellipse(eccentricity, semi_major_axis, theta, inclination):
    """
    Calculate the x, y, z coordinates of an inclined ellipse based on eccentricity,
    semi-major axis, and inclination.
    """
    b = semi_major_axis * np.sqrt(1 - eccentricity**2)
    r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(inclination)
    z = r * np.sin(theta) * np.sin(inclination)
    return x, y, z

def calculate_3d_kuiper_belt(inner_radius, outer_radius, num_points, inclination):
    """
    Generate random 3D coordinates for the Kuiper Belt points within a torus shape.
    """
    radii = np.random.uniform(inner_radius, outer_radius, num_points)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    
    x = radii * np.cos(theta)
    y = radii * np.sin(theta) * np.cos(inclination)
    z = radii * np.sin(theta) * np.sin(inclination) * np.sin(phi)
    
    return x, y, z

# Constants
ORBIT_POINTS = 1000
PLANET_ORBITS = [0.39, 0.72, 1.0, 1.52, 5.2, 9.5, 19.2, 30]
PLUTO_ECCENTRICITY = 0.25
PLUTO_SEMI_MAJOR_AXIS = 39.5  # Average distance in AU
PLUTO_INCLINATION = np.radians(17)
KUIPER_BELT_INNER = 30
KUIPER_BELT_OUTER = 50
KUIPER_BELT_POINTS = 20000

theta = np.linspace(0, 2 * np.pi, ORBIT_POINTS)

# Generate the 3D orbit of Pluto and Kuiper Belt
x, y, z = calculate_3d_ellipse(PLUTO_ECCENTRICITY, PLUTO_SEMI_MAJOR_AXIS, theta, PLUTO_INCLINATION)
kuiper_belt_x, kuiper_belt_y, kuiper_belt_z = calculate_3d_kuiper_belt(KUIPER_BELT_INNER, KUIPER_BELT_OUTER, KUIPER_BELT_POINTS, PLUTO_INCLINATION)

# Define viewing angles
view_angles = [
    (90, 0),  # top-down view
    (45, 300),  # elevation and azimuth
    (30, 210),  # elevation and azimuth
    (20, 120)  # elevation and azimuth
]

# Loop through each view angle to plot and save the figure
for angle, image_file in zip(view_angles, [
    "pluto_orbit_3d_top_down_view.jpg",
    "pluto_orbit_3d_view_45_300.jpg",
    "pluto_orbit_3d_view_30_210.jpg",
    "pluto_orbit_3d_view_20_120.jpg"
]):
    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0, 0, 0, color='yellow', s=100, label='Sun')  # Sun
    for orbit in PLANET_ORBITS:
        circle_x, circle_y = calculate_ellipse(0, orbit, theta)  # Planets' orbits
        ax.plot(circle_x, circle_y, 0, color='black')
    ax.plot(x, y, z, color='blue', label="Pluto's Orbit")  # Pluto's orbit
    ax.scatter(kuiper_belt_x, kuiper_belt_y, kuiper_belt_z, color='darkgray', s=1, alpha=0.5)  # Kuiper Belt
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.view_init(elev=angle[0], azim=angle[1])
    ax.set_title('3D Representation of Pluto’s Orbit and the Kuiper Belt', fontsize=20)
    ax.legend()
    plt.axis('off')  # Removes the axes for a cleaner look
    plt.savefig(image_file, bbox_inches='tight')
    plt.close()  # Close the plot to free memory
