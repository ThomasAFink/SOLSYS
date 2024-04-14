from matplotlib import pyplot as plt
import numpy as np
from scipy.constants import G, pi
import astropy.units as u
from astropy.constants import M_sun
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from numpy.polynomial import Polynomial




import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Orbital parameters for 'Oumuamua
eccentricity = 1.2
perihelion = 0.25  # closest approach to the sun in astronomical units
inclination = np.radians(122.74)  # convert degrees to radians
semi_major_axis = -1.279  # for a hyperbolic trajectory, semi-major axis is negative

# Function to calculate position in the orbit
def calculate_orbit(e, q, num_points=1000):
    # theta range for one branch of the hyperbola
    theta = np.linspace(-np.pi, np.pi, num_points)
    # r = q * (1 + e) / (1 + e*cos(theta)) for hyperbolic trajectory
    r = q * (1 + e) / (1 + e * np.cos(theta))
    return r, theta

# Generate the orbit
r, theta = calculate_orbit(eccentricity, perihelion)

# Convert polar coordinates to Cartesian coordinates in the plane of the orbit
x = r * np.cos(theta)
y = r * np.sin(theta)
z = np.zeros_like(x)  # No out-of-plane component initially

# Rotate the orbit to account for the inclination
x_rot = x
y_rot = y * np.cos(inclination) - z * np.sin(inclination)
z_rot = y * np.sin(inclination) + z * np.cos(inclination)

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_rot, y_rot, z_rot, label='Orbit of \'Oumuamua')
ax.scatter([0], [0], [0], color='yellow', label='Sun')  # Sun at the origin

# Setting labels and legend
ax.set_xlabel('X in AU')
ax.set_ylabel('Y in AU')
ax.set_zlabel('Z in AU')
ax.legend()

plt.show()