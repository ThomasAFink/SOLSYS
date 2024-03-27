import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import re
from scipy.spatial.transform import Rotation as R

# Function to parse RA from "14h 29m 43.0s" to degrees
def parse_ra_to_degrees(ra_str):
    if not isinstance(ra_str, str):
        return None  # Or handle the non-string case as needed
    match = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', ra_str)
    if match:
        hours, minutes, seconds = map(float, match.groups())
        degrees = 15 * (hours + minutes / 60 + seconds / 3600)
        return degrees
    return None

def calculate_3d_orbit(semi_major_axis, eccentricity, inclination, num_points=100):
    inclination = np.radians(inclination)
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(inclination)
    z = r * np.sin(theta) * np.sin(inclination)
    return x, y, z

def parse_ra_dec_safe(ra_str, dec_str):
    if not isinstance(ra_str, str) or not isinstance(dec_str, str):
        return None, None
    ra_pattern = re.compile(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s')
    dec_pattern = re.compile(r'([+-]?\d+)°\s*(\d+)′\s*(\d+(?:\.\d*)?)″')
    ra_match = ra_pattern.search(ra_str)
    dec_match = dec_pattern.search(dec_str)
    if ra_match and dec_match:
        ra = float(ra_match.group(1)) + float(ra_match.group(2)) / 60.0 + float(ra_match.group(3)) / 3600.0
        dec = float(dec_match.group(1)) + float(dec_match.group(2)) / 60.0 + float(dec_match.group(3)) / 3600.0
        ra *= 15.0  # Convert RA to degrees
        if '-' in dec_match.group(1):
            dec *= -1
        return ra, dec
    else:
        return None, None

def ra_dec_to_3d(ra_deg, dec_deg, distance):
    # Convert RA and DEC from degrees to radians
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    # Convert spherical (RA, DEC, distance) to Cartesian (x, y, z)
    x = distance * np.cos(dec_rad) * np.cos(ra_rad)
    y = distance * np.cos(dec_rad) * np.sin(ra_rad)
    z = distance * np.sin(dec_rad)
    return x, y, z


def generate_belt_points(radius_inner, radius_outer, thickness, num_points):
    """
    Generate points within a spherical shell with thickness.
    :param radius_inner: Inner radius of the shell in AU.
    :param radius_outer: Outer radius of the shell in AU.
    :param thickness: Thickness of the shell in AU.
    :param num_points: Number of points to generate.
    :return: x, y, z coordinates of the points.
    """
    phi = np.random.uniform(0, 2 * np.pi, num_points)  # Azimuthal angle
    costheta = np.random.uniform(-1, 1, num_points)  # Cosine of polar angle for uniform distribution
    # Adjusting for thickness: Distribute radii within [radius - thickness/2, radius + thickness/2]
    radii = np.random.uniform(radius_inner - thickness / 2, radius_outer + thickness / 2, num_points)
    
    theta = np.arccos(costheta)
    r = radii  # Directly use the adjusted radii including thickness

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

def generate_lagrange_points(semi_major_axis, inclination, num_points, leading=True):
    """
    Generate points around Jupiter's L4 or L5 Lagrange points in 3D.
    :param semi_major_axis: Semi-major axis of Jupiter's orbit in AU.
    :param inclination: Orbital inclination in degrees.
    :param num_points: Number of points to generate.
    :param leading: True for L4 (leading), False for L5 (trailing).
    :return: x, y, z coordinates of the points.
    """
    lag_angle = np.radians(60 if leading else 300)  # L4 (60°) or L5 (300°)
    inclination = np.radians(inclination)
    r = semi_major_axis

    theta = np.random.uniform(0, 2 * np.pi, num_points)
    phi = np.random.uniform(-np.pi / 6, np.pi / 6, num_points) + inclination  # Spread around the inclination angle

    x = r * np.cos(lag_angle) + r * np.cos(theta) * np.sin(phi)
    y = r * np.sin(lag_angle) + r * np.sin(theta) * np.sin(phi)
    z = r * np.sin(phi) * np.cos(phi)

    return x, y, z

def hildas_cluster_distribution(semi_major_axis, eccentricity, inclination, num_clusters, num_points_per_cluster):
    """
    Generate points for the Hildas group with an expanded coverage primarily inside Jupiter's orbit.
    
    :param semi_major_axis: Semi-major axis of Jupiter's orbit in AU.
    :param eccentricity: Eccentricity of Jupiter's orbit.
    :param inclination: Orbital inclination in degrees of Jupiter's orbit.
    :param num_clusters: Number of clusters to simulate the Hildas group.
    :param num_points_per_cluster: Number of points to generate around each of the cluster points.
    :return: x, y, z coordinates of the points.
    """
    inclination = np.radians(inclination)
    
    # Define broader displacement for clusters
    radial_displacement = 0.8  # AU, increased radial displacement for a broader coverage
    vertical_displacement = 0.05  # AU, for vertical spread due to inclination
    
    cluster_angles = np.radians([60, 180, 300])
    
    x_total, y_total, z_total = [], [], []
    
    for angle in cluster_angles:
        r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(angle)) - radial_displacement / 2
        for _ in range(num_points_per_cluster):
            # Generate points with broader displacement around cluster centers
            displacement_radius = np.random.uniform(0, radial_displacement)
            displacement_angle = np.random.uniform(0, 2 * np.pi)
            
            x_disp = displacement_radius * np.cos(displacement_angle)
            y_disp = displacement_radius * np.sin(displacement_angle)
            
            x = r * np.cos(angle) + x_disp
            y = r * np.sin(angle) + y_disp
            z = np.sin(inclination) * np.random.uniform(-vertical_displacement, vertical_displacement)
            
            x_total.append(x)
            y_total.append(y)
            z_total.append(z)
    
    return np.array(x_total), np.array(y_total), np.array(z_total)

def hildas_cluster_bands(cluster_points, num_interpolation_points, spread_radius, bow_factor):
    """
    Spread out and interpolate points between given cluster centers, with an added bowing effect towards Jupiter's orbit.
    
    :param cluster_points: Coordinates of the cluster centers as (x, y, z) tuples.
    :param num_interpolation_points: Number of points to interpolate between each cluster center.
    :param spread_radius: Radius within which to spread out the points around the interpolation line.
    :param bow_factor: Factor to control the bowing effect towards Jupiter's orbit.
    :return: x, y, z coordinates of the spread out points with bowing effect.
    """
    interpolated_x, interpolated_y, interpolated_z = [], [], []
    
    num_clusters = len(cluster_points)
    for i in range(num_clusters):
        start_cluster = cluster_points[i]
        end_cluster = cluster_points[(i + 1) % num_clusters]
        
        for j in range(1, num_interpolation_points + 1):
            t = j / (num_interpolation_points + 1)
            base_point = (1 - t) * np.array(start_cluster) + t * np.array(end_cluster)
            
            # Calculate a bowing effect that increases the spread as we move away from cluster centers
            bowing_effect = bow_factor * np.sin(np.pi * t)
            
            # Random spread with additional bowing effect
            spread_distance = np.random.uniform(-spread_radius, spread_radius) + bowing_effect
            
            # Calculate the direction towards Jupiter's orbit (assumed at origin for simplicity)
            direction_to_jupiter = -base_point / np.linalg.norm(base_point)
            spread_point = base_point + direction_to_jupiter * spread_distance
            
            interpolated_x.append(spread_point[0])
            interpolated_y.append(spread_point[1])
            interpolated_z.append(spread_point[2])
    
    return np.array(interpolated_x), np.array(interpolated_y), np.array(interpolated_z)

def calculate_hyperbolic_orbit_parabolic_segment_3d(eccentricity, semi_major_axis, inclination, num_points=1000):
    """
    Calculate and rotate the 3D parabolic segment of a hyperbolic trajectory to ensure it passes through the Solar System
    near the specified point and ends at Vega.
    """
    # Adjust the range of theta to ensure the trajectory extends outward from the solar system
    theta = np.linspace(-np.pi/2, np.pi/2, num_points)
    
    # Calculate the initial parabolic segment
    r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(np.radians(inclination))
    z = r * np.sin(theta) * np.sin(np.radians(inclination))

    # Specify the point through which the trajectory must pass (near the solar system center)
    solar_system_pass_point = np.array([0.24742940409, 0.24742940409, 0])

    # Vega's position relative to the Solar System (origin)
    vega_x, vega_y, vega_z = 198133.63647947184, -1218652.875270329, 992106.3881115752

    # Normalize the initial and final direction vectors
    initial_direction = np.array([x[-1], y[-1], z[-1]]) - np.array([x[0], y[0], z[0]])
    final_direction = np.array([vega_x, vega_y, vega_z]) - solar_system_pass_point
    print(final_direction)
    #[  198133.38905007 -1218653.12269973   992106.38811158]
    initial_direction_norm = initial_direction / np.linalg.norm(initial_direction)
    final_direction_norm = final_direction / np.linalg.norm(final_direction)

    # Calculate the rotation needed to align the initial trajectory with the final direction
    rotation_axis = np.cross(initial_direction_norm, final_direction_norm)
    rotation_angle = np.arccos(np.dot(initial_direction_norm, final_direction_norm))
    
    # Rodrigues' rotation formula for rotation matrix
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    rotation_matrix = np.eye(3) + np.sin(rotation_angle) * K + (1 - np.cos(rotation_angle)) * np.dot(K, K)

    # Apply rotation to align with the final direction
    rotated_trajectory = np.dot(rotation_matrix, np.vstack((x, y, z)))

    # Adjust the trajectory to pass through the specified solar system point
    # This involves translating the trajectory
    trajectory_offset = solar_system_pass_point - rotated_trajectory[:,0]  # Adjustment from the first point
    rotated_and_translated_trajectory = rotated_trajectory + trajectory_offset[:, np.newaxis]

    # Scale the trajectory to ensure it ends exactly at Vega's position
    end_point_vector = np.array([vega_x, vega_y, vega_z]) - rotated_and_translated_trajectory[:, -1]
    scale_factor = np.linalg.norm(end_point_vector) / np.linalg.norm(rotated_and_translated_trajectory[:, -1] - rotated_and_translated_trajectory[:, 0])
    scaled_trajectory = rotated_and_translated_trajectory * scale_factor

    # Adjust final position to match Vega exactly (due to potential scaling imprecision)
    final_adjustment = np.array([vega_x, vega_y, vega_z]) - scaled_trajectory[:, -1]
    adjusted_trajectory = scaled_trajectory + final_adjustment[:, np.newaxis]

    return adjusted_trajectory[0], adjusted_trajectory[1], adjusted_trajectory[2]







axis_limits = [(-3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
               (-6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
               (-70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
               (-50000, 50000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
               (-280000, 280000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
               (-632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
               (-1584188.9811, 1584188.9811, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years'),
               (-1897232.3126, 1897232.3126, 80, 'solar_system_with_nearest_stars_30', 'Interstellar Neighbors Within 30 Light Years')]

stars_data = pd.read_csv('data/nearby_stars_30.csv')  # Correct the path if necessary
# Assuming 'RA_degrees' and 'DEC_degrees' are already processed and included in the CSV
# If not, you'll need to add code here to process them as needed

# Process star data for 3D coordinates
stars_data['RA_deg'], stars_data['DEC_deg'] = zip(*stars_data.apply(lambda row: parse_ra_dec_safe(row['RA'], row['Dec']), axis=1))
stars_data['Distance (AU)'] = stars_data['Distance (ly)'] * 63241  # Convert light-years to AU
stars_data['x'], stars_data['y'], stars_data['z'] = zip(*stars_data.apply(lambda row: ra_dec_to_3d(row['RA_deg'], row['DEC_deg'], row['Distance (AU)']), axis=1))

# Updated PLANET_DATA with colors, diameters (in kilometers), and orbital periods (in days)
PLANET_DATA = {
    'Mercury': {'a': 0.39, 'e': 0.205, 'i': 7, 'color': 'gray', 'diameter': 4879, 'period': 88},
    'Venus': {'a': 0.72, 'e': 0.007, 'i': 3.4, 'color': 'yellow', 'diameter': 12104, 'period': 224.7},
    'Earth': {'a': 1.00, 'e': 0.017, 'i': 0, 'color': 'blue', 'diameter': 12742, 'period': 365.2},
    'Mars': {'a': 1.52, 'e': 0.093, 'i': 1.85, 'color': 'red', 'diameter': 6779, 'period': 687},
    'Jupiter': {'a': 5.20, 'e': 0.048, 'i': 1.3, 'color': 'orange', 'diameter': 139822, 'period': 4331},
    'Saturn': {'a': 9.58, 'e': 0.056, 'i': 2.49, 'color': 'gold', 'diameter': 116464, 'period': 10747},
    'Uranus': {'a': 19.22, 'e': 0.046, 'i': 0.77, 'color': 'lightblue', 'diameter': 50724, 'period': 30589},
    'Neptune': {'a': 30.05, 'e': 0.010, 'i': 1.77, 'color': 'blue', 'diameter': 49244, 'period': 59800},
    'Pluto': {'a': 39.48, 'e': 0.248, 'i': 17.16, 'color': 'brown', 'diameter': 2376, 'period': 90560}
}

JUPITER_SEMI_MAJOR_AXIS = 5.2
JUPITER_INCLINATION = 1.3
JUPITER_ECCENTRICITY = 0.0489

labeled_star_systems = set()
for i, limit in enumerate(axis_limits):

    # Generate 3D plots
    fig = plt.figure(figsize=(39, 39))

    labeled_star_systems.clear()

    ASTEROID_BELT_INNER = 2.2
    ASTEROID_BELT_OUTER = 3.2
    KUIPER_BELT_INNER = 30
    KUIPER_BELT_OUTER = 50
    OORT_CLOUD_INNER = 2000
    OORT_CLOUD_OUTER = 50000
    THICKNESS = 0.05

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
        ASTEROID_BELT_POINTS = 200
        TROJANS_GREEKS_POINTS = 100
        HILDAS_POINTS = 50
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

    elif  limit[3] == 'solar_system_with_nearest_stars_25' or limit[3] == 'solar_system_with_nearest_stars_30':
        ASTEROID_BELT_POINTS = 1
        TROJANS_GREEKS_POINTS = 1
        HILDAS_POINTS = 1
        KUIPER_BELT_POINTS = 10
        OORT_CLOUD_POINTS = 1000

    ax = fig.add_subplot(111, projection='3d')

    ax.scatter([0], [0], [0], color='yellow', s=50)
    ax.view_init(elev=25, azim=120)

    asteroid_x, asteroid_y, asteroid_z = generate_belt_points(ASTEROID_BELT_INNER, ASTEROID_BELT_OUTER, THICKNESS, ASTEROID_BELT_POINTS)
    ax.scatter(asteroid_x, asteroid_y, asteroid_z, color='gray', s=1)

    trojans_x, trojans_y, trojans_z = generate_lagrange_points(JUPITER_SEMI_MAJOR_AXIS, JUPITER_INCLINATION, TROJANS_GREEKS_POINTS, leading=True)
    greeks_x, greeks_y, greeks_z = generate_lagrange_points(JUPITER_SEMI_MAJOR_AXIS, JUPITER_INCLINATION, TROJANS_GREEKS_POINTS, leading=False)
    ax.scatter(trojans_x, trojans_y, trojans_z, color='gray', s=1)
    ax.scatter(greeks_x, greeks_y, greeks_z, color='gray', s=1)

    hildas_x, hildas_y, hildas_z = hildas_cluster_distribution(
    JUPITER_SEMI_MAJOR_AXIS, JUPITER_ECCENTRICITY, JUPITER_INCLINATION, 3, max(int(HILDAS_POINTS / 4), 1))
    ax.scatter(hildas_x, hildas_y, hildas_z, color='gray', s=1)

    # Ensure the indices used are within bounds
    num_points_total = len(hildas_x)
    cluster_indices = [0, num_points_total//3, 2*num_points_total//3]

    bowed_x, bowed_y, bowed_z = hildas_cluster_bands(
        [(hildas_x[i], hildas_y[i], hildas_z[i]) for i in cluster_indices], HILDAS_POINTS, spread_radius=0.5, bow_factor=-1.75)
    ax.scatter(bowed_x, bowed_y, bowed_z, color='gray', s=1)

    kuiper_x, kuiper_y, kuiper_z = generate_belt_points(KUIPER_BELT_INNER, KUIPER_BELT_OUTER, THICKNESS, KUIPER_BELT_POINTS)
    ax.scatter(kuiper_x, kuiper_y, kuiper_z, color='gray', s=1)

    oort_x, oort_y, oort_z = generate_belt_points(OORT_CLOUD_INNER, OORT_CLOUD_OUTER, THICKNESS * 5, OORT_CLOUD_POINTS // 5)  # Less dense for visualization
    ax.scatter(oort_x, oort_y, oort_z, color='gray', s=1)

    for planet, data in PLANET_DATA.items():
        x, y, z = calculate_3d_orbit(data['a'], data['e'], data['i'], 1000)
        ax.plot(x, y, z, color="black")  # Orbit path

        # Select a random index for x, y, z to simulate a random position in the orbit
        if planet == "Jupiter":
            ax.scatter(x[50], y[50], z[50], color=data['color'], s=int(10+(data['diameter']/2500)))
        else:
            random_index = random.randint(-360, 360)
            ax.scatter(x[random_index], y[random_index], z[random_index], color=data['color'], s=int(10+(data['diameter']/2500)))

    
    # Add 'Oumuamua's orbit calculation to your plot
    oumuamua_eccentricity = 1.2
    oumuamua_inclination = 122.74
    oumuamua_semi_major_axis = -1.279

    # Inside your plotting loop, after plotting the stars
    oumuamua_x, oumuamua_y, oumuamua_z = calculate_hyperbolic_orbit_parabolic_segment_3d(oumuamua_eccentricity, oumuamua_semi_major_axis, oumuamua_inclination, 5000)
    ax.plot(oumuamua_x, oumuamua_y, oumuamua_z, '--', color='darkred')
    

    # Plot stars within the current view limit, if applicable
    view_limit = limit[1]  # Assuming this is the maximal distance we're considering in AU
    stars_range = stars_data[stars_data['Distance (AU)'] <= view_limit]
    print(stars_range)
    for index, row in stars_range.iterrows():
        if 'Vega' in row['System']:
            print("Vega: ", row['x'], row['y'], row['z'])
            ax.scatter(row['x'], row['y'], row['z'], color='silver', s=500, alpha=.9)
        else:
            ax.scatter(row['x'], row['y'], row['z'], color='orange', s=80, alpha=.9)


    for index, row in stars_range.iterrows():
        if row['System'][:20] not in labeled_star_systems:
            # Trim and place a text label slightly offset from the star's position
            ax.text(row['x'], row['y'], row['z'], row['System'][:20], fontsize=10, ha='left', va='bottom')
            labeled_star_systems.add(row['System'][:10])


    # Setting axis limits and labels
    ax.set_xlim(limit[0], limit[1])
    ax.set_ylim(limit[0], limit[1])
    ax.set_zlim(limit[0], limit[1])
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.legend()

    plt.title(f'{limit[4]}', fontsize=limit[2], pad=50)
    plt.savefig(f"output/3d/{i}_{limit[3]}.jpg", dpi=300)

    plt.clf()  # Clear the current figure for the next loop iteration

plt.close()
