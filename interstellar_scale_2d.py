import random
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_pluto_ellipse(eccentricity, semi_major_axis, theta):
    pluto_b = semi_major_axis * np.sqrt(1 - eccentricity**2)
    pluto_r = (semi_major_axis * (1 - eccentricity**2)) / (1 + eccentricity * np.cos(theta))
    pluto_x = pluto_r * np.cos(theta)
    pluto_y = pluto_r * np.sin(theta)
    return pluto_x, pluto_y

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

def calculate_2d_orbit(semi_major_axis, eccentricity, inclination, num_points=100):
    inclination = np.radians(inclination)
    theta = np.linspace(0, 2 * np.pi, num_points)
    r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(theta))
    x = r * np.cos(theta)
    y = r * np.sin(theta) * np.cos(inclination)
    return x, y

# Load the stars data
stars_data_path = 'data/nearby_stars_25.csv'  # Update this path as necessary
stars_data = pd.read_csv(stars_data_path)

# Convert light-years to AU (approximate; 1 light-year ≈ 63,241 AU)
stars_data['Distance (AU)'] = stars_data['Distance (ly)'] * 63241

stars_data['RA_degrees'] = stars_data['RA'].apply(parse_ra_to_degrees)
stars_data['RA_rad'] = np.radians(stars_data['RA_degrees'])

# Assuming 'Distance (AU)' is already in your data, or convert 'Distance (ly)' to 'Distance (AU)'
stars_data['x'] = np.cos(stars_data['RA_rad']) * stars_data['Distance (AU)']
stars_data['y'] = np.sin(stars_data['RA_rad']) * stars_data['Distance (AU)']

axis_limits = [(-3.5, 3.5, 80, 'inner_solar_system', 'Inner Solar System'),
               (-6, 6, 80, 'inner_solar_system_with_jupiter', 'Inner Solar System With Jupiter'),
               (-70, 70, 80, 'solar_system_with_kuiper_belt', 'Solar System With Kuiper Belt'),
               (-100000, 100000, 80, 'solar_system_with_oort_cloud', 'Solar System With Oort Cloud'),
               (-280000, 125000, 80, 'solar_system_with_alpha_centauri', 'Solar System with Alpha Centauri'),
               (-632410.77088, 632410.77088, 80, 'solar_system_with_nearest_stars_10', 'Interstellar Neighbors Within 10 Light Years'),
               (-1581026.9272, 1581026.9272, 80, 'solar_system_with_nearest_stars_25', 'Interstellar Neighbors Within 25 Light Years')]
labeled_star_systems = set()

for i, limit in enumerate(axis_limits):
    
    labeled_star_systems.clear()

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
    
    ORBIT_POINTS = 1000
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
        HILDAS_POINTS = 100
        KUIPER_BELT_POINTS = 10000
        OORT_CLOUD_POINTS = 50000

    elif limit[3] == 'solar_system_with_kuiper_belt':
        ASTEROID_BELT_POINTS = 500
        TROJANS_GREEKS_POINTS = 20
        HILDAS_POINTS = 15
        KUIPER_BELT_POINTS = 10000
        OORT_CLOUD_POINTS = 50000
    
    elif limit[3] == 'solar_system_with_oort_cloud':
        ASTEROID_BELT_POINTS = 20
        TROJANS_GREEKS_POINTS = 10
        HILDAS_POINTS = 100
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

    asteroid_belt_r = np.random.uniform(ASTEROID_BELT_INNER, ASTEROID_BELT_OUTER, ASTEROID_BELT_POINTS)
    asteroid_belt_theta = np.random.uniform(0, 2 * np.pi, ASTEROID_BELT_POINTS)
    asteroid_belt_x = asteroid_belt_r * np.cos(asteroid_belt_theta)
    asteroid_belt_y = asteroid_belt_r * np.sin(asteroid_belt_theta)
    trojans_r = np.random.uniform(JUPITER_SEMI_MAJOR_AXIS - TROJANS_GREEKS_WIDTH, JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH, TROJANS_GREEKS_POINTS)
    greeks_r = np.random.uniform(JUPITER_SEMI_MAJOR_AXIS - TROJANS_GREEKS_WIDTH, JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH, TROJANS_GREEKS_POINTS)
    trojans_theta = np.linspace(TROJANS_GREEKS_ANGLE - TROJANS_GREEKS_SPREAD / 2, TROJANS_GREEKS_ANGLE + TROJANS_GREEKS_SPREAD / 2, TROJANS_GREEKS_POINTS)
    greeks_theta = np.linspace(TROJANS_GREEKS_ANGLE + np.pi - TROJANS_GREEKS_SPREAD / 2, TROJANS_GREEKS_ANGLE + np.pi + TROJANS_GREEKS_SPREAD / 2, TROJANS_GREEKS_POINTS) + 20.15
    trojans_x = trojans_r * np.cos(trojans_theta)
    trojans_y = trojans_r * np.sin(trojans_theta)
    greeks_x = greeks_r * np.cos(greeks_theta)
    greeks_y = greeks_r * np.sin(greeks_theta)
    theta = np.linspace(0, 2 * np.pi, ORBIT_POINTS)
    pluto_x, pluto_y = calculate_pluto_ellipse(PLUTO_ECCENTRICITY, PLUTO_SEMI_MAJOR_AXIS, theta)
    kuiper_belt_r = np.random.uniform(KUIPER_BELT_INNER, KUIPER_BELT_OUTER, KUIPER_BELT_POINTS)
    kuiper_belt_theta = np.random.uniform(0, 2 * np.pi, KUIPER_BELT_POINTS)
    kuiper_belt_x = kuiper_belt_r * np.cos(kuiper_belt_theta)
    kuiper_belt_y = kuiper_belt_r * np.sin(kuiper_belt_theta)
    oort_cloud_r = np.random.uniform(OORT_CLOUD_INNER, OORT_CLOUD_OUTER, OORT_CLOUD_POINTS)
    oort_cloud_theta = np.random.uniform(0, 2 * np.pi, OORT_CLOUD_POINTS)
    oort_cloud_x = oort_cloud_r * np.cos(oort_cloud_theta)
    oort_cloud_y = oort_cloud_r * np.sin(oort_cloud_theta)

    fig, ax = plt.subplots(figsize=(39, 39))

    if limit[3] == 'inner_solar_system_with_jupiter' or limit[3] == 'inner_solar_system':
        ax.plot(0, 0, 'o', markersize=75, color='yellow')
    else:
        ax.plot(0, 0, 'o', markersize=8, color='yellow')

    for planet, data in PLANET_DATA.items():
        x, y = calculate_2d_orbit(data['a'], data['e'], data['i'], 1000)
        ax.plot(x, y, color="black")  # Orbit path

        # Select a random index for x, y, z to simulate a random position in the orbit
        if planet == "Jupiter":
            if limit[3] == 'inner_solar_system_with_jupiter' or limit[3] == 'inner_solar_system':
                ax.scatter(x[50], y[50], color=data['color'], s=int(10+(data['diameter']/100)))
            else:
                ax.scatter(x[50], y[50], color=data['color'], s=int(10+(data['diameter']/1000)))
        else:
            random_index = random.randint(-360, 360)
            if limit[3] == 'inner_solar_system_with_jupiter' or limit[3] == 'inner_solar_system':
                ax.scatter(x[random_index], y[random_index], color=data['color'], s=int(10+(data['diameter']/100)))
            else:
                ax.scatter(x[random_index], y[random_index], color=data['color'], s=int(10+(data['diameter']/1000)))

        
        ax.plot(pluto_x, pluto_y, color='blue')
        ax.plot(PLUTO_PERIHELION, 0, 'bo')
        ax.plot(-PLUTO_APHELION, 0, 'bo')
        ax.scatter(asteroid_belt_x, asteroid_belt_y, color='gray', s=5)
        ax.scatter(trojans_x, trojans_y, color='gray', s=5)
        ax.scatter(greeks_x, greeks_y, color='gray', s=5)

        hildas_x, hildas_y, hildas_z = hildas_cluster_distribution(
        JUPITER_SEMI_MAJOR_AXIS, JUPITER_ECCENTRICITY, JUPITER_INCLINATION, 3, max(int(HILDAS_POINTS / 4), 1))
        ax.scatter(hildas_x, hildas_y, color='lightgray', s=5)

        # Ensure the indices used are within bounds
        num_points_total = len(hildas_x)
        cluster_indices = [0, num_points_total//3, 2*num_points_total//3]

        bowed_x, bowed_y, bowed_z = hildas_cluster_bands(
            [(hildas_x[i], hildas_y[i], hildas_z[i]) for i in cluster_indices], HILDAS_POINTS, spread_radius=0.5, bow_factor= -1.75)     
        ax.scatter(bowed_x, bowed_y, color='lightgray', s=5)
    
    ax.scatter(kuiper_belt_x, kuiper_belt_y, color='gray', s=5)   
    ax.scatter(oort_cloud_x, oort_cloud_y, color='gray', s=5)  # Oort Cloud
    if limit[3] == 'solar_system_with_alpha_centauri' or limit[3] == 'solar_system_with_nearest_stars_10' or limit[3] == 'solar_system_with_nearest_stars_25':
        
        if limit[3] == 'solar_system_with_nearest_stars_10':
            stars_range = stars_data[stars_data['Distance (ly)'] <= 10]
        elif limit[3] == 'solar_system_with_nearest_stars_25':
            stars_range = stars_data[stars_data['Distance (ly)'] <= 25]
        else:
            stars_range = stars_data[stars_data['Distance (ly)'] <= 5]
        
        for index, row in stars_range.iterrows():
            # Plot each star as an orange dot
            ax.plot(row['x'], row['y'], 'o', markersize=15, color='orange')
            
            if row['System'][:20] not in labeled_star_systems:
                ax.text(row['x'] + 0.01 * (limit[1] - limit[0]), row['y'], row['System'][:20], fontsize=20, ha='left', va='center')
                labeled_star_systems.add(row['System'])

    ax.set_xlim(limit[0], limit[1])
    ax.set_ylim(limit[0], limit[1])
    ax.set_aspect('equal', 'box')
    ax.axis('off')
    
    font_size = 48
    if limit[3] == 'inner_solar_system':
        ax.annotate('Asteroid Belt (2.2-3.2 AU)', xy=(ASTEROID_BELT_OUTER, 0), xytext=(ASTEROID_BELT_INNER+0.1, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
    if limit[3] == 'inner_solar_system_with_jupiter':
        ax.annotate('Asteroid Belt (2.2-3.2 AU)', xy=(ASTEROID_BELT_OUTER, 0), xytext=(ASTEROID_BELT_INNER+0.1, 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate('Hildas', xy=(-HILDAS_OUTER, 0), xytext=(-HILDAS_OUTER-1, -2.5),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate('Trojans', xy=(3, -(JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH)+1), xytext=(2, -(JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH)-1),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate('Greeks', xy=(4, (JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH)-1.75), xytext=(3, (JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH)),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)        
    if limit[3] == 'solar_system_with_kuiper_belt':
        ax.annotate('Kuiper Belt (50 AU)', xy=(KUIPER_BELT_OUTER, 0), xytext=(KUIPER_BELT_OUTER+5, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate("Pluto's aphelion (49.5 AU)", xy=(-PLUTO_APHELION, 0), xytext=(-PLUTO_APHELION-25, 10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate("Pluto's perihelion (29.7 AU)", xy=(PLUTO_PERIHELION, 0), xytext=(PLUTO_PERIHELION+10, -10),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
    if limit[3] == 'solar_system_with_oort_cloud':
        ax.annotate('Kuiper Belt (50 AU)', xy=(-KUIPER_BELT_OUTER+2500, 3500), xytext=(KUIPER_BELT_OUTER-90000, -90000),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate('Oort Cloud (100000 AU)', xy=(100000, 5), xytext=(70000, 25000),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)   
    if limit[3] == 'solar_system_with_alpha_centauri':
        ax.annotate('Kuiper Belt (50 AU)', xy=(-KUIPER_BELT_OUTER+1000, 3500), xytext=(KUIPER_BELT_OUTER-90000, -125000),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)
        ax.annotate('Oort Cloud (100000 AU)', xy=(-100000, 5), xytext=(-180000, -25000),
            arrowprops=dict(facecolor='black', shrink=0.05), fontsize=font_size)  

    plt.title(f'{limit[4]}', fontsize=limit[2], pad=50)
    plt.savefig(f"output/2d/{i}_{limit[3]}.jpg", dpi=300)

plt.close(fig)