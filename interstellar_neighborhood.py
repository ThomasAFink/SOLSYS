import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re

# Load the data from the uploaded CSV file
stars_data_path = 'nearby_stars_25.csv'
stars_data = pd.read_csv(stars_data_path)
stars_data = stars_data[stars_data['Distance (ly)'] <= 10]

# Function to parse the right ascension and declination from the data, adjusted to handle non-string inputs
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

# Function to convert spherical coordinates (RA, Dec, Distance) to Cartesian (X, Y, Z)
def spherical_to_cartesian(ra_deg, dec_deg, distance):
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
    x = distance * np.cos(ra_rad) * np.cos(dec_rad)
    y = distance * np.sin(ra_rad) * np.cos(dec_rad)
    z = distance * np.sin(dec_rad)
    return x, y, z

# Exclude the Sun from transformation, then calculate Cartesian coordinates
stars_data = stars_data.iloc[1:].copy()
stars_data['RA_deg'], stars_data['Dec_deg'] = zip(*stars_data.apply(lambda row: parse_ra_dec_safe(row['RA'], row['Dec']), axis=1))
stars_data = stars_data.dropna(subset=['RA_deg', 'Dec_deg'])
stars_data['x'], stars_data['y'], stars_data['z'] = zip(*stars_data.apply(lambda row: spherical_to_cartesian(row['RA_deg'], row['Dec_deg'], row['Distance (ly)']), axis=1))

# Ensure there are no NaN or Inf values in the calculations for axes limits
valid_coordinates = stars_data[['x', 'y', 'z']].dropna()
max_range = np.max(np.abs(valid_coordinates.values))

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('black')

# Plot each star
for index, star in valid_coordinates.iterrows():
    star_name_or_system = stars_data.loc[index, 'Star or (sub-) brown dwarf'] if pd.notnull(stars_data.loc[index, 'Star or (sub-) brown dwarf']) else stars_data.loc[index, 'System']
    color = 'yellow' if 'Sun' in star_name_or_system else 'white'
    ax.scatter(star['x'], star['y'], star['z'], color=color, s=100, depthshade=True, marker='o')
    label = star_name_or_system
    ax.text(star['x'], star['y'], star['z'], label, color='white', fontsize=5, ha='center')

# Place the Sun at the center
ax.scatter(0, 0, 0, color='yellow', s=200, depthshade=True, marker='o')
ax.text(0, 0, 0, "Sun", color='white', fontsize=8, ha='center')

# Adjusting axes limits to center the plot
ax.set_xlim([-max_range, max_range])
ax.set_ylim([-max_range, max_range])
ax.set_zlim([-max_range, max_range])

ax.axis('off')
ax.set_title('3D Interstellar Neighborhood with Spheres', color='white')

plt.show()
