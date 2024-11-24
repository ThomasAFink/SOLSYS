import random
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os

@dataclass
class PlanetData:
    """Data structure for planet properties"""
    semi_major_axis: float  # a
    eccentricity: float    # e
    inclination: float     # i
    color: str
    diameter: int
    period: float

class AstronomicalConstants:
    """Constants for astronomical calculations"""
    PLUTO_PERIHELION = 29.7
    PLUTO_APHELION = 49.5
    ASTEROID_BELT_INNER = 2.2
    ASTEROID_BELT_OUTER = 3.2
    KUIPER_BELT_INNER = 30
    KUIPER_BELT_OUTER = 50
    JUPITER_SEMI_MAJOR_AXIS = 5.2
    JUPITER_INCLINATION = 1.3
    JUPITER_ECCENTRICITY = 0.0489
    OORT_CLOUD_INNER = 2000
    OORT_CLOUD_OUTER = 100000
    LIGHT_YEAR_TO_AU = 63241
    OUMUAMUA_ECCENTRICITY = 1.2
    OUMUAMUA_SEMI_MAJOR_AXIS = -1.279
    OUMUAMUA_INCLINATION = 122.74

class OrbitalMechanics:
    
    def parse_ra_to_degrees(ra_str: str) -> Optional[float]:
        """Convert Right Ascension string to degrees"""
        if not isinstance(ra_str, str):
            return None
        match = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', ra_str)
        if not match:
            return None
        hours, minutes, seconds = map(float, match.groups())
        return 15 * (hours + minutes / 60 + seconds / 3600)

    def calculate_2d_orbit(semi_major_axis: float, eccentricity: float, 
                        inclination: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 2D orbital coordinates"""
        inclination_rad = np.radians(inclination)
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(inclination_rad)
        return x, y

    def calculate_hyperbolic_orbit(eccentricity: float, semi_major_axis: float, 
                                inclination: float, vega_ra: str = "18h 36m 56s",
                                num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate and rotate hyperbolic orbit to point towards Vega
        """
        # Calculate initial hyperbolic orbit
        inclination_rad = np.radians(inclination)
        theta = np.linspace(-np.arccos(-1/eccentricity) + 0.0000000001,
                        np.arccos(-1/eccentricity) - 0.0000000001, num_points)
        r = semi_major_axis * (1 - eccentricity**2) / (1 + eccentricity * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(inclination_rad)

        # Calculate Vega's position
        vega_ra_deg = OrbitalMechanics.parse_ra_to_degrees(vega_ra)
        vega_distance_au = 25.04 * 63241  # Convert light years to AU
        vega_x = vega_distance_au * np.cos(np.radians(vega_ra_deg))
        vega_y = vega_distance_au * np.sin(np.radians(vega_ra_deg))

        # Calculate rotation angle to align with Vega
        oumuamua_direction = np.array([x[-1], y[-1]])  # Direction at the last point
        vega_direction = np.array([vega_x, vega_y])  # Direction to Vega
        rotation_angle = np.arctan2(vega_direction[1], vega_direction[0]) - np.arctan2(oumuamua_direction[1], oumuamua_direction[0])

        # Apply rotation
        cos_angle = np.cos(rotation_angle)
        sin_angle = np.sin(rotation_angle)
        x_rotated = x * cos_angle - y * sin_angle
        y_rotated = x * sin_angle + y * cos_angle

        return x_rotated, y_rotated


class SolarSystemVisualizer:
    def __init__(self, stars_data_path: str):
        self.constants = AstronomicalConstants()
        self.stars_data = self._load_stars_data(stars_data_path)
        self.planet_data = self._init_planet_data()
        self.labeled_star_systems = set()
        
    def _load_stars_data(self, path: str) -> pd.DataFrame:
        """Load and process stars data"""
        df = pd.read_csv(path)
        df['Distance (AU)'] = df['Distance (ly)'] * self.constants.LIGHT_YEAR_TO_AU
        df['RA_degrees'] = df['RA'].apply(OrbitalMechanics.parse_ra_to_degrees)
        df['RA_rad'] = np.radians(df['RA_degrees'])
        df['x'] = np.cos(df['RA_rad']) * df['Distance (AU)']
        df['y'] = np.sin(df['RA_rad']) * df['Distance (AU)']
        return df

    def _init_planet_data(self) -> Dict[str, PlanetData]:
        """Initialize planet data"""
        return {
            'Mercury': PlanetData(0.39, 0.205, 7, 'gray', 4879, 88),
            'Venus': PlanetData(0.72, 0.007, 3.4, 'yellow', 12104, 224.7),
            'Earth': PlanetData(1.00, 0.017, 0, 'blue', 12742, 365.2),
            'Mars': PlanetData(1.52, 0.093, 1.85, 'red', 6779, 687),
            'Jupiter': PlanetData(5.20, 0.048, 1.3, 'orange', 139822, 4331),
            'Saturn': PlanetData(9.58, 0.056, 2.49, 'gold', 116464, 10747),
            'Uranus': PlanetData(19.22, 0.046, 0.77, 'lightblue', 50724, 30589),
            'Neptune': PlanetData(30.05, 0.010, 1.77, 'blue', 49244, 59800),
            'Pluto': PlanetData(39.48, 0.248, 17.16, 'brown', 2376, 90560)
        }

    def _calculate_points_density(self, view_type: str) -> Dict[str, int]:
        """Calculate point density based on view type"""
        base_densities = {
            '0_inner_solar_system': (20000, 4000, 4000, 10000, 50000),
            '1_inner_solar_system_with_jupiter': (10000, 2000, 1000, 10000, 50000),
            '2_solar_system_with_kuiper_belt': (500, 20, 15, 10000, 50000),
            '3_solar_system_with_oort_cloud': (20, 10, 100, 100, 50000),
            '4_solar_system_with_alpha_centauri': (10, 5, 5, 50, 5000),
            '5_solar_system_with_nearest_stars_10': (2, 2, 2, 20, 2000),
            'default': (1, 1, 1, 10, 1000)
        }
        
        densities = base_densities.get(view_type, base_densities['default'])
        return {
            'asteroid_belt': densities[0],
            'trojans_greeks': densities[1],
            'hildas': densities[2],
            'kuiper_belt': densities[3],
            'oort_cloud': densities[4]
        }

    def _plot_sun(self, ax: plt.Axes, view_type: str):
        """Plot the Sun"""
        markersize = 75 if view_type in ['0_inner_solar_system', '1_inner_solar_system_with_jupiter'] else 8
        ax.plot(0, 0, 'o', markersize=markersize, color='yellow')

    def _plot_planets(self, ax: plt.Axes, view_type: str):
        """Plot planets and their orbits"""
        for name, data in self.planet_data.items():
            x, y = OrbitalMechanics.calculate_2d_orbit(data.semi_major_axis, data.eccentricity, data.inclination, 1000)
            ax.plot(x, y, color="black")  # Orbit path

            scale_factor = 100 if view_type in ['0_inner_solar_system', '1_inner_solar_system_with_jupiter'] else 1000
            marker_size = int(10 + (data.diameter / scale_factor))

            if name == "Jupiter":
                ax.scatter(x[50], y[50], color=data.color, s=marker_size)
            else:
                random_index = random.randint(-360, 360)
                ax.scatter(x[random_index], y[random_index], color=data.color, s=marker_size)

    def _plot_hildas_group(self, ax: plt.Axes, points: int):
        """Plot Hilda asteroids in their characteristic triangular configuration"""
        HILDAS_INNER = self.constants.ASTEROID_BELT_OUTER + 0.25
        HILDAS_OUTER = self.constants.JUPITER_SEMI_MAJOR_AXIS - 0.25
        
        # Calculate coordinates for triangle vertices
        cluster_points = max(points // 3, 1)  # Points per cluster
        
        # For each vertex and connecting segments
        angles = np.array([0, 2*np.pi/3, 4*np.pi/3])  # Three main cluster positions
        
        # Plot main clusters
        for angle in angles:
            # Create a cluster at each vertex
            r = np.random.uniform(HILDAS_INNER, HILDAS_OUTER, cluster_points)
            spread = np.pi/12  # Tighter spread (15 degrees) around cluster centers
            theta = angle + np.random.normal(0, spread, cluster_points)
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            ax.scatter(x, y, color='#AAAAAA', s=5)
            
            # Add connecting segments between vertices
            segment_points = cluster_points
            for i in range(segment_points):
                # Random point along the connecting line
                t = np.random.uniform(0, 1)
                r = np.random.uniform(HILDAS_INNER, HILDAS_OUTER)
                
                # Get position along the segment
                next_angle = angles[(np.where(angles == angle)[0][0] + 1) % 3]
                theta = angle + t * (next_angle - angle)
                
                # Add some spread perpendicular to the segment
                spread_distance = np.random.normal(0, 0.2)  # Perpendicular spread
                
                # Calculate base position
                x = r * np.cos(theta)
                y = r * np.sin(theta)
                
                # Add perpendicular spread
                perp_angle = theta + np.pi/2
                x += spread_distance * np.cos(perp_angle)
                y += spread_distance * np.sin(perp_angle)
                
                ax.scatter(x, y, color='#AAAAAA', s=5)

    def _plot_belts_and_clouds(self, ax: plt.Axes, points: Dict[str, int]):
        """Plot all asteroid groups, belts, and clouds"""
        # Asteroid Belt
        r = np.random.uniform(self.constants.ASTEROID_BELT_INNER, 
                            self.constants.ASTEROID_BELT_OUTER, 
                            points['asteroid_belt'])
        theta = np.random.uniform(0, 2 * np.pi, points['asteroid_belt'])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter(x, y, color='gray', s=5)

        # Trojans and Greeks (Jupiter's Lagrange points)
        TROJANS_GREEKS_ANGLE = np.deg2rad(60)
        TROJANS_GREEKS_SPREAD = np.pi / 3
        TROJANS_GREEKS_WIDTH = 0.5

        trojans_r = np.random.uniform(self.constants.JUPITER_SEMI_MAJOR_AXIS - TROJANS_GREEKS_WIDTH,
                                    self.constants.JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH,
                                    points['trojans_greeks'])
        greeks_r = np.random.uniform(self.constants.JUPITER_SEMI_MAJOR_AXIS - TROJANS_GREEKS_WIDTH,
                                    self.constants.JUPITER_SEMI_MAJOR_AXIS + TROJANS_GREEKS_WIDTH,
                                    points['trojans_greeks'])

        # Calculate positions
        trojans_theta = np.linspace(TROJANS_GREEKS_ANGLE - TROJANS_GREEKS_SPREAD / 2,
                                TROJANS_GREEKS_ANGLE + TROJANS_GREEKS_SPREAD / 2,
                                points['trojans_greeks'])
        greeks_theta = np.linspace(TROJANS_GREEKS_ANGLE + np.pi - TROJANS_GREEKS_SPREAD / 2,
                                TROJANS_GREEKS_ANGLE + np.pi + TROJANS_GREEKS_SPREAD / 2,
                                points['trojans_greeks']) + 20.15

        # Plot Trojans and Greeks
        ax.scatter(trojans_r * np.cos(trojans_theta), 
                trojans_r * np.sin(trojans_theta),
                color='gray', s=5)
        ax.scatter(greeks_r * np.cos(greeks_theta), 
                greeks_r * np.sin(greeks_theta),
                color='gray', s=5)

        # Plot Hildas
        self._plot_hildas_group(ax, points['hildas'])

        # Kuiper Belt
        r = np.random.uniform(self.constants.KUIPER_BELT_INNER, 
                            self.constants.KUIPER_BELT_OUTER, 
                            points['kuiper_belt'])
        theta = np.random.uniform(0, 2 * np.pi, points['kuiper_belt'])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter(x, y, color='gray', s=5)

        # Oort Cloud
        r = np.random.uniform(self.constants.OORT_CLOUD_INNER, 
                            self.constants.OORT_CLOUD_OUTER, 
                            points['oort_cloud'])
        theta = np.random.uniform(0, 2 * np.pi, points['oort_cloud'])
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        ax.scatter(x, y, color='gray', s=5)

        # Plot Oumuamua's orbit with rotation towards Vega
        x, y = OrbitalMechanics.calculate_hyperbolic_orbit(
            self.constants.OUMUAMUA_ECCENTRICITY,
            self.constants.OUMUAMUA_SEMI_MAJOR_AXIS,
            self.constants.OUMUAMUA_INCLINATION,
            vega_ra="18h 36m 56s"  # Vega's right ascension
        )
        ax.plot(x, y, '--', color='darkred')
        
    def _get_view_limits(self, view_type: str) -> Tuple[float, float]:
        """Get view limits for the given view type"""
        view_limits = {
            '0_inner_solar_system': (-3.5, 3.5),
            '1_inner_solar_system_with_jupiter': (-6, 6),
            '2_solar_system_with_kuiper_belt': (-70, 70),
            '3_solar_system_with_oort_cloud': (-100000, 100000),
            '4_solar_system_with_alpha_centauri': (-280000, 125000),
            '5_solar_system_with_nearest_stars_10': (-632410.77088, 632410.77088),
            '6_solar_system_with_nearest_stars_25': (-1584189.9811, 1584189.9811),
            '7_solar_system_with_nearest_stars_30': (-1897232.3126, 1897232.3126)
        }
        return view_limits.get(view_type, (-3.5, 3.5))

    def _plot_nearby_stars(self, ax: plt.Axes, view_type: str):
        """Plot nearby stars based on view type"""
        max_distance = {
            '5_solar_system_with_nearest_stars_10': 10,
            '6_solar_system_with_nearest_stars_25': 25.05,
            '7_solar_system_with_nearest_stars_30': 30,
            '4_solar_system_with_alpha_centauri': 5
        }.get(view_type, 25.05)

        stars_range = self.stars_data[self.stars_data['Distance (ly)'] <= max_distance]
        
        for _, row in stars_range.iterrows():
            # Plot star
            if row['System'].startswith('Vega'):
                ax.plot(row['x'], row['y'], 'o', markersize=30, color='silver')
            else:
                ax.plot(row['x'], row['y'], 'o', markersize=15, color='orange')

            # Add label with offset based on view scale
            limits = self._get_view_limits(view_type)
            offset = 0.01 * (limits[1] - limits[0])  # Calculate offset based on view scale
            
            ax.text(row['x'] + offset, row['y'], row['System'][:20], 
                   fontsize=20, ha='left', va='center')

    def _set_plot_properties(self, ax: plt.Axes, view_type: str):
        """Set plot properties based on view type"""
        view_limits = {
            '0_inner_solar_system': (-3.5, 3.5),
            '1_inner_solar_system_with_jupiter': (-6, 6),
            '2_solar_system_with_kuiper_belt': (-70, 70),
            '3_solar_system_with_oort_cloud': (-100000, 100000),
            '4_solar_system_with_alpha_centauri': (-280000, 125000),
            '5_solar_system_with_nearest_stars_10': (-632410.77088, 632410.77088),
            '6_solar_system_with_nearest_stars_25': (-1584189.9811, 1584189.9811),
            '7_solar_system_with_nearest_stars_30': (-1897232.3126, 1897232.3126)
        }
        
        limits = view_limits.get(view_type, (-3.5, 3.5))
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        
        title_map = {
            '0_inner_solar_system': 'Inner Solar System',
            '1_inner_solar_system_with_jupiter': 'Inner Solar System With Jupiter',
            '2_solar_system_with_kuiper_belt': 'Solar System With Kuiper Belt',
            '3_solar_system_with_oort_cloud': 'Solar System With Oort Cloud',
            '4_solar_system_with_alpha_centauri': 'Solar System with Alpha Centauri',
            '5_solar_system_with_nearest_stars_10': 'Interstellar Neighbors Within 10 Light Years',
            '6_solar_system_with_nearest_stars_25': 'Interstellar Neighbors Within 25 Light Years',
            '7_solar_system_with_nearest_stars_30': 'Interstellar Neighbors Within 30 Light Years'
        }
        
        plt.title(title_map.get(view_type, 'Solar System'), fontsize=80, pad=50)
 
 
    def _add_labels(self, ax: plt.Axes, view_type: str):
        """Add labels to solar system features based on view type"""
        font_size = 48  # Base font size for labels
        
        label_configs = {
            '0_inner_solar_system': [
                ('Asteroid Belt (2.2-3.2 AU)', (self.constants.ASTEROID_BELT_OUTER, 0), 
                (self.constants.ASTEROID_BELT_INNER+0.1, 1.5)),
                ('Oumuamua Orbit', (-1, -0.90), (-3, -0.5))
            ],
            '1_inner_solar_system_with_jupiter': [
                ('Asteroid Belt (2.2-3.2 AU)', (self.constants.ASTEROID_BELT_OUTER, 0), 
                (self.constants.ASTEROID_BELT_INNER+0.1, 2)),
                ('Hildas', (-self.constants.JUPITER_SEMI_MAJOR_AXIS+0.5, 0), 
                (-self.constants.JUPITER_SEMI_MAJOR_AXIS-1, -2.5)),
                ('Trojans', (3, -(self.constants.JUPITER_SEMI_MAJOR_AXIS + 1)+1), 
                (2, -(self.constants.JUPITER_SEMI_MAJOR_AXIS + 1)-1)),
                ('Greeks', (4, (self.constants.JUPITER_SEMI_MAJOR_AXIS + 1)-1.75), 
                (3, (self.constants.JUPITER_SEMI_MAJOR_AXIS + 1))),
                ('Oumuamua Orbit', (-3.5, -5), (-6, -4.5))
            ],
            '2_solar_system_with_kuiper_belt': [
                ('Kuiper Belt (50 AU)', (self.constants.KUIPER_BELT_OUTER, 0), 
                (self.constants.KUIPER_BELT_OUTER+5, 10)),
                ("Pluto's aphelion (49.5 AU)", (-self.constants.PLUTO_APHELION, 0), 
                (-self.constants.PLUTO_APHELION-25, 10)),
                ("Pluto's perihelion (29.7 AU)", (self.constants.PLUTO_PERIHELION, 0), 
                (self.constants.PLUTO_PERIHELION+10, -10)),
                ('Oumuamua Orbit', (-34, -55), (-60, -45))
            ],
            '3_solar_system_with_oort_cloud': [
                ('Kuiper Belt (50 AU)', (self.constants.KUIPER_BELT_OUTER-3500, -4000),
                (self.constants.KUIPER_BELT_OUTER+80000, 90000)),
                ('Oort Cloud (100000 AU)', (100000, 5), (70000, 25000)),
                ('Oumuamua Orbit', (-60000, -100000), (-110000, -110000))
            ],
            '4_solar_system_with_alpha_centauri': [
                ('Kuiper Belt (50 AU)', (self.constants.KUIPER_BELT_OUTER-3500, -4000),
                (self.constants.KUIPER_BELT_OUTER+80000, 110000)),
                ('Oort Cloud (100000 AU)', (-100000, 5), (-180000, 25000)),
                ('Oumuamua Orbit', (-65000, -110000), (-180000, -130000))
            ]
        }
        
        if view_type in label_configs:
            for label_text, xy, xytext in label_configs[view_type]:
                ax.annotate(
                    label_text, 
                    xy=xy, 
                    xytext=xytext,
                    fontsize=font_size,
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )

    def create_visualization(self, view_type: str, save_path: str):
        """Create and save solar system visualization"""
        fig, ax = plt.subplots(figsize=(39, 39))
        points = self._calculate_points_density(view_type)
        
        self._plot_sun(ax, view_type)
        self._plot_planets(ax, view_type)
        self._plot_belts_and_clouds(ax, points)
        
        if 'nearest_stars' in view_type or 'alpha_centauri' in view_type:
            self._plot_nearby_stars(ax, view_type)
        
        self._add_labels(ax, view_type)  # Add labels after plotting all elements
        self._set_plot_properties(ax, view_type)
        
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    
     
def create_output_directory():
    """Create output directory if it doesn't exist"""
    os.makedirs('output/2d', exist_ok=True)

def generate_all_views(visualizer):
    """Generate all different scale views of the solar system"""
    views = [
        ('0_inner_solar_system', 'Inner Solar System (±3.5 AU)'),
        ('1_inner_solar_system_with_jupiter', 'Solar System to Jupiter (±6 AU)'),
        ('2_solar_system_with_kuiper_belt', 'Solar System with Kuiper Belt (±70 AU)'),
        ('3_solar_system_with_oort_cloud', 'Solar System with Oort Cloud (±100,000 AU)'),
        ('4_solar_system_with_alpha_centauri', 'Local Space with Alpha Centauri (±280,000 AU)'),
        ('5_solar_system_with_nearest_stars_10', 'Stars within 10 Light Years (±632,410 AU)'),
        ('6_solar_system_with_nearest_stars_25', 'Stars within 25 Light Years (±1,584,190 AU)'),
        ('7_solar_system_with_nearest_stars_30', 'Stars within 30 Light Years (±1,897,232 AU)')
    ]
    
    for view_type, description in views:
        output_path = f'output/2d/{view_type}.jpg'
        print(f'Generating {description}...')
        visualizer.create_visualization(view_type, output_path)
        print(f'Saved to {output_path}')

# Usage example
if __name__ == "__main__":
    visualizer = SolarSystemVisualizer('data/nearby_stars_30.csv')
    visualizer.create_visualization('0_inner_solar_system', 'output/2d/0_inner_solar_system.jpg')
    

    create_output_directory()
    visualizer = SolarSystemVisualizer('data/nearby_stars_30.csv')
    generate_all_views(visualizer)
    print("All visualizations completed!")