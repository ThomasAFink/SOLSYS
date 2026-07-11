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
    PLUTO_SEMI_MAJOR_AXIS = 39.482
    PLUTO_ECCENTRICITY = 0.2488
    PLUTO_PERIHELION = PLUTO_SEMI_MAJOR_AXIS * (1 - PLUTO_ECCENTRICITY)  # ~29.66 AU
    PLUTO_APHELION = PLUTO_SEMI_MAJOR_AXIS * (1 + PLUTO_ECCENTRICITY)    # ~49.31 AU
    ASTEROID_BELT_INNER = 2.2
    ASTEROID_BELT_OUTER = 3.2
    KUIPER_BELT_INNER = 30
    KUIPER_BELT_OUTER = 55
    JUPITER_SEMI_MAJOR_AXIS = 5.2
    JUPITER_INCLINATION = 1.3
    JUPITER_ECCENTRICITY = 0.0489
    OORT_CLOUD_INNER = 2000
    OORT_CLOUD_OUTER = 100000
    LIGHT_YEAR_TO_AU = 63241.077
    OUMUAMUA_ECCENTRICITY = 1.2011
    OUMUAMUA_PERIHELION = 0.2559
    OUMUAMUA_INCLINATION = 122.74
    OUMUAMUA_LONGITUDE_ASCENDING_NODE = 24.60
    OUMUAMUA_ARGUMENT_OF_PERIHELION = 241.69

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

    def parse_ra_dec(ra_str: str, dec_str: str) -> Tuple[Optional[float], Optional[float]]:
        """Convert RA/Dec strings to degrees (J2000 equatorial)."""
        if not isinstance(ra_str, str) or not isinstance(dec_str, str):
            return None, None
        ra_match = re.match(r'(\d+)h\s*(\d+)m\s*(\d+(?:\.\d*)?)s', ra_str)
        dec_normalized = dec_str.replace('−', '-').replace('–', '-')
        dec_match = re.match(r'([+-]?\d+)°\s*(\d+)′\s*(\d+(?:\.\d*)?)″', dec_normalized)
        if not ra_match or not dec_match:
            return None, None
        ra = 15 * (float(ra_match.group(1)) + float(ra_match.group(2)) / 60 +
                   float(ra_match.group(3)) / 3600)
        dec_sign = -1 if dec_match.group(1).startswith('-') else 1
        dec = dec_sign * (abs(float(dec_match.group(1))) + float(dec_match.group(2)) / 60 +
                            float(dec_match.group(3)) / 3600)
        return ra, dec

    def ra_dec_to_cartesian(ra_deg: float, dec_deg: float, distance_au: float) -> Tuple[float, float, float]:
        """Convert equatorial spherical coordinates to heliocentric Cartesian AU."""
        ra_rad = np.radians(ra_deg)
        dec_rad = np.radians(dec_deg)
        x = distance_au * np.cos(dec_rad) * np.cos(ra_rad)
        y = distance_au * np.cos(dec_rad) * np.sin(ra_rad)
        z = distance_au * np.sin(dec_rad)
        return x, y, z

    def calculate_2d_orbit(semi_major_axis: float, eccentricity: float, 
                        inclination: float, num_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate 2D orbital coordinates"""
        inclination_rad = np.radians(inclination)
        theta = np.linspace(0, 2 * np.pi, num_points)
        r = semi_major_axis * (1 - eccentricity ** 2) / (1 + eccentricity * np.cos(theta))
        x = r * np.cos(theta)
        y = r * np.sin(theta) * np.cos(inclination_rad)
        return x, y

    def calculate_hyperbolic_orbit(perihelion: float, eccentricity: float,
                                 inclination: float, longitude_ascending_node: float,
                                 argument_of_perihelion: float,
                                 num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate a hyperbolic trajectory from standard orbital elements."""
        x, y, z = OrbitalMechanics.calculate_hyperbolic_orbit_3d(
            perihelion, eccentricity, inclination,
            longitude_ascending_node, argument_of_perihelion, num_points
        )
        return x, y, z

    def calculate_hyperbolic_orbit_3d(perihelion: float, eccentricity: float,
                                      inclination: float, longitude_ascending_node: float,
                                      argument_of_perihelion: float,
                                      num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Heliocentric ecliptic coordinates for e > 1, using true anomaly."""
        i = np.radians(inclination)
        Omega = np.radians(longitude_ascending_node)
        omega = np.radians(argument_of_perihelion)
        nu_max = np.arccos(-1 / eccentricity) - 1e-6
        nu = np.linspace(-nu_max, nu_max, num_points)
        r = perihelion * (1 + eccentricity) / (1 + eccentricity * np.cos(nu))

        x_p = r * np.cos(nu)
        y_p = r * np.sin(nu)

        # Rotate perifocal frame into ecliptic coordinates (Rz(Omega) Rx(i) Rz(omega))
        x = ((np.cos(Omega) * np.cos(omega) - np.sin(Omega) * np.sin(omega) * np.cos(i)) * x_p +
             (-np.cos(Omega) * np.sin(omega) - np.sin(Omega) * np.cos(omega) * np.cos(i)) * y_p)
        y = ((np.sin(Omega) * np.cos(omega) + np.cos(Omega) * np.sin(omega) * np.cos(i)) * x_p +
             (-np.sin(Omega) * np.sin(omega) + np.cos(Omega) * np.cos(omega) * np.cos(i)) * y_p)
        z = (np.sin(omega) * np.sin(i) * x_p + np.cos(omega) * np.sin(i) * y_p)
        return x, y, z


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
        ra_dec = df.apply(lambda row: OrbitalMechanics.parse_ra_dec(row['RA'], row['Dec']), axis=1)
        df['RA_degrees'] = ra_dec.apply(lambda pair: pair[0])
        df['Dec_degrees'] = ra_dec.apply(lambda pair: pair[1])
        coords = df.apply(
            lambda row: OrbitalMechanics.ra_dec_to_cartesian(
                row['RA_degrees'], row['Dec_degrees'], row['Distance (AU)']
            ) if pd.notna(row['RA_degrees']) and pd.notna(row['Dec_degrees']) else (np.nan, np.nan, np.nan),
            axis=1
        )
        df['x'] = coords.apply(lambda c: c[0])
        df['y'] = coords.apply(lambda c: c[1])
        df['z'] = coords.apply(lambda c: c[2])
        return df

    def _init_planet_data(self) -> Dict[str, PlanetData]:
        """Initialize planet data"""
        return {
            'Mercury': PlanetData(0.387, 0.205, 7.0, 'gray', 4879, 88),
            'Venus': PlanetData(0.723, 0.007, 3.4, 'yellow', 12104, 224.7),
            'Earth': PlanetData(1.00, 0.017, 0, 'blue', 12742, 365.2),
            'Mars': PlanetData(1.52, 0.093, 1.85, 'red', 6779, 687),
            'Jupiter': PlanetData(5.20, 0.048, 1.3, 'orange', 139822, 4331),
            'Saturn': PlanetData(9.58, 0.056, 2.49, 'gold', 116464, 10747),
            'Uranus': PlanetData(19.22, 0.046, 0.77, 'lightblue', 50724, 30589),
            'Neptune': PlanetData(30.05, 0.010, 1.77, 'blue', 49244, 59800),
            'Pluto': PlanetData(self.constants.PLUTO_SEMI_MAJOR_AXIS, self.constants.PLUTO_ECCENTRICITY,
                                17.16, 'brown', 2376, 90560)
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
                random_index = random.randint(0, len(x) - 1)
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
                                points['trojans_greeks'])

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

        oumuamua_x, oumuamua_y = self._get_oumuamua_trajectory()
        ax.plot(oumuamua_x, oumuamua_y, '--', color='darkred', label="'Oumuamua hyperbolic trajectory")

    def _get_oumuamua_trajectory(self, num_points: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        x, y, _ = OrbitalMechanics.calculate_hyperbolic_orbit(
            self.constants.OUMUAMUA_PERIHELION,
            self.constants.OUMUAMUA_ECCENTRICITY,
            self.constants.OUMUAMUA_INCLINATION,
            self.constants.OUMUAMUA_LONGITUDE_ASCENDING_NODE,
            self.constants.OUMUAMUA_ARGUMENT_OF_PERIHELION,
            num_points=num_points,
        )
        return x, y

    def _oumuamua_inbound_direction(self) -> Tuple[float, float]:
        """Unit vector of 'Oumuamua's inbound asymptote projected onto the xy plane."""
        x, y = self._get_oumuamua_trajectory()
        r = np.sqrt(x**2 + y**2)
        far_idx = 0
        magnitude = r[far_idx]
        if magnitude == 0:
            return 0.0, -1.0
        return float(x[far_idx] / magnitude), float(y[far_idx] / magnitude)

    def _oumuamua_label_text_position(self, anchor: Tuple[float, float],
                                      direction: Tuple[float, float],
                                      span: float) -> Tuple[float, float]:
        """Offset label perpendicular to the trajectory so the arrow crosses the curve."""
        perp = (-direction[1], direction[0])
        if perp[0] * anchor[0] + perp[1] * anchor[1] < 0:
            perp = (-perp[0], -perp[1])
        text_offset = 0.10 * span
        return (anchor[0] + perp[0] * text_offset,
                anchor[1] + perp[1] * text_offset)

    def _oumuamua_label_for_view(self, view_type: str) -> Tuple[str, Tuple[float, float], Tuple[float, float]]:
        """Label text and arrow anchors aligned to the real hyperbolic trajectory."""
        limits = self._get_view_limits(view_type)
        span = limits[1] - limits[0]
        label = "'Oumuamua hyperbolic trajectory"
        zoomed_out_views = {
            '2_solar_system_with_kuiper_belt',
            '3_solar_system_with_oort_cloud',
            '4_solar_system_with_alpha_centauri',
        }

        if view_type in zoomed_out_views:
            direction = self._oumuamua_inbound_direction()
            target_radii = {
                '2_solar_system_with_kuiper_belt': 55,
                '3_solar_system_with_oort_cloud': 45000,
                '4_solar_system_with_alpha_centauri': 45000,
            }
            target_r = target_radii[view_type]
            anchor = (direction[0] * target_r, direction[1] * target_r)
            text = self._oumuamua_label_text_position(anchor, direction, span)
            return label, anchor, text

        x, y = self._get_oumuamua_trajectory()
        r = np.sqrt(x**2 + y**2)
        visible = (x >= limits[0]) & (x <= limits[1]) & (y >= limits[0]) & (y <= limits[1])
        if visible.any():
            visible_idx = np.where(visible)[0]
            idx = int(visible_idx[len(visible_idx) // 3])
            anchor = (float(x[idx]), float(y[idx]))
        else:
            idx = int(np.argmax(r))
            anchor = (float(x[idx]), float(y[idx]))

        offset = 0.08 * span
        text = (anchor[0] - offset, anchor[1] - offset)
        return label, anchor, text
        
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

        stars_range = self.stars_data[
            (self.stars_data['Distance (ly)'] <= max_distance) &
            self.stars_data['x'].notna() &
            self.stars_data['y'].notna()
        ]
        
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
 
 
    def _jupiter_lagrange_labels(self) -> List[Tuple[str, Tuple[float, float], Tuple[float, float]]]:
        """Annotation anchors for Jupiter's L4 (Trojans) and L5 (Greeks) clusters."""
        a = self.constants.JUPITER_SEMI_MAJOR_AXIS
        trojan_angle = np.deg2rad(60)
        greek_angle = np.deg2rad(240)

        trojan_anchor = (a * np.cos(trojan_angle), a * np.sin(trojan_angle))
        greek_anchor = (a * np.cos(greek_angle), a * np.sin(greek_angle))

        return [
            ('Trojans', trojan_anchor, (trojan_anchor[0] + 1.0, trojan_anchor[1] + 0.5)),
            ('Greeks', greek_anchor, (greek_anchor[0] - 1.0, greek_anchor[1] - 2.2)),
        ]

    def _add_labels(self, ax: plt.Axes, view_type: str):
        """Add labels to solar system features based on view type"""
        font_size = 48  # Base font size for labels
        
        label_configs = {
            '0_inner_solar_system': [
                ('Asteroid Belt (2.2-3.2 AU)', (self.constants.ASTEROID_BELT_OUTER, 0), 
                (self.constants.ASTEROID_BELT_INNER+0.1, 1.5)),
            ],
            '1_inner_solar_system_with_jupiter': [
                ('Asteroid Belt (2.2-3.2 AU)', (self.constants.ASTEROID_BELT_OUTER, 0), 
                (self.constants.ASTEROID_BELT_INNER+0.1, 2)),
                ('Hildas', (-self.constants.JUPITER_SEMI_MAJOR_AXIS+0.5, 0), 
                (-self.constants.JUPITER_SEMI_MAJOR_AXIS-1, -2.5)),
            ],
            '2_solar_system_with_kuiper_belt': [
                (f'Kuiper Belt ({self.constants.KUIPER_BELT_INNER}-{self.constants.KUIPER_BELT_OUTER} AU)',
                (self.constants.KUIPER_BELT_OUTER, 0),
                (self.constants.KUIPER_BELT_OUTER+5, 10)),
                (f"Pluto's aphelion ({self.constants.PLUTO_APHELION:.1f} AU)",
                (-self.constants.PLUTO_APHELION, 0),
                (-self.constants.PLUTO_APHELION-25, 10)),
                (f"Pluto's perihelion ({self.constants.PLUTO_PERIHELION:.1f} AU)",
                (self.constants.PLUTO_PERIHELION, 0),
                (self.constants.PLUTO_PERIHELION+10, -10)),
            ],
            '3_solar_system_with_oort_cloud': [
                (f'Kuiper Belt ({self.constants.KUIPER_BELT_INNER}-{self.constants.KUIPER_BELT_OUTER} AU)',
                (self.constants.KUIPER_BELT_OUTER-3500, -4000),
                (self.constants.KUIPER_BELT_OUTER+80000, 90000)),
                ('Oort Cloud (100000 AU)', (100000, 5), (70000, 25000)),
            ],
            '4_solar_system_with_alpha_centauri': [
                (f'Kuiper Belt ({self.constants.KUIPER_BELT_INNER}-{self.constants.KUIPER_BELT_OUTER} AU)',
                (self.constants.KUIPER_BELT_OUTER-3500, -4000),
                (self.constants.KUIPER_BELT_OUTER+80000, 110000)),
                ('Oort Cloud (100000 AU)', (-100000, 5), (-180000, 25000)),
            ]
        }

        if view_type == '1_inner_solar_system_with_jupiter':
            label_configs.setdefault(view_type, []).extend(self._jupiter_lagrange_labels())

        oumuamua_views = {
            '0_inner_solar_system',
            '1_inner_solar_system_with_jupiter',
            '2_solar_system_with_kuiper_belt',
            '3_solar_system_with_oort_cloud',
            '4_solar_system_with_alpha_centauri',
        }
        if view_type in oumuamua_views:
            label_configs.setdefault(view_type, []).append(self._oumuamua_label_for_view(view_type))
        
        if view_type in label_configs:
            for label_text, xy, xytext in label_configs[view_type]:
                ax.annotate(
                    label_text, 
                    xy=xy, 
                    xytext=xytext,
                    fontsize=font_size,
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )

    def _get_vega_position(self) -> Tuple[float, float]:
        vega = self.stars_data[self.stars_data['System'].str.startswith('Vega', na=False)].iloc[0]
        return float(vega['x']), float(vega['y'])

    def _oumuamua_vega_sky_separation_degrees(self) -> float:
        x, y, z = OrbitalMechanics.calculate_hyperbolic_orbit(
            self.constants.OUMUAMUA_PERIHELION,
            self.constants.OUMUAMUA_ECCENTRICITY,
            self.constants.OUMUAMUA_INCLINATION,
            self.constants.OUMUAMUA_LONGITUDE_ASCENDING_NODE,
            self.constants.OUMUAMUA_ARGUMENT_OF_PERIHELION,
            num_points=5000,
        )
        r = np.sqrt(x**2 + y**2 + z**2)
        inbound = np.array([x[0] / r[0], y[0] / r[0], z[0] / r[0]])
        vega = self.stars_data[self.stars_data['System'].str.startswith('Vega', na=False)].iloc[0]
        vega_dir = np.array([vega['x'], vega['y'], vega['z']], dtype=float)
        vega_dir /= np.linalg.norm(vega_dir)
        return float(np.degrees(np.arccos(np.clip(np.dot(inbound, vega_dir), -1.0, 1.0))))

    def _oumuamua_vega_view_limits(self) -> Tuple[float, float, float, float]:
        """Axis limits framing the Sun and Vega at 25 light years."""
        vega_x, vega_y = self._get_vega_position()
        padding = 0.18
        x_min = min(0.0, vega_x) - abs(vega_x) * padding
        x_max = max(0.0, vega_x) + abs(vega_x) * padding
        y_min = min(0.0, vega_y) - abs(vega_y) * padding
        y_max = max(0.0, vega_y) + abs(vega_y) * padding
        return x_min, x_max, y_min, y_max

    def create_oumuamua_vega_visualization(self, save_path: str):
        """
        Context view of 'Oumuamua's true hyperbolic path relative to Vega at 25 ly.
        Vega lies near the inbound asymptote in the sky, but is not the origin.
        """
        fig, ax = plt.subplots(figsize=(39, 39))
        vega_x, vega_y = self._get_vega_position()
        vega_distance = np.hypot(vega_x, vega_y)
        separation = self._oumuamua_vega_sky_separation_degrees()
        inbound = self._oumuamua_inbound_direction()

        ax.plot(0, 0, 'o', markersize=20, color='yellow')

        oumuamua_x, oumuamua_y = self._get_oumuamua_trajectory()
        ax.plot(oumuamua_x, oumuamua_y, '--', color='darkred', linewidth=1.5)

        asymptote_length = vega_distance * 0.42
        asymptote_end = (inbound[0] * asymptote_length, inbound[1] * asymptote_length)
        ax.plot([0, asymptote_end[0]], [0, asymptote_end[1]], '--', color='darkred', linewidth=2)

        ax.plot([0, vega_x], [0, vega_y], ':', color='gray', linewidth=1.5)

        stars_range = self.stars_data[
            (self.stars_data['Distance (ly)'] <= 25.05) &
            self.stars_data['x'].notna() &
            self.stars_data['y'].notna()
        ]
        for _, row in stars_range.iterrows():
            if row['System'].startswith('Vega'):
                ax.plot(row['x'], row['y'], 'o', markersize=40, color='silver')
                ax.text(row['x'] + 15000, row['y'], 'Vega (25 ly)', fontsize=36, ha='left', va='center')
            else:
                ax.plot(row['x'], row['y'], 'o', markersize=10, color='orange', alpha=0.6)

        font_size = 36
        asymptote_anchor = (asymptote_end[0] * 0.72, asymptote_end[1] * 0.72)
        asymptote_text = self._oumuamua_label_text_position(
            asymptote_anchor, inbound, vega_distance)
        ax.annotate(
            "'Oumuamua inbound asymptote",
            xy=asymptote_anchor,
            xytext=asymptote_text,
            fontsize=font_size,
            arrowprops=dict(facecolor='black', shrink=0.05),
        )
        ax.annotate(
            f"Sky direction near Vega (~{separation:.0f}° away) — not from Vega",
            xy=(vega_x * 0.55, vega_y * 0.55),
            xytext=(vega_x * 0.35, vega_y * 0.75),
            fontsize=font_size,
            arrowprops=dict(facecolor='black', shrink=0.05, linestyle=':'),
        )

        x_min, x_max, y_min, y_max = self._oumuamua_vega_view_limits()
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal', 'box')
        ax.axis('off')
        plt.title("'Oumuamua and Vega — Interstellar Context (25 light years)", fontsize=72, pad=50)

        plt.savefig(save_path, dpi=300)
        plt.close(fig)

    def create_visualization(self, view_type: str, save_path: str):
        """Create and save solar system visualization"""
        if view_type == '8_oumuamua_and_vega_25':
            self.create_oumuamua_vega_visualization(save_path)
            return

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
        ('7_solar_system_with_nearest_stars_30', 'Stars within 30 Light Years (±1,897,232 AU)'),
        ('8_oumuamua_and_vega_25', "'Oumuamua and Vega (25 light years)"),
    ]
    
    for view_type, description in views:
        output_path = (
            'output/2d/8_oumuamua_origin_vega_system_25.jpg'
            if view_type == '8_oumuamua_and_vega_25'
            else f'output/2d/{view_type}.jpg'
        )
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