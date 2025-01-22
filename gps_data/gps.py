from fitparse import FitFile
import gpxpy
import gpxpy.gpx
import pandas as pd
import folium
from folium import LayerControl
import geopandas as gpd
from shapely.geometry import Point

# Helper function to convert raw GPS data (if in semicircles) to degrees
def semicircles_to_degrees(value):
    return value * (180 / 2**31) if pd.notnull(value) else None

# Load and parse the .FIT file
fit_file_path = 'Morning_Run.fit'  # Replace with your FIT file path
fit_data = FitFile(fit_file_path)

# Extract GPS data from the FIT file
fit_records = []
for record in fit_data.get_messages('record'):
    record_data = {}
    for data in record:
        record_data[data.name] = data.value
    fit_records.append(record_data)

fit_df = pd.DataFrame(fit_records)
fit_df['timestamp'] = pd.to_datetime(fit_df['timestamp'], errors='coerce')

# Convert raw GPS values to degrees
fit_df['latitude'] = fit_df['position_lat'].apply(semicircles_to_degrees)
fit_df['longitude'] = fit_df['position_long'].apply(semicircles_to_degrees)

# Load and parse the .GPX file
gpx_file_path = 'Morning_Run.gpx'  # Replace with your GPX file path
with open(gpx_file_path, 'r') as gpx_file:
    gpx = gpxpy.parse(gpx_file)

# Extract GPS data from the GPX file
gpx_points = []
for track in gpx.tracks:
    for segment in track.segments:
        for point in segment.points:
            gpx_points.append({
                'latitude': point.latitude,
                'longitude': point.longitude,
                'elevation': point.elevation,
                'time': pd.to_datetime(point.time)
            })

gpx_df = pd.DataFrame(gpx_points)

# Combine and convert to GeoDataFrames
fit_df = fit_df.dropna(subset=['latitude', 'longitude'])
fit_gdf = gpd.GeoDataFrame(
    fit_df, geometry=gpd.points_from_xy(fit_df.longitude, fit_df.latitude), crs="EPSG:4326"
)

gpx_gdf = gpd.GeoDataFrame(
    gpx_df, geometry=gpd.points_from_xy(gpx_df.longitude, gpx_df.latitude), crs="EPSG:4326"
)

# Calculate the center of the map based on both datasets
mean_latitude = (fit_gdf['latitude'].mean() + gpx_gdf['latitude'].mean()) / 2
mean_longitude = (fit_gdf['longitude'].mean() + gpx_gdf['longitude'].mean()) / 2

# Create an interactive map
m = folium.Map(location=[mean_latitude, mean_longitude], zoom_start=14, tiles="OpenStreetMap")

# Add FIT data to the map
fit_layer = folium.FeatureGroup(name="FIT Data")
for _, row in fit_gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='blue',
        fill=True,
        fill_color='blue',
        fill_opacity=0.7
    ).add_to(fit_layer)
fit_layer.add_to(m)

# Add GPX data to the map
gpx_layer = folium.FeatureGroup(name="GPX Data")
for _, row in gpx_gdf.iterrows():
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=2,
        color='red',
        fill=True,
        fill_color='red',
        fill_opacity=0.7
    ).add_to(gpx_layer)
gpx_layer.add_to(m)

# Add layer control
LayerControl().add_to(m)

# Save the map to an HTML file
map_output_path = 'gps_comparison_map.html'
m.save(map_output_path)

print(f"Interactive map saved to {map_output_path}. Open this file in a web browser to explore the data.")

