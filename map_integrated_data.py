import pandas as pd
import numpy as np
import folium
import os

# Load final integrated dataset
csv_path = "output/integrated_fire_aqi_health.csv"
df = pd.read_csv(csv_path)

# Simulate lat/lon (since original health data doesn't have it)
# We'll cluster them around the fire prediction location
base_lat, base_lon = 38.5, -122.0  # Napa/Sonoma, CA

np.random.seed(42)
df["Latitude"] = base_lat + (np.random.rand(len(df)) - 0.5)
df["Longitude"] = base_lon + (np.random.rand(len(df)) - 0.5)

# Setup map
m = folium.Map(location=[base_lat, base_lon], zoom_start=7)

# Function to get color by health class
def get_color(severity):
    colors = ['green', 'yellow', 'orange', 'red', 'darkred']
    return colors[int(severity)] if 0 <= severity <= 4 else "gray"

# Add markers
for _, row in df.sample(n=300).iterrows():  # sample to reduce clutter
    popup = (
        f"AQI: {row['AQI']:.1f}<br>"
        f"PM2.5: {row['PM2_5']:.1f}<br>"
        f"FireScore: {row['FireMaskScore']:.2f}<br>"
        f"HealthScore: {row['HealthImpactScore']:.1f}<br>"
        f"HealthClass: {int(row['HealthImpactClass'])}"
    )
    folium.CircleMarker(
        location=[row["Latitude"], row["Longitude"]],
        radius=6,
        color=get_color(row["HealthImpactClass"]),
        fill=True,
        fill_opacity=0.8,
        popup=folium.Popup(popup, max_width=300)
    ).add_to(m)

# Save map
os.makedirs("output/maps", exist_ok=True)
out_path = "output/maps/integrated_health_aqi_fire_map.html"
m.save(out_path)

print(f"Integrated pipeline map saved to: {out_path}")
