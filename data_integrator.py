import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.dataset_builder import load_dataset
import numpy as np

# Step 1: Load AQI & Health datasets
aqi_df = pd.read_csv("data/aqi_daily_1980_to_2021.csv")
health_df = pd.read_csv("data/air_quality_health_impact_data.csv")

# Step 2: Load trained FireMask CNN
model = load_model("saved_models/firemask_cnn.h5")

# Step 3: Load one FireMask prediction
def preprocess(batch):
    features = [
        "PrevFireMask", "NDVI", "DroughtIndex", "Humidity", "TemperatureMin",
        "TemperatureMax", "WindSpeed", "WindDirection", "Precipitation",
        "Elevation", "PopulationDensity", "EnergyReleaseComponent"
    ]
    x = tf.stack([tf.reshape(batch[f], [-1, 64, 64]) for f in features], axis=-1)
    y = tf.reshape(batch["FireMask"], [-1, 64, 64, 1])
    return x, y

dataset = load_dataset("data/wildfire_tf_data", batch_size=1)
dataset = dataset.map(preprocess)

# Get 1 sample
for x, y_true in dataset.take(1):
    fire_pred = model.predict(x)[0, ..., 0]
    fire_score = fire_pred.mean()
    break

# Step 4: Simulate location of fire region
fire_lat, fire_lon = 38.5, -122.0  # e.g., Northern CA (Napa/Sonoma)

# Step 5: Filter nearby AQI rows
tolerance = 1.0
nearby_aqi = aqi_df[
    (aqi_df['Latitude'].between(fire_lat - tolerance, fire_lat + tolerance)) &
    (aqi_df['Longitude'].between(fire_lon - tolerance, fire_lon + tolerance)) &
    (aqi_df['Defining Parameter'] == "PM2.5")
].copy()

# Add fire score column
nearby_aqi["FireMaskScore"] = fire_score

# Step 6: Merge with health impact data (simulate same location filtering)
# You can expand this by mapping specific counties later
merged_df = health_df.copy()
merged_df["FireMaskScore"] = fire_score

# Step 7: Save final integrated dataset
output_path = "output/integrated_fire_aqi_health.csv"
import os
os.makedirs("output", exist_ok=True)
merged_df.to_csv(output_path, index=False)

print(f"Integrated dataset created with FireMask → AQI → Health.\nSaved to: {output_path}")
