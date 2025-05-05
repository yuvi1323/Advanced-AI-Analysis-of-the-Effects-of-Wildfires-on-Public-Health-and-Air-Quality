import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Step 1: Load the AQI dataset
aqi_path = "data/aqi_daily_1980_to_2021.csv"

if not os.path.exists(aqi_path):
    raise FileNotFoundError(f"AQI data not found at: {aqi_path}")

df = pd.read_csv(aqi_path)

# Step 2: Preview structure
print("✅ AQI dataset loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head())

# Step 3: Check unique states and pollutants
print("\nStates:", df['State Name'].nunique())
print("Example states:", df['State Name'].unique()[:5])
print("\nPollutants:", df['Defining Parameter'].unique())

# Step 4: Filter for one state (e.g., California) and pollutant (PM2.5)
ca_pm25 = df[(df['State Name'] == 'California') & (df['Defining Parameter'] == 'PM2.5')]

# Step 5: Convert date and sort
ca_pm25 = ca_pm25.copy()
ca_pm25['Date'] = pd.to_datetime(ca_pm25['Date'])


# Step 6: Plot AQI trend
plt.figure(figsize=(12, 5))
sns.lineplot(data=ca_pm25, x='Date', y='AQI', label='PM2.5 AQI')
plt.title('California PM2.5 AQI Over Time')
plt.xlabel('Date')
plt.ylabel('AQI')
plt.grid(True)
plt.tight_layout()
plt.show()

# Step 7: Load trained model
model = load_model("saved_models/firemask_cnn.h5")

# Step 8: Load fire data to simulate predictions
from utils.dataset_builder import load_dataset

# Load 1 sample batch for demo
fire_dataset = load_dataset("data/wildfire_tf_data", batch_size=1)

# Same preprocessing as training
def preprocess(batch):
    features = [
        "PrevFireMask", "NDVI", "DroughtIndex", "Humidity", "TemperatureMin",
        "TemperatureMax", "WindSpeed", "WindDirection", "Precipitation",
        "Elevation", "PopulationDensity", "EnergyReleaseComponent"
    ]
    x = tf.stack([tf.reshape(batch[f], [-1, 64, 64]) for f in features], axis=-1)
    y = tf.reshape(batch["FireMask"], [-1, 64, 64, 1])
    return x, y

fire_dataset = fire_dataset.map(preprocess)

# Step 9: Predict firemask for one record
for x_batch, y_true in fire_dataset.take(1):
    y_pred = model.predict(x_batch)[0, ..., 0]  # shape (64,64)
    y_pred_binary = (y_pred > 0.5).astype(int)  # thresholded fire region

    # Step 10: Simulate a region & match AQI
    # We'll assume lat/lon center around some known wildfire-prone area
    example_lat = 38.5  # somewhere in Northern California
    example_lon = -122.0

    # Step 11: Filter AQI entries nearby (simple bounding box)
    tolerance = 1.0
    nearby_aqi = df[
        (df['Latitude'].between(example_lat - tolerance, example_lat + tolerance)) &
        (df['Longitude'].between(example_lon - tolerance, example_lon + tolerance)) &
        (df['Defining Parameter'] == 'PM2.5')
    ]

    print(f"\nNearby AQI rows around fire region ({example_lat}, {example_lon}):", nearby_aqi.shape[0])
    print(nearby_aqi[['Date', 'AQI', 'County Name']].head())

    # Optional: Plot AQI distribution around fire region
    plt.figure(figsize=(10, 5))
    sns.histplot(data=nearby_aqi, x='AQI', bins=30, kde=True)
    plt.title("AQI Distribution Around Fire Region")
    plt.xlabel("PM2.5 AQI")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    break
