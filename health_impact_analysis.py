import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Load the dataset
health_path = "data/air_quality_health_impact_data.csv"

if not os.path.exists(health_path):
    raise FileNotFoundError(f"Health impact data not found at: {health_path}")

df = pd.read_csv(health_path)

# Step 2: Preview structure
print("Health Impact dataset loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head())

# Step 3: Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Step 4: Convert 'HealthImpactClass' to categorical (if not already)
if df["HealthImpactClass"].dtype != "object":
    df["HealthImpactClass"] = df["HealthImpactClass"].astype("category")

# Step 5: Correlation matrix (AQI/pollutants vs health impact)
corr_columns = [
    "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed",
    "RespiratoryCases", "CardiovascularCases",
    "HospitalAdmissions", "HealthImpactScore"
]

corr_df = df[corr_columns].dropna()

plt.figure(figsize=(12, 8))
sns.heatmap(corr_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
plt.title("Correlation between Air Quality, Weather & Health Impact")
plt.tight_layout()
plt.show()

# Step 6: Plot health impact by AQI range
plt.figure(figsize=(10, 6))
sns.boxplot(x="HealthImpactClass", y="HealthImpactScore", data=df)
plt.title("Health Impact Score by Impact Class")
plt.grid(True)
plt.tight_layout()
plt.show()