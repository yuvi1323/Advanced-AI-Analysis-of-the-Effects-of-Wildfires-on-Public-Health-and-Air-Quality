import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Step 1: Load merged dataset
data_path = "output/integrated_fire_aqi_health.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

df = pd.read_csv(data_path)

# Step 2: Define features and target
features = [
    "FireMaskScore", "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed"
]
target = "HealthImpactScore"

X = df[features]
y = df[target]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 4: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Model Trained!")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Error: {mae:.2f}")

# Step 6: Plot feature importance
importances = model.feature_importances_
feat_df = pd.DataFrame({"Feature": features, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_df, palette="viridis")
plt.title("Feature Importance for Predicting HealthImpactScore")
plt.tight_layout()

# Save figure
os.makedirs("output/figures", exist_ok=True)
plt.savefig("output/figures/feature_importance_health_score.png", dpi=300)
plt.show()