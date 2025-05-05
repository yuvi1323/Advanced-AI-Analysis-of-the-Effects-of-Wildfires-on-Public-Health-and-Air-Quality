import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load dataset
data_path = "output/integrated_fire_aqi_health.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(f"File not found: {data_path}")

df = pd.read_csv(data_path)

# Step 2: Prepare features and labels
features = [
    "FireMaskScore", "AQI", "PM10", "PM2_5", "NO2", "SO2", "O3",
    "Temperature", "Humidity", "WindSpeed"
]
target = "HealthImpactClass"

X = df[features]
y = df[target].astype(int)  # Ensure integer labels

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Step 4: Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Step 5: Predict & evaluate
y_pred = clf.predict(X_test)

print("✅ Classification Model Trained\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Confusion matrix
cm = confusion_matrix(y_test, y_pred)
labels = sorted(df[target].unique())

plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap="Blues", values_format='d')
plt.title("Confusion Matrix: HealthImpactClass Prediction")
plt.tight_layout()

# Save plot
os.makedirs("output/figures", exist_ok=True)
plt.savefig("output/figures/confusion_matrix_health_class.png", dpi=300)
plt.show()