🧠 Core Components
🔥 Fire Spread Prediction (CNN)
- Input: Previous FireMask, Temperature, NDVI, Wind, Elevation, etc.
- Output: Predicted FireMask for the next day
- Trained on ~18,000 TFRecord samples

🌫️ AQI Trend Analysis
- Analyzed PM2.5 pollution levels across California (1980–2021)
- Found strong seasonal spikes during wildfire months
- PM2.5 and AQI highly correlate with public health degradation

🏥 Health Impact Modeling
- Regression Model: Predicts continuous HealthImpactScore
- Classification Model: Categorizes into HealthImpactClass (0–4)
- Top features: AQI, PM2.5, FireMaskScore

🗺️ Geo-Visualization
- Interactive map built using `folium`
- Plots AQI, Fire, and Health impact together
- Helps identify high-risk zones in California

📊 Sample Outputs
- 🔥 Fire Mask Prediction Heatmap  
- 📈 PM2.5 AQI Trends (CA)  
- 💡 Feature Importance Bar Chart  
- 📉 Confusion Matrix for Health Classes  
- 🗺️ Folium-based interactive map of affected regions

🛠 Tech Stack
- Python 3.11**
- TensorFlow/Keras** – Deep Learning (CNN)
- Pandas / NumPy** – Data Manipulation
- Seaborn / Matplotlib** – Plotting
- scikit-learn** – Regression and Classification
- Folium** – Interactive map visualization
- PyCharm** – IDE used on Mac M1
- GitHub + Lucidchart – Version control and diagrams
