import tensorflow as tf
from utils.dataset_builder import load_dataset

# Step 1: Load dataset
dataset = load_dataset("data/wildfire_tf_data", batch_size=32)

# Step 2: Prepare (X, y)
def preprocess(batch):
    features = [
        "PrevFireMask", "NDVI", "DroughtIndex", "Humidity", "TemperatureMin",
        "TemperatureMax", "WindSpeed", "WindDirection", "Precipitation",
        "Elevation", "PopulationDensity", "EnergyReleaseComponent"
    ]
    # Stack selected input features along last axis (depth)
    x = tf.stack([tf.reshape(batch[f], [-1, 64, 64]) for f in features], axis=-1)
    y = tf.reshape(batch["FireMask"], [-1, 64, 64, 1])
    return x, y

train_dataset = dataset.map(preprocess)

# Step 3: Build a CNN model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(64, 64, 12)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.UpSampling2D((2,2)),
        tf.keras.layers.Conv2D(1, (3,3), activation='sigmoid', padding='same'),
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()
model.summary()

# Step 4: Train the model
model.fit(train_dataset, epochs=5)

# Step 5: Save the trained model
model.save("saved_models/firemask_cnn.h5")
print("Model saved successfully to saved_models/")
