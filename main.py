from utils.dataset_builder import load_dataset

# Load and print one batch
dataset = load_dataset("data/wildfire_tf_data")

print("TFRecord dataset loaded successfully! Sample batch shapes:")
for batch in dataset.take(1):
    for key, value in batch.items():
        print(f"{key}: shape = {value.shape}")