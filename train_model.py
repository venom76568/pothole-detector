import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from detect_potholes import detect_potholes  # Assuming detect_potholes is in the same directory

# Create results directory
os.makedirs("results", exist_ok=True)

# Constants for dataset and training
TRAIN_DIR = './train'
TEST_DIR = './test'
VAL_DIR = './val'
EPOCHS = 100
IMAGE_SIZE = (640, 640)  
NUM_SAMPLES = 1200  

# Simulate Ground Truth Labels (1 for "Pothole", 0 for "No Pothole")
np.random.seed(42)

# Load Dataset Function
def load_images_from_directory(directory, num_samples):
    images = []
    labels = []
    image_files = [f for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = np.random.choice(image_files, num_samples, replace=False)
    
    for image_file in selected_files:
        image_path = os.path.join(directory, image_file)
        image = Image.open(image_path).resize(IMAGE_SIZE)
        image = np.array(image)  # Convert to numpy array
        label = 1 if 'pothole' in image_file.lower() else 0  # Assuming 'pothole' in filename indicates a pothole
        images.append(image)
        labels.append(label)
    
    return np.array(images), np.array(labels)

# Load training, validation, and test datasets
train_images, train_labels = load_images_from_directory(TRAIN_DIR, NUM_SAMPLES)
val_images, val_labels = load_images_from_directory(VAL_DIR, NUM_SAMPLES)
test_images, test_labels = load_images_from_directory(TEST_DIR, NUM_SAMPLES)

# Simulate Model Predictions (random for illustration)
model_predictions_prob = np.random.uniform(0.0, 1.0, NUM_SAMPLES)
predictions = (model_predictions_prob >= 0.5).astype(int)

# Simulate Metric Trend (used for loss trends)
def simulate_metric_trend(base=5.0, scale=0.5, shift=0.1):
    noise = np.random.normal(0, scale, EPOCHS)
    trend = base * np.exp(-np.linspace(0, 5, EPOCHS)) + shift
    return np.clip(trend + noise, shift, None)

# Simulate Sigmoid Curve for Metrics (used for precision, recall, etc.)
def simulate_curve_sigmoid(start=0.4, end=0.9, noise_level=0.02, clip_min=0.4, clip_max=0.95):
    sigmoid = lambda x: start + (end - start) / (1 + np.exp(-x))
    x = np.linspace(-6, 6, EPOCHS)
    curve = sigmoid(x) + np.random.normal(0, noise_level, EPOCHS)
    return np.clip(curve, clip_min, clip_max)

# Simulated Metrics with complex equations:
train_loss = simulate_metric_trend(base=5.2, scale=0.6, shift=0.1)
val_loss = simulate_metric_trend(base=4.7, scale=0.6, shift=0.1)
precision = simulate_curve_sigmoid(0.4, 0.9, noise_level=0.02, clip_min=0.4, clip_max=0.95)
recall = simulate_curve_sigmoid(0.3, 0.88, noise_level=0.03, clip_min=0.3, clip_max=0.93)
map50 = simulate_curve_sigmoid(0.2, 0.85, noise_level=0.02, clip_min=0.2, clip_max=0.9)

# Compute Precision, Recall, F1 Score
precision_value = precision_score(test_labels, predictions)
recall_value = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)

# Generate Confusion Matrix dynamically
tn, fp, fn, tp = confusion_matrix(test_labels, predictions).ravel()

# Save metrics to JSON (simulate the same structure as before)
metrics = {
    "epochs": EPOCHS,
    "results": [],
    "best_epoch": int(np.argmin(val_loss)),
    "final": {
        "train_loss": float(train_loss[-1]),
        "val_loss": float(val_loss[-1]),
        "precision": float(precision_value),
        "recall": float(recall_value),
        "f1_score": float(f1),
        "map50": float(np.mean(map50)),  # Assuming MAP50 is the average of the simulated curve
    },
    "timestamp": datetime.now().isoformat()
}

for epoch in range(EPOCHS):
    metrics["results"].append({
        "epoch": epoch + 1,
        "train_loss": float(train_loss[epoch]),
        "val_loss": float(val_loss[epoch]),
        "precision": float(precision[epoch]),  # Assuming you want to plot precision over epochs
        "recall": float(recall[epoch]),  # Same for recall
        "map50": float(map50[epoch])  # Same for MAP50
    })

with open("results/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Plot 1: Training/Validation Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss", linewidth=2)
plt.plot(val_loss, label="Validation Loss", linestyle="--", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Simulated Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png")

# Plot 2: Precision, Recall, F1 Score over Epochs
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, EPOCHS-1, EPOCHS), precision, label="Precision", linewidth=2)
plt.plot(np.linspace(0, EPOCHS-1, EPOCHS), recall, label="Recall", linewidth=2)
plt.plot(np.linspace(0, EPOCHS-1, EPOCHS), f1, label="F1 Score", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Precision, Recall, and F1 Score over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/metrics_curve.png")

# Plot 3: MAP50 over Epochs
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, EPOCHS-1, EPOCHS), map50, label="MAP50", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("MAP50 over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/map50_curve.png")

# Plot 4: Confusion Matrix (Generated from predictions and ground truth)
conf_matrix = np.array([[tn, fp], [fn, tp]])

plt.figure(figsize=(5, 4))
plt.imshow(conf_matrix, cmap='Blues')
plt.colorbar()
plt.xticks([0, 1], ["No Pothole", "Pothole"])
plt.yticks([0, 1], ["No Pothole", "Pothole"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix.png")

# Call detect_potholes function (assuming this function does something meaningful, like detecting potholes)
# detected_potholes = detect_potholes('path_to_image_or_data')
# print(detected_potholes)

# Print results summary
print(f"✅ Simulated training complete. Metrics and plots saved in 'results/'")
print(f"Precision: {precision_value:.4f}, Recall: {recall_value:.4f}, F1 Score: {f1:.4f}, MAP50: {np.mean(map50):.4f}")
