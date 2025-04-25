import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Create results directory
os.makedirs("results", exist_ok=True)

# Simulate 100 epochs of training
EPOCHS = 100
np.random.seed(42)

def noisy_curve(base, noise_scale=0.03, floor=0.1):
    noise = np.random.normal(0, noise_scale, EPOCHS)
    trend = base * np.exp(-np.linspace(0, 5, EPOCHS)) + floor
    return np.clip(trend + noise, floor, None)

# Simulated training data
train_loss = noisy_curve(base=5.0)
val_loss = noisy_curve(base=4.5)
precision = np.clip(np.linspace(0.4, 0.9, EPOCHS) + np.random.normal(0, 0.02, EPOCHS), 0.4, 0.95)
recall = np.clip(np.linspace(0.3, 0.88, EPOCHS) + np.random.normal(0, 0.03, EPOCHS), 0.3, 0.93)
map50 = np.clip(np.linspace(0.2, 0.85, EPOCHS) + np.random.normal(0, 0.02, EPOCHS), 0.2, 0.9)

# Save simulated metrics to JSON (like YOLOv8 format)
metrics = {
    "epochs": EPOCHS,
    "results": [],
    "best_epoch": int(np.argmin(val_loss)),
    "final": {
        "train_loss": float(train_loss[-1]),
        "val_loss": float(val_loss[-1]),
        "precision": float(precision[-1]),
        "recall": float(recall[-1]),
        "map50": float(map50[-1]),
    },
    "timestamp": datetime.now().isoformat()
}

for epoch in range(EPOCHS):
    metrics["results"].append({
        "epoch": epoch + 1,
        "train_loss": float(train_loss[epoch]),
        "val_loss": float(val_loss[epoch]),
        "precision": float(precision[epoch]),
        "recall": float(recall[epoch]),
        "map50": float(map50[epoch])
    })

with open("results/training_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# Plot 1: Training/Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label="Train Loss", linewidth=2)
plt.plot(val_loss, label="Val Loss", linestyle="--", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("YOLOv8 Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/loss_curve.png")

# Plot 2: Precision, Recall, mAP50
plt.figure(figsize=(10, 5))
plt.plot(precision, label="Precision", linewidth=2)
plt.plot(recall, label="Recall", linewidth=2)
plt.plot(map50, label="mAP@0.5", linewidth=2)
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Precision, Recall, and mAP over Epochs")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/metrics_curve.png")

# Plot 3: Confusion Matrix (realistic layout)
conf_matrix = np.array([[78, 12], [9, 101]])
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

print("✅ Simulated training complete. Metrics and plots saved in 'results/'")
