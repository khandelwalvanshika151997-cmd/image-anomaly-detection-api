import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models, transforms
from tqdm import tqdm

# ---------------- CONFIG ----------------
ARTIFACT_DIR = "artifacts"
TEST_NORMAL_DIR = "data_1/test/normal"
TEST_ANOMALY_DIR = "data_1/test/anomaly"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- LOAD ARTIFACTS ----------------
mean_embedding = np.load(f"{ARTIFACT_DIR}/mean_embedding.npy")
threshold = np.load(f"{ARTIFACT_DIR}/threshold.npy")

mean_embedding = torch.tensor(mean_embedding).to(DEVICE)

# ---------------- TRANSFORMS ----------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def load_image(path):
    return transform(Image.open(path).convert("RGB"))

# ---------------- MODEL ----------------
model = models.resnet18(pretrained=True)
model.fc = torch.nn.Identity()
model.to(DEVICE)
model.eval()

# ---------------- SCORING FUNCTION ----------------
def anomaly_score(img_path):
    with torch.no_grad():
        x = load_image(img_path).unsqueeze(0).to(DEVICE)
        emb = model(x)
        score = torch.norm(emb - mean_embedding)
        return score.item()

# ---------------- RUN VALIDATION ----------------
normal_scores = []
anomaly_scores = []

print("Scoring normal images...")
for fname in tqdm(os.listdir(TEST_NORMAL_DIR)):
    path = os.path.join(TEST_NORMAL_DIR, fname)
    normal_scores.append(anomaly_score(path))

print("Scoring anomaly images...")
for defect in os.listdir(TEST_ANOMALY_DIR):
    defect_dir = os.path.join(TEST_ANOMALY_DIR, defect)
    for fname in os.listdir(defect_dir):
        path = os.path.join(defect_dir, fname)
        anomaly_scores.append(anomaly_score(path))

# ---------------- PLOTS ----------------
plt.figure(figsize=(10, 6))
plt.hist(normal_scores, bins=50, alpha=0.7, label="Normal")
plt.hist(anomaly_scores, bins=50, alpha=0.7, label="Anomaly")
plt.axvline(threshold, color="red", linestyle="--", label="Threshold")
plt.xlabel("Anomaly Score (L2 Distance)")
plt.ylabel("Count")
plt.title("Anomaly Score Distribution")
plt.legend()
plt.show()

# ---------------- SIMPLE METRICS ----------------
normal_above_thresh = sum(s > threshold for s in normal_scores)
anomaly_below_thresh = sum(s <= threshold for s in anomaly_scores)

print("\n--- Validation Summary ---")
print(f"Normal images flagged as anomaly: {normal_above_thresh}/{len(normal_scores)}")
print(f"Anomalies missed as normal: {anomaly_below_thresh}/{len(anomaly_scores)}")
