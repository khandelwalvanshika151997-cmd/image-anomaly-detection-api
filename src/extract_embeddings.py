import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from torchvision import models, transforms

# ---------------- CONFIG ----------------
DATA_DIR = "data_1/train/normal"
ARTIFACT_DIR = "artifacts"
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

# ---------------- EMBEDDING EXTRACTION ----------------
embeddings = []

with torch.no_grad():
    for fname in tqdm(os.listdir(DATA_DIR)):
        img_path = os.path.join(DATA_DIR, fname)
        x = load_image(img_path).unsqueeze(0).to(DEVICE)
        emb = model(x).cpu().numpy()
        embeddings.append(emb)

embeddings = np.vstack(embeddings)

# ---------------- STATISTICS ----------------
mean_embedding = embeddings.mean(axis=0)
distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
threshold = np.percentile(distances, 90)

# ---------------- SAVE ----------------
np.save(os.path.join(ARTIFACT_DIR, "mean_embedding.npy"), mean_embedding)
np.save(os.path.join(ARTIFACT_DIR, "threshold.npy"), threshold)

print("✅ Training complete")
print("Embeddings shape:", embeddings.shape)
print("Threshold:", threshold)

