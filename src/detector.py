import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms

class AnomalyDetector:
    def __init__(self, mean_path, threshold_path, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load artifacts
        self.mean_embedding = torch.tensor(
            np.load(mean_path), device=self.device
        )
        self.threshold = float(np.load(threshold_path))

        # Load model
        self.model = models.resnet18(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.to(self.device)
        self.model.eval()

        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _preprocess(self, image: Image.Image):
        return self.transform(image).unsqueeze(0).to(self.device)

    def score(self, image: Image.Image) -> float:
        with torch.no_grad():
            x = self._preprocess(image)
            emb = self.model(x)
            score = torch.norm(emb - self.mean_embedding)
            return score.item()

    def predict(self, image: Image.Image) -> dict:
        score = self.score(image)
        is_anomaly = score > self.threshold

        return {
            "anomaly_score": score,
            "threshold": self.threshold,
            "is_anomaly": bool(is_anomaly)
        }
