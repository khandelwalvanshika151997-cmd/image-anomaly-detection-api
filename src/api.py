from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

from src.detector import AnomalyDetector

app = FastAPI(title="Image Anomaly Detection API")

# Load detector ONCE at startup
detector = AnomalyDetector(
    mean_path="artifacts/mean_embedding.npy",
    threshold_path="artifacts/threshold.npy"
)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    result = detector.predict(image)
    return result
