# Image Anomaly Detection API (Computer Vision)

End-to-end deep learning system for detecting visual anomalies in images,
deployed as a real-time REST API using FastAPI.

## Problem Statement

In manufacturing and quality-control pipelines, defective samples are rare and
labeled anomaly data is often unavailable. Traditional supervised learning
approaches fail in such scenarios due to severe class imbalance and lack of
ground-truth labels.

This project addresses the problem of detecting anomalous images by learning
only from normal samples during training, making it suitable for real-world
industrial inspection and monitoring use cases.


## Solution Overview

The system learns a compact representation of normal images using a pretrained
convolutional neural network (CNN) as a feature extractor.

During training, embeddings are extracted from normal images and aggregated to
estimate a reference representation of the normal data distribution.
A distance-based threshold is then computed to separate normal and anomalous
samples.

At inference time, incoming images are converted into embeddings and scored
based on their distance from the learned normal representation. Images exceeding
the threshold are classified as anomalies.

The entire pipeline is deployed as a real-time REST API using FastAPI, enabling
easy integration into production systems.


## Architecture

The system follows a two-stage pipeline: a training phase using only normal
images, and an inference phase for real-time anomaly detection.

### Training Phase
Normal Images  
→ Pretrained CNN Feature Extractor  
→ Feature Embedding Extraction  
→ Mean Embedding Computation  
→ Threshold Estimation  

### Inference Phase
Input Image  
→ Pretrained CNN Feature Extractor  
→ Feature Embedding  
→ Distance Scoring  
→ Anomaly Decision  


## Demo

### API Documentation (Swagger UI)

The FastAPI service exposes interactive API documentation using Swagger UI,
allowing users to test the anomaly detection endpoint directly from the browser.

![Swagger UI](assets/swagger_ui.png)

### Sample Prediction

Uploading an image to the `/predict` endpoint returns an anomaly score,
the learned threshold, and the final anomaly decision.

![Prediction Example](assets/prediction_example.png)


## API Usage

### Predict Anomaly

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.png"

Response:
{
  "anomaly_score": 18.94,
  "threshold": 5.14,
  "is_anomaly": true
}

## Project Structure

image-anomaly-detection-api/
├── artifacts/                  # Saved mean embedding & threshold
│   ├── mean_embedding.npy
│   └── threshold.npy
│
├── assets/                     # Screenshots for README / demo
│   ├── swagger_ui.png
│   └── prediction_example.png
│
├── data_1/                     # Dataset
│   ├── train/                  # Normal images (training)
│   └── test/                   # Normal & anomalous images (evaluation)
│
├── src/
│   ├── __init__.py
│   ├── detector.py             # OOP anomaly detector logic
│   ├── extract_embeddings.py   # Feature extraction & statistics
│   ├── validate.py             # Offline validation & metrics
│   └── api.py                  # FastAPI inference service
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
The project is structured to clearly separate training, validation, and deployment logic, following production-ready machine learning best practices.


## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/khandelwalvanshika151997-cmd/image-anomaly-detection-api.git
cd image-anomaly-detection-api

### 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

### 3. Install dependencies
pip install -r requirements.txt


### 4. Start the FastAPI server
uvicorn src.api:app --reload

### 5. Open Swagger UI
Visit:  http://127.0.0.1:8000/docs


## Results & Observations

- The model achieves a low false-positive rate on normal images.
- Distance-based thresholding allows flexible control over anomaly sensitivity.
- The approach performs well in scenarios where anomalous samples are rare or unavailable.
- The system is suitable for real-time inspection pipelines where missing anomalies is more costly than false alarms.

## Future Improvements

- Support additional distance metrics such as cosine similarity and Mahalanobis distance.
- Add batch inference support for high-throughput use cases.
- Introduce model versioning and monitoring for production deployments.
- Dockerize the application for easier cloud deployment.
- Extend the system to provide anomaly localization heatmaps.
