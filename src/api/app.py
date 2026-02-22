from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

from src.models.model import SimpleCNN

import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
request_count = 0
app = FastAPI()

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SimpleCNN()
model.load_state_dict(torch.load("model.pt", map_location=device))
model.to(device)
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

classes = ["cat", "dog"]


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start_time = time.time()
    global request_count
    request_count += 1

    logger.info("Received prediction request")

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]

    prediction = classes[int(probs.argmax())]

    latency = time.time() - start_time

    logger.info(f"Prediction: {prediction}, Latency: {latency:.4f}s")

    return {
        "prediction": prediction,
        "probabilities": probs.tolist(),
        "latency": latency
    }

@app.get("/metrics")
def metrics():
    return {
        "total_requests": request_count
    }