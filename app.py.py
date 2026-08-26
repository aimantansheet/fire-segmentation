from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import numpy as np
import cv2
import io
from PIL import Image

from model import predict

app = FastAPI(title="Fire Segmentation API")

@app.get("/")
def home():
    return {"message": "Fire Segmentation API is running"}

@app.post("/predict")
async def segmentation(file: UploadFile = File(...)):

    image_bytes = await file.read()

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    image = np.array(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    mask = predict(image)

    mask = (mask * 255).astype(np.uint8)

    output = Image.fromarray(mask)

    buffer = io.BytesIO()

    output.save(buffer, format="PNG")

    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/png")