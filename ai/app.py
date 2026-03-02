from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import shutil
import os
import uuid

app = FastAPI()

# Load model (best.pt is one level up)
model = YOLO("../best.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    filename = f"{uuid.uuid4()}.jpg"
    filepath = f"uploads/{filename}"

    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model(filepath)

    annotated_path = f"uploads/annotated_{filename}"
    results[0].save(filename=annotated_path)

    return {"annotated_image": annotated_path}