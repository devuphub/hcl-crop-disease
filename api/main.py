"""
FastAPI REST API Service for Crop Disease Detection
Provides production-ready endpoints for image prediction, model health, and class metadata.
"""

import os
import sys
import io
from typing import Optional
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models.predictor import CropDiseasePredictor


app = FastAPI(
    title="AgriScan AI - Crop Disease Detection API",
    description="High-performance deep learning REST API for automated crop pathology classification and treatment recommendations.",
    version="1.0.0"
)

# Enable CORS for cross-origin frontend apps (React, Flutter, Vue, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy loading of predictor
predictor = None


def get_predictor() -> CropDiseasePredictor:
    global predictor
    if predictor is None:
        predictor = CropDiseasePredictor()
    return predictor


@app.get("/")
def root():
    return {
        "service": "AgriScan AI - Crop Disease Detection API",
        "status": "online",
        "docs": "/docs",
        "redoc": "/redoc"
    }


@app.get("/health")
def health_check():
    p = get_predictor()
    return {
        "status": "healthy",
        "device": p.device,
        "num_classes": len(p.class_names),
        "model_loaded": p.model is not None
    }


@app.get("/classes")
def get_classes():
    p = get_predictor()
    formatted = []
    for c in p.class_names:
        info = p.disease_info.get(c, {})
        formatted.append({
            "class_key": c,
            "crop": info.get("crop", c.split("___")[0]),
            "disease": info.get("disease", c.split("___")[1].replace("_", " ")),
            "is_healthy": info.get("is_healthy", "healthy" in c.lower())
        })
    return {"total_classes": len(formatted), "classes": formatted}


@app.post("/predict")
async def predict_crop_disease(
    file: UploadFile = File(..., description="Leaf image file (JPEG, PNG, WEBP)"),
    top_k: int = Form(3, description="Number of top candidate diseases to return")
):
    # Validate mime type if present
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a valid image format.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {str(e)}")

    p = get_predictor()
    # Run prediction (omit Grad-CAM numpy array in JSON response)
    result = p.predict(image, top_k=top_k, generate_heatmap=False)

    return JSONResponse(content={
        "status": "success",
        "filename": file.filename,
        "primary_prediction": result["primary_prediction"],
        "top_predictions": result["top_predictions"],
        "severity": result["severity"],
        "status_badge": result["status"],
        "advisory": result["details"]
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
