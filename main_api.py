"""Ship & Maritime Monitoring — FastAPI Microservice.

Technology  : FastAPI, YOLOv8, Python
Methodology : Object Detection → Feature Extraction → JSON Payload Return
Objectives  : Expose detection pipeline as a high performance microservice.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import List, TypedDict

from fastapi import FastAPI, File, HTTPException, Query, UploadFile

import pipeline

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application."""
    logger.info("Starting up API service...")
    pipeline.initialize_model()
    yield
    logger.info("Shutting down API service...")

app = FastAPI(
    title="Ship & Maritime Monitoring API",
    description="FastAPI service for ship detection using YOLOv8",
    version="1.0.0",
    lifespan=lifespan
)


class RiskDict(TypedDict):
    level: str
    emoji: str


class EnrichedDetection(TypedDict):
    bbox: List[int]
    label: str
    confidence: float
    size: str
    risk: RiskDict


@app.get("/")
def read_root():
    """Health check and service info."""
    return {
        "status": "online",
        "model": "yolov8n.pt",
        "supported_classes": ["boat"]
    }


@app.post("/analyze", response_model=List[EnrichedDetection])
async def analyze(
    file: UploadFile = File(...),
    min_confidence: float = Query(0.50, ge=0.0, le=1.0)
):
    """Analyze uploaded image using YOLOv8 and return enriched detection payload."""
    content_type = file.content_type or ""
    filename = file.filename or ""
    if not (content_type.startswith("image/") or filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = await file.read()
        logger.info(f"Received image '{filename}' for analysis. Minimum confidence: {min_confidence}")
        
        # Decode and run pipeline
        image_array, detections = pipeline.run_pipeline(image_bytes, min_confidence)
        
        img_h, img_w = image_array.shape[:2]
        enriched: List[EnrichedDetection] = []
        
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = det["label"]
            confidence = det["confidence"]
            
            # Fetch size label using pipeline logic
            size = pipeline.size_label(x1, y1, x2, y2, img_h, img_w)
            
            # Fetch risk levels
            risk_info = pipeline.VESSEL_RISK.get(label, ("Unknown", "⚪"))
            
            enriched.append({
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "confidence": confidence,
                "size": size,
                "risk": {
                    "level": risk_info[0],
                    "emoji": risk_info[1]
                }
            })
            
        logger.info(f"Successfully processed '{filename}'. Returned {len(enriched)} detections.")
        return enriched
    except Exception as e:
        logger.exception(f"Pipeline processing failed for file '{filename}'")
        raise HTTPException(status_code=500, detail=f"Pipeline processing failed: {str(e)}")
