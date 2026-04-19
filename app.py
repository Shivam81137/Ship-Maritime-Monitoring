"""Streamlit UI for Ship and Maritime Monitoring with SAR image inputs."""

from __future__ import annotations

from typing import List, Tuple, TypedDict

import numpy as np
import streamlit as st
from PIL import Image
import cv2


class Detection(TypedDict):
    bbox: Tuple[int, int, int, int]
    label: str
    confidence: float


def format_confidence(value: float) -> str:
    """Format confidence score for UI output."""

    return f"{value:.2f}"


st.set_page_config(page_title="Ship & Maritime Monitoring", layout="wide")


def mock_sar_pipeline(image: np.ndarray) -> List[Detection]:
    """Simulate image processing -> object detection -> classification pipeline.

    TODO: Replace this mock implementation with your trained CNN model pipeline:
      1) SAR image preprocessing
      2) Ship/object detection
      3) Ship-type classification
    """

    height, width = image.shape[:2]
    return [
        {
            "bbox": (
                int(width * 0.12),
                int(height * 0.20),
                int(width * 0.36),
                int(height * 0.48),
            ),
            "label": "Cargo",
            "confidence": 0.94,
        },
        {
            "bbox": (
                int(width * 0.56),
                int(height * 0.32),
                int(width * 0.86),
                int(height * 0.67),
            ),
            "label": "Tanker",
            "confidence": 0.89,
        },
    ]


def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw detection bounding boxes and labels on an image."""

    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = detection["label"]
        confidence = detection["confidence"]
        text = f"{label} ({format_confidence(confidence)})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
        cv2.putText(
            annotated,
            text,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


st.title("🚢 Ship and Maritime Monitoring Dashboard")
st.caption("CNN + SAR workflow: Image processing -> Object detection -> Classification")

uploaded_file = st.file_uploader(
    "Upload SAR image data",
    type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
)

if uploaded_file is None:
    st.info("Upload a SAR image to run the monitoring pipeline.")
else:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(pil_image)

    st.subheader("Main Display Area")
    left_col, right_col = st.columns(2)
    with left_col:
        st.image(pil_image, caption="Original uploaded image", use_container_width=True)

    detections = mock_sar_pipeline(image_np)
    annotated_image = draw_detections(image_np, detections)
    with right_col:
        st.image(
            annotated_image,
            caption="Detection and classification output",
            channels="RGB",
            use_container_width=True,
        )

    st.subheader("Dashboard Metrics")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.metric("Maritime Traffic Count", f"{len(detections)} ships")
    with metric_col2:
        # TODO: Replace with security/anomaly score from your production model + rules engine.
        st.metric("Security Status", "Mock: Normal", "No high-risk detection")
    with metric_col3:
        # TODO: Replace with analytics from historical detections/trade intelligence system.
        st.metric("Trade Analysis Stats", "2 route segments", "Stable throughput")

    st.subheader("Detection Summary")
    st.dataframe(
        [
            {
                "Ship Type": d["label"],
                "Confidence": format_confidence(d["confidence"]),
                "Bounding Box (x1,y1,x2,y2)": d["bbox"],
            }
            for d in detections
        ],
        use_container_width=True,
    )
