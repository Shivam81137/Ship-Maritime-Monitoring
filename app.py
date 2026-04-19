"""Streamlit UI for Ship and Maritime Monitoring over SAR imagery."""

from __future__ import annotations

from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import streamlit as st
from PIL import Image


BBox = Tuple[int, int, int, int]
LABEL_BACKGROUND_HEIGHT = 28
ELEVATED_SECURITY_THRESHOLD = 5
DAILY_MOVEMENT_MULTIPLIER = 12


class Detection(TypedDict):
    """Structured detection output: bbox=(x1, y1, x2, y2)."""

    bbox: BBox
    label: str
    confidence: float


def run_mock_pipeline(image_array: np.ndarray) -> List[Detection]:
    """Simulate SAR image processing, detection, and classification.

    TODO: Replace this mock implementation with your trained CNN pipeline:
    1) Preprocess image for model input
    2) Run object detection model
    3) Run/attach ship-classification outputs
    4) Return model-driven boxes, labels, and confidence scores
    """
    height, width = image_array.shape[:2]
    return [
        {
            "bbox": (
                int(0.10 * width),
                int(0.18 * height),
                int(0.38 * width),
                int(0.48 * height),
            ),
            "label": "Cargo",
            "confidence": 0.92,
        },
        {
            "bbox": (
                int(0.52 * width),
                int(0.35 * height),
                int(0.84 * width),
                int(0.72 * height),
            ),
            "label": "Tanker",
            "confidence": 0.88,
        },
    ]


def draw_detections(image_array: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw mock detections on a copy of the input image."""
    annotated = image_array.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        label = str(detection["label"])
        confidence = float(detection["confidence"])
        text = f"{label} ({confidence:.2f})"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.rectangle(
            annotated,
            (x1, max(y1 - LABEL_BACKGROUND_HEIGHT, 0)),
            (x2, y1),
            (0, 220, 0),
            -1,
        )
        cv2.putText(
            annotated,
            text,
            (x1 + 6, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def build_metrics(detections: List[Detection]) -> Tuple[int, str, str]:
    """Generate dashboard metrics for maritime operations."""
    traffic_count = len(detections)
    security_status = (
        "Normal" if traffic_count < ELEVATED_SECURITY_THRESHOLD else "Elevated"
    )
    trade_stat = (
        f"{traffic_count * DAILY_MOVEMENT_MULTIPLIER} estimated vessel movements/day"
    )
    return traffic_count, security_status, trade_stat


def main() -> None:
    """Render Streamlit application."""
    st.set_page_config(page_title="Ship & Maritime Monitoring", layout="wide")
    st.title("🚢 Ship and Maritime Monitoring System")
    st.caption("SAR image analysis pipeline: Image Processing → Object Detection → Classification")

    uploaded_file = st.file_uploader(
        "Upload SAR image data",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF",
    )

    if uploaded_file is None:
        st.info("Upload a SAR image to view detections and monitoring metrics.")
        return

    original_image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(original_image)

    st.subheader("Image Analysis")
    col_original, col_annotated = st.columns(2)

    with col_original:
        st.markdown("**Original Uploaded Image**")
        st.image(original_image, use_container_width=True)

    # TODO: Replace with your CNN inference endpoint / local model execution.
    detections = run_mock_pipeline(image_array)
    annotated_array = draw_detections(image_array, detections)

    with col_annotated:
        st.markdown("**Detection Results**")
        st.image(annotated_array, use_container_width=True)

    st.subheader("Maritime Monitoring Dashboard")
    traffic_count, security_status, trade_stat = build_metrics(detections)
    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Maritime Traffic Count", traffic_count)
    metric_col_2.metric("Security Status", security_status)
    metric_col_3.metric("Trade Analysis Stats", trade_stat)

    st.markdown("### Detection Summary")
    for idx, detection in enumerate(detections, start=1):
        st.write(
            f"{idx}. **{detection['label']}** detected "
            f"with confidence **{float(detection['confidence']):.2f}**"
        )


if __name__ == "__main__":
    main()
