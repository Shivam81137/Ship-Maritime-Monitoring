"""Streamlit UI for Ship and Maritime Monitoring using SAR imagery.

This app is intentionally built with a mock pipeline so a trained CNN model
can be integrated later with minimal changes.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np
import streamlit as st
from PIL import Image


ShipDetection = Dict[str, object]


def mock_methodology_pipeline(image_bgr: np.ndarray) -> List[ShipDetection]:
    """Mock pipeline: image processing -> detection -> classification.

    TODO: Replace this function with the actual trained CNN model inference.
    Suggested integration points:
      1) Preprocess SAR image for model input.
      2) Run object detection model to get bounding boxes.
      3) Run/attach ship-type classification and confidence scores.
    """

    height, width = image_bgr.shape[:2]

    # Dummy detections scaled to image dimensions.
    return [
        {
            "bbox": [int(width * 0.12), int(height * 0.18), int(width * 0.28), int(height * 0.22)],
            "ship_type": "Cargo",
            "confidence": 0.93,
        },
        {
            "bbox": [int(width * 0.58), int(height * 0.40), int(width * 0.25), int(height * 0.20)],
            "ship_type": "Tanker",
            "confidence": 0.88,
        },
    ]


def draw_detection_results(image_bgr: np.ndarray, detections: List[ShipDetection]) -> np.ndarray:
    """Draw detection bounding boxes and labels on the image."""

    output = image_bgr.copy()
    for det in detections:
        x, y, w, h = det["bbox"]
        label = f"{det['ship_type']} ({det['confidence']:.2f})"

        cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            output,
            label,
            (x, max(y - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    return output


def security_status(detections: List[ShipDetection]) -> str:
    """Simple status placeholder for dashboard display.

    TODO: Replace with real security risk logic tied to model output and rules.
    """

    return "Nominal" if len(detections) <= 3 else "Review Required"


def trade_analysis_text(detections: List[ShipDetection]) -> str:
    """Simple trade analytics placeholder from detected ship mix."""

    cargo_count = sum(1 for d in detections if d["ship_type"] == "Cargo")
    tanker_count = sum(1 for d in detections if d["ship_type"] == "Tanker")
    return f"Cargo: {cargo_count} | Tanker: {tanker_count}"


def main() -> None:
    st.set_page_config(page_title="Ship & Maritime Monitoring", layout="wide")
    st.title("🚢 Ship & Maritime Monitoring (SAR)")
    st.caption("Methodology: Image Processing → Object Detection → Classification")

    uploaded_file = st.file_uploader(
        "Upload SAR Image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Upload SAR image data for ship detection preview.",
    )

    if uploaded_file is None:
        st.info("Please upload an image file to run the monitoring pipeline.")
        return

    pil_image = Image.open(uploaded_file).convert("RGB")
    image_rgb = np.array(pil_image)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    detections = mock_methodology_pipeline(image_bgr)
    detected_bgr = draw_detection_results(image_bgr, detections)
    detected_rgb = cv2.cvtColor(detected_bgr, cv2.COLOR_BGR2RGB)

    left, right = st.columns(2)
    with left:
        st.subheader("Original SAR Image")
        st.image(image_rgb, use_container_width=True)
    with right:
        st.subheader("Detection Results")
        st.image(detected_rgb, use_container_width=True)

    st.subheader("Operational Dashboard")
    metric_1, metric_2, metric_3 = st.columns(3)
    with metric_1:
        st.metric("Maritime Traffic Count", len(detections))
    with metric_2:
        st.metric("Security Status", security_status(detections))
    with metric_3:
        st.metric("Trade Analysis Stats", trade_analysis_text(detections))

    with st.expander("Detection Output (Mock)", expanded=False):
        st.json(detections)


if __name__ == "__main__":
    main()
