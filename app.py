"""Ship & Maritime Monitoring — Streamlit dashboard.

Technology  : CNN, SAR data, Python
Methodology : Image processing → Object Detection → Classification
Objectives  : Monitor maritime traffic | Enhance security | Support global trade analysis
"""

from __future__ import annotations

import os
import logging
from io import BytesIO

import streamlit as st
import numpy as np
import cv2  # type: ignore[import-untyped]
import requests
from PIL import Image

# ── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/analyze")

from pipeline import (
    VESSEL_TYPES,
    VESSEL_EMOJI,
    VESSEL_RISK,
    VESSEL_DESCRIPTION,
    ELEVATED_THRESHOLD,
    MOVEMENTS_PER_VESSEL,
    draw_detections,
    build_metrics,
    size_label,
    position_label,
    annotated_png_bytes,
    csv_report,
)

@st.cache_data(show_spinner=False)
def get_image_array(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes locally into an RGB numpy array."""
    pil_img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.array(pil_img)

@st.cache_data(show_spinner=False, ttl=3600)
def call_analyze_api(image_bytes: bytes, filename: str, content_type: str, min_conf: float):
    """Send image to the FastAPI /analyze endpoint."""
    logger.info(f"Sending request to API at {API_URL} for {filename}")
    files = {"file": (filename, image_bytes, content_type or "image/png")}
    params = {"min_confidence": min_conf}
    try:
        response = requests.post(API_URL, files=files, params=params, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error communicating with API: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title="Ship & Maritime Monitoring",
        page_icon="🚢",
        layout="wide",
    )

    # Hide Streamlit's Deploy button and hamburger toolbar
    st.markdown(
        """
        <style>
        [data-testid="stToolbar"]      { display: none !important; }
        [data-testid="stDeployButton"] { display: none !important; }
        #MainMenu                      { display: none !important; }
        footer                         { display: none !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Controls")
        min_conf = st.slider(
            "Minimum Confidence Threshold",
            min_value=0.40, max_value=0.95, value=0.50, step=0.05,
            help="Filter out detections below this confidence score.",
        )
        st.divider()

        st.markdown("### 📖 Project Info")
        st.markdown(
            "**Technology:** CNN · SAR data · Python\n\n"
            "**Methodology:**\n"
            "Image Processing → Object Detection → Classification\n\n"
            "**Objectives:**\n"
            "- Monitor maritime traffic\n"
            "- Enhance security\n"
            "- Support global trade analysis"
        )
        st.divider()

        st.markdown("### 🚢 Vessel Risk Legend")
        for label, (risk, dot) in VESSEL_RISK.items():
            emoji = VESSEL_EMOJI.get(label, "")
            st.markdown(f"{dot} {emoji} **{label}** — {risk}")

    # ── Main Header ──────────────────────────────────────────────────────────
    st.title("🚢 Ship & Maritime Monitoring")
    st.caption(
        "**Technology:** CNN · SAR data · Python  |  "
        "**Methodology:** Image Processing → Object Detection → Classification  |  "
        "**Objectives:** Monitor maritime traffic · Enhance security · Support global trade analysis"
    )
    st.divider()

    # ── File uploader ────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload a SAR / optical ship image",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        help="Supported formats: PNG, JPG, JPEG, BMP, TIF, TIFF",
    )

    if uploaded_file is None:
        st.info("📡 Upload a SAR image to view detections and monitoring metrics.")
        return

    # ── Run pipeline ─────────────────────────────────────────────────────────
    image_bytes = uploaded_file.getvalue()
    image_array = get_image_array(image_bytes)
    
    with st.spinner("Analysing image via FastAPI microservice…"):
        try:
            detections = call_analyze_api(
                image_bytes, 
                uploaded_file.name, 
                uploaded_file.type, 
                min_conf
            )
        except Exception as e:
            st.error(f"Failed to connect to the analysis service at `{API_URL}`. Please ensure the backend is running.")
            with st.expander("Show detailed error"):
                st.code(str(e))
            return

    img_h, img_w = image_array.shape[:2]
    annotated_array = draw_detections(image_array, detections)

    # ── Side-by-side images ──────────────────────────────────────────────────
    col_orig, col_ann = st.columns(2)
    with col_orig:
        st.markdown("**Original Image**")
        st.image(image_array, use_container_width=True)
    with col_ann:
        st.markdown("**Detection Results**")
        st.image(annotated_array, use_container_width=True)

    st.divider()

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab_overview, tab_vessels, tab_analytics, tab_export = st.tabs(
        ["📊 Overview", "🚢 Vessel Details", "📈 Analytics", "📥 Export"]
    )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ════════════════════════════════════════════════════════════════════════
    with tab_overview:
        traffic_count, security_status, trade_stat = build_metrics(detections)

        st.markdown("## Maritime Monitoring Dashboard")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🚢 Vessels Detected",    traffic_count)
        m2.metric("🔒 Security Status",     security_status)
        m3.metric("📈 Trade Analysis",      trade_stat)
        m4.metric("🖼️ Image Resolution",   f"{img_w} × {img_h} px")

        st.divider()

        # Image metadata
        with st.expander("🖼️ Image Metadata", expanded=False):
            file_kb = len(image_bytes) / 1024
            gray    = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            brightness = float(np.mean(gray))
            edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0) * 100)

            ic1, ic2, ic3, ic4 = st.columns(4)
            ic1.metric("File Size",       f"{file_kb:.1f} KB")
            ic2.metric("Dimensions",      f"{img_w} × {img_h}")
            ic3.metric("Mean Brightness", f"{brightness:.1f} / 255")
            ic4.metric("Edge Density",    f"{edge_density:.1f} %")

        # Detection summary list
        st.markdown("### Detection Summary")
        if not detections:
            st.warning("No vessels meet the current confidence threshold. Try lowering it in the sidebar.")
        else:
            for idx, det in enumerate(detections, start=1):
                risk_data = det.get("risk", {})
                risk = risk_data.get("level", "Unknown")
                dot = risk_data.get("emoji", "⚪")
                emoji = VESSEL_EMOJI.get(det["label"], "🚢")
                st.markdown(
                    f"{idx}. {emoji} **{det['label']}** — "
                    f"confidence **{float(det['confidence']):.2f}** — "
                    f"risk {dot} **{risk}**"
                )

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — VESSEL DETAILS
    # ════════════════════════════════════════════════════════════════════════
    with tab_vessels:
        st.markdown("## Vessel Detail Cards")
        if not detections:
            st.warning("No vessels detected above the current confidence threshold.")
        else:
            for idx, det in enumerate(detections, start=1):
                label = det["label"]
                conf  = float(det["confidence"])
                x1, y1, x2, y2 = det["bbox"]
                emoji = VESSEL_EMOJI.get(label, "🚢")
                risk_data = det.get("risk", {})
                risk = risk_data.get("level", "Unknown")
                dot = risk_data.get("emoji", "⚪")
                size_lbl = det.get("size", size_label(x1, y1, x2, y2, img_h, img_w))
                pos_lbl  = position_label(x1, y1, x2, y2, img_h, img_w)
                desc     = VESSEL_DESCRIPTION.get(label, "")

                with st.expander(f"{emoji}  Vessel {idx} — {label}  ({conf:.2f})", expanded=(idx == 1)):
                    vc1, vc2, vc3 = st.columns(3)
                    vc1.metric("Vessel Type",   f"{emoji} {label}")
                    vc2.metric("Confidence",    f"{conf:.0%}")
                    vc3.metric("Risk Level",    f"{dot} {risk}")

                    vc4, vc5, vc6 = st.columns(3)
                    vc4.metric("Est. Size",     size_lbl)
                    vc5.metric("Image Position", pos_lbl)
                    vc6.metric("Bounding Box",  f"{x2-x1} × {y2-y1} px")

                    st.caption(f"ℹ️ {desc}")

                    # Crop of the detected region
                    roi = image_array[y1:y2, x1:x2]
                    if roi.size > 0:
                        st.image(roi, caption=f"Detected region — {label}", width=280)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 3 — ANALYTICS
    # ════════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.markdown("## Detection Analytics")

        if not detections:
            st.info("No detections to chart. Upload an image and adjust the threshold.")
        else:
            ac1, ac2 = st.columns(2)

            # ── Confidence bar chart ──────────────────────────────────────
            with ac1:
                st.markdown("### Confidence per Vessel")
                conf_data = {
                    f"{VESSEL_EMOJI.get(d['label'],'')} {d['label']}": float(d["confidence"])
                    for d in detections
                }
                st.bar_chart(conf_data, color="#00DC00", height=300)

            # ── Risk distribution ────────────────────────────────────────
            with ac2:
                st.markdown("### Risk Distribution")
                risk_counts: dict = {"High": 0, "Medium": 0, "Low": 0}
                for d in detections:
                    r = d.get("risk", {}).get("level", "Low")
                    if r in risk_counts:
                        risk_counts[r] += 1
                st.bar_chart(
                    {"🔴 High": risk_counts["High"],
                     "🟡 Medium": risk_counts["Medium"],
                     "🟢 Low": risk_counts["Low"]},
                    color="#e74c3c",
                    height=300,
                )

            st.divider()

            # ── Image channel histograms ─────────────────────────────────
            st.markdown("### Image Channel Histograms")
            hc1, hc2, hc3 = st.columns(3)
            channel_pairs = [("Red Channel",   0, "#e74c3c"),
                             ("Green Channel", 1, "#2ecc71"),
                             ("Blue Channel",  2, "#3498db")]
            for col_ctx, (ch_name, ch_idx, ch_color) in zip([hc1, hc2, hc3], channel_pairs):
                with col_ctx:
                    st.markdown(f"**{ch_name}**")
                    hist, bins = np.histogram(image_array[:, :, ch_idx].flatten(), bins=32, range=(0, 256))
                    hist_dict  = {str(int(bins[i])): int(hist[i]) for i in range(len(hist))}
                    st.bar_chart(hist_dict, color=ch_color, height=200)

            st.divider()

            # ── Summary statistics table ─────────────────────────────────
            st.markdown("### Detection Statistics Table")
            rows = []
            for i, det in enumerate(detections, 1):
                x1, y1, x2, y2 = det["bbox"]
                risk_data = det.get("risk", {})
                risk = risk_data.get("level", "Unknown")
                dot = risk_data.get("emoji", "⚪")
                rows.append({
                    "#":         i,
                    "Vessel":    f"{VESSEL_EMOJI.get(det['label'],'')} {det['label']}",
                    "Confidence": f"{float(det['confidence']):.2f}",
                    "Risk":       f"{dot} {risk}",
                    "Est. Size":  det.get("size", size_label(x1, y1, x2, y2, img_h, img_w)),
                    "Position":   position_label(x1, y1, x2, y2, img_h, img_w),
                    "Box (px)":   f"{x2-x1} × {y2-y1}",
                })
            st.dataframe(rows, use_container_width=True, hide_index=True)

    # ════════════════════════════════════════════════════════════════════════
    # TAB 4 — EXPORT
    # ════════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown("## Export Results")

        ec1, ec2 = st.columns(2)

        with ec1:
            st.markdown("### 🖼️ Annotated Image")
            st.caption("Download the detection overlay image as PNG.")
            png_bytes = annotated_png_bytes(annotated_array)
            st.download_button(
                label="⬇️ Download Annotated PNG",
                data=png_bytes,
                file_name=f"detections_{uploaded_file.name.rsplit('.', 1)[0]}.png",
                mime="image/png",
                use_container_width=True,
            )
            st.image(annotated_array, caption="Preview", use_container_width=True)

        with ec2:
            st.markdown("### 📄 Detection Report (CSV)")
            st.caption("Download all vessel detections as a CSV file.")
            csv_str = csv_report(detections, img_h, img_w)
            st.download_button(
                label="⬇️ Download CSV Report",
                data=csv_str,
                file_name=f"report_{uploaded_file.name.rsplit('.', 1)[0]}.csv",
                mime="text/csv",
                use_container_width=True,
            )
            if detections:
                st.markdown("**CSV Preview:**")
                st.code(csv_str, language="text")
            else:
                st.info("No detections to export.")


if __name__ == "__main__":
    main()
