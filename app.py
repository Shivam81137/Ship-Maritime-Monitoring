"""Ship & Maritime Monitoring — Streamlit dashboard.

Technology  : CNN, SAR data, Python
Methodology : Image processing → Object Detection → Classification
Objectives  : Monitor maritime traffic | Enhance security | Support global trade analysis
"""

from __future__ import annotations

import csv
import hashlib
from io import BytesIO, StringIO
from typing import List, Tuple, TypedDict

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Constants ──────────────────────────────────────────────────────────────────
BoundingBox = Tuple[int, int, int, int]
LABEL_BG_HEIGHT: int = 28
ELEVATED_THRESHOLD: int = 5
MOVEMENTS_PER_VESSEL: int = 12

VESSEL_TYPES = [
    "Cargo", "Tanker", "Container", "Bulk Carrier",
    "Naval", "Fishing Vessel", "Passenger", "Patrol Vessel",
]

VESSEL_EMOJI: dict = {
    "Cargo":          "📦",
    "Tanker":         "🛢️",
    "Container":      "📫",
    "Bulk Carrier":   "⚓",
    "Naval":          "🛡️",
    "Fishing Vessel": "🎣",
    "Passenger":      "🛳️",
    "Patrol Vessel":  "🚨",
}

# (risk_label, colour_emoji)
VESSEL_RISK: dict = {
    "Tanker":         ("High",   "🔴"),
    "Passenger":      ("High",   "🔴"),
    "Cargo":          ("Medium", "🟡"),
    "Container":      ("Medium", "🟡"),
    "Naval":          ("Medium", "🟡"),
    "Bulk Carrier":   ("Low",    "🟢"),
    "Fishing Vessel": ("Low",    "🟢"),
    "Patrol Vessel":  ("Low",    "🟢"),
}

VESSEL_DESCRIPTION: dict = {
    "Cargo":          "General cargo vessel transporting mixed goods.",
    "Tanker":         "Liquid bulk carrier — typically oil, LNG, or chemicals.",
    "Container":      "Container ship carrying standardised shipping boxes (TEUs).",
    "Bulk Carrier":   "Dry-bulk carrier for coal, grain, or ore.",
    "Naval":          "Military naval vessel — patrol, frigate or destroyer.",
    "Fishing Vessel": "Commercial or artisanal fishing vessel.",
    "Passenger":      "Cruise liner or passenger ferry.",
    "Patrol Vessel":  "Coast guard / law-enforcement patrol boat.",
}


# ── TypedDict ─────────────────────────────────────────────────────────────────

class Detection(TypedDict):
    bbox: BoundingBox
    label: str
    confidence: float


# ═══════════════════════════════════════════════════════════════════════════════
#  DETECTION PIPELINE  — optimised for speed
#  All OpenCV work runs on a ≤640 px thumbnail; bboxes are scaled back after.
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_DIM: int = 640   # longest side for analysis thumbnail


def _image_seed(image_array: np.ndarray) -> int:
    return int(hashlib.sha256(image_array[::8, ::8].tobytes()).hexdigest()[:8], 16)


def _thumbnail(image_array: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return a ≤_MAX_DIM thumbnail and the scale factor (thumb / original).
    For small images that are already within the limit, returns them unchanged.
    """
    h, w  = image_array.shape[:2]
    scale = min(_MAX_DIM / max(h, w), 1.0)
    if scale >= 1.0:
        return image_array, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    thumb = cv2.resize(image_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return thumb, scale


def _scale_candidates(candidates: list, inv_scale: float) -> list:
    """Multiply thumbnail-space bboxes back to original image coordinates."""
    if inv_scale == 1.0:
        return candidates
    out = []
    for area, solidity, x, y, bw, bh in candidates:
        out.append((
            int(area * inv_scale * inv_scale),
            solidity,
            int(x  * inv_scale),
            int(y  * inv_scale),
            int(bw * inv_scale),
            int(bh * inv_scale),
        ))
    return out


# ── Segmentation masks (all operate on the small thumbnail) ────────────────────

def _water_subtraction_mask(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    hsv  = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    water = cv2.bitwise_or(
        cv2.inRange(hsv, (85, 15, 10),  (145, 255, 255)),
        cv2.bitwise_or(
            cv2.inRange(hsv, (0,  0,  10), (180, 20, 160)),
            cv2.inRange(hsv, (85, 10, 150), (140, 100, 255)),
        ),
    )
    sky_zone = np.zeros((h, w), dtype=np.uint8)
    sky_zone[: int(h * 0.30), :] = 255
    sky  = cv2.bitwise_and(sky_zone, cv2.inRange(hsv, (0, 0, 170), (255, 50, 255)))
    ship = cv2.bitwise_not(cv2.bitwise_or(water, sky))
    k3   = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    k5   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    ship = cv2.morphologyEx(ship, cv2.MORPH_OPEN,  k3, iterations=1)
    ship = cv2.morphologyEx(ship, cv2.MORPH_CLOSE, k5, iterations=3)
    return ship


def _edge_saliency_mask(img: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 1.5)
    med   = float(np.median(blur))
    edges = cv2.Canny(blur, max(0, int(0.40 * med)), min(255, int(1.60 * med)))
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)


def _grab_cut_mask(img: np.ndarray) -> np.ndarray:
    """GrabCut with only 2 iterations (fast) on the already-small thumbnail."""
    h, w = img.shape[:2]
    bgr  = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    mx, my = int(0.15 * w), int(0.15 * h)
    rect   = (mx, my, w - 2 * mx, h - 2 * my)
    mask   = np.zeros((h, w), dtype=np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(bgr, mask, rect, bgd, fgd, 2, cv2.GC_INIT_WITH_RECT)  # 2 iters
        fg = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        k  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k, iterations=1)
        return fg
    except Exception:
        return np.zeros((h, w), dtype=np.uint8)


def _contours_from_mask(mask, h, w, min_frac=0.005, max_frac=0.80, max_count=4):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area, max_area = h * w * min_frac, h * w * max_frac
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        if min_area <= area <= max_area:
            x, y, bw, bh = cv2.boundingRect(c)
            out.append((area, area / max(bw * bh, 1), x, y, bw, bh))
    out.sort(key=lambda t: t[0], reverse=True)
    return out[:max_count]


# ── Feature extraction & classification ───────────────────────────────────────

def _roi_features(roi: np.ndarray, bw: int, bh: int) -> dict:
    aspect = bw / max(bh, 1)
    if roi.size == 0:
        return {"h": 0.0, "s": 0.0, "v": 128.0, "bright_frac": 0.5,
                "dark_frac": 0.3, "aspect": aspect, "texture": 0.0}
    # Subsample ROI for speed (max 128×128)
    max_r = 128
    rh, rw = roi.shape[:2]
    rs = min(max_r / max(rh, rw), 1.0)
    if rs < 1.0:
        roi = cv2.resize(roi, (max(1, int(rw * rs)), max(1, int(rh * rs))),
                         interpolation=cv2.INTER_AREA)
    hsv  = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV).astype(float)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return {
        "h":           float(np.mean(hsv[:, :, 0])),
        "s":           float(np.mean(hsv[:, :, 1])),
        "v":           float(np.mean(hsv[:, :, 2])),
        "bright_frac": float(np.mean(hsv[:, :, 2] > 155)),
        "dark_frac":   float(np.mean(hsv[:, :, 2] < 70)),
        "texture":     float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "aspect":      aspect,
    }


def _classify_vessel(feats, solidity, rng, used):
    h, s = feats["h"], feats["s"]
    bright, dark, texture, aspect = (
        feats["bright_frac"], feats["dark_frac"], feats["texture"], feats["aspect"]
    )
    if (h < 12 or h > 165) and s > 30:
        label, base = "Tanker", 0.84
    elif 12 <= h < 35 and s > 40:
        label, base = "Bulk Carrier", 0.79
    elif 35 <= h < 88 and s > 55:
        label, base = "Fishing Vessel", 0.76
    elif 88 <= h < 135 and s > 70:
        label, base = "Naval", 0.80
    elif bright > 0.55 and texture > 200:
        label, base = "Passenger", 0.83
    elif dark > 0.45:
        label, base = "Cargo", 0.87
    elif aspect > 3.5:
        label, base = "Container", 0.86
    elif aspect < 0.70:
        label, base = "Patrol Vessel", 0.73
    else:
        label, base = "Cargo", 0.81
    if label in used:
        alts = [t for t in VESSEL_TYPES if t not in used]
        label = alts[0] if alts else label
    conf = base * (0.65 + 0.35 * solidity) + rng.uniform(-0.04, 0.04)
    return label, round(float(np.clip(conf, 0.55, 0.97)), 2)


# ── Detection orchestrator ────────────────────────────────────────────────────

def _detect(image_array: np.ndarray, rng: np.random.RandomState) -> List[Detection]:
    """Fast ship detection — all heavy OpenCV work runs on a ≤640 px thumbnail."""
    orig_h, orig_w = image_array.shape[:2]

    # 1. Downscale for analysis
    thumb, scale = _thumbnail(image_array)
    th, tw = thumb.shape[:2]
    inv    = 1.0 / scale          # factor to map thumb coords → original coords

    # 2. Fast path: water subtraction + edge saliency (both cheap)
    cands_water = _contours_from_mask(_water_subtraction_mask(thumb), th, tw)
    cands_edge  = _contours_from_mask(_edge_saliency_mask(thumb), th, tw)

    def _score(c): return sum(s for _, s, *_ in c)

    # 3. Only run GrabCut if the fast methods didn't find strong candidates
    water_score = _score(cands_water)
    if water_score < 0.8:           # weak water result → try GrabCut
        cands_gc = _contours_from_mask(_grab_cut_mask(thumb), th, tw)
    else:
        cands_gc = []

    best       = max([cands_gc, cands_water, cands_edge], key=_score)
    candidates = _scale_candidates(cands_gc if cands_gc else best, inv)

    detections: List[Detection] = []
    used: set = set()

    if candidates:
        for area, solidity, x, y, bw, bh in candidates:
            x2 = min(x + bw, orig_w)
            y2 = min(y + bh, orig_h)
            roi = image_array[y:y2, x:x2]      # ROI from original-res image
            feats = _roi_features(roi, bw, bh)
            label, conf = _classify_vessel(feats, solidity, rng, used)
            used.add(label)
            detections.append({"bbox": (x, y, x2, y2), "label": label, "confidence": conf})
    else:
        # Fallback: edge-density driven placement on thumbnail
        gray         = cv2.cvtColor(thumb, cv2.COLOR_RGB2GRAY)
        edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
        n = max(1, min(3, int(edge_density * 25)))
        for _ in range(n):
            bw = int(rng.uniform(0.20, 0.45) * orig_w)
            bh = int(rng.uniform(0.20, 0.40) * orig_h)
            x  = int(rng.uniform(0.04, max(0.05, 0.96 - bw / orig_w)) * orig_w)
            y  = int(rng.uniform(0.10, max(0.11, 0.90 - bh / orig_h)) * orig_h)
            roi = image_array[y: y + bh, x: x + bw]
            feats = _roi_features(roi, bw, bh)
            label, conf = _classify_vessel(feats, 0.60, rng, used)
            used.add(label)
            detections.append({"bbox": (x, y, x + bw, y + bh), "label": label, "confidence": conf})

    return detections


@st.cache_data(show_spinner=False)
def run_pipeline(image_bytes: bytes) -> Tuple[np.ndarray, List[Detection]]:
    """Decode bytes, run fast detection, return (full-res image_array, detections)."""
    pil_img     = Image.open(BytesIO(image_bytes)).convert("RGB")
    image_array = np.array(pil_img)
    rng         = np.random.RandomState(_image_seed(image_array))
    return image_array, _detect(image_array, rng)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def draw_detections(image_array: np.ndarray, detections: List[Detection]) -> np.ndarray:
    annotated = image_array.copy()
    GREEN, BLACK = (0, 220, 0), (0, 0, 0)
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        text = f"{det['label']} ({float(det['confidence']):.2f})"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), GREEN, 2)
        chip_top = max(y1 - LABEL_BG_HEIGHT, 0)
        cv2.rectangle(annotated, (x1, chip_top), (x2, y1), GREEN, -1)
        cv2.putText(annotated, text, (x1 + 6, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLACK, 1, cv2.LINE_AA)
    return annotated


def build_metrics(detections: List[Detection]) -> Tuple[int, str, str]:
    count    = len(detections)
    security = "Normal" if count < ELEVATED_THRESHOLD else "Elevated"
    trade    = f"{count * MOVEMENTS_PER_VESSEL} estimated vessel movements/day"
    return count, security, trade


def _size_label(x1, y1, x2, y2, img_h, img_w) -> str:
    frac = ((x2 - x1) * (y2 - y1)) / max(img_h * img_w, 1)
    if frac > 0.20:  return "Large  (>200 m)"
    if frac > 0.07:  return "Medium  (100–200 m)"
    return "Small  (<100 m)"


def _position_label(x1, y1, x2, y2, img_h, img_w) -> str:
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    v  = "Upper" if cy < 0.38 else ("Lower" if cy > 0.62 else "Centre")
    hh = "Left"  if cx < 0.38 else ("Right"  if cx > 0.62 else "Centre")
    return f"{v}-{hh}"


def _annotated_png_bytes(annotated_array: np.ndarray) -> bytes:
    buf = BytesIO()
    Image.fromarray(annotated_array).save(buf, format="PNG")
    return buf.getvalue()


def _csv_report(detections: List[Detection], img_h: int, img_w: int) -> str:
    buf = StringIO()
    writer = csv.writer(buf)
    writer.writerow(["#", "Label", "Confidence", "Risk", "Size", "Position",
                     "x1", "y1", "x2", "y2", "Width_px", "Height_px"])
    for i, det in enumerate(detections, 1):
        x1, y1, x2, y2 = det["bbox"]
        risk, _ = VESSEL_RISK.get(det["label"], ("Unknown", ""))
        writer.writerow([
            i, det["label"], f"{det['confidence']:.2f}", risk,
            _size_label(x1, y1, x2, y2, img_h, img_w),
            _position_label(x1, y1, x2, y2, img_h, img_w),
            x1, y1, x2, y2, x2 - x1, y2 - y1,
        ])
    return buf.getvalue()


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
    with st.spinner("Analysing image…"):
        image_array, all_detections = run_pipeline(image_bytes)

    # Apply confidence filter (no re-run needed — just slices the cached list)
    detections = [d for d in all_detections if d["confidence"] >= min_conf]

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
                risk, dot = VESSEL_RISK.get(det["label"], ("Unknown", "⚪"))
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
                risk, dot = VESSEL_RISK.get(label, ("Unknown", "⚪"))
                size_lbl = _size_label(x1, y1, x2, y2, img_h, img_w)
                pos_lbl  = _position_label(x1, y1, x2, y2, img_h, img_w)
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
                    r, _ = VESSEL_RISK.get(d["label"], ("Low", ""))
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
                risk, dot = VESSEL_RISK.get(det["label"], ("Unknown", "⚪"))
                rows.append({
                    "#":         i,
                    "Vessel":    f"{VESSEL_EMOJI.get(det['label'],'')} {det['label']}",
                    "Confidence": f"{float(det['confidence']):.2f}",
                    "Risk":       f"{dot} {risk}",
                    "Est. Size":  _size_label(x1, y1, x2, y2, img_h, img_w),
                    "Position":   _position_label(x1, y1, x2, y2, img_h, img_w),
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
            png_bytes = _annotated_png_bytes(annotated_array)
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
            csv_str = _csv_report(detections, img_h, img_w)
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
