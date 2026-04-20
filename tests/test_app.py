"""Unit tests for Ship & Maritime Monitoring helpers."""

from __future__ import annotations

import importlib
import sys
import unittest
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

# Stub out streamlit so the app module can be imported without a running server.
if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

app = importlib.import_module("app")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_bytes(width: int = 200, height: int = 150, colour: tuple = (30, 80, 140)) -> bytes:
    """Return PNG bytes for a solid-colour image."""
    arr = np.full((height, width, 3), colour, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _blue_ocean_bytes(width: int = 320, height: int = 240) -> bytes:
    """Image that looks like open ocean — triggers fewer ship detections."""
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    arr[:, :] = (30, 100, 180)   # solid HSV-blue → water
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Pipeline tests ─────────────────────────────────────────────────────────────

class PipelineTests(unittest.TestCase):
    def test_returns_tuple_of_array_and_list(self) -> None:
        img_bytes = _make_bytes()
        image_array, detections = app.run_pipeline(img_bytes)
        self.assertIsInstance(image_array, np.ndarray)
        self.assertIsInstance(detections, list)

    def test_detection_schema(self) -> None:
        img_bytes = _make_bytes()
        _, detections = app.run_pipeline(img_bytes)
        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("label", det)
            self.assertIn("confidence", det)
            x1, y1, x2, y2 = det["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertGreater(x2, x1)
            self.assertGreater(y2, y1)

    def test_bboxes_within_image_bounds(self) -> None:
        img_bytes = _make_bytes(width=300, height=200)
        image_array, detections = app.run_pipeline(img_bytes)
        h, w = image_array.shape[:2]
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            self.assertLessEqual(x2, w, "x2 exceeds image width")
            self.assertLessEqual(y2, h, "y2 exceeds image height")

    def test_same_image_same_result(self) -> None:
        img_bytes = _make_bytes(colour=(50, 120, 200))
        _, d1 = app.run_pipeline(img_bytes)
        _, d2 = app.run_pipeline(img_bytes)
        self.assertEqual(d1, d2, "Same image should produce identical detections (cached).")

    def test_confidence_in_valid_range(self) -> None:
        img_bytes = _make_bytes()
        _, detections = app.run_pipeline(img_bytes)
        for det in detections:
            self.assertGreaterEqual(det["confidence"], 0.50)
            self.assertLessEqual(det["confidence"], 1.00)

    def test_labels_are_known_vessel_types(self) -> None:
        img_bytes = _make_bytes()
        _, detections = app.run_pipeline(img_bytes)
        for det in detections:
            self.assertIn(det["label"], app.VESSEL_TYPES)

    def test_no_duplicate_labels_per_image(self) -> None:
        img_bytes = _make_bytes()
        _, detections = app.run_pipeline(img_bytes)
        labels = [d["label"] for d in detections]
        self.assertEqual(len(labels), len(set(labels)), "Duplicate vessel labels found.")


# ── Drawing tests ──────────────────────────────────────────────────────────────

class DrawDetectionsTests(unittest.TestCase):
    def test_empty_detection_list_returns_copy(self) -> None:
        image = np.zeros((120, 180, 3), dtype=np.uint8)
        result = app.draw_detections(image, [])
        self.assertTrue(np.array_equal(result, image))

    def test_original_not_mutated(self) -> None:
        image = np.zeros((120, 180, 3), dtype=np.uint8)
        original_copy = image.copy()
        _, dets = app.run_pipeline(_make_bytes(180, 120))
        app.draw_detections(image, dets)
        self.assertTrue(np.array_equal(image, original_copy))

    def test_annotated_differs_from_original(self) -> None:
        image = np.zeros((150, 200, 3), dtype=np.uint8)
        det: app.Detection = {"bbox": (10, 10, 100, 80), "label": "Cargo", "confidence": 0.90}
        annotated = app.draw_detections(image, [det])
        self.assertFalse(np.array_equal(annotated, image))


# ── Metrics tests ──────────────────────────────────────────────────────────────

class BuildMetricsTests(unittest.TestCase):
    def _det(self) -> app.Detection:
        return {"bbox": (0, 0, 1, 1), "label": "Cargo", "confidence": 0.90}

    def test_normal_security(self) -> None:
        count, status, trade = app.build_metrics([self._det()])
        self.assertEqual(count, 1)
        self.assertEqual(status, "Normal")
        self.assertEqual(trade, f"{1 * app.MOVEMENTS_PER_VESSEL} estimated vessel movements/day")

    def test_elevated_security(self) -> None:
        dets = [self._det() for _ in range(app.ELEVATED_THRESHOLD)]
        _, status, _ = app.build_metrics(dets)
        self.assertEqual(status, "Elevated")

    def test_trade_stat_scales_with_count(self) -> None:
        for n in range(1, 6):
            _, _, trade = app.build_metrics([self._det()] * n)
            self.assertIn(str(n * app.MOVEMENTS_PER_VESSEL), trade)


if __name__ == "__main__":
    unittest.main()
