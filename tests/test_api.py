"""Integration tests for FastAPI microservice endpoints."""

from __future__ import annotations

import unittest
from io import BytesIO

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from main_api import app


def _make_bytes(width: int = 200, height: int = 150, colour: tuple = (30, 80, 140)) -> bytes:
    """Return PNG bytes for a solid-colour image."""
    arr = np.full((height, width, 3), colour, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_root_endpoint(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "online")
        self.assertEqual(data.get("model"), "yolov8n.pt")

    def test_analyze_endpoint_valid_image(self) -> None:
        img_bytes = _make_bytes()
        response = self.client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            params={"min_confidence": 0.50}
        )
        self.assertEqual(response.status_code, 200)
        detections = response.json()
        self.assertIsInstance(detections, list)
        
        for det in detections:
            self.assertIn("bbox", det)
            self.assertIn("label", det)
            self.assertIn("confidence", det)
            self.assertIn("size", det)
            self.assertIn("risk", det)
            
            self.assertIsInstance(det["bbox"], list)
            self.assertEqual(len(det["bbox"]), 4)
            self.assertIsInstance(det["label"], str)
            self.assertIsInstance(det["confidence"], float)
            self.assertIsInstance(det["size"], str)
            self.assertIsInstance(det["risk"], dict)
            self.assertIn("level", det["risk"])
            self.assertIn("emoji", det["risk"])

    def test_analyze_endpoint_confidence_filter(self) -> None:
        img_bytes = _make_bytes()
        # Set confidence threshold to 1.0 (should filter out all detections)
        response = self.client.post(
            "/analyze",
            files={"file": ("test.png", img_bytes, "image/png")},
            params={"min_confidence": 1.0}
        )
        self.assertEqual(response.status_code, 200)
        detections = response.json()
        self.assertEqual(len(detections), 0)

    def test_analyze_endpoint_invalid_file_type(self) -> None:
        response = self.client.post(
            "/analyze",
            files={"file": ("test.txt", b"Hello, World!", "text/plain")}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("must be an image", data["detail"])


if __name__ == "__main__":
    unittest.main()
