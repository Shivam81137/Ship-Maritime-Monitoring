"""Focused tests for Streamlit maritime monitoring helpers."""

from __future__ import annotations

import importlib
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np


if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = MagicMock()

app = importlib.import_module("app")


class AppHelperTests(unittest.TestCase):
    def test_run_mock_pipeline_returns_expected_structure(self) -> None:
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        detections = app.run_mock_pipeline(image)

        self.assertEqual(len(detections), 2)
        for detection in detections:
            self.assertIn("bbox", detection)
            self.assertIn("label", detection)
            self.assertIn("confidence", detection)
            x1, y1, x2, y2 = detection["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLessEqual(x2, image.shape[1])
            self.assertLessEqual(y2, image.shape[0])

    def test_draw_detections_returns_new_array_and_handles_empty(self) -> None:
        image = np.zeros((120, 180, 3), dtype=np.uint8)
        original_copy = image.copy()

        empty_output = app.draw_detections(image, [])
        self.assertTrue(np.array_equal(empty_output, original_copy))
        self.assertTrue(np.array_equal(image, original_copy))

        detections = app.run_mock_pipeline(image)
        annotated = app.draw_detections(image, detections)
        self.assertFalse(np.array_equal(annotated, original_copy))
        self.assertTrue(np.array_equal(image, original_copy))

    def test_build_metrics_security_status_thresholds(self) -> None:
        normal_metrics = app.build_metrics([{"bbox": (0, 0, 1, 1), "label": "Cargo", "confidence": 0.9}])
        self.assertEqual(normal_metrics[0], 1)
        self.assertEqual(normal_metrics[1], "Normal")
        expected_trade = f"{1 * app.DAILY_MOVEMENT_MULTIPLIER} estimated vessel movements/day"
        self.assertEqual(normal_metrics[2], expected_trade)

        elevated_input = [
            {"bbox": (0, 0, 1, 1), "label": "Cargo", "confidence": 0.9}
            for _ in range(app.ELEVATED_SECURITY_THRESHOLD)
        ]
        elevated_metrics = app.build_metrics(elevated_input)
        self.assertEqual(elevated_metrics[1], "Elevated")


if __name__ == "__main__":
    unittest.main()
