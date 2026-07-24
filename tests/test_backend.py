import unittest

import numpy as np

from backend.app import DetectionService
from backend.features import FEATURE_COUNT, normalize_landmarks


class FeatureTests(unittest.TestCase):
    def test_normalization_is_wrist_relative_and_scaled(self):
        points = np.zeros((21, 3), dtype=np.float32)
        points[:, 0] = np.arange(21) + 10
        points[:, 1] = np.arange(21) * 2 + 5
        normalized = normalize_landmarks(points.reshape(-1)).reshape(21, 3)
        self.assertEqual(normalized.size, FEATURE_COUNT)
        np.testing.assert_allclose(normalized[0], 0)
        self.assertAlmostEqual(
            float(np.max(np.linalg.norm(normalized[:, :2], axis=1))), 1.0, places=6
        )


class ActionTests(unittest.TestCase):
    def setUp(self):
        self.service = DetectionService()

    def test_sentence_actions_and_settings(self):
        self.service.handle_action({"action": "append_text", "value": "hello"})
        self.assertEqual(self.service.snapshot(False)["sentence"], "HELLO ")
        self.service.handle_action({"action": "backspace"})
        self.assertEqual(self.service.snapshot(False)["sentence"], "HELLO")
        self.service.handle_action({"action": "space"})
        self.service.handle_action({"action": "set_threshold", "value": 0.82})
        self.service.handle_action({"action": "set_stability", "value": 18})
        snapshot = self.service.snapshot(False)
        self.assertEqual(snapshot["sentence"], "HELLO ")
        self.assertEqual(snapshot["threshold"], 0.82)
        self.assertEqual(snapshot["stable_target"], 18)

    def test_word_suggestions(self):
        self.service.sentence = list("HEL")
        suggestions = self.service.snapshot(False)["suggestions"]
        self.assertIn("HELLO", suggestions)

    def test_accept_current_sign_action(self):
        self.service.sign = "A"
        self.service.handle_action({"action": "accept_current"})
        snapshot = self.service.snapshot(False)
        self.assertEqual(snapshot["sentence"], "A")
        self.assertEqual(snapshot["accepted"], "A")


if __name__ == "__main__":
    unittest.main()
