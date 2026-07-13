"""Feature preparation shared by training and live inference."""

from __future__ import annotations

import numpy as np


LANDMARK_COUNT = 21
FEATURE_COUNT = LANDMARK_COUNT * 3


def normalize_landmarks(values: np.ndarray | list[float]) -> np.ndarray:
    """Return wrist-relative, scale-normalized 3D hand landmarks."""
    points = np.asarray(values, dtype=np.float32).reshape(LANDMARK_COUNT, 3).copy()
    points -= points[0]

    scale = float(np.max(np.linalg.norm(points[:, :2], axis=1)))
    if scale > 1e-8:
        points /= scale

    return points.reshape(FEATURE_COUNT)


def normalize_batch(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1, LANDMARK_COUNT, 3)
    return np.vstack([normalize_landmarks(sample) for sample in array])
