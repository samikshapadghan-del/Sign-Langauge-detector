"""Optional desktop preview using the same model pipeline as the web app."""

from __future__ import annotations

import pickle
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from features import normalize_landmarks


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "model" / "sign_model.pkl"
LANDMARKER_FILE = BASE_DIR / "hand_landmarker.task"


def run_detector() -> None:
    with MODEL_FILE.open("rb") as handle:
        model = pickle.load(handle)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera 0 could not be opened.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )

    print("Desktop detector running. Press Q to quit.")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )

                label = "No hand"
                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]
                    values = [value for point in hand for value in (point.x, point.y, point.z)]
                    features = normalize_landmarks(values)
                    probabilities = model.predict_proba([features])[0]
                    index = int(np.argmax(probabilities))
                    label = f"{model.classes_[index]}  {probabilities[index]:.0%}"

                cv2.putText(
                    frame, label, (20, 42), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (30, 230, 180), 2, cv2.LINE_AA,
                )
                cv2.imshow("Sign Language Detector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_detector()
