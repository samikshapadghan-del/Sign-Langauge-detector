"""Collect additional alphabet landmark samples from the local camera."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import mediapipe as mp


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR.parent / "data" / "landmarks.csv"
LANDMARKER_FILE = BASE_DIR / "hand_landmarker.task"
SIGNS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def collect(sign: str, samples: int) -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Camera 0 could not be opened.")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
    )
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    recording = False

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while count < samples:
                ok, frame = camera.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                )

                if recording and result.hand_landmarks:
                    row = [sign]
                    for point in result.hand_landmarks[0]:
                        row.extend((point.x, point.y, point.z))
                    with DATA_FILE.open("a", newline="", encoding="utf-8") as handle:
                        csv.writer(handle).writerow(row)
                    count += 1

                color = (30, 220, 170) if recording else (80, 120, 255)
                cv2.putText(
                    frame, f"{sign}: {count}/{samples}", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
                )
                cv2.putText(
                    frame, "SPACE start/pause | Q quit", (15, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                )
                cv2.imshow("Collect Sign Landmarks", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(" "):
                    recording = not recording
                elif key == ord("q"):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"Saved {count} samples for {sign} to {DATA_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("sign", choices=SIGNS, type=str.upper)
    parser.add_argument("--samples", type=int, default=200)
    arguments = parser.parse_args()
    collect(arguments.sign, max(1, arguments.samples))
