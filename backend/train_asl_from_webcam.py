"""Collect ASL alphabet samples from your webcam and retrain the detector.

Use the ASL poster in `frontend/assets/asl_alphabet.jpg` as the hand-shape guide.
This records MediaPipe hand landmarks into `data/landmarks.csv`, then calls
`train_model.py` to build `backend/model/sign_model.pkl`.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import time
from pathlib import Path

import cv2
import mediapipe as mp

from train_model import train


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "data" / "landmarks.csv"
BACKUP_FILE = PROJECT_DIR / "data" / "landmarks.backup.csv"
LANDMARKER_FILE = BASE_DIR / "hand_landmarker.task"
LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def draw_panel(frame, letter: str, count: int, samples: int, recording: bool) -> None:
    color = (40, 230, 180) if recording else (80, 140, 255)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 110), (8, 12, 28), -1)
    cv2.putText(
        frame,
        f"ASL letter: {letter}    Samples: {count}/{samples}",
        (18, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "SPACE start/pause | N next letter | Q quit",
        (18, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (235, 240, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Match the ASL alphabet poster. Vary distance/angle slightly while recording.",
        (18, 98),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (180, 205, 255),
        1,
        cv2.LINE_AA,
    )


def collect(samples_per_letter: int, append: bool) -> None:
    DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_FILE.exists() and not append:
        shutil.copy2(DATA_FILE, BACKUP_FILE)
        DATA_FILE.unlink()
        print(f"Backed up old training data to {BACKUP_FILE}")

    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam. Close Camera/Zoom/Teams and try again.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    )

    print("ASL webcam collection starting.")
    print("Use the app's ASL poster as your guide: frontend/assets/asl_alphabet.jpg")
    print("For each letter, press SPACE to record. Press N to skip/advance.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            for letter in LETTERS:
                count = 0
                recording = False
                last_saved = 0.0
                print(f"\nShow ASL letter {letter}. Press SPACE when ready.")

                while count < samples_per_letter:
                    ok, frame = camera.read()
                    if not ok:
                        raise RuntimeError("Webcam stopped returning frames.")

                    frame = cv2.flip(frame, 1)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    result = landmarker.detect(
                        mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    )

                    if result.hand_landmarks:
                        h, w = frame.shape[:2]
                        for point in result.hand_landmarks[0]:
                            cv2.circle(
                                frame,
                                (int(point.x * w), int(point.y * h)),
                                4,
                                (40, 230, 180),
                                -1,
                                cv2.LINE_AA,
                            )

                        now = time.monotonic()
                        if recording and now - last_saved >= 0.035:
                            row = [letter]
                            for point in result.hand_landmarks[0]:
                                row.extend((point.x, point.y, point.z))
                            with DATA_FILE.open("a", newline="", encoding="utf-8") as handle:
                                csv.writer(handle).writerow(row)
                            count += 1
                            last_saved = now

                    draw_panel(frame, letter, count, samples_per_letter, recording)
                    cv2.imshow("ASL Training Collector", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord(" "):
                        recording = not recording
                    elif key == ord("n"):
                        print(f"Skipped remaining samples for {letter}.")
                        break
                    elif key == ord("q"):
                        raise KeyboardInterrupt

                print(f"Collected {count} samples for {letter}.")
    finally:
        camera.release()
        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect ASL webcam samples and retrain.")
    parser.add_argument("--samples", type=int, default=100, help="Samples per letter.")
    parser.add_argument("--append", action="store_true", help="Append to existing landmarks.csv.")
    args = parser.parse_args()

    collect(max(10, args.samples), args.append)
    print("\nTraining ASL model from collected landmarks...")
    train()
    print("\nDone. Restart the app with .\\run.ps1")


if __name__ == "__main__":
    main()
