import cv2
import mediapipe as mp
import csv
import os
import urllib.request

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

DATASET_DIR = "../dataset"
OUTPUT_CSV  = "../data/landmarks.csv"
os.makedirs("../data", exist_ok=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def extract_all():
    signs = sorted(os.listdir(DATASET_DIR))
    print(f"Found signs: {signs}")

    total = 0
    skipped = 0

    # Clear old CSV
    open(OUTPUT_CSV, "w").close()

    with HandLandmarker.create_from_options(options) as landmarker:
        for sign in signs:
            sign_dir = os.path.join(DATASET_DIR, sign)
            if not os.path.isdir(sign_dir):
                continue

            images = os.listdir(sign_dir)
            sign_count = 0
            print(f"Processing {sign} ({len(images)} images)...")

            for img_file in images:
                img_path = os.path.join(sign_dir, img_file)
                frame = cv2.imread(img_path)
                if frame is None:
                    skipped += 1
                    continue

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect(mp_image)

                if result.hand_landmarks:
                    row = [sign]
                    for lm in result.hand_landmarks[0]:
                        row.extend([lm.x, lm.y, lm.z])

                    with open(OUTPUT_CSV, "a", newline="") as f:
                        csv.writer(f).writerow(row)

                    sign_count += 1
                    total += 1
                else:
                    skipped += 1

            print(f"  ✓ {sign}: {sign_count} landmarks extracted")

    print(f"\nDone! Total: {total} samples, Skipped: {skipped}")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_all()