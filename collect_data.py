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
    print("Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )
    print("Downloaded!")

DATA_FILE = "../data/landmarks.csv"
os.makedirs("../data", exist_ok=True)

SIGNS = [
    "A","B","C","D","E","F","G","H","I","J","K","L","M",
    "N","O","P","Q","R","S","T","U","V","W","X","Y","Z"
]

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def collect(sign_label, samples=200):
    cap = cv2.VideoCapture(0)
    count = 0
    recording = False
    print(f"\nGet ready to sign: {sign_label}")
    print("Press SPACE to start recording, Q to quit")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for lm in hand:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                if recording:
                    row = [sign_label]
                    for lm in result.hand_landmarks[0]:
                        row.extend([lm.x, lm.y, lm.z])
                    with open(DATA_FILE, "a", newline="") as f:
                        csv.writer(f).writerow(row)
                    count += 1
                    if count >= samples:
                        print(f"Done! Collected {samples} samples for {sign_label}")
                        break

            status = f"Sign: {sign_label} | {count}/{samples}"
            color = (0, 255, 0) if recording else (0, 0, 255)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, "SPACE=Start  Q=Quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow("Collect Data - ISL", frame)
            key = cv2.waitKey(1)
            if key == ord(" "):
                recording = True
                print("Recording started!")
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("=== ISL Data Collector ===")
    print("Available signs:", SIGNS)
    sign = input("Enter sign name exactly as shown: ").strip()
    if sign in SIGNS:
        collect(sign)
    else:
        print(f"'{sign}' not in list!")