
import cv2
import mediapipe as mp
import pickle

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

MODEL_PATH = "hand_landmarker.task"

with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("model/labels.pkl", "rb") as f:
    labels = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

def run_detector():
    cap = cv2.VideoCapture(0)
    sentence = []
    last_sign = ""
    stable_count = 0
    STABLE_THRESHOLD = 20

    print("Live Detection Running!")
    print("SPACE = add space | C = clear sentence | Q = quit")

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            sign_text = ""
            confidence = 0.0

            if result.hand_landmarks:
                # Draw landmarks
                for hand in result.hand_landmarks:
                    for lm in hand:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                # Predict
                row = []
                for lm in result.hand_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z])

                sign_text = model.predict([row])[0]
                confidence = max(model.predict_proba([row])[0])

                # Stable detection
                if sign_text == last_sign:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_sign = sign_text

                if stable_count == STABLE_THRESHOLD:
                    sentence.append(sign_text)
                    stable_count = 0

            # Display
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {sign_text}  ({confidence:.0%})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Text: {''.join(sentence[-20:])}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("ISL Sign Detector", frame)
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            if key == ord("c"):
                sentence = []
            if key == ord(" "):
                sentence.append(" ")

    cap.release()
    cv2.destroyAllWindows()
    print("Final text:", "".join(sentence))

if __name__ == "__main__":
    run_detector()