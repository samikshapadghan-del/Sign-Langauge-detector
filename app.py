from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import cv2
import mediapipe as mp
import pickle
import base64
import asyncio
import threading
import time

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

with open("model/sign_model.pkl", "rb") as f:
    model = pickle.load(f)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

state = {
    "frame": None,
    "sign": "",
    "confidence": 0.0,
    "sentence": [],
}

def camera_loop():
    cap = cv2.VideoCapture(0)
    print("Camera opened:", cap.isOpened())
    last_sign = ""
    stable_count = 0
    last_added_time = 0

    with HandLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            sign_text = ""
            confidence = 0.0

            if result.hand_landmarks:
                for hand in result.hand_landmarks:
                    for lm in hand:
                        h, w, _ = frame.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                row = []
                for lm in result.hand_landmarks[0]:
                    row.extend([lm.x, lm.y, lm.z])

                sign_text = model.predict([row])[0]
                confidence = float(max(model.predict_proba([row])[0]))

                if sign_text == last_sign:
                    stable_count += 1
                else:
                    stable_count = 0
                    last_sign = sign_text

                now = time.time()
                if stable_count >= 40 and confidence > 0.75 and (now - last_added_time) > 1.5:
                    state["sentence"].append(sign_text)
                    last_added_time = now
                    stable_count = 0

            _, buffer = cv2.imencode(".jpg", frame)
            state["frame"] = base64.b64encode(buffer).decode("utf-8")
            state["sign"] = sign_text
            state["confidence"] = confidence

t = threading.Thread(target=camera_loop, daemon=True)
t.start()
print("Camera thread started!")

@app.get("/")
async def index():
    return FileResponse("../frontend/index.html")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected!")
    try:
        while True:
            if state["frame"]:
                await websocket.send_json({
                    "frame": state["frame"],
                    "sign": state["sign"],
                    "confidence": state["confidence"],
                    "sentence": "".join(state["sentence"][-30:])
                })
            try:
                data = await asyncio.wait_for(
                    websocket.receive_json(), timeout=0.01
                )
                if data.get("action") == "clear":
                    state["sentence"] = []
                elif data.get("action") == "space":
                    state["sentence"].append(" ")
                elif data.get("action") == "backspace":
                    if state["sentence"]:
                        state["sentence"].pop()
            except:
                pass
            await asyncio.sleep(0.033)
    except Exception as e:
        print(f"Client disconnected: {e}")