"""FastAPI backend for the Sign Language Detector dashboard."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import pickle
import threading
import time
from collections import Counter, deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

try:
    from .features import FEATURE_COUNT, normalize_landmarks
except ImportError:  # pragma: no cover - allows running the module directly during local development
    from features import FEATURE_COUNT, normalize_landmarks


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
LOGGER = logging.getLogger("sign-language-detector")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
FRONTEND_FILE = Path(os.getenv("FRONTEND_FILE", PROJECT_DIR / "frontend" / "index.html"))
FRONTEND_ASSETS = Path(os.getenv("FRONTEND_ASSETS", PROJECT_DIR / "frontend" / "assets"))
MODEL_FILE = Path(os.getenv("SIGN_MODEL_PATH", BASE_DIR / "model" / "sign_model.pkl"))
METADATA_FILE = Path(os.getenv("MODEL_METADATA_PATH", BASE_DIR / "model" / "metadata.json"))
LANDMARKER_FILE = Path(os.getenv("HAND_LANDMARKER_PATH", BASE_DIR / "hand_landmarker.task"))
CAMERA_ENABLED = os.getenv("CAMERA_ENABLED", "true").lower() not in {"0", "false", "no"}

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

WORD_BANK = (
    "HELLO", "HELP", "HOME", "HOSPITAL", "HUNGRY", "HAPPY",
    "PLEASE", "THANK", "THANKS", "SORRY", "YES", "NO", "WATER",
    "FOOD", "FAMILY", "FRIEND", "GOOD", "MORNING", "NIGHT", "NAME",
    "NEED", "PAIN", "DOCTOR", "MEDICINE", "BATHROOM", "WHERE", "WHAT",
    "WHEN", "WHO", "WHY", "HOW", "STOP", "START", "MORE", "FINISHED",
    "LOVE", "WORK", "SCHOOL", "CALL", "EMERGENCY", "SAFE", "DANGER",
)


def clamp(value: Any, minimum: float, maximum: float) -> float:
    try:
        return max(minimum, min(maximum, float(value)))
    except (TypeError, ValueError):
        return minimum


class DetectionService:
    def __init__(self) -> None:
        if not MODEL_FILE.exists():
            raise FileNotFoundError(f"Model not found: {MODEL_FILE}. Run train_model.py first.")
        if not LANDMARKER_FILE.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_FILE}.")

        with MODEL_FILE.open("rb") as handle:
            self.model = pickle.load(handle)
        if getattr(self.model, "n_features_in_", FEATURE_COUNT) != FEATURE_COUNT:
            raise ValueError("The classifier does not use the expected 63 landmark features.")

        self.metadata = {}
        if METADATA_FILE.exists():
            self.metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.camera: cv2.VideoCapture | None = None

        self.frame: str | None = None
        self.sign = ""
        self.confidence = 0.0
        self.sentence: list[str] = []
        self.hand_detected = False
        self.camera_status = "starting"
        self.camera_error = ""
        self.camera_index: int | None = None
        self.fps = 0.0
        self.paused = False

        self.threshold = 0.25
        self.stability_frames = 5
        self.cooldown_seconds = 0.35
        self.release_frames = 5

        self.candidate = ""
        self.stable_count = 0
        self.locked_sign = ""
        self.change_count = 0
        self.missing_count = 0
        self.last_commit_time = 0.0
        self.event_id = 0
        self.accepted = ""
        self.prediction_window: deque[tuple[str, float]] = deque(maxlen=3)
        self.debug_message = "Starting detector"

    def start(self) -> None:
        if not CAMERA_ENABLED:
            with self.lock:
                self.camera_status = "disabled"
                self.camera_error = "Camera is disabled by CAMERA_ENABLED=false."
                self.debug_message = "Camera disabled for this deployment."
            return
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._camera_loop, name="sign-camera", daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)
        if self.camera is not None:
            self.camera.release()

    def _open_camera(self) -> cv2.VideoCapture | None:
        backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
        for index in (0, 1, 2):
            for backend in backends:
                camera = cv2.VideoCapture(index, backend)
                if camera.isOpened():
                    ok, _ = camera.read()
                    if ok:
                        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
                        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
                        camera.set(cv2.CAP_PROP_FPS, 30)
                        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        self.camera_index = index
                        return camera
                camera.release()
        return None

    def _camera_loop(self) -> None:
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.25,
            min_hand_presence_confidence=0.25,
        )

        self.camera = self._open_camera()
        if self.camera is None:
            with self.lock:
                self.camera_status = "error"
                self.camera_error = "No camera could be opened. Close other camera apps and restart."
            LOGGER.error(self.camera_error)
            return

        with self.lock:
            self.camera_status = "ready"
            self.camera_error = ""
        LOGGER.info("Camera opened successfully: index %s", self.camera_index)

        frame_count = 0
        fps_started = time.perf_counter()

        try:
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                while not self.stop_event.is_set():
                    ok, frame = self.camera.read()
                    if not ok:
                        with self.lock:
                            self.camera_status = "error"
                            self.camera_error = "The camera stopped returning frames."
                        time.sleep(0.1)
                        continue

                    frame = cv2.flip(frame, 1)
                    with self.lock:
                        paused = self.paused

                    if paused:
                        self._clear_live_prediction()
                        cv2.putText(
                            frame, "DETECTION PAUSED", (25, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (80, 190, 255), 2,
                        )
                    else:
                        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        result = landmarker.detect(
                            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                        )
                        self._process_result(result, frame)

                    ok, buffer = cv2.imencode(
                        ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 78]
                    )
                    if ok:
                        encoded = base64.b64encode(buffer).decode("ascii")
                        with self.lock:
                            self.frame = encoded
                            self.camera_status = "ready"

                    frame_count += 1
                    elapsed = time.perf_counter() - fps_started
                    if elapsed >= 1.0:
                        with self.lock:
                            self.fps = round(frame_count / elapsed, 1)
                        frame_count = 0
                        fps_started = time.perf_counter()
        except Exception as error:
            LOGGER.exception("Camera worker failed")
            with self.lock:
                self.camera_status = "error"
                self.camera_error = str(error)
        finally:
            if self.camera is not None:
                self.camera.release()
            LOGGER.info("Camera worker stopped")

    def _process_result(self, result: Any, frame: np.ndarray) -> None:
        if not result.hand_landmarks:
            self._handle_missing_hand()
            return

        hand = result.hand_landmarks[0]
        height, width = frame.shape[:2]
        pixels = [(int(point.x * width), int(point.y * height)) for point in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(frame, pixels[start], pixels[end], (109, 95, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            radius = 5 if index in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(frame, point, radius, (39, 224, 190), -1, cv2.LINE_AA)

        raw_features = [coordinate for point in hand for coordinate in (point.x, point.y, point.z)]
        features = normalize_landmarks(raw_features)
        probabilities = self.model.predict_proba([features])[0]
        best_index = int(np.argmax(probabilities))
        raw_sign = str(self.model.classes_[best_index])
        raw_confidence = float(probabilities[best_index])

        self.prediction_window.append((raw_sign, raw_confidence))
        votes = Counter(sign for sign, _ in self.prediction_window)
        sign = votes.most_common(1)[0][0]
        matching_confidences = [
            confidence for vote, confidence in self.prediction_window if vote == sign
        ]
        confidence = float(sum(matching_confidences) / len(matching_confidences))

        with self.lock:
            self.hand_detected = True
            self.sign = sign
            self.confidence = confidence
            self.missing_count = 0
            self.debug_message = (
                "Hold steady"
                if confidence >= self.threshold
                else f"Low confidence ({confidence:.0%}). Move closer or lower threshold."
            )
            self._update_stability(sign, confidence)

    def _update_stability(self, sign: str, confidence: float) -> None:
        if confidence < self.threshold:
            self.candidate = ""
            self.stable_count = 0
            return

        if self.locked_sign:
            if sign == self.locked_sign:
                self.change_count = 0
                return
            self.change_count += 1
            if self.change_count < 4:
                return
            self.locked_sign = ""
            self.change_count = 0
            self.candidate = sign
            self.stable_count = 1
            return

        if sign == self.candidate:
            self.stable_count += 1
        else:
            self.candidate = sign
            self.stable_count = 1

        now = time.monotonic()
        if (
            self.stable_count >= self.stability_frames
            and now - self.last_commit_time >= self.cooldown_seconds
        ):
            self._append_text(sign)
            self.accepted = sign
            self.event_id += 1
            self.last_commit_time = now
            self.locked_sign = sign
            self.candidate = ""
            self.stable_count = 0

    def _handle_missing_hand(self) -> None:
        with self.lock:
            self.hand_detected = False
            self.sign = ""
            self.confidence = 0.0
            self.candidate = ""
            self.stable_count = 0
            self.prediction_window.clear()
            self.missing_count += 1
            self.debug_message = "No hand detected. Keep one hand fully visible in the camera."
            if self.missing_count >= self.release_frames:
                self.locked_sign = ""
                self.change_count = 0

    def _clear_live_prediction(self) -> None:
        with self.lock:
            self.hand_detected = False
            self.sign = ""
            self.confidence = 0.0
            self.candidate = ""
            self.stable_count = 0
            self.locked_sign = ""
            self.prediction_window.clear()
            self.debug_message = "Detection paused"

    def _append_text(self, text: str) -> None:
        safe_text = "".join(character for character in str(text) if character.isprintable())
        if not safe_text:
            return
        self.sentence.extend(list(safe_text[:120]))
        self.sentence = self.sentence[-240:]

    def _suggestions(self) -> list[str]:
        text = "".join(self.sentence)
        prefix = text.rsplit(" ", 1)[-1].strip().upper()
        if not prefix:
            return []
        return [word for word in WORD_BANK if word.startswith(prefix) and word != prefix][:5]

    def handle_action(self, payload: dict[str, Any]) -> None:
        action = str(payload.get("action", ""))
        with self.lock:
            if action == "clear":
                self.sentence.clear()
                self.event_id += 1
                self.accepted = ""
            elif action == "space":
                if self.sentence and self.sentence[-1] != " ":
                    self._append_text(" ")
                    self.event_id += 1
                    self.accepted = " "
            elif action == "backspace":
                if self.sentence:
                    self.sentence.pop()
                    self.event_id += 1
                    self.accepted = ""
            elif action == "accept_current":
                if self.sign:
                    self._append_text(self.sign)
                    self.accepted = self.sign
                    self.event_id += 1
                    self.last_commit_time = time.monotonic()
                    self.locked_sign = self.sign
                    self.candidate = ""
                    self.stable_count = 0
            elif action == "set_threshold":
                self.threshold = clamp(payload.get("value"), 0.05, 0.99)
            elif action in {"set_stability", "set_stability_frames"}:
                self.stability_frames = int(clamp(payload.get("value"), 1, 90))
                self.stable_count = min(self.stable_count, self.stability_frames)
            elif action == "set_cooldown":
                self.cooldown_seconds = clamp(payload.get("value"), 0.2, 3.0)
            elif action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused = False
            elif action == "toggle_pause":
                self.paused = not self.paused
            elif action in {"append_text", "append_phrase"}:
                text = str(payload.get("value", ""))
                if text:
                    if self.sentence and self.sentence[-1] != " ":
                        self._append_text(" ")
                    self._append_text(text.upper())
                    self._append_text(" ")
                    self.event_id += 1
                    self.accepted = text.upper()
            elif action == "replace_current_word":
                word = str(payload.get("value", "")).strip().upper()
                if word and word.isalpha():
                    text = "".join(self.sentence)
                    head = text.rsplit(" ", 1)[0] + " " if " " in text else ""
                    self.sentence = list((head + word + " ")[-240:])
                    self.event_id += 1
                    self.accepted = word

    def snapshot(self, include_frame: bool = True) -> dict[str, Any]:
        with self.lock:
            progress = min(1.0, self.stable_count / max(1, self.stability_frames))
            return {
                "frame": self.frame if include_frame else None,
                "sign": self.sign,
                "confidence": self.confidence,
                "sentence": "".join(self.sentence),
                "stable_count": self.stable_count,
                "stable_target": self.stability_frames,
                "stability": progress,
                "threshold": self.threshold,
                "cooldown": self.cooldown_seconds,
                "fps": self.fps,
                "hand_detected": self.hand_detected,
                "camera_status": self.camera_status,
                "camera_error": self.camera_error,
                "camera_index": self.camera_index,
                "paused": self.paused,
                "event_id": self.event_id,
                "accepted": self.accepted,
                "debug_message": self.debug_message,
                "suggestions": self._suggestions(),
            }


service = DetectionService()


@asynccontextmanager
async def lifespan(_: FastAPI):
    service.start()
    yield
    service.stop()


app = FastAPI(
    title="Sign Language Detector",
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS), name="assets")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(FRONTEND_FILE)


@app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Response:
    return Response(status_code=204)


@app.get("/api/health")
async def health() -> dict[str, Any]:
    state = service.snapshot(include_frame=False)
    return {
        "status": "ok" if state["camera_status"] in {"ready", "disabled"} else "degraded",
        "project": "Sign Language Detector",
        "camera": state["camera_status"],
        "camera_error": state["camera_error"],
        "model": service.metadata,
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send_updates() -> None:
        while True:
            await websocket.send_json(service.snapshot())
            await asyncio.sleep(0.05)

    async def receive_actions() -> None:
        while True:
            payload = await websocket.receive_json()
            if isinstance(payload, dict):
                service.handle_action(payload)

    sender = asyncio.create_task(send_updates())
    receiver = asyncio.create_task(receive_actions())
    try:
        done, pending = await asyncio.wait(
            {sender, receiver}, return_when=asyncio.FIRST_EXCEPTION
        )
        for task in done:
            task.result()
        for task in pending:
            task.cancel()
    except (WebSocketDisconnect, RuntimeError):
        pass
    finally:
        sender.cancel()
        receiver.cancel()
