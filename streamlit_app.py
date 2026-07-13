import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

from backend.features import normalize_landmarks

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "backend" / "model" / "sign_model.pkl"
METADATA_FILE = BASE_DIR / "backend" / "model" / "metadata.json"
LANDMARKER_FILE = BASE_DIR / "backend" / "hand_landmarker.task"

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")

if not MODEL_FILE.exists():
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()
if not LANDMARKER_FILE.exists():
    st.error(f"MediaPipe model file not found: {LANDMARKER_FILE}")
    st.stop()

metadata: dict[str, Any] = {}
if METADATA_FILE.exists():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

@st.cache_resource
def load_detector() -> tuple[Any, Any]:
    with MODEL_FILE.open("rb") as handle:
        model = pickle.load(handle)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return model, landmarker


try:
    model, landmarker = load_detector()
except Exception as exc:  # pragma: no cover - depends on runtime environment
    st.error(f"Could not initialize the detector: {exc}")
    st.stop()

st.title("🤟 Sign Language Detector")
st.caption("Upload an image to test the trained sign classifier.")

if "sentence" not in st.session_state:
    st.session_state.sentence = []
if "last_sign" not in st.session_state:
    st.session_state.last_sign = ""


def annotate_frame(frame: np.ndarray) -> tuple[np.ndarray, str, float]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    annotated = frame.copy()
    sign_text = ""
    confidence = 0.0

    if result.hand_landmarks:
        hand = result.hand_landmarks[0]
        height, width = annotated.shape[:2]
        pixels = [(int(point.x * width), int(point.y * height)) for point in hand]
        for start, end in HAND_CONNECTIONS:
            cv2.line(annotated, pixels[start], pixels[end], (109, 95, 255), 2, cv2.LINE_AA)
        for index, point in enumerate(pixels):
            radius = 5 if index in (0, 4, 8, 12, 16, 20) else 3
            cv2.circle(annotated, point, radius, (39, 224, 190), -1, cv2.LINE_AA)

        features = normalize_landmarks(
            [coordinate for point in hand for coordinate in (point.x, point.y, point.z)]
        )
        probabilities = model.predict_proba([features])[0]
        best_index = int(np.argmax(probabilities))
        sign_text = str(model.classes_[best_index])
        confidence = float(probabilities[best_index])

    return annotated, sign_text, confidence


uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

image_bytes = None
if uploaded is not None:
    image_bytes = uploaded.getvalue()

if image_bytes:
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("The uploaded file could not be read as an image.")
    else:
        with st.spinner("Running prediction..."):
            annotated, sign_text, confidence = annotate_frame(frame)

        if sign_text:
            if sign_text == st.session_state.last_sign:
                st.session_state.sentence.append(sign_text)
            st.session_state.last_sign = sign_text

        st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted sign", sign_text or "No hand detected")
        with col2:
            st.metric("Confidence", f"{confidence:.2%}" if confidence else "0.00%")

        st.write("Current sentence:")
        st.write("".join(st.session_state.sentence[-20:]))

        if st.button("Clear sentence"):
            st.session_state.sentence = []
            st.session_state.last_sign = ""
else:
    st.info("Upload a photo or use the webcam to begin detection.")

st.markdown("### Deployment note")
st.info(
    "This Streamlit version is tuned for cloud deployment and uses uploaded images rather than a local webcam."
)
