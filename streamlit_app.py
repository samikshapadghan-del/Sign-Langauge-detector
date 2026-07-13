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


@st.cache_resource
def load_pipeline() -> tuple[Any, dict[str, Any], Any]:
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")
    if not LANDMARKER_FILE.exists():
        raise FileNotFoundError(f"MediaPipe model not found: {LANDMARKER_FILE}")

    with MODEL_FILE.open("rb") as handle:
        model = pickle.load(handle)

    metadata: dict[str, Any] = {}
    if METADATA_FILE.exists():
        metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(LANDMARKER_FILE)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.25,
        min_hand_presence_confidence=0.25,
    )
    landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    return model, metadata, landmarker


def predict_from_image(model: Any, landmarker: Any, image: np.ndarray) -> tuple[str | None, float]:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    if not result.hand_landmarks:
        return None, 0.0

    hand = result.hand_landmarks[0]
    values = [coordinate for point in hand for coordinate in (point.x, point.y, point.z)]
    features = normalize_landmarks(values)
    probabilities = model.predict_proba([features])[0]
    best_index = int(np.argmax(probabilities))
    sign = str(model.classes_[best_index])
    confidence = float(probabilities[best_index])
    return sign, confidence


def update_sentence(
    sentence: list[str],
    sign: str | None,
    confidence: float,
    threshold: float,
    last_sign: str,
    stable_count: int,
) -> tuple[list[str], str, int]:
    if sign is None or confidence < threshold:
        return sentence, "", 0

    if sign == last_sign:
        stable_count += 1
    else:
        stable_count = 1
        last_sign = sign

    if stable_count >= 3:
        if sentence and sentence[-1] != " ":
            sentence.append(" ")
        sentence.append(sign)
        stable_count = 0
        last_sign = ""

    return sentence, last_sign, stable_count


def main() -> None:
    st.title("🤟 Sign Language Detector")
    st.caption("This Streamlit deployment now uses the same trained model and landmark pipeline as the FastAPI backend.")

    try:
        model, metadata, landmarker = load_pipeline()
    except Exception as exc:  # pragma: no cover - handled in the UI
        st.error(f"Could not load detector: {exc}")
        st.stop()

    st.subheader("Model metadata")
    st.json(metadata)

    if "sentence" not in st.session_state:
        st.session_state.sentence = []
    if "last_sign" not in st.session_state:
        st.session_state.last_sign = ""
    if "stable_count" not in st.session_state:
        st.session_state.stable_count = 0

    uploaded_file = st.file_uploader("Upload an image for sign classification", type=["png", "jpg", "jpeg"])
    if uploaded_file is None:
        st.info("Upload a clear image with one hand visible to classify a sign.")
        return

    file_bytes = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        st.error("The uploaded image could not be decoded.")
        return

    col_left, col_right = st.columns(2)
    with col_left:
        st.image(image, channels="BGR", caption="Uploaded image")

    with col_right:
        sign, confidence = predict_from_image(model, landmarker, image)
        if sign is None:
            st.warning("No hand detected. Try a clearer image with one hand fully visible.")
        else:
            st.metric("Predicted sign", sign)
            st.metric("Confidence", f"{confidence:.0%}")
            st.session_state.sentence, st.session_state.last_sign, st.session_state.stable_count = update_sentence(
                st.session_state.sentence,
                sign,
                confidence,
                0.25,
                st.session_state.last_sign,
                st.session_state.stable_count,
            )
            st.text_area("Current sentence", "".join(st.session_state.sentence), height=120)

    st.caption("This deployment uses image inference rather than a live webcam stream.")


if __name__ == "__main__":
    main()
