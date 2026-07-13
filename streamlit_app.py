import json
import pickle
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "backend" / "model" / "sign_model.pkl"
METADATA_FILE = BASE_DIR / "backend" / "model" / "metadata.json"

st.set_page_config(page_title="Sign Language Detector", page_icon="🤟", layout="wide")

if not MODEL_FILE.exists():
    st.error(f"Model file not found: {MODEL_FILE}")
    st.stop()

metadata: dict[str, Any] = {}
if METADATA_FILE.exists():
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))

st.title("🤟 Sign Language Detector")
st.caption("This deployment uses the trained model metadata and is ready for image-based inference.")

st.info(
    "The previous detector initialization path was too heavy for Streamlit Cloud. "
    "This version keeps the app lightweight and focuses on a stable deployment shell."
)

st.write("Model metadata:")
st.json(metadata)

st.markdown("### Next step")
st.write("Upload a sample image in a future version to run actual sign classification.")
