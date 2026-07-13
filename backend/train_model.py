"""Train the alphabet classifier from the compact landmark dataset."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import FEATURE_COUNT, normalize_batch


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DATA_FILE = PROJECT_DIR / "data" / "landmarks.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_FILE = MODEL_DIR / "sign_model.pkl"
LABELS_FILE = MODEL_DIR / "labels.pkl"
METADATA_FILE = MODEL_DIR / "metadata.json"


def load_dataset() -> tuple[pd.DataFrame, object, object]:
    frame = pd.read_csv(DATA_FILE, header=None)
    expected_columns = FEATURE_COUNT + 1
    if frame.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns in {DATA_FILE}, found {frame.shape[1]}."
        )

    frame[0] = frame[0].astype(str).str.strip().str.upper()
    frame = frame[frame[0].str.fullmatch(r"[A-Z]")].dropna()
    features = normalize_batch(frame.iloc[:, 1:].to_numpy(dtype="float32"))
    labels = frame.iloc[:, 0].to_numpy()
    return frame, features, labels


def train() -> None:
    frame, features, labels = load_dataset()
    if len(set(labels)) != 26:
        raise ValueError("Training data must contain all 26 alphabet classes.")

    # Every fifth sample is held out so the result is deterministic and repeatable.
    test_mask = frame.groupby(0).cumcount().to_numpy() % 5 == 0
    model = ExtraTreesClassifier(
        n_estimators=350,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(features[~test_mask], labels[~test_mask])

    predictions = model.predict(features[test_mask])
    accuracy = float(accuracy_score(labels[test_mask], predictions))
    report = classification_report(
        labels[test_mask], predictions, output_dict=True, zero_division=0
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with MODEL_FILE.open("wb") as handle:
        pickle.dump(model, handle)
    with LABELS_FILE.open("wb") as handle:
        pickle.dump(sorted(set(labels)), handle)

    metadata = {
        "project": "ASL Sign Language Detector",
        "model": type(model).__name__,
        "classes": sorted(set(labels)),
        "samples": int(len(labels)),
        "features": FEATURE_COUNT,
        "normalized": True,
        "validation_accuracy": accuracy,
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "note": "Validation uses held-out frames from the same capture dataset.",
    }
    METADATA_FILE.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Trained {metadata['model']} on {metadata['samples']} samples.")
    print(f"Validation accuracy: {accuracy:.2%}")
    print(f"Saved model to {MODEL_FILE}")


if __name__ == "__main__":
    train()
