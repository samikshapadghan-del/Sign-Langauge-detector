import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

DATA_FILE = "../data/landmarks.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_FILE, header=None, low_memory=False)

    # Keep only rows where label is a letter (A-Z)
    df = df[df.iloc[:, 0].apply(lambda x: str(x).strip().isalpha())]

    # Drop rows with any missing values
    df = df.dropna()

    # Force all landmark columns to numeric, drop bad rows
    df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce')
    df = df.dropna()

    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    print(f"Total samples: {len(X)}")
    print(f"Signs found: {sorted(set(y))}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training model... (may take 2-5 mins)")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"Accuracy: {accuracy * 100:.2f}%")

    with open(f"{MODEL_DIR}/sign_model.pkl", "wb") as f:
        pickle.dump(model, f)

    labels = sorted(set(y))
    with open(f"{MODEL_DIR}/labels.pkl", "wb") as f:
        pickle.dump(labels, f)

    print("Model saved to model/sign_model.pkl")

if __name__ == "__main__":
    train()