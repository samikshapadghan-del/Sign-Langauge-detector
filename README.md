# ASL Sign Language Detector

An accessible American Sign Language alphabet detector that converts live hand signs into text and speech.

## Features

- Live ASL A-Z alphabet recognition from 21 MediaPipe hand landmarks
- Wrist-relative normalized classifier with confidence and stability filtering
- Sentence builder with duplicate-sign release protection
- Word completion suggestions and common communication phrases
- Text-to-speech, optional automatic speech, copy, and transcript download
- Pause/resume, fullscreen camera, adjustable confidence, stability, and cooldown
- Responsive dark/light glass dashboard with detection and capture animations
- Camera and model health endpoint at `http://localhost:5000/api/health`

## Run

From PowerShell in this folder:

```powershell
.\run.ps1
```

Then open `http://localhost:5000`.

If PowerShell blocks local scripts for the current terminal, run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run.ps1
```

## Manual Setup

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements.txt
cd backend
..\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 5000
```

## Training

The app includes your ASL alphabet poster at `frontend/assets/asl_alphabet.jpg`.

A single poster image is a reference, not a full training dataset. For reliable detection, collect webcam landmarks of your own hand while matching the poster:

```powershell
cd "C:\Users\lanke\OneDrive\Desktop\Sign Language Detector (Project2)\backend"
..\.venv\Scripts\python.exe train_asl_from_webcam.py --samples 100
```

Controls in the training window:

- `SPACE`: start/pause recording for the current letter
- `N`: skip to the next letter
- `Q`: quit

After the script finishes, restart the app with `.\run.ps1`.

You can also retrain from the existing compact landmark file with:

```powershell
cd backend
..\.venv\Scripts\python.exe train_model.py
```

The removed `own-data-preprocessed` image dataset is not required at runtime.

## Scope and Accuracy

This version recognizes static ASL alphabet signs and builds words from those letters. Full continuous ASL word or sentence recognition requires a separate, labeled temporal video dataset and a sequence model; the current project does not claim that capability. Dynamic letters like J and Z are harder because they involve motion.

The included validation score is measured on held-out frames from the same capture dataset. Real-world accuracy varies by signer, lighting, camera angle, and hand position.
