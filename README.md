# NEUROSENSE-MULTI FUNCTIONAL BRAIN ATTENTION SYSTEM

# 🧠 NeuroSense

> **Academic Research Prototype**: Multi-sensor cognitive attention estimation, browser-based contactless webcam monitoring, and pediatric EEG seizure research.

---

## 📌 Project Overview

**NeuroSense** is a comprehensive brain-computer interface (BCI) research and monitoring platform designed to explore cognitive focus, mental fatigue, and electroencephalogram (EEG) signal processing. It combines physiological signal modeling, machine learning with explainable AI (XAI), and edge-based computer vision.

### Key Objectives
* **Cognitive Attention Tracking**: Classifies mental focus states (`Focused`, `Moderate/Neutral`, `Distracted`) using frequency band powers extracted from EEG alongside heart rate and motion levels.
* **Calibrated Machine Learning**: Uses Sigmoid-calibrated XGBoost to output reliable probability distributions, paired with SHAP (SHapley Additive exPlanations) for interpretability.
* **Contactless Behavioral Tracking**: Uses client-side computer vision (MediaPipe 478-point Face Mesh) to calculate Eye Aspect Ratio (EAR) for blink rate, eye closure duration, and head orientation—without uploading video frames.
* **Clinical EEG Research Gateway**: Implements a fail-closed, governance-driven framework for evaluating pediatric epilepsy recordings from the CHB-MIT PhysioNet cohort.

---

## 🏗️ Architecture & Technology Stack

```
                                  NEUROSENSE
                                      │
     ┌────────────────────────────────┼────────────────────────────────┐
     ▼                                ▼                                ▼
[Track 1: Attention ML]      [Track 2: Webcam Vision]     [Track 3: Seizure Research]
• Butterworth bandpass       • MediaPipe Face Mesh (478)  • CHB-MIT pediatric cohort
• PSD (Theta, Alpha, Beta)   • Eye Aspect Ratio (EAR)     • Patient-disjoint splits
• Sensor Fusion Heuristic    • Blink & fatigue tracking   • Signal quality gates
• Calibrated XGBoost + SHAP  • Contactless rPPG pulse     • Retrospective evaluation
```

* **Frontend**: Streamlit 1.x / custom interactive JavaScript components.
* **Backend API**: FastAPI, Pydantic v2, Uvicorn.
* **Database & Auth**: SQLite, Argon2id password hashing, user-scoped session tokens.
* **Signal Processing & ML**: SciPy, NumPy, Scikit-Learn, XGBoost, SHAP, Joblib.
* **Computer Vision**: Client-side MediaPipe Face Mesh in vanilla JavaScript (100% on-device).

---

## 🚀 Quick Start Guide

### 1. Prerequisites
* Python 3.10 – 3.12
* Recommended: Virtual environment (`.venv`)

### 2. Environment Setup
Activate your virtual environment and install the required packages:
```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 3. Generate ML Model Weights
To initialize the attention models on the synthetic benchmark dataset:
```powershell
python -m src.model.train
python -m src.model.train_xgboost
python -m src.model.calibrate_xgboost
```

### 4. Run the Streamlit Dashboard
Launch the primary user interface:
```powershell
streamlit run dashboard/app.py
```
* Access the app in your browser at `http://localhost:8501`.
* Create an account on the **Sign Up** tab, or log in with your credentials.

### 5. Run the Backend API (Optional)
To run the standalone FastAPI service:
```powershell
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000 --reload
```
* Interactive API documentation is available at `http://localhost:8000/docs`.

---

## 📂 Repository Directory Map

```
NeuroSense/
├── backend/                  # FastAPI backend service
│   ├── api/main.py           # REST endpoints (/auth, /predict, /sessions)
│   ├── schemas/              # Pydantic request/response models
│   ├── security/             # Argon2 password hashing and token generation
│   ├── services/             # ML and sensor inference services
│   └── storage/              # SQLite database and repository layers
├── dashboard/                # Streamlit web dashboard
│   ├── app.py                # Main authenticated dashboard router
│   ├── seizure_lab.py        # EEG Seizure Research workspace
│   ├── webcam.py             # Bi-directional webcam preview component
│   └── static/               # Client-side JavaScript (MediaPipe, EAR, rPPG)
├── data/
│   ├── database/             # SQLite database (neurosense.db)
│   └── processed/            # Benchmark feature training datasets
├── docs/                     # Technical specifications and checkpoint reports
├── models/                   # Serialized ML models (XGBoost, calibrated estimators)
├── seizure_research/         # CHB-MIT ingestion, partitioning, and audit tools
├── src/                      # Core acquisition, filtering, and model training
│   ├── acquisition/          # Synthetic EEG, heart rate, and motion generators
│   ├── features/             # Band power extraction and sensor fusion logic
│   ├── model/                # Model definitions, calibration, and SHAP explainers
│   └── preprocessing/        # Butterworth bandpass filters
└── tests/                    # Comprehensive unit and integration test suite
```

---

## 🔬 Academic & Medical Disclaimer

> **IMPORTANT**: NeuroSense is an **academic research prototype and educational software framework**. It is **not** a certified medical device and is **not** intended for clinical diagnosis, patient monitoring, treatment, or emergency response. Simulated signals and synthetic benchmark outputs do not reflect real physiological conditions.
