import joblib
import pandas as pd
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "attention_model.pkl"
_model = None


def _get_model():
    global _model
    if _model is not None:
        return _model
    if not MODEL_PATH.exists():
        from src.model.train import train_model
        train_model()
    _model = joblib.load(MODEL_PATH)
    return _model


def predict_attention(theta, alpha, beta, heart_rate, motion_level):
    model = _get_model()
    data = pd.DataFrame([{
        "theta": theta,
        "alpha": alpha,
        "beta": beta,
        "heart_rate": heart_rate,
        "motion_level": motion_level
    }])

    prediction = model.predict(data)[0]

    probabilities = model.predict_proba(data)[0]

    confidence = max(probabilities) * 100

    return prediction, round(confidence, 1)


if __name__ == "__main__":
    prediction, confidence = predict_attention(
        theta=20,
        alpha=80,
        beta=70,
        heart_rate=75,
        motion_level=0.25
    )

    print("Prediction:", prediction)
    print("Confidence:", confidence, "%")
