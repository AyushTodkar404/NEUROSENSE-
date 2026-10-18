from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "calibrated_xgboost_synthetic_1m.pkl"
ENCODER_PATH = (
    ROOT
    / "models"
    / "calibrated_xgboost_synthetic_1m_label_encoder.pkl"
)
FALLBACK_MODEL_PATH = ROOT / "models" / "calibrated_xgboost_model.pkl"
FALLBACK_ENCODER_PATH = ROOT / "models" / "calibrated_label_encoder.pkl"


class MLService:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.model_version = "calibrated_xgboost_v1"

    def _ensure_loaded(self):
        if self.model is not None and self.encoder is not None:
            return

        if MODEL_PATH.exists() and ENCODER_PATH.exists():
            self.model = joblib.load(MODEL_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
            self.model_version = "synthetic-1m-calibrated-v1"
        else:
            if not FALLBACK_MODEL_PATH.exists() or not FALLBACK_ENCODER_PATH.exists():
                from src.model.calibrate_xgboost import train_and_calibrate
                self.model, self.encoder = train_and_calibrate()
            else:
                self.model = joblib.load(FALLBACK_MODEL_PATH)
                self.encoder = joblib.load(FALLBACK_ENCODER_PATH)
            self.model_version = "calibrated_xgboost_v1"

    def predict(
        self,
        theta,
        alpha,
        beta,
        heart_rate,
        motion_level,
    ):
        self._ensure_loaded()
        features = pd.DataFrame(
            [[
                theta,
                alpha,
                beta,
                heart_rate,
                motion_level,
            ]],
            columns=[
                "theta",
                "alpha",
                "beta",
                "heart_rate",
                "motion_level",
            ],
        )

        probabilities = self.model.predict_proba(features)[0]

        probability_map = {
            label: float(probability)
            for label, probability in zip(
                self.encoder.classes_,
                probabilities,
            )
        }

        focused = probability_map.get("Focused", 0.0)
        neutral = probability_map.get("Neutral", 0.0)
        distracted = probability_map.get("Distracted", 0.0)

        predicted_index = int(probabilities.argmax())
        state = self.encoder.inverse_transform(
            [predicted_index]
        )[0]

        confidence = float(probabilities[predicted_index]) * 100

        # Blueprint attention score
        attention_score = 100 * (
            focused + 0.5 * neutral
        )

        return {
            "state": state,
            "confidence": round(confidence, 1),
            "attention_score": round(attention_score, 1),
            "focused_probability": focused,
            "neutral_probability": neutral,
            "distracted_probability": distracted,
            "model_version": self.model_version,
        }


ml_service = MLService()
