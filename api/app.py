"""FastAPI application for prediction."""
from typing import List, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from src.predict import load_baseline_model, load_transformer_model, predict_baseline, predict_transformer
import os

app = FastAPI(title="Ticket Classifier API")


class PredictRequest(BaseModel):
    text: str
    model: str = "best"


@app.post("/predict")
def predict(req: PredictRequest) -> Dict[str, Any]:
    text = req.text
    # Simple loader: if a transformer model exists at models/best_transformer, use it.
    transformer_dir = os.path.join("models", "best_transformer")
    baseline_path = os.path.join("models", "baseline_logreg.joblib")
    if req.model == "transformer" and os.path.exists(transformer_dir):
        tokenizer, model = load_transformer_model(transformer_dir)
        out = predict_transformer(tokenizer, model, [text])
        return {"category": out["predictions"][0], "probs": out["probs"][0]}
    if os.path.exists(baseline_path):
        clf = load_baseline_model(baseline_path)
        out = predict_baseline(clf, [text])
        return {"category": out["predictions"][0], "probs": out["probs"][0]}
    return {"error": "No model available"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
