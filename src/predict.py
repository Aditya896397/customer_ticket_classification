"""Prediction helpers for serving models."""
from typing import List, Dict, Any
import os
import joblib
import numpy as np

def load_baseline_model(path: str):
    return joblib.load(path)


def load_transformer_model(model_dir: str, device: str = "cpu"):
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch
    except Exception as e:
        raise RuntimeError("Transformers and torch are required to load transformer models") from e
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    return tokenizer, model


def predict_baseline(model, texts: List[str]) -> Dict[str, Any]:
    probs = model.predict_proba(texts)
    classes = model.classes_
    top_idx = probs.argmax(axis=1)
    return {"predictions": classes[top_idx].tolist(), "probs": probs.tolist()}


def predict_transformer(tokenizer, model, texts: List[str], device: str = "cpu") -> Dict[str, Any]:
    try:
        import torch
    except Exception:
        raise RuntimeError("torch is required for transformer predictions")
    enc = tokenizer(texts, truncation=True, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
        logits = out.logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
    labels = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(probs.shape[1])}
    pred_labels = [labels[int(p)] for p in preds]
    return {"predictions": pred_labels, "probs": probs.tolist()}
