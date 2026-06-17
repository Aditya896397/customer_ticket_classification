"""Explanation utilities using SHAP."""
from typing import Any, List
import shap
import numpy as np


def explain_text_model(predict_fn, texts: List[str], nsamples: int = 100) -> Any:
    """Generate SHAP explanations for a text prediction function.

    Args:
        predict_fn: Callable that accepts list[str] and returns probability array.
        texts: List of texts to explain.
        nsamples: SHAP sampling parameter.
    Returns:
        SHAP explanation object.
    """
    # Use a KernelExplainer or Partition explainer depending on model speed.
    # Here we use a simple KernelExplainer over a small background sample.
    background = texts[:min(50, len(texts))]
    explainer = shap.Explainer(lambda x: predict_fn(x), shap.maskers.Text(), output_names=None)
    shap_values = explainer(texts)
    return shap_values
