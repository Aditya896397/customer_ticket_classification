"""Streamlit dashboard for model inspection and prediction."""
import streamlit as st
import pandas as pd
from typing import Any
from src.predict import load_baseline_model, load_transformer_model, predict_baseline, predict_transformer
import os

st.set_page_config(page_title="Ticket Classifier Dashboard")

st.title("Intelligent Ticket Classifier")

uploaded = st.file_uploader("Upload CSV with `text` and `label` columns", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    st.write(df.head())

st.subheader("Real-time prediction")
text = st.text_area("Ticket text")
model_choice = st.selectbox("Model", ["baseline", "transformer"])
if st.button("Predict") and text:
    if model_choice == "baseline":
        path = os.path.join("models", "baseline_logreg.joblib")
        if os.path.exists(path):
            clf = load_baseline_model(path)
            out = predict_baseline(clf, [text])
            st.write(out)
        else:
            st.error("Baseline model not found")
    else:
        td = os.path.join("models", "best_transformer")
        if os.path.exists(td):
            tokenizer, model = load_transformer_model(td)
            out = predict_transformer(tokenizer, model, [text])
            st.write(out)
        else:
            st.error("Transformer model not found")

st.markdown("---")
st.write("Dashboard features: upload CSV, compare models, view SHAP (coming).")
