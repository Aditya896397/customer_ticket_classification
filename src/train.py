"""Training utilities for baseline and Transformer models."""
from typing import Tuple, Dict, Any
import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


def stratified_split(df, label_col: str = "label", test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42):
    """Return train/val/test stratified splits.

    Args:
        df: DataFrame with label column.
    """
    # If any class has fewer than 2 samples, stratified split will fail; fall back to random split.
    counts = df[label_col].value_counts()
    if counts.min() < 2:
        # fallback to non-stratified split
        train_val, test = train_test_split(df, test_size=test_size, random_state=random_state)
        rel_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=rel_val, random_state=random_state)
    else:
        train_val, test = train_test_split(df, test_size=test_size, stratify=df[label_col], random_state=random_state)
        rel_val = val_size / (1 - test_size)
        train, val = train_test_split(train_val, test_size=rel_val, stratify=train_val[label_col], random_state=random_state)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def train_baselines(train_df, val_df, text_col: str = "text_clean", label_col: str = "label", out_dir: str = "models") -> Dict[str, Any]:
    """Train logistic regression and random forest baselines with TF-IDF.

    Saves models to `out_dir` and returns metrics.
    """
    os.makedirs(out_dir, exist_ok=True)
    X_train = train_df[text_col].values
    y_train = train_df[label_col].values
    X_val = val_df[text_col].values
    y_val = val_df[label_col].values

    pipeline = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)), ("clf", LogisticRegression(max_iter=1000))])
    param_grid = {"clf__C": [0.1, 1.0, 5.0]}
    # adapt cv for small datasets
    import pandas as _pd
    min_count = _pd.Series(y_train).value_counts().min()
    if min_count >= 3:
        cv_splits = 3
    elif min_count >= 2:
        cv_splits = 2
    else:
        cv_splits = 0
    if cv_splits >= 2:
        gs = GridSearchCV(pipeline, param_grid, cv=cv_splits, n_jobs=2, verbose=1)
        gs.fit(X_train, y_train)
        best_lr = gs.best_estimator_
    else:
        # not enough samples per class for CV; fit default pipeline
        pipeline.fit(X_train, y_train)
        best_lr = pipeline
    joblib.dump(best_lr, os.path.join(out_dir, "baseline_logreg.joblib"))
    preds = best_lr.predict(X_val)
    report_lr = classification_report(y_val, preds, output_dict=True)

    rf_pipe = Pipeline([("tfidf", TfidfVectorizer(max_features=20000)), ("clf", RandomForestClassifier(n_jobs=2))])
    rf_params = {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 20]}
    if cv_splits >= 2:
        gs_rf = GridSearchCV(rf_pipe, rf_params, cv=cv_splits, n_jobs=2, verbose=1)
        gs_rf.fit(X_train, y_train)
        best_rf = gs_rf.best_estimator_
    else:
        rf_pipe.fit(X_train, y_train)
        best_rf = rf_pipe
    joblib.dump(best_rf, os.path.join(out_dir, "baseline_rf.joblib"))
    preds_rf = best_rf.predict(X_val)
    report_rf = classification_report(y_val, preds_rf, output_dict=True)

    return {"logreg": report_lr, "random_forest": report_rf}


def fine_tune_transformer():
    """Placeholder for Transformer fine-tuning using HuggingFace Trainer.

    Implement dataset conversion to `datasets.Dataset`, tokenization, Trainer args, and logging with MLflow.
    """
    raise NotImplementedError("Transformer fine-tuning function is a scaffold; implement training loop using HF Trainer.")
