"""Data loading and preprocessing utilities."""
from typing import Tuple
import re
import pandas as pd


def load_dataset(path: str, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Load CSV dataset and ensure required columns exist.

    Args:
        path: Path to CSV file.
        text_col: Column name containing text.
        label_col: Column name containing labels.

    Returns:
        Cleaned pandas DataFrame.
    """
    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"CSV must contain '{text_col}' and '{label_col}' columns")
    df = df[[text_col, label_col]].copy()
    df = df.dropna(subset=[text_col, label_col])
    df[text_col] = df[text_col].astype(str)
    df[label_col] = df[label_col].astype(str)
    return df


def clean_text(text: str) -> str:
    """Basic text cleaner: lowercases, removes urls, non-alphanumerics.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_dataframe(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """Apply cleaning to dataframe and return a copy with cleaned text.

    Args:
        df: Input DataFrame.
        text_col: Name of text column.
        label_col: Name of label column.

    Returns:
        DataFrame with `clean_text` applied as `text_clean`.
    """
    out = df.copy()
    out["text_clean"] = out[text_col].fillna("").astype(str).apply(clean_text)
    out = out[out["text_clean"].str.len() > 0]
    return out
