from src.preprocessing import clean_text, load_dataset, preprocess_dataframe
import pandas as pd


def test_clean_text():
    s = "Hello WORLD!! Visit https://example.com"
    out = clean_text(s)
    assert "hello" in out and "visit" in out


def test_preprocess_dataframe(tmp_path):
    p = tmp_path / "data.csv"
    df = pd.DataFrame({"text": ["Hi there", None], "label": ["greet", "x"]})
    df.to_csv(p, index=False)
    df2 = load_dataset(str(p))
    out = preprocess_dataframe(df2)
    assert "text_clean" in out.columns
