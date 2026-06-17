"""Script to run baseline training on a CSV dataset.

Usage: python -m src.run_train_baseline
"""
from src.preprocessing import load_dataset, preprocess_dataframe
from src.train import stratified_split, train_baselines
import os


def main():
    data_path = os.path.join("data", "sample_tickets.csv")
    print(f"Loading dataset from {data_path}")
    df = load_dataset(data_path, text_col="text", label_col="label")
    dfp = preprocess_dataframe(df, text_col="text", label_col="label")
    train, val, test = stratified_split(dfp, label_col="label", test_size=0.2, val_size=0.1)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    out = train_baselines(train, val, text_col="text_clean", label_col="label", out_dir="models")
    print("Training completed. Reports:")
    for k, v in out.items():
        print(f"--- {k} ---")
        print(v)


if __name__ == "__main__":
    main()
