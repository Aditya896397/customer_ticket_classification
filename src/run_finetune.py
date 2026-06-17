"""Fine-tune BERT and RoBERTa on the provided CSV and save models.

This script performs quick CPU fine-tuning with 1 epoch for demo purposes.
"""
from typing import Dict
import os
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.preprocessing import load_dataset, preprocess_dataframe


def prepare_dataset(df, text_col="text_clean", label_col="label"):
    labels = sorted(df[label_col].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    ds = Dataset.from_pandas(df[[text_col, label_col]].rename(columns={text_col: "text", label_col: "label"}))
    ds = ds.map(lambda x: {"label": label2id[x["label"]]}, remove_columns=[])  # ensure label numeric
    return ds, labels


def tokenize_dataset(tokenizer, ds):
    def fn(ex):
        return tokenizer(ex["text"], truncation=True, padding=True)
    return ds.map(fn, batched=False)


def finetune_model(model_name: str, ds_train, ds_val, labels, out_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(labels))
    tokenized_train = ds_train.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)
    tokenized_val = ds_val.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length", max_length=128), batched=True)

    tokenized_train = tokenized_train.remove_columns([col for col in tokenized_train.column_names if col not in ["input_ids","attention_mask","label"]])
    tokenized_val = tokenized_val.remove_columns([col for col in tokenized_val.column_names if col not in ["input_ids","attention_mask","label"]])

    tokenized_train.set_format(type="torch")
    tokenized_val.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir=out_dir,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=10,
        learning_rate=2e-5,
        fp16=False,
        disable_tqdm=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
    )

    trainer.train()
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)


def main():
    data_path = os.path.join("data", "sample_tickets.csv")
    df = load_dataset(data_path, text_col="text", label_col="label")
    dfp = preprocess_dataframe(df, text_col="text", label_col="label")
    # split manually: 70/20/10
    df_shuf = dfp.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df_shuf)
    n_train = max(1, int(0.7 * n))
    n_val = max(1, int(0.2 * n))
    train = df_shuf.iloc[:n_train]
    val = df_shuf.iloc[n_train:n_train + n_val]
    print(f"Train {len(train)}, Val {len(val)}")

    ds_train, labels = prepare_dataset(train)
    ds_val, _ = prepare_dataset(val)

    models = {
        "bert": "prajjwal1/bert-tiny",
        "roberta": "distilroberta-base",
    }

    best_score = -1.0
    best_dir = None

    for name, model_id in models.items():
        out_dir = os.path.join("models", name)
        os.makedirs(out_dir, exist_ok=True)
        print(f"Fine-tuning {name} ({model_id}) -> {out_dir}")
        try:
            finetune_model(model_id, ds_train, ds_val, labels, out_dir)
            # crude evaluation: load model and predict on val
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            tok = AutoTokenizer.from_pretrained(out_dir)
            m = AutoModelForSequenceClassification.from_pretrained(out_dir)
            m.eval()
            inputs = tok(list(val["text"].astype(str)), truncation=True, padding=True, return_tensors="pt")
            with torch.no_grad():
                out = m(**{k: v for k, v in inputs.items()})
                probs = torch.softmax(out.logits, dim=1).numpy()
                preds = probs.argmax(axis=1)
            acc = (preds == np.array([labels.index(l) for l in val["label"]])).mean() if len(val) > 0 else 0.0
            print(f"Validation acc for {name}: {acc}")
            if acc > best_score:
                best_score = acc
                best_dir = out_dir
        except Exception as e:
            print(f"Failed to fine-tune {name}: {e}")

    if best_dir:
        # copy best to models/best_transformer
        import shutil
        dst = os.path.join("models", "best_transformer")
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(best_dir, dst)
        print(f"Best transformer copied to {dst}")
    else:
        print("No transformer models trained successfully.")


if __name__ == "__main__":
    main()
