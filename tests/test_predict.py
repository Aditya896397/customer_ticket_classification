from src.predict import predict_baseline
from sklearn.dummy import DummyClassifier
import numpy as np


def test_predict_baseline():
    clf = DummyClassifier(strategy="most_frequent")
    X = ["a", "b", "c"]
    y = ["x", "x", "x"]
    clf.fit(X, y)
    out = predict_baseline(clf, ["a test"])
    assert "predictions" in out and "probs" in out
