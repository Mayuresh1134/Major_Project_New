"""Train and evaluate the genre classifier.

Input: the GTZAN feature tables from the Kaggle dataset "GTZAN Dataset - Music Genre
Classification" (features_3_sec.csv and features_30_sec.csv). They are not included in
this repository; download them and pass their paths:

    python -m ml.train_genre path/to/features_3_sec.csv path/to/features_30_sec.csv

Every score is from 5-fold cross-validation grouped by song: all ten 3-second slices of a
song are in the same fold, so the model is always tested on songs it never saw.
Writes models/genre_mlp.npz (weights for numpy) and models/genre_metrics.json.
Needs scikit-learn and pandas (pip install -r requirements-train.txt).
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from museassist.genre import FEATURES, MODELS  # noqa: E402

warnings.filterwarnings("ignore")
SEEDS = (0, 1, 2)
MLP_ARGS = dict(hidden_layer_sizes=(256, 128), alpha=1e-2, max_iter=300, early_stopping=True)


class Ensemble:
    """A StandardScaler shared by several MLPs whose probabilities are averaged."""

    def __init__(self, seeds=SEEDS):
        self.seeds = seeds

    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.nets = [MLPClassifier(random_state=s, **MLP_ARGS).fit(Xs, y) for s in self.seeds]
        self.classes_ = self.nets[0].classes_
        return self

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        return np.mean([m.predict_proba(Xs) for m in self.nets], axis=0)


def song_of(filename: str) -> str:
    return ".".join(filename.split(".")[:2])  # blues.00004.3.wav -> blues.00004


def evaluate(make_model, df, cols, by_song: bool):
    X, y, g = df[cols].values, df.label.values, df.song.values
    folds = StratifiedGroupKFold(5, shuffle=True, random_state=0)
    accs, y_true, y_pred = [], [], []
    for tr, te in folds.split(X, y, g):
        m = make_model().fit(X[tr], y[tr])
        P = pd.DataFrame(m.predict_proba(X[te]), columns=m.classes_)
        if by_song:
            P["song"], P["y"] = g[te], y[te]
            P = P.groupby("song").agg({**{c: "mean" for c in m.classes_}, "y": "first"})
            truth = P.pop("y").values
        else:
            truth = y[te]
        pred = np.array(m.classes_)[P[list(m.classes_)].values.argmax(axis=1)]
        accs.append(float(np.mean(pred == truth)))
        y_true += list(truth)
        y_pred += list(pred)
    return float(np.mean(accs)), float(np.std(accs)), np.array(y_true), np.array(y_pred)


def main(path3: str, path30: str | None = None):
    d3 = pd.read_csv(path3)
    d3["song"] = d3.filename.map(song_of)
    rows = []
    if path30:
        d30 = pd.read_csv(path30)
        d30["song"] = d30.filename.map(song_of)
        mf13 = [f"mfcc{i}_mean" for i in range(1, 14)]
        old = lambda: make_pipeline(StandardScaler(), MLPClassifier(  # noqa: E731
            (256, 128, 64), max_iter=600, early_stopping=True, random_state=0))
        a, s, *_ = evaluate(old, d30, mf13, by_song=False)
        rows.append({"model": "13 MFCC averages over 30 s, neural network (the first version's approach)",
                     "accuracy": a, "std": s})
        print(f"{rows[-1]['model']}: {a:.3f} ± {s:.3f}", flush=True)
    logit = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))  # noqa: E731
    a, s, *_ = evaluate(logit, d3, FEATURES, by_song=True)
    rows.append({"model": "57 features per 3 s slice, logistic regression, averaged per song",
                 "accuracy": a, "std": s})
    print(f"{rows[-1]['model']}: {a:.3f} ± {s:.3f}", flush=True)
    t = time.time()
    a, s, yt, yp = evaluate(Ensemble, d3, FEATURES, by_song=True)
    rows.append({"model": "57 features per 3 s slice, 3 neural networks, averaged per song (used in the app)",
                 "accuracy": a, "std": s})
    print(f"{rows[-1]['model']}: {a:.3f} ± {s:.3f} ({time.time() - t:.0f}s)", flush=True)

    classes = sorted(d3.label.unique())
    cm = confusion_matrix(yt, yp, labels=classes)
    per_genre = {c: round(float(cm[i, i] / cm[i].sum()), 3) for i, c in enumerate(classes)}

    final = Ensemble().fit(d3[FEATURES].values, d3.label.values)
    arrays = {"classes": np.array(final.classes_), "mean": final.scaler.mean_,
              "scale": final.scaler.scale_, "n_nets": np.array(len(final.nets))}
    for k, net in enumerate(final.nets):
        for j, (W, b) in enumerate(zip(net.coefs_, net.intercepts_)):
            arrays[f"net{k}_W{j}"] = W.astype(np.float32)
            arrays[f"net{k}_b{j}"] = b.astype(np.float32)
    MODELS.mkdir(exist_ok=True)
    np.savez_compressed(MODELS / "genre_mlp.npz", **arrays)
    report = {
        "dataset": "GTZAN (1,000 songs, 10 genres, 30 s each), features_3_sec.csv",
        "songs": int(d3.song.nunique()), "slices": int(len(d3)),
        "method": "5-fold cross-validation grouped by song",
        "comparison": [{**r, "accuracy": round(r["accuracy"], 3), "std": round(r["std"], 3)} for r in rows],
        "per_genre": per_genre,
        "labels": classes,
        "confusion": cm.tolist(),
    }
    (MODELS / "genre_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(per_genre, indent=1))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(*sys.argv[1:3])
